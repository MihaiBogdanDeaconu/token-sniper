from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scripts.train_vgib import (
    check_loaded_tasks,
    generate_candidates_via_retrieval,
    load_reranked_tasks,
    load_split_records,
)
from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.vectorstore import EntityVectorStore
from token_sniper.rl.rewards import LLMJudgeReward
from token_sniper.rl.trainer import GRPOConfig, GRPOTrainer, load_warm_start_policy
from token_sniper.vgib_selector.features import ChromaSourceEncoder

if TYPE_CHECKING:
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class RLData:
    train_tasks: list[RerankedVGIBTask]
    val_tasks: list[RerankedVGIBTask]
    base_commits: dict[str, str]
    train_dropped: dict[str, str]
    val_dropped: dict[str, str]
    load_time_s: float


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("configs/phases/phase0_tasks.json"),
        help="Phase 0 task manifest JSON.",
    )
    parser.add_argument(
        "--split-json", type=Path, required=True,
        help=(
            "Train/val split shared with the reranker and VGIB "
            "(configs/phases/reranker_split.json). The candidate records "
            "must match it exactly."
        ),
    )
    parser.add_argument(
        "--reranker-dir", type=Path, default=None,
        help=(
            "Directory with train_inputs.json + val_inputs.json from the "
            "reranker stage. Optional — when absent, candidates are generated "
            "on-the-fly via hybrid retrieval (requires ChromaDB + SSGs)."
        ),
    )
    parser.add_argument(
        "--ssg-dir", type=Path, default=Path("data/ssg"),
        help="Root directory of SSG JSON files (repo__name/commit12/ssg.json).",
    )
    parser.add_argument(
        "--chromadb-dir", type=Path, default=Path("data/chromadb"),
        help="ChromaDB persist directory with pre-indexed entity embeddings.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/rl_grpo"),
        help="Directory for checkpoints and summary.",
    )
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=None,
        help="Node feature cache, keyed by content. Keep it outside --output-dir so it "
             "survives a re-clone, and share it with train_vgib.py "
             "(default: <output-dir>/feature_cache).",
    )
    parser.add_argument(
        "--issue-cache-dir", type=Path, default=None,
        help="Issue embedding cache (default: <output-dir>/issue_cache).",
    )
    parser.add_argument("--dataset", default="SWE-bench-Live/SWE-bench-Live")
    parser.add_argument("--split", default="verified")
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    parser.add_argument(
        "--embedding-model", default="jinaai/jina-code-embeddings-0.5b",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)


def add_verified_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-config", type=Path, default=Path("configs/base.yaml"),
        help="YAML whose models.agent block configures the agent model and server "
             "(TOKEN_SNIPER_AGENT_MODEL / TOKEN_SNIPER_AGENT_BASE_URL override it).",
    )
    parser.add_argument(
        "--agent-max-turns", type=int, default=30,
        help="Agent turn budget per verified rollout (the Docker eval uses 100).",
    )
    parser.add_argument(
        "--verified-workers", type=int, default=8,
        help="Verified rollouts run concurrently (one agent container + one harness run each).",
    )
    parser.add_argument(
        "--verified-work-dir", type=Path, default=None,
        help="Per-rollout agent logs and harness reports (default: <output-dir>/verified_rollouts).",
    )
    parser.add_argument(
        "--verified-token-beta", type=float, default=0.0,
        help="Verified reward = pass - beta * token_ratio; 0 keeps it a pure pass/fail.",
    )
    parser.add_argument(
        "--harness-timeout", type=int, default=1800,
        help="SWE-bench harness per-instance timeout (seconds).",
    )


def load_rl_data(args: argparse.Namespace) -> RLData:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_phase0_manifest(args.manifest)
    task_lookup: dict[str, dict] = {
        t.instance_id: {
            "repo": t.repo,
            "base_commit": t.base_commit,
            "gold_patch_files": t.gold_patch_files,
        }
        for t in manifest.tasks
    }
    logger.info("Manifest: %d tasks loaded", len(task_lookup))

    embed_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        name=args.embedding_model,
        dimension=args.embedding_dim,
    )
    embedder = create_embedder(embed_cfg)
    logger.info("Embedder: %s (dim=%d)", args.embedding_model, embed_cfg.dimension)

    vector_store = EntityVectorStore(
        persist_directory=str(args.chromadb_dir),
        collection_prefix="token_sniper",
    )

    all_records: list[dict] = []

    if args.reranker_dir is not None:
        train_inputs_path = args.reranker_dir / "train_inputs.json"
        val_inputs_path = args.reranker_dir / "val_inputs.json"

        if train_inputs_path.exists() and val_inputs_path.exists():
            train_rec = json.loads(train_inputs_path.read_text(encoding="utf-8"))
            val_rec = json.loads(val_inputs_path.read_text(encoding="utf-8"))
            all_records = train_rec + val_rec
            logger.info(
                "Reranker outputs: %d train + %d val = %d total",
                len(train_rec), len(val_rec), len(all_records),
            )
        else:
            logger.warning(
                "--reranker-dir %s does not contain train/val inputs — "
                "falling back to retrieval",
                args.reranker_dir,
            )

    if not all_records:
        cached_path = output_dir / "retrieval_candidates.json"
        if cached_path.exists():
            logger.info("Found cached retrieval candidates at %s", cached_path)
            all_records = json.loads(cached_path.read_text(encoding="utf-8"))
        else:
            logger.info(
                "No reranker outputs or cache — generating candidates via hybrid retrieval. "
                "This requires ChromaDB at %s and SSGs at %s.",
                args.chromadb_dir, args.ssg_dir,
            )
            all_records = generate_candidates_via_retrieval(
                manifest_tasks=manifest.tasks,
                ssg_dir=args.ssg_dir,
                embedder=embedder,
                vector_store=vector_store,
                dataset_name=args.dataset,
                split=args.split,
            )
            cached_path.write_text(
                json.dumps(all_records, indent=2), encoding="utf-8",
            )
            logger.info(
                "Generated %d candidate records — cached to %s",
                len(all_records), cached_path,
            )

    if not all_records:
        raise SystemExit("No candidate records found. Check --ssg-dir and --chromadb-dir.")

    train_records, val_records = load_split_records(args.split_json, all_records)
    logger.info(
        "Split from %s: %d train, %d val",
        args.split_json, len(train_records), len(val_records),
    )

    encoder = ChromaSourceEncoder(
        chromadb_dir=args.chromadb_dir,
        collection_prefix="token_sniper",
        model_name=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )

    feature_cache_dir = args.feature_cache_dir or output_dir / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    issue_cache_dir = args.issue_cache_dir or output_dir / "issue_cache"
    issue_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Feature cache: %s | issue cache: %s", feature_cache_dir, issue_cache_dir)

    logger.info("=== Loading VGIB tasks ===")
    t0 = time.monotonic()
    loaded: dict[str, tuple[list[RerankedVGIBTask], dict[str, str]]] = {}
    for label, records in (("TRAIN", train_records), ("VAL", val_records)):
        tasks, dropped = load_reranked_tasks(
            reranker_inputs=records,
            task_lookup=task_lookup,
            ssg_dir=args.ssg_dir,
            encoder=encoder,
            embedder=embedder,
            feature_cache_dir=feature_cache_dir,
            issue_cache_dir=issue_cache_dir,
            dataset_name=args.dataset,
            split=args.split,
            label=label,
        )
        check_loaded_tasks(records, tasks, dropped, label)
        loaded[label] = (tasks, dropped)
    load_time = time.monotonic() - t0

    train_tasks, train_dropped = loaded["TRAIN"]
    val_tasks, val_dropped = loaded["VAL"]
    logger.info(
        "Task loading: %.1fs — %d train + %d val tasks",
        load_time, len(train_tasks), len(val_tasks),
    )
    if not train_tasks:
        raise SystemExit(
            "No training tasks loaded. Verify --ssg-dir, --chromadb-dir, and --manifest are correct.",
        )

    return RLData(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        base_commits={iid: info["base_commit"] for iid, info in task_lookup.items()},
        train_dropped=train_dropped,
        val_dropped=val_dropped,
        load_time_s=load_time,
    )


def build_verified_reward(
    args: argparse.Namespace,
    *,
    base_commits: dict[str, str],
) -> LLMJudgeReward:
    from token_sniper.agent.config import load_base_config
    from token_sniper.agent.providers import OpenAIChatModelClient
    from token_sniper.evaluation.config import HarnessConfig
    from token_sniper.evaluation.swebench import SWEbenchHarnessRunner
    from token_sniper.rl.verified import SWEBENCH_LIVE_NAMESPACE, DockerAgentBackend

    model_config = load_base_config(args.agent_config).models.agent
    logger.info(
        "Verified reward: agent %s at %s, %d turns, %d workers",
        model_config.name, model_config.base_url, args.agent_max_turns, args.verified_workers,
    )
    harness = SWEbenchHarnessRunner(HarnessConfig(
        dataset_name=args.dataset,
        split=args.split,
        max_workers=1,
        namespace=SWEBENCH_LIVE_NAMESPACE,
        timeout_seconds=args.harness_timeout,
    ))
    backend = DockerAgentBackend(
        client=OpenAIChatModelClient(model_config),
        model_config=model_config,
        max_turns=args.agent_max_turns,
        dataset_name=args.dataset,
        split=args.split,
    )
    return LLMJudgeReward(
        agent_backend=backend,
        harness=harness,
        model_name=model_config.name,
        ssg_dir=args.ssg_dir,
        base_commits=base_commits,
        work_dir=args.verified_work_dir or args.output_dir / "verified_rollouts",
        dataset_name=args.dataset,
        split=args.split,
        max_workers=args.verified_workers,
        token_beta=args.verified_token_beta,
    )


def subsample(tasks: list[RerankedVGIBTask], n: int | None, seed: int) -> list[RerankedVGIBTask]:
    if n is None:
        return list(tasks)
    if n > len(tasks):
        raise ValueError(f"asked for {n} tasks, only {len(tasks)} loaded")
    ordered = sorted(tasks, key=lambda t: t.instance_id)
    return random.Random(seed).sample(ordered, n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning of VGIB selector (Phase RL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(parser)

    parser.add_argument(
        "--warm-start", type=Path, required=True,
        help="Path to Phase C best.pt checkpoint (required).",
    )

    parser.add_argument(
        "--group-size", type=int, default=8,
        help="Number of rollouts per task per epoch (G in GRPO).",
    )
    parser.add_argument(
        "--clip-eps", type=float, default=0.2,
        help="PPO clipping epsilon — caps per-step policy change.",
    )
    parser.add_argument(
        "--kl-coeff", type=float, default=0.1,
        help="Initial KL penalty coefficient (adaptively adjusted).",
    )
    parser.add_argument(
        "--kl-target", type=float, default=0.01,
        help="Target per-node KL divergence from the reference (Phase C) policy.",
    )
    parser.add_argument(
        "--entropy-coeff", type=float, default=0.1,
        help="Entropy bonus coefficient — encourages exploration.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate (10x lower than Phase C supervised pretrain).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience (epochs without val improvement).",
    )

    parser.add_argument(
        "--reward", choices=("proxy", "verified"), default="proxy",
        help="Training reward: the zero-cost proxy, or the verified agent + FAIL_TO_PASS reward.",
    )
    parser.add_argument(
        "--reward-beta", type=float, default=0.5,
        help="Token compression penalty weight β.",
    )
    parser.add_argument(
        "--reward-trace-weight", type=float, default=0.1,
        help="Trace-edge connectivity bonus weight.",
    )
    parser.add_argument(
        "--reward-recall-bonus", type=float, default=0.5,
        help="Smooth joint quality bonus scale: recall_bonus * recall * (1 - token_ratio).",
    )
    add_verified_args(parser)
    parser.add_argument(
        "--max-train-tasks", type=int, default=None,
        help="Train on this many train-split tasks (seeded sample); all when omitted.",
    )
    parser.add_argument(
        "--max-val-tasks", type=int, default=None,
        help="Validate on this many val-split tasks (seeded sample); all when omitted.",
    )

    parser.add_argument(
        "--tau-start", type=float, default=1.0,
        help="Initial Gumbel-softmax temperature (exploration).",
    )
    parser.add_argument(
        "--tau-end", type=float, default=0.5,
        help="Final temperature after annealing (exploitation).",
    )

    parser.add_argument(
        "--rollout-temperature", type=float, default=2.0,
        help=(
            "Temperature during rollout sampling (higher → more diverse masks). "
            "Decoupled from the policy inference tau: only affects Bernoulli "
            "sampling diversity within each G-group."
        ),
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=32,
        help=(
            "Accumulate gradients over this many tasks before optimizer.step(). "
            "Reduces optimizer overhead and improves GPU utilisation."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device string. Auto-detected if omitted (cuda > mps > cpu).",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    data = load_rl_data(args)
    train_tasks = subsample(data.train_tasks, args.max_train_tasks, args.seed)
    loaded_val_tasks = subsample(data.val_tasks, args.max_val_tasks, args.seed)
    n_val_loaded = len(loaded_val_tasks)

    total_nodes = sum(len(t.features.node_ids) for t in train_tasks)
    total_gold = sum(int(t.gold_file_mask.sum().item()) for t in train_tasks)
    total_gold_nodes = sum(int(t.gold_node_mask.sum().item()) for t in train_tasks)
    logger.info(
        "Train dataset: %d tasks, %d nodes (avg %.0f/task), %d gold files",
        len(train_tasks),
        total_nodes,
        total_nodes / len(train_tasks),
        total_gold,
    )

    logger.info("=== Loading Phase C warm-start from %s ===", args.warm_start)
    if not args.warm_start.exists():
        raise SystemExit(f"Warm-start checkpoint not found: {args.warm_start}")
    policy = load_warm_start_policy(args.warm_start, device=args.device)

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        group_size=args.group_size,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        kl_target=args.kl_target,
        entropy_coeff=args.entropy_coeff,
        lr=args.lr,
        weight_decay=1e-5,
        epochs=args.epochs,
        patience=args.patience,
        grad_accum_steps=args.grad_accum_steps,
        reward_beta=args.reward_beta,
        reward_trace_weight=args.reward_trace_weight,
        reward_recall_bonus=args.reward_recall_bonus,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        rollout_temperature=args.rollout_temperature,
        seed=args.seed,
        device=args.device,
        mlflow_experiment="phase_rl_grpo",
        reward_source=args.reward,
    )

    verified_reward = (
        build_verified_reward(args, base_commits=data.base_commits)
        if args.reward == "verified" else None
    )
    trainer = GRPOTrainer(policy=policy, cfg=grpo_cfg, verified_reward=verified_reward)

    logger.info("=== Starting GRPO Training (reward=%s) ===", args.reward)
    t0 = time.monotonic()
    best_path = trainer.train(train_tasks, val_tasks=loaded_val_tasks)
    train_time = time.monotonic() - t0
    logger.info(
        "GRPO training complete in %.1fs — best checkpoint: %s",
        train_time, best_path,
    )

    summary: dict = {
        "mode": "grpo",
        "reward": args.reward,
        "warm_start": str(args.warm_start),
        "split_json": str(args.split_json) if args.split_json else None,
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": n_val_loaded,
        "total_train_nodes": total_nodes,
        "avg_nodes_per_task": round(total_nodes / len(train_tasks), 1),
        "total_gold_files": total_gold,
        "total_gold_nodes": total_gold_nodes,
        "dropped_without_gold_node": {"train": data.train_dropped, "val": data.val_dropped},
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "load_time_s": round(data.load_time_s, 1),
        "train_time_s": round(train_time, 1),
        "config": {
            "group_size": args.group_size,
            "clip_eps": args.clip_eps,
            "kl_coeff": args.kl_coeff,
            "kl_target": args.kl_target,
            "entropy_coeff": args.entropy_coeff,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "grad_accum_steps": args.grad_accum_steps,
            "reward_beta": args.reward_beta,
            "reward_trace_weight": args.reward_trace_weight,
            "reward_recall_bonus": args.reward_recall_bonus,
            "tau_start": args.tau_start,
            "tau_end": args.tau_end,
            "rollout_temperature": args.rollout_temperature,
            "seed": args.seed,
            "max_train_tasks": args.max_train_tasks,
            "max_val_tasks": args.max_val_tasks,
        },
        "best_checkpoint": str(best_path),
    }
    if args.reward == "verified":
        summary["verified"] = {
            "agent_config": str(args.agent_config),
            "agent_max_turns": args.agent_max_turns,
            "verified_workers": args.verified_workers,
            "verified_token_beta": args.verified_token_beta,
            "harness_timeout": args.harness_timeout,
        }

    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        summary["best_epoch"] = ckpt.get("epoch")
        summary["best_val_reward"] = ckpt.get("best_val_reward")

    (output_dir / "rl_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    logger.info("=== Phase RL Complete ===")
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
