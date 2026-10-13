from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from scripts.train_rl import (
    add_data_args,
    add_verified_args,
    build_verified_reward,
    load_rl_data,
    subsample,
)
from token_sniper.rl.rewards import ProxyReward
from token_sniper.rl.rollout import RolloutBuffer
from token_sniper.rl.trainer import disable_dropout, load_warm_start_policy

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "instance_id", "rollout_idx", "proxy_reward", "recall", "token_ratio",
    "verified", "resolved", "patch_produced", "patch_applied", "n_spans", "n_files",
    "context_chars", "prompt_tokens", "completion_tokens", "agent_wall_s", "harness_wall_s",
    "rollout_dir",
]


def sample_rollouts(
    *,
    checkpoint: Path,
    tasks: list,
    group_size: int,
    rollout_temperature: float,
    proxy: ProxyReward,
    seed: int,
    device: str | None,
) -> dict:
    policy = load_warm_start_policy(checkpoint, device=device)
    disable_dropout(policy)
    dev = policy.device
    torch.manual_seed(seed)
    buffer = RolloutBuffer(group_size=group_size)
    rollouts = []
    for task in tasks:
        group = buffer.collect_group(policy, task, proxy, dev, rollout_temperature=rollout_temperature)
        for idx, traj in enumerate(group):
            rollouts.append({
                "instance_id": task.instance_id,
                "rollout_idx": idx,
                "actions": traj.actions.to(torch.uint8),
                "proxy_reward": traj.reward,
                "recall": traj.reward_signal.recall,
                "token_ratio": traj.reward_signal.token_ratio,
            })
    return {"rollouts": rollouts}


def _auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows: list[dict]) -> dict:
    from scipy import stats

    verified = [float(r["verified"]) for r in rows]
    proxy = [float(r["proxy_reward"]) for r in rows]
    recall = [float(r["recall"]) for r in rows]
    n_pass = int(sum(verified))
    out: dict = {
        "n_rollouts": len(rows),
        "n_tasks": len({r["instance_id"] for r in rows}),
        "pass_rate": n_pass / len(rows),
        "patch_rate": sum(r["patch_produced"] in (True, "True") for r in rows) / len(rows),
    }
    if 0 < n_pass < len(rows):
        for name, score in (("proxy", proxy), ("recall", recall)):
            if len(set(score)) == 1:
                out[f"{name}_note"] = f"{name} is constant over all rollouts; correlations undefined"
                continue
            pb = stats.pointbiserialr(verified, score)
            sp = stats.spearmanr(verified, score)
            out[f"{name}_pointbiserial_r"] = float(pb.statistic)
            out[f"{name}_pointbiserial_p"] = float(pb.pvalue)
            out[f"{name}_spearman_rho"] = float(sp.statistic)
            out[f"{name}_spearman_p"] = float(sp.pvalue)
            out[f"{name}_auc"] = _auc(
                [s for s, v in zip(score, verified, strict=True) if v],
                [s for s, v in zip(score, verified, strict=True) if not v],
            )
    else:
        out["note"] = "verified outcome is constant over all rollouts; correlations undefined"

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["instance_id"]].append(r)
    mixed_aucs = []
    for group in groups.values():
        pos = [float(r["proxy_reward"]) for r in group if float(r["verified"])]
        neg = [float(r["proxy_reward"]) for r in group if not float(r["verified"])]
        if pos and neg:
            mixed_aucs.append(_auc(pos, neg))
    out["n_groups"] = len(groups)
    out["groups_all_pass"] = sum(all(float(r["verified"]) for r in g) for g in groups.values())
    out["groups_all_fail"] = sum(not any(float(r["verified"]) for r in g) for g in groups.values())
    out["groups_mixed"] = len(mixed_aucs)
    out["frac_groups_with_verified_signal"] = len(mixed_aucs) / len(groups)
    out["within_group_proxy_auc_mean"] = (
        sum(mixed_aucs) / len(mixed_aucs) if mixed_aucs else None
    )
    return out


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(
        description="Proxy vs verified reward correlation on fixed rollouts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(parser)
    add_verified_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Policy to sample rollouts from (Phase C best.pt or GRPO best_rl.pt).")
    parser.add_argument("--task-split", choices=("train", "val"), default="val")
    parser.add_argument("--num-tasks", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=2.0)
    parser.add_argument("--reward-beta", type=float, default=0.5)
    parser.add_argument("--reward-trace-weight", type=float, default=0.1)
    parser.add_argument("--reward-recall-bonus", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rollouts_path = out / "rollouts.pt"
    csv_path = out / "rollouts.csv"

    data = load_rl_data(args)
    pool = data.val_tasks if args.task_split == "val" else data.train_tasks
    tasks = subsample(pool, args.num_tasks, args.seed)
    by_id = {t.instance_id: t for t in tasks}

    config = {
        "checkpoint": str(args.checkpoint),
        "task_split": args.task_split,
        "instance_ids": sorted(by_id),
        "group_size": args.group_size,
        "rollout_temperature": args.rollout_temperature,
        "reward": [args.reward_beta, args.reward_trace_weight, args.reward_recall_bonus],
        "seed": args.seed,
    }
    if rollouts_path.exists():
        saved = torch.load(rollouts_path, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise SystemExit(
                f"{rollouts_path} was sampled with a different config:\n"
                f"  saved: {saved['config']}\n  now:   {config}\n"
                "use a fresh --output-dir"
            )
        logger.info("Reusing %d rollouts from %s", len(saved["rollouts"]), rollouts_path)
    else:
        proxy = ProxyReward(
            beta=args.reward_beta, trace_weight=args.reward_trace_weight,
            recall_bonus=args.reward_recall_bonus,
        )
        saved = sample_rollouts(
            checkpoint=args.checkpoint, tasks=tasks, group_size=args.group_size,
            rollout_temperature=args.rollout_temperature, proxy=proxy,
            seed=args.seed, device=args.device,
        )
        saved["config"] = config
        torch.save(saved, rollouts_path)
        logger.info("Sampled %d rollouts -> %s", len(saved["rollouts"]), rollouts_path)

    done = {(r["instance_id"], int(r["rollout_idx"])) for r in _read_rows(csv_path)}
    pending = [r for r in saved["rollouts"] if (r["instance_id"], r["rollout_idx"]) not in done]
    logger.info("%d rollouts done, %d pending", len(done), len(pending))

    if pending:
        judge = build_verified_reward(args, base_commits=data.base_commits)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            step = args.verified_workers
            for lo in range(0, len(pending), step):
                chunk = pending[lo:lo + step]
                outcomes = judge.run_many(
                    [(by_id[r["instance_id"]], r["actions"].float()) for r in chunk],
                )
                for r, o in zip(chunk, outcomes, strict=True):
                    writer.writerow({
                        "instance_id": r["instance_id"], "rollout_idx": r["rollout_idx"],
                        "proxy_reward": r["proxy_reward"], "recall": r["recall"],
                        "token_ratio": r["token_ratio"], "verified": int(o.passed),
                        "resolved": o.resolved, "patch_produced": o.patch_produced,
                        "patch_applied": o.patch_applied, "n_spans": o.n_spans,
                        "n_files": o.n_files, "context_chars": o.context_chars,
                        "prompt_tokens": o.prompt_tokens, "completion_tokens": o.completion_tokens,
                        "agent_wall_s": round(o.agent_wall_s, 1),
                        "harness_wall_s": round(o.harness_wall_s, 1), "rollout_dir": o.rollout_dir,
                    })
                fh.flush()
                logger.info("verified %d/%d pending rollouts", min(lo + step, len(pending)), len(pending))

    rows = _read_rows(csv_path)
    if len(rows) != len(saved["rollouts"]):
        raise SystemExit(f"{csv_path} has {len(rows)} rows, expected {len(saved['rollouts'])}")
    summary = {"config": config, **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
