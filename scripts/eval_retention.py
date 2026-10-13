from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from scripts.eval_candidates import (
    DATASETS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    collection_repo,
    ssg_path,
)
from token_sniper.evaluation.eval_stages import (
    candidates_path,
    check_ssg_matches,
    read_record,
    retention_path,
    sha256_file,
    write_record,
)
from token_sniper.evaluation.locagent_ranking import entity_keys
from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.evaluation.swebench import load_swebench_task_metadata
from token_sniper.graph.io import read_ssg_json
from token_sniper.retrieval.config import EmbeddingConfig
from token_sniper.retrieval.embeddings import create_embedder
from token_sniper.retrieval.reranker import FEATURE_NAMES
from token_sniper.rl.trainer import load_warm_start_policy
from token_sniper.vgib_selector.dataset import load_reranked_vgib_task
from token_sniper.vgib_selector.features import ChromaSourceEncoder
from token_sniper.vgib_selector.token_counts import (
    count_ssg_tokens,
    load_tokenizer,
    read_token_sidecar,
    sidecar_path,
    write_token_sidecar,
)

logger = logging.getLogger(__name__)


def retention_header(checkpoint_sha256: str, reranker_sha256: str, candidates_sha256: str, top_k: int) -> dict:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "reranker_sha256": reranker_sha256,
        "candidates_sha256": candidates_sha256,
        "top_k": top_k,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Eval stage B: cached reranker order and VGIB p_v")
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ssg-dir", type=Path, required=True)
    parser.add_argument("--chromadb-dir", type=Path, required=True)
    parser.add_argument("--repos-dir", type=Path, required=True, help="Bare repo cache: <repos-dir>/owner__name")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-name", default=None,
                        help="Cache namespace under --cache-root (default: the dataset key)")
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_name, split = DATASETS[args.dataset_key]
    cache = args.cache_root / (args.cache_name or args.dataset_key)
    tasks = load_phase0_manifest(args.manifest).tasks

    reranker_sha = sha256_file(args.reranker_model)
    booster = xgb.Booster()
    booster.load_model(str(args.reranker_model))
    device = torch.device(args.device)
    checkpoints = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoints[name] = (sha256_file(Path(path)), load_warm_start_policy(path, device=args.device).eval())
        logger.info("Checkpoint %s: %s (sha256 %s)", name, path, checkpoints[name][0])

    embedder = create_embedder(EmbeddingConfig(
        provider="sentence-transformers", name=EMBEDDING_MODEL, dimension=EMBEDDING_DIM, device=args.device,
    ))
    encoder = ChromaSourceEncoder(
        chromadb_dir=args.chromadb_dir, collection_prefix="token_sniper",
        model_name=EMBEDDING_MODEL, embedding_dim=EMBEDDING_DIM,
    )
    tokenizer = None

    t_start = time.monotonic()
    for n, task in enumerate(tasks, 1):
        cpath = candidates_path(cache, task.instance_id)
        if not cpath.exists():
            raise FileNotFoundError(f"no stage-A record {cpath}: run scripts/eval_candidates.py first")
        cheader, cand = read_record(cpath)
        if "excluded" in cand:
            continue

        candidates_sha = sha256_file(cpath)
        todo = []
        for name, (ckpt_sha, _) in checkpoints.items():
            header = retention_header(ckpt_sha, reranker_sha, candidates_sha, args.top_k)
            out = retention_path(cache, ckpt_sha, task.instance_id)
            if out.exists():
                read_record(out, header)
            else:
                todo.append((name, out, header))
        if not todo:
            continue

        ssg = ssg_path(args.ssg_dir, task)
        check_ssg_matches(cpath, cheader, ssg)
        graph = read_ssg_json(ssg)
        if not sidecar_path(ssg).exists():
            tokenizer = tokenizer or load_tokenizer()
            git_dir = args.repos_dir / task.repo.replace("/", "__")
            if not git_dir.is_dir():
                raise FileNotFoundError(f"no cached repo for {task.repo} at {git_dir}")
            write_token_sidecar(ssg, count_ssg_tokens(graph, git_dir, task.base_commit, tokenizer))
        node_tokens = read_token_sidecar(ssg)

        pool_files = cand["candidate_files"]
        t_rerank = time.perf_counter()
        scores = booster.predict(xgb.DMatrix(np.asarray(cand["feature_matrix"]), feature_names=FEATURE_NAMES))
        rerank_ms = (time.perf_counter() - t_rerank) * 1000
        ranked = sorted(zip(pool_files, scores.tolist(), strict=True), key=lambda pair: -pair[1])
        topk_files = [fp for fp, _ in ranked[:args.top_k]]
        pool_set = set(pool_files)
        pool_tokens = sum(node_tokens[nid] for nid, data in graph.nodes(data=True) if data["file_path"] in pool_set)

        issue_text = load_swebench_task_metadata(
            task.instance_id, dataset_name=dataset_name, split=split,
        ).problem_statement
        t_input = time.perf_counter()
        vgib_task = load_reranked_vgib_task(
            instance_id=task.instance_id, repo=task.repo, issue_text=issue_text,
            candidate_files=topk_files, gold_files=[], graph=graph, encoder=encoder, embedder=embedder,
            feature_cache_dir=args.cache_root / "vgib_features", issue_cache_dir=args.cache_root / "issue_emb",
            collection_repo=collection_repo(task),
        )
        vgib_input_ms = (time.perf_counter() - t_input) * 1000
        node_ids = vgib_task.features.node_ids

        for name, out, header in todo:
            p_v: list[float] = []
            t_forward = time.perf_counter()
            if node_ids:
                with torch.no_grad():
                    p_v = checkpoints[name][1](vgib_task, device).retention_probs.cpu().tolist()
            forward_ms = (time.perf_counter() - t_forward) * 1000
            nodes = []
            for nid, prob in zip(node_ids, p_v, strict=True):
                keys = entity_keys(nid, graph.nodes[nid])
                nodes.append({
                    "id": nid,
                    "file": graph.nodes[nid]["file_path"],
                    "module_key": keys[0] if keys else None,
                    "entity_key": keys[1] if keys else None,
                    "p_v": prob,
                    "tokens": node_tokens[nid],
                })
            write_record(out, header, {
                "ranked_files": ranked, "pool_tokens": pool_tokens, "nodes": nodes,
                "selector_latency_ms": rerank_ms + forward_ms, "vgib_input_ms": vgib_input_ms,
                "selector_device": str(device),
            })
        logger.info("[%d/%d] %s: %d top-%d nodes, %.0fs elapsed",
                    n, len(tasks), task.instance_id, len(node_ids), args.top_k, time.monotonic() - t_start)


if __name__ == "__main__":
    main()
