from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from token_sniper.edge_pred.candidates import CandidateSet, edge_index_from_graph, generate_candidates
from token_sniper.edge_pred.features import EdgePredEncoder, NodeFeatureBundle, SignatureEncoder, build_node_features
from token_sniper.graph.io import read_ssg_json

logger = logging.getLogger(__name__)


@dataclass
class TaskGraph:
    instance_id: str
    repo: str
    node_ids: list[str]
    features: NodeFeatureBundle
    edge_index: torch.Tensor
    positive_pairs: torch.Tensor
    candidate_pairs: torch.Tensor
    trace_edges: set[tuple[str, str]] = field(default_factory=set)

    def sample_negatives(self, n: int, rng: random.Random) -> torch.Tensor:
        cands = self.candidate_pairs
        if cands.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)
        pos_set = {tuple(p.tolist()) for p in self.positive_pairs}
        trace_pair_set = set()
        name_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        for a, b in self.trace_edges:
            if a in name_to_idx and b in name_to_idx:
                trace_pair_set.add((name_to_idx[a], name_to_idx[b]))

        negs: list[tuple[int, int]] = []
        tries = 0
        max_tries = max(10 * n, 2 * len(cands))
        indices = list(range(len(cands)))
        rng.shuffle(indices)
        for idx in indices:
            tries += 1
            if tries > max_tries:
                break
            pair = tuple(cands[idx].tolist())
            if pair in pos_set or pair in trace_pair_set:
                continue
            negs.append(pair)
            if len(negs) >= n:
                break
        return torch.tensor(negs, dtype=torch.long) if negs else torch.empty((0, 2), dtype=torch.long)


def load_task_graph(
    *,
    instance_id: str,
    repo: str,
    ssg_path: Path,
    trace_path: Path,
    encoder: EdgePredEncoder,
    feature_cache_dir: Path | None = None,
    candidate_cache_dir: Path | None = None,
    max_fs_hops: int = 3,
) -> TaskGraph:
    graph = read_ssg_json(ssg_path)
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_edges_raw: set[tuple[str, str]] = {
        (a, b) for a, b, *_ in trace_payload.get("edges", [])
    }

    commit_short = ssg_path.parent.name
    collection_repo = f"{repo}@{commit_short}"
    features = build_node_features(
        graph, encoder=encoder, cache_dir=feature_cache_dir, collection_repo=collection_repo,
    )
    node_ids = features.node_ids
    name_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    edge_index = edge_index_from_graph(graph, node_ids)

    positives: list[tuple[int, int]] = []
    for a, b in trace_edges_raw:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        if graph.has_edge(a, b):
            data = graph.get_edge_data(a, b) or {}
            if any(d.get("edge_type") == "calls" for d in data.values()):
                continue
        positives.append((name_to_idx[a], name_to_idx[b]))

    cand_path = (
        candidate_cache_dir / f"candidates_{ssg_path.parent.parent.name}_{ssg_path.parent.name}.pt"
        if candidate_cache_dir is not None
        else None
    )
    if cand_path is not None and cand_path.exists():
        candidates = CandidateSet(pairs=torch.load(cand_path, weights_only=False)["pairs"])
        logger.info("Loaded %d cached candidates from %s", len(candidates), cand_path)
    else:
        candidates = generate_candidates(graph, node_ids, max_fs_hops=max_fs_hops)
        if cand_path is not None:
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"pairs": candidates.pairs}, cand_path)

    pos_tensor = (
        torch.tensor(positives, dtype=torch.long).reshape(-1, 2)
        if positives
        else torch.empty((0, 2), dtype=torch.long)
    )

    cand_set = set(candidates.pairs)
    merged_pairs = list(candidates.pairs)
    positives_added = 0
    for p in positives:
        if p not in cand_set:
            merged_pairs.append(p)
            cand_set.add(p)
            positives_added += 1
    if positives_added > 0:
        logger.info(
            "[%s] %d/%d positive edges were outside candidate set and were injected",
            instance_id, positives_added, len(positives),
        )

    cand_tensor = (
        torch.tensor(merged_pairs, dtype=torch.long).reshape(-1, 2)
        if merged_pairs
        else torch.empty((0, 2), dtype=torch.long)
    )

    return TaskGraph(
        instance_id=instance_id,
        repo=repo,
        node_ids=node_ids,
        features=features,
        edge_index=edge_index,
        positive_pairs=pos_tensor,
        candidate_pairs=cand_tensor,
        trace_edges=trace_edges_raw,
    )


def load_task_graph_for_scoring(
    *,
    instance_id: str,
    repo: str,
    ssg_path: Path,
    encoder: EdgePredEncoder,
    feature_cache_dir: Path | None = None,
    candidate_cache_dir: Path | None = None,
    max_fs_hops: int = 3,
) -> TaskGraph:
    graph = read_ssg_json(ssg_path)

    commit_short = ssg_path.parent.name
    collection_repo = f"{repo}@{commit_short}"
    features = build_node_features(
        graph, encoder=encoder, cache_dir=feature_cache_dir, collection_repo=collection_repo,
    )
    node_ids = features.node_ids
    edge_index = edge_index_from_graph(graph, node_ids)

    cand_path = (
        candidate_cache_dir / f"candidates_{ssg_path.parent.parent.name}_{ssg_path.parent.name}.pt"
        if candidate_cache_dir is not None
        else None
    )
    if cand_path is not None and cand_path.exists():
        candidates = CandidateSet(pairs=torch.load(cand_path, weights_only=False)["pairs"])
        logger.info("Loaded %d cached candidates from %s", len(candidates), cand_path)
    else:
        candidates = generate_candidates(graph, node_ids, max_fs_hops=max_fs_hops)
        if cand_path is not None:
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"pairs": candidates.pairs}, cand_path)

    cand_tensor = (
        torch.tensor(candidates.pairs, dtype=torch.long).reshape(-1, 2)
        if candidates.pairs
        else torch.empty((0, 2), dtype=torch.long)
    )

    return TaskGraph(
        instance_id=instance_id,
        repo=repo,
        node_ids=node_ids,
        features=features,
        edge_index=edge_index,
        positive_pairs=torch.empty((0, 2), dtype=torch.long),
        candidate_pairs=cand_tensor,
    )


def stratified_split(
    manifest: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    traces_dir: Path | None = None,
    ssg_dir: Path | None = None,
    min_trace_edges: int = 1,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)

    eligible = list(manifest)
    if traces_dir is not None or ssg_dir is not None:
        filtered: list[dict[str, Any]] = []
        for task in manifest:
            iid = task["instance_id"]
            if traces_dir is not None:
                trace_path = traces_dir / f"{iid}.json"
                if not trace_path.exists():
                    logger.debug("[split] %s excluded: no trace file", iid)
                    continue
                try:
                    td = json.loads(trace_path.read_text(encoding="utf-8"))
                    if len(td.get("edges", [])) < min_trace_edges:
                        logger.debug("[split] %s excluded: %d edges < %d",
                                     iid, len(td.get("edges", [])), min_trace_edges)
                        continue
                except Exception:
                    logger.debug("[split] %s excluded: unreadable trace", iid)
                    continue
            if ssg_dir is not None:
                repo_slug = task["repo"].replace("/", "__")
                commit_short = task["base_commit"][:12]
                ssg_path = ssg_dir / repo_slug / commit_short / "ssg.json"
                if not ssg_path.exists():
                    logger.debug("[split] %s excluded: no SSG", iid)
                    continue
            filtered.append(task)
        skipped = len(manifest) - len(filtered)
        if skipped:
            logger.info(
                "[split] Excluded %d/%d tasks without valid trace/SSG data",
                skipped, len(manifest),
            )
        eligible = filtered

    by_repo: dict[str, list[str]] = {}
    for task in eligible:
        by_repo.setdefault(task["repo"], []).append(task["instance_id"])

    target_train = max(1, round(len(eligible) * train_ratio))
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)

    train: list[str] = []
    test: list[str] = []

    for repo in repos:
        if len(train) < target_train:
            train.extend(by_repo[repo])
        else:
            test.extend(by_repo[repo])

    if not test:
        last_repo_tasks = by_repo[repos[-1]]
        for tid in last_repo_tasks:
            train.remove(tid)
            test.append(tid)

    logger.info(
        "[split] %d train (%d repos) + %d test (%d repos) from %d eligible tasks",
        len(train),
        len({t["repo"] for t in eligible if t["instance_id"] in set(train)}),
        len(test),
        len({t["repo"] for t in eligible if t["instance_id"] in set(test)}),
        len(eligible),
    )
    return sorted(train), sorted(test)
