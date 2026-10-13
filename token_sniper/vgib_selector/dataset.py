from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

import torch

from token_sniper.vgib_selector.features import (
    VGIBEncoder,
    VGIBFeatureBundle,
    build_vgib_features,
    edge_index_from_graph,
    is_file_node_id,
    lexical_node_features,
)

logger = logging.getLogger(__name__)


@dataclass
class VGIBTask:
    instance_id: str
    repo: str
    issue_text: str
    gold_files: list[str]
    gold_node_ids: set[str]
    features: VGIBFeatureBundle
    edge_index: torch.Tensor
    task_embedding: torch.Tensor


@dataclass
class VGIBBatch:
    task: VGIBTask
    positive_task_emb: torch.Tensor
    negative_task_embs: list[torch.Tensor]


def _load_or_compute_issue_embedding(
    instance_id: str,
    issue_text: str,
    embedder: Any | None,
    cache_dir: Path | None,
    embedding_dim: int | None = None,
) -> torch.Tensor:
    import json
    from pathlib import Path as _Path

    if cache_dir is not None:
        cache_path = _Path(cache_dir) / f"issue_emb_{instance_id}.json"
        if cache_path.exists():
            vec = json.loads(cache_path.read_text(encoding="utf-8"))
            return torch.tensor(vec, dtype=torch.float32)
    else:
        cache_path = None

    if embedder is None:
        raise RuntimeError(
            f"No cached issue embedding for {instance_id} and no embedder provided. "
            "Either supply an embedder or pre-compute the cache."
        )

    if not issue_text.strip():
        dim = embedding_dim or getattr(embedder, "dimension", None)
        if dim is None:
            from token_sniper.vgib_selector.backbone import EMBEDDING_DIM
            dim = EMBEDDING_DIM

        logger.warning("Empty issue text for %s — using zero vector (dim=%d)", instance_id, dim)
        zero = [0.0] * dim
        tensor = torch.tensor(zero, dtype=torch.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(zero), encoding="utf-8")
        return tensor

    emb_list = embedder.embed_query(issue_text)
    tensor = torch.tensor(emb_list, dtype=torch.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(emb_list), encoding="utf-8")
        logger.info("Cached issue embedding for %s → %s", instance_id, cache_path)

    return tensor


def load_vgib_task(
    *,
    manifest_entry: dict[str, Any],
    ssg_path: Path,
    repo_root: Path | None,
    encoder: VGIBEncoder,
    embedder: Any | None = None,
    feature_cache_dir: Path | None = None,
    issue_cache_dir: Path | None = None,
    collection_repo: str | None = None,
) -> VGIBTask:
    instance_id = manifest_entry["instance_id"]
    repo = manifest_entry["repo"]
    gold_files = manifest_entry.get("gold_patch_files", [])

    from token_sniper.graph.io import read_ssg_json

    graph = read_ssg_json(ssg_path)
    features = build_vgib_features(
        graph,
        encoder=encoder,
        repo_root=repo_root,
        cache_dir=feature_cache_dir,
        collection_repo=collection_repo,
    )
    edge_index = edge_index_from_graph(graph, features.node_ids)

    gold_set: set[str] = set()
    for nid in features.node_ids:
        if is_file_node_id(nid):
            continue
        node_data = graph.nodes[nid]
        if node_data.get("file_path") in gold_files:
            gold_set.add(nid)

    issue_text = manifest_entry.get("problem_statement", "")
    if not issue_text.strip():
        try:
            from token_sniper.evaluation.swebench import load_swebench_task_metadata

            meta = load_swebench_task_metadata(instance_id)
            issue_text = meta.problem_statement or ""
            logger.info("Loaded problem_statement from SWE-bench for %s", instance_id)
        except Exception:
            logger.warning("Could not fetch problem_statement from SWE-bench for %s", instance_id)

    effective_issue_cache = issue_cache_dir or feature_cache_dir
    task_emb = _load_or_compute_issue_embedding(
        instance_id,
        issue_text,
        embedder,
        effective_issue_cache,
    )

    return VGIBTask(
        instance_id=instance_id,
        repo=repo,
        issue_text=issue_text,
        gold_files=gold_files,
        gold_node_ids=gold_set,
        features=features,
        edge_index=edge_index,
        task_embedding=task_emb,
    )


def gold_labels_tensor(node_ids: list[str], gold_node_ids: set[str]) -> torch.Tensor:
    return torch.tensor(
        [1.0 if nid in gold_node_ids else 0.0 for nid in node_ids],
        dtype=torch.float32,
    )


def build_contrastive_batches(
    tasks: list[VGIBTask],
    *,
    negatives_per_task: int = 3,
    rng: random.Random,
) -> list[VGIBBatch]:
    by_repo: dict[str, list[VGIBTask]] = {}
    for t in tasks:
        by_repo.setdefault(t.repo, []).append(t)

    batches: list[VGIBBatch] = []
    for task in tasks:
        same_repo = [t for t in by_repo.get(task.repo, []) if t.instance_id != task.instance_id]
        cross_repo = [t for t in tasks if t.repo != task.repo]

        neg_samples: list[VGIBTask] = []
        n_same = min(negatives_per_task, len(same_repo))
        if n_same > 0:
            neg_samples.extend(rng.sample(same_repo, n_same))

        n_remaining = negatives_per_task - len(neg_samples)
        if n_remaining > 0 and cross_repo:
            n_cross = min(n_remaining, len(cross_repo))
            neg_samples.extend(rng.sample(cross_repo, n_cross))

        batches.append(
            VGIBBatch(
                task=task,
                positive_task_emb=task.task_embedding,
                negative_task_embs=[t.task_embedding for t in neg_samples],
            )
        )
    return batches


def stratified_split(
    manifest: list[dict[str, Any]],
    *,
    train_size: int = 35,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)

    by_repo: dict[str, list[str]] = {}
    for task in manifest:
        by_repo.setdefault(task["repo"], []).append(task["instance_id"])

    actual_train_size = min(train_size, len(manifest) - 1)
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)

    target_test = len(manifest) - actual_train_size
    train: list[str] = []
    test: list[str] = []

    for repo in repos:
        if len(test) < target_test:
            test.extend(by_repo[repo])
        else:
            train.extend(by_repo[repo])

    while len(train) < actual_train_size and test:
        train.append(test.pop(0))
    while len(train) > actual_train_size:
        test.append(train.pop())
    return sorted(train), sorted(test)


@dataclass
class RerankedVGIBTask:
    instance_id: str
    repo: str
    gold_files: list[str]
    candidate_files: list[str]
    features: VGIBFeatureBundle
    edge_index: torch.Tensor
    task_embedding: torch.Tensor
    gold_file_mask: torch.Tensor
    node_to_file: torch.Tensor
    token_counts: torch.Tensor | None
    """Exclusive per-node token counts (see ``token_counts.py``); ``None`` when loaded for inference."""

    gold_node_mask: torch.Tensor | None = None
    """[N] 1.0 for nodes owning lines the gold patch changes (see ``gold_nodes.py``).

    ``None`` when loaded for inference, which never reads it.  Training needs
    it: the compression loss scores each gold *node*, not each gold file."""

    trace_edge_mask: torch.Tensor | None = None
    """Optional [E_trace, 2] tensor of (src_idx, dst_idx) pairs representing
    trace execution edges mapped to indices within ``features.node_ids``.
    Used by the RL proxy reward to compute trace-connectivity bonuses.
    Default ``None`` — Phase C training is entirely unaffected."""

    lexical_features: torch.Tensor | None = None
    """[N, len(LEXICAL_FEATURE_NAMES)] issue-conditioned node features
    (``features.lexical_node_features``); always set by the loader."""

    prior_ppr: torch.Tensor | None = None
    """[N] personalized PageRank from the issue's seed entities over the full
    SSG (see :func:`structural_prior_inputs`).  Set only when training with the
    structural prior."""

    prior_seed_mask: torch.Tensor | None = None
    """[N] 1.0 for seed entities and their one-hop SSG neighbours; set with ``prior_ppr``."""


def load_reranked_vgib_task(
    *,
    instance_id: str,
    repo: str,
    issue_text: str,
    candidate_files: list[str],
    gold_files: list[str],
    graph: Any,
    encoder: VGIBEncoder,
    embedder: Any | None = None,
    feature_cache_dir: Any | None = None,
    issue_cache_dir: Any | None = None,
    collection_repo: str | None = None,
    node_tokens: Mapping[str, int] | None = None,
    gold_node_ids: set[str] | None = None,
) -> RerankedVGIBTask:
    import networkx as nx

    candidate_set = set(candidate_files)

    subgraph_nodes = []
    for nid, data in graph.nodes(data=True):
        if is_file_node_id(nid):
            fp = nid[len("file:"):]
            if fp in candidate_set:
                subgraph_nodes.append(nid)
        else:
            fp = data.get("file_path", "")
            if fp in candidate_set:
                subgraph_nodes.append(nid)

    subgraph = graph.subgraph(subgraph_nodes).copy()

    if subgraph.number_of_nodes() == 0:
        logger.warning("Empty subgraph for %s — no candidate nodes found in SSG", instance_id)
        subgraph = nx.MultiDiGraph()

    features = build_vgib_features(
        subgraph,
        encoder=encoder,
        repo_root=None,
        cache_dir=feature_cache_dir,
        collection_repo=collection_repo,
    )
    edge_index = edge_index_from_graph(subgraph, features.node_ids)

    task_emb = _load_or_compute_issue_embedding(
        instance_id, issue_text, embedder, issue_cache_dir or feature_cache_dir,
    )
    lexical = lexical_node_features(
        issue_text=issue_text,
        graph=subgraph,
        node_ids=features.node_ids,
        source_texts=features.source_texts,
        signature_emb=features.signature_emb,
        issue_emb=task_emb,
    )

    file_vocab_ordered = sorted(candidate_set)
    file_to_idx = {fp: i for i, fp in enumerate(file_vocab_ordered)}
    n_files = len(file_vocab_ordered)

    gold_file_mask = torch.zeros(n_files, dtype=torch.float32)
    for gf in gold_files:
        if gf in file_to_idx:
            gold_file_mask[file_to_idx[gf]] = 1.0

    node_to_file = torch.zeros(len(features.node_ids), dtype=torch.long)

    for i, nid in enumerate(features.node_ids):
        if is_file_node_id(nid):
            fp = nid[len("file:"):]
        else:
            fp = subgraph.nodes[nid].get("file_path", "") if nid in subgraph else ""
        if fp in file_to_idx:
            node_to_file[i] = file_to_idx[fp]

    token_counts = None
    if node_tokens is not None:
        missing = [nid for nid in features.node_ids if nid not in node_tokens]
        if missing:
            raise KeyError(f"{instance_id}: {len(missing)} nodes have no token count, e.g. {missing[:3]}")
        token_counts = torch.tensor([float(node_tokens[nid]) for nid in features.node_ids], dtype=torch.float32)

    gold_node_mask = None
    if gold_node_ids is not None:
        gold_node_mask = torch.tensor(
            [1.0 if nid in gold_node_ids else 0.0 for nid in features.node_ids], dtype=torch.float32,
        )

    return RerankedVGIBTask(
        instance_id=instance_id,
        repo=repo,
        gold_files=gold_files,
        candidate_files=candidate_files,
        features=features,
        edge_index=edge_index,
        task_embedding=task_emb,
        gold_file_mask=gold_file_mask,
        node_to_file=node_to_file,
        token_counts=token_counts,
        gold_node_mask=gold_node_mask,
        lexical_features=lexical,
    )


def structural_prior_inputs(
    graph: Any,
    seed_ids: list[str],
    node_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    import networkx as nx

    seeds = [s for s in seed_ids if s in graph]
    ppr = torch.zeros(len(node_ids), dtype=torch.float32)
    near = torch.zeros(len(node_ids), dtype=torch.float32)
    if not seeds:
        return ppr, near

    undirected = nx.Graph(graph.to_undirected(as_view=True))
    scores = nx.pagerank(
        undirected, alpha=0.85, personalization={s: 1.0 / len(seeds) for s in seeds},
        max_iter=200, tol=1e-6,
    )
    neighbourhood = set(seeds)
    for s in seeds:
        neighbourhood.update(undirected.neighbors(s))
    for i, nid in enumerate(node_ids):
        ppr[i] = scores.get(nid, 0.0)
        near[i] = float(nid in neighbourhood)
    return ppr, near
