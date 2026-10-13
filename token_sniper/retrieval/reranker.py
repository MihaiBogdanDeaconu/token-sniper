from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from token_sniper.graph.schema import FILE_NODE_PREFIX, is_file_node_id
from token_sniper.retrieval.retriever import RetrievalResult, RetrievedEntity

if TYPE_CHECKING:
    from token_sniper.retrieval.embeddings import Embedder
    from token_sniper.retrieval.vectorstore import EntityVectorStore

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "knn_score_max",
    "knn_score_mean",
    "knn_entity_count",
    "ppr_score",
    "call_distance_min",
    "import_edges_to_entry",
    "calls_edge_count",
    "file_pagerank",
    "in_degree",
    "out_degree",
    "importers_count",
    "entities_in_file",
    "file_clustering_coeff",
    "file_name_in_issue",
    "dir_name_in_issue",
    "entity_name_mentions",
    "traceback_mention",
    "path_depth",
]


@dataclass(frozen=True)
class CandidateFileFeatures:
    file_path: str
    knn_score_max: float = 0.0
    knn_score_mean: float = 0.0
    knn_entity_count: int = 0
    ppr_score: float = 0.0
    call_distance_min: float = float("inf")
    import_edges_to_entry: int = 0
    calls_edge_count: int = 0
    file_pagerank: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    importers_count: int = 0
    entities_in_file: int = 0
    file_clustering_coeff: float = 0.0
    file_name_in_issue: float = 0.0
    dir_name_in_issue: float = 0.0
    entity_name_mentions: int = 0
    traceback_mention: float = 0.0
    path_depth: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.knn_score_max,
            self.knn_score_mean,
            self.knn_entity_count,
            self.ppr_score,
            min(self.call_distance_min, 100.0),
            self.import_edges_to_entry,
            self.calls_edge_count,
            self.file_pagerank,
            self.in_degree,
            self.out_degree,
            self.importers_count,
            self.entities_in_file,
            self.file_clustering_coeff,
            self.file_name_in_issue,
            self.dir_name_in_issue,
            self.entity_name_mentions,
            self.traceback_mention,
            self.path_depth,
        ], dtype=np.float32)


@dataclass
class RerankerInput:
    instance_id: str
    candidate_features: list[CandidateFileFeatures] = field(default_factory=list)
    gold_files: list[str] = field(default_factory=list)
    retrieved_entities: list[RetrievedEntity] = field(default_factory=list)
    """The merged dense+BM25 entities the candidates came from, best first.

    Kept so graph expansion (which seeds on the top entities, not on files) can
    be reproduced or re-run at a different depth without redoing retrieval."""

    def to_feature_matrix(self) -> np.ndarray:
        if not self.candidate_features:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.stack([f.to_array() for f in self.candidate_features])

    def relevance_labels(self) -> np.ndarray:
        gold_set = set(self.gold_files)
        return np.array(
            [1.0 if f.file_path in gold_set else 0.0 for f in self.candidate_features],
            dtype=np.float32,
        )


def _compute_text_features(
    file_path: str,
    entity_node_ids: list[str],
    issue_text_lower: str,
    traceback_files: set[str],
) -> tuple[float, float, int, float, int]:
    import re
    from pathlib import PurePosixPath

    p = PurePosixPath(file_path)
    stem = p.stem
    path_depth = len(p.parts)

    _GENERIC = {"__init__", "utils", "base", "core", "main", "test", "tests",
                "conf", "setup", "helpers", "common", "types", "models", "views"}
    file_name_in_issue = 0.0
    if stem not in _GENERIC and len(stem) >= 3:
        if re.search(r'\b' + re.escape(stem) + r'\b', issue_text_lower):
            file_name_in_issue = 1.0

    dir_name_in_issue = 0.0
    if len(p.parts) >= 2:
        parent = p.parts[-2]
        if parent not in _GENERIC and len(parent) >= 3:
            if re.search(r'\b' + re.escape(parent) + r'\b', issue_text_lower):
                dir_name_in_issue = 1.0

    entity_mentions = 0
    for nid in entity_node_ids:
        short_name = nid.rsplit(".", 1)[-1] if "." in nid else nid
        if len(short_name) >= 4 and short_name.lower() in issue_text_lower:
            entity_mentions += 1

    traceback_mention = 1.0 if p.name in traceback_files else 0.0

    return file_name_in_issue, dir_name_in_issue, entity_mentions, traceback_mention, path_depth


def _extract_traceback_files(issue_text: str) -> set[str]:
    import re
    patterns = [
        r'File "([^"]+\.py)"',
        r"File '([^']+\.py)'",
        r'(\S+\.py):\d+',
    ]
    files: set[str] = set()
    for pat in patterns:
        for m in re.findall(pat, issue_text):
            from pathlib import PurePosixPath
            files.add(PurePosixPath(m).name)
    return files


def extract_candidate_features(
    *,
    knn_result: RetrievalResult,
    graph: nx.MultiDiGraph[Any],
    entry_point_node_ids: list[str],
    issue_text: str = "",
    ppr_scores: dict[str, float] | None = None,
    pagerank_scores: dict[str, float] | None = None,
) -> list[CandidateFileFeatures]:
    undirected = nx.Graph(graph.to_undirected(as_view=True))
    issue_text_lower = issue_text.lower()
    traceback_files = _extract_traceback_files(issue_text) if issue_text else set()

    if pagerank_scores is None:
        try:
            pagerank_scores = nx.pagerank(undirected, alpha=0.85, max_iter=100, tol=1e-4)
        except nx.PowerIterationFailedConvergence:
            pagerank_scores = {}

    if ppr_scores is None and entry_point_node_ids:
        personalization = {n: 1.0 / len(entry_point_node_ids) for n in entry_point_node_ids}
        try:
            ppr_scores = nx.pagerank(
                undirected, alpha=0.85, personalization=personalization,
                max_iter=200, tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            ppr_scores = {}
    ppr_scores = ppr_scores or {}

    file_to_entities: dict[str, list[str]] = {}
    for nid, data in graph.nodes(data=True):
        if is_file_node_id(nid):
            continue
        fp = data.get("file_path", "")
        if fp:
            file_to_entities.setdefault(fp, []).append(nid)

    entry_files = set()
    for ep_id in entry_point_node_ids:
        if ep_id in graph:
            fp = graph.nodes[ep_id].get("file_path", "")
            if fp:
                entry_files.add(fp)

    call_distances = _compute_call_distances(graph, entry_point_node_ids, max_depth=10)

    file_node_map: dict[str, str] = {}
    for nid in graph.nodes:
        if is_file_node_id(nid):
            fp = nid[len(FILE_NODE_PREFIX):]
            file_node_map[fp] = nid

    importers_cache: dict[str, int] = {}
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") == "imports":
            v_fp = graph.nodes[v].get("file_path", "") if v in graph else ""
            if v_fp:
                importers_cache[v_fp] = importers_cache.get(v_fp, 0) + 1

    knn_by_file: dict[str, list[RetrievedEntity]] = {}
    for entity in knn_result.entities:
        knn_by_file.setdefault(entity.file_path, []).append(entity)

    features: list[CandidateFileFeatures] = []
    for file_path in knn_result.file_paths:
        entities = knn_by_file.get(file_path, [])
        entity_scores = [e.score for e in entities]
        entity_node_ids = file_to_entities.get(file_path, [])

        min_call_dist = float("inf")
        for nid in entity_node_ids:
            if nid in call_distances:
                min_call_dist = min(min_call_dist, call_distances[nid])

        import_edges = 0
        for nid in entity_node_ids:
            for ep_id in entry_point_node_ids:
                if graph.has_edge(nid, ep_id) or graph.has_edge(ep_id, nid):
                    edge_data_list = list(graph.get_edge_data(nid, ep_id, default={}).values())
                    edge_data_list += list(graph.get_edge_data(ep_id, nid, default={}).values())
                    import_edges += sum(
                        1 for ed in edge_data_list if ed.get("edge_type") == "imports"
                    )

        calls_count = 0
        for nid in entity_node_ids:
            for _, _, ed in graph.edges(nid, data=True):
                if ed.get("edge_type") == "calls":
                    calls_count += 1

        file_nid = file_node_map.get(file_path, "")
        file_pr = pagerank_scores.get(file_nid, 0.0) if file_nid else 0.0

        file_ppr = 0.0
        for nid in entity_node_ids:
            file_ppr = max(file_ppr, ppr_scores.get(nid, 0.0))

        in_deg = sum(graph.in_degree(nid) for nid in entity_node_ids if nid in graph)
        out_deg = sum(graph.out_degree(nid) for nid in entity_node_ids if nid in graph)

        clust = 0.0
        if file_nid and file_nid in undirected:
            clust = nx.clustering(undirected, file_nid)

        fname_in_issue, dir_in_issue, ent_mentions, tb_mention, pdepth = (
            _compute_text_features(
                file_path, entity_node_ids, issue_text_lower, traceback_files,
            )
        )

        features.append(CandidateFileFeatures(
            file_path=file_path,
            knn_score_max=max(entity_scores) if entity_scores else 0.0,
            knn_score_mean=float(np.mean(entity_scores)) if entity_scores else 0.0,
            knn_entity_count=len(entities),
            ppr_score=file_ppr,
            call_distance_min=min_call_dist,
            import_edges_to_entry=import_edges,
            calls_edge_count=calls_count,
            file_pagerank=file_pr,
            in_degree=in_deg,
            out_degree=out_deg,
            importers_count=importers_cache.get(file_path, 0),
            entities_in_file=len(entity_node_ids),
            file_clustering_coeff=clust,
            file_name_in_issue=fname_in_issue,
            dir_name_in_issue=dir_in_issue,
            entity_name_mentions=ent_mentions,
            traceback_mention=tb_mention,
            path_depth=pdepth,
        ))

    return features


def _compute_call_distances(
    graph: nx.MultiDiGraph[Any],
    sources: list[str],
    max_depth: int,
) -> dict[str, int]:
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for s in sources:
        if s in graph:
            visited[s] = 0
            queue.append((s, 0))

    while queue:
        node, dist = queue.popleft()
        if dist >= max_depth:
            continue
        for _, neighbor, data in graph.edges(node, data=True):
            if data.get("edge_type") != "calls":
                continue
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
        for pred, _, data in graph.in_edges(node, data=True):
            if data.get("edge_type") != "calls":
                continue
            if pred not in visited:
                visited[pred] = dist + 1
                queue.append((pred, dist + 1))

    return visited


class GraphReranker:
    def __init__(self, model_path: Path | None = None) -> None:
        self._model: Any = None
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        import xgboost as xgb

        self._model = xgb.Booster()
        self._model.load_model(str(model_path))
        logger.info("Loaded reranker model from %s", model_path)

    def save(self, model_path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(model_path))

    def rerank(
        self,
        candidate_features: list[CandidateFileFeatures],
        *,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        if not candidate_features:
            return []

        if self._model is None:
            return [
                (f.file_path, f.knn_score_max) for f in candidate_features[:top_k]
            ]

        import xgboost as xgb

        X = np.stack([f.to_array() for f in candidate_features])
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        scores = self._model.predict(dmat)

        ranked = sorted(
            zip(candidate_features, scores),
            key=lambda pair: -pair[1],
        )
        return [(f.file_path, float(s)) for f, s in ranked[:top_k]]

    def fit(
        self,
        train_inputs: list[RerankerInput],
        *,
        val_inputs: list[RerankerInput] | None = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        eval_metric: str = "ndcg@10",
    ) -> dict[str, Any]:
        import xgboost as xgb

        X_train, y_train, groups_train = _build_ltr_dataset(train_inputs)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
        dtrain.set_group(groups_train)

        params: dict[str, Any] = {
            "objective": "rank:pairwise",
            "eval_metric": eval_metric,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
        if val_inputs:
            X_val, y_val, groups_val = _build_ltr_dataset(val_inputs)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)
            dval.set_group(groups_val)
            evals.append((dval, "val"))

        evals_result: dict[str, dict[str, list[float]]] = {}
        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20 if val_inputs else None,
            verbose_eval=10,
        )

        importance = self._model.get_score(importance_type="gain")
        logger.info("Feature importance (gain): %s", importance)

        return {
            "n_train_tasks": len(train_inputs),
            "n_val_tasks": len(val_inputs) if val_inputs else 0,
            "n_train_samples": len(y_train),
            "importance": importance,
            "evals_result": evals_result,
        }


class NeuralReranker:
    def __init__(self, feature_dim: int = len(FEATURE_NAMES), hidden: int = 128) -> None:
        import torch
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self._feature_dim = feature_dim

    def score(self, features: list[CandidateFileFeatures]) -> list[float]:
        import torch

        if not features:
            return []
        X = torch.tensor(
            np.stack([f.to_array() for f in features]),
            dtype=torch.float32,
        )
        self.net.eval()
        with torch.no_grad():
            scores = self.net(X).squeeze(-1)
        return scores.tolist()

    def fit(
        self,
        train_inputs: list[RerankerInput],
        *,
        val_inputs: list[RerankerInput] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val_loss = float("inf")
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
            for inp in train_inputs:
                X = torch.tensor(inp.to_feature_matrix(), dtype=torch.float32, device=device)
                y = torch.tensor(inp.relevance_labels(), dtype=torch.float32, device=device)
                if y.sum() == 0 or len(y) < 2:
                    continue

                opt.zero_grad()
                scores = self.net(X).squeeze(-1)
                loss = _listwise_loss(scores, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            val_loss = 0.0
            if val_inputs:
                self.net.eval()
                with torch.no_grad():
                    n_val = 0
                    for inp in val_inputs:
                        X = torch.tensor(
                            inp.to_feature_matrix(), dtype=torch.float32, device=device,
                        )
                        y = torch.tensor(
                            inp.relevance_labels(), dtype=torch.float32, device=device,
                        )
                        if y.sum() == 0 or len(y) < 2:
                            continue
                        scores = self.net(X).squeeze(-1)
                        val_loss += _listwise_loss(scores, y).item()
                        n_val += 1
                    val_loss /= max(n_val, 1)

            history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})
            if epoch % 10 == 0:
                logger.info(
                    "epoch %03d | train=%.4f | val=%.4f", epoch, avg_loss, val_loss,
                )

            if val_inputs and val_loss < best_val_loss:
                best_val_loss = val_loss

        return {"best_val_loss": best_val_loss, "history": history}


def _listwise_loss(scores: Any, relevance: Any) -> Any:
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(scores, dim=-1)
    target_dist = relevance / relevance.sum().clamp(min=1.0)
    return -(target_dist * log_probs).sum()


def _build_ltr_dataset(
    inputs: list[RerankerInput],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    groups: list[int] = []

    for inp in inputs:
        X = inp.to_feature_matrix()
        y = inp.relevance_labels()
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        groups.append(len(y))

    if not all_X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), []

    return np.concatenate(all_X), np.concatenate(all_y), groups
