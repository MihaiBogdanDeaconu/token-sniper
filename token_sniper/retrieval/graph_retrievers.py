from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import networkx as nx

from token_sniper.graph.schema import is_file_node_id
from token_sniper.retrieval.entry_points import EntryPointResult, identify_entry_points
from token_sniper.retrieval.retriever import RetrievalResult, RetrievedEntity

if TYPE_CHECKING:
    from token_sniper.retrieval.embeddings import Embedder
    from token_sniper.retrieval.vectorstore import EntityVectorStore

logger = logging.getLogger(__name__)


class GraphBFSRetriever:
    def __init__(
        self,
        *,
        graph: nx.MultiDiGraph[Any],
        entry_point_method: str = "embedding",
        max_hops: int = 2,
        n_entry_points: int = 3,
        embedder: Embedder | None = None,
        vector_store: EntityVectorStore | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._graph = graph
        self._undirected = graph.to_undirected(as_view=True)
        self._entry_point_method = entry_point_method
        self._max_hops = max_hops
        self._n_entry_points = n_entry_points
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm_client = llm_client

    def retrieve(
        self,
        *,
        issue_text: str,
        repo: str,
        k: int = 20,
        where_filter: dict[str, Any] | None = None,
        pre_computed_entry_points: EntryPointResult | None = None,
    ) -> tuple[RetrievalResult, EntryPointResult]:
        t0 = time.perf_counter()

        if pre_computed_entry_points is not None:
            ep_result = pre_computed_entry_points
        else:
            ep_result = identify_entry_points(
                method=self._entry_point_method,
                issue_text=issue_text,
                repo=repo,
                graph=self._graph,
                n_candidates=self._n_entry_points,
                embedder=self._embedder,
                vector_store=self._vector_store,
                llm_client=self._llm_client,
            )

        seed_nodes = ep_result.matched_node_ids
        if not seed_nodes:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return (
                RetrievalResult(
                    strategy="graph_bfs",
                    repo=repo,
                    query=issue_text[:500],
                    k=k,
                    entities=[],
                    total_indexed=self._graph.number_of_nodes(),
                    latency_ms=round(latency_ms, 1),
                ),
                ep_result,
            )

        visited_with_dist = _bfs_multi_source(
            self._undirected, seed_nodes, max_depth=self._max_hops
        )

        entity_entries: list[tuple[str, int, int]] = []
        for node_id, dist in visited_with_dist.items():
            if is_file_node_id(node_id):
                continue
            if node_id not in self._graph:
                continue
            node_data = self._graph.nodes[node_id]
            if node_data.get("node_type") == "module_preamble":
                continue
            if where_filter and not _matches_filter(node_data, where_filter):
                continue
            degree = self._undirected.degree(node_id)
            entity_entries.append((node_id, dist, degree))

        entity_entries.sort(key=lambda x: (x[1], -x[2]))

        entities: list[RetrievedEntity] = []
        for rank, (node_id, dist, _degree) in enumerate(entity_entries[:k], start=1):
            node_data = self._graph.nodes[node_id]
            score = 1.0 / (1.0 + dist)
            entities.append(
                RetrievedEntity(
                    entity_id=node_id,
                    score=score,
                    rank=rank,
                    entity_type=str(node_data.get("node_type", "")),
                    file_path=str(node_data.get("file_path", "")),
                    fully_qualified_name=str(node_data.get("fully_qualified_name", node_id)),
                    start_line=int(node_data.get("start_line", 0)),
                    end_line=int(node_data.get("end_line", 0)),
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "BFS retrieval: %d seeds, %d hops → %d entities in %.0fms",
            len(seed_nodes),
            self._max_hops,
            len(entities),
            latency_ms,
        )

        return (
            RetrievalResult(
                strategy="graph_bfs",
                repo=repo,
                query=issue_text[:500],
                k=k,
                entities=entities,
                total_indexed=self._graph.number_of_nodes(),
                latency_ms=round(latency_ms, 1),
            ),
            ep_result,
        )


class GraphPPRRetriever:
    def __init__(
        self,
        *,
        graph: nx.MultiDiGraph[Any],
        entry_point_method: str = "embedding",
        alpha: float = 0.15,
        n_entry_points: int = 3,
        embedder: Embedder | None = None,
        vector_store: EntityVectorStore | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._graph = graph
        self._alpha = alpha
        self._entry_point_method = entry_point_method
        self._n_entry_points = n_entry_points
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm_client = llm_client

    def retrieve(
        self,
        *,
        issue_text: str,
        repo: str,
        k: int = 20,
        where_filter: dict[str, Any] | None = None,
        pre_computed_entry_points: EntryPointResult | None = None,
    ) -> tuple[RetrievalResult, EntryPointResult]:
        t0 = time.perf_counter()

        if pre_computed_entry_points is not None:
            ep_result = pre_computed_entry_points
        else:
            ep_result = identify_entry_points(
                method=self._entry_point_method,
                issue_text=issue_text,
                repo=repo,
                graph=self._graph,
                n_candidates=self._n_entry_points,
                embedder=self._embedder,
                vector_store=self._vector_store,
                llm_client=self._llm_client,
            )

        seed_nodes = ep_result.matched_node_ids
        if not seed_nodes:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return (
                RetrievalResult(
                    strategy="graph_ppr",
                    repo=repo,
                    query=issue_text[:500],
                    k=k,
                    entities=[],
                    total_indexed=self._graph.number_of_nodes(),
                    latency_ms=round(latency_ms, 1),
                ),
                ep_result,
            )

        personalization = {node: 1.0 / len(seed_nodes) for node in seed_nodes}

        damping_factor = 1.0 - self._alpha
        try:
            undirected = self._graph.to_undirected(as_view=True)
            ppr_scores = nx.pagerank(
                undirected,
                alpha=damping_factor,
                personalization=personalization,
                max_iter=200,
                tol=1e-06,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR did not converge; using partial results")
            ppr_scores = nx.pagerank(
                undirected,
                alpha=damping_factor,
                personalization=personalization,
                max_iter=500,
                tol=1e-04,
            )

        entity_scores: list[tuple[str, float]] = []
        for node_id, score in ppr_scores.items():
            if is_file_node_id(node_id):
                continue
            if node_id not in self._graph:
                continue
            node_data = self._graph.nodes[node_id]
            if node_data.get("node_type") == "module_preamble":
                continue
            if where_filter and not _matches_filter(node_data, where_filter):
                continue
            entity_scores.append((node_id, score))

        entity_scores.sort(key=lambda x: -x[1])

        entities: list[RetrievedEntity] = []
        for rank, (node_id, score) in enumerate(entity_scores[:k], start=1):
            node_data = self._graph.nodes[node_id]
            entities.append(
                RetrievedEntity(
                    entity_id=node_id,
                    score=score,
                    rank=rank,
                    entity_type=str(node_data.get("node_type", "")),
                    file_path=str(node_data.get("file_path", "")),
                    fully_qualified_name=str(node_data.get("fully_qualified_name", node_id)),
                    start_line=int(node_data.get("start_line", 0)),
                    end_line=int(node_data.get("end_line", 0)),
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "PPR retrieval: %d seeds, alpha=%.2f → %d entities in %.0fms",
            len(seed_nodes),
            self._alpha,
            len(entities),
            latency_ms,
        )

        return (
            RetrievalResult(
                strategy="graph_ppr",
                repo=repo,
                query=issue_text[:500],
                k=k,
                entities=entities,
                total_indexed=self._graph.number_of_nodes(),
                latency_ms=round(latency_ms, 1),
            ),
            ep_result,
        )


def _bfs_multi_source(
    graph: nx.Graph[Any],
    sources: list[str],
    max_depth: int,
) -> dict[str, int]:
    visited: dict[str, int] = {s: 0 for s in sources if s in graph}
    queue: deque[tuple[str, int]] = deque((s, 0) for s in sources if s in graph)

    while queue:
        node, dist = queue.popleft()
        if dist >= max_depth:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    return visited


def _matches_filter(node_data: dict[str, Any], where_filter: dict[str, Any]) -> bool:
    return all(node_data.get(key) == value for key, value in where_filter.items())
