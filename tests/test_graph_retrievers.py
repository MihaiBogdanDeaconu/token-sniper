from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Any

import networkx as nx
import pytest

from token_sniper.graph.schema import file_node_id
from token_sniper.retrieval.entry_points import (
    EntryPointResult,
    _parse_fqn_response,
    identify_entry_points,
    identify_entry_points_by_embedding,
)
from token_sniper.retrieval.graph_retrievers import (
    GraphBFSRetriever,
    GraphPPRRetriever,
    _bfs_multi_source,
)
from token_sniper.retrieval.retriever import RetrievalResult


def _build_test_graph() -> nx.MultiDiGraph:
    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()

    g.add_node(file_node_id("pkg/mod.py"), node_type="file", file_path="pkg/mod.py", start_line=1, end_line=50)
    g.add_node(file_node_id("pkg/other.py"), node_type="file", file_path="pkg/other.py", start_line=1, end_line=30)

    entities = {
        "pkg.mod.ClassA": ("class", "pkg/mod.py", 1, 20),
        "pkg.mod.ClassA.__init__": ("method", "pkg/mod.py", 2, 5),
        "pkg.mod.ClassA.process": ("method", "pkg/mod.py", 7, 15),
        "pkg.mod.helper": ("function", "pkg/mod.py", 22, 30),
        "pkg.mod.util": ("function", "pkg/mod.py", 32, 40),
    }
    for fqn, (etype, fpath, sl, el) in entities.items():
        g.add_node(fqn, node_type=etype, file_path=fpath, start_line=sl, end_line=el, fully_qualified_name=fqn)

    entities2 = {
        "pkg.other.ClassB": ("class", "pkg/other.py", 1, 15),
        "pkg.other.ClassB.run": ("method", "pkg/other.py", 5, 12),
        "pkg.other.standalone": ("function", "pkg/other.py", 17, 25),
    }
    for fqn, (etype, fpath, sl, el) in entities2.items():
        g.add_node(fqn, node_type=etype, file_path=fpath, start_line=sl, end_line=el, fully_qualified_name=fqn)

    g.add_edge(file_node_id("pkg/mod.py"), "pkg.mod.ClassA", edge_type="contains")
    g.add_edge("pkg.mod.ClassA", "pkg.mod.ClassA.__init__", edge_type="contains")
    g.add_edge("pkg.mod.ClassA", "pkg.mod.ClassA.process", edge_type="contains")
    g.add_edge(file_node_id("pkg/mod.py"), "pkg.mod.helper", edge_type="contains")
    g.add_edge(file_node_id("pkg/mod.py"), "pkg.mod.util", edge_type="contains")

    g.add_edge(file_node_id("pkg/other.py"), "pkg.other.ClassB", edge_type="contains")
    g.add_edge("pkg.other.ClassB", "pkg.other.ClassB.run", edge_type="contains")
    g.add_edge(file_node_id("pkg/other.py"), "pkg.other.standalone", edge_type="contains")

    g.add_edge("pkg.other.ClassB", "pkg.mod.ClassA", edge_type="inherits")

    g.add_edge("pkg.other.ClassB.run", "pkg.mod.helper", edge_type="calls")

    g.add_edge(file_node_id("pkg/other.py"), file_node_id("pkg/mod.py"), edge_type="imports")

    return g


class FakeEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[hash(t) % 100 / 100.0 + i / 1000.0 for i in range(self._dim)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [hash(text) % 100 / 100.0 + i / 1000.0 for i in range(self._dim)]


class TestBFSMultiSource:
    def test_single_source(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        result = _bfs_multi_source(g, ["A"], max_depth=2)
        assert result == {"A": 0, "B": 1, "C": 2}
        assert "D" not in result

    def test_multi_source(self) -> None:
        g = nx.Graph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "D")])
        result = _bfs_multi_source(g, ["A", "E"], max_depth=1)
        assert result["A"] == 0
        assert result["B"] == 1
        assert result["E"] == 0
        assert result["D"] == 1

    def test_missing_source_ignored(self) -> None:
        g = nx.Graph()
        g.add_edge("A", "B")
        result = _bfs_multi_source(g, ["MISSING", "A"], max_depth=1)
        assert result == {"A": 0, "B": 1}

    def test_zero_depth(self) -> None:
        g = nx.Graph()
        g.add_edge("A", "B")
        result = _bfs_multi_source(g, ["A"], max_depth=0)
        assert result == {"A": 0}

    def test_empty_sources(self) -> None:
        g = nx.Graph()
        g.add_edge("A", "B")
        result = _bfs_multi_source(g, [], max_depth=2)
        assert result == {}


class TestGraphBFSRetriever:
    def test_bfs_from_known_seed(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        retriever = GraphBFSRetriever(
            graph=graph,
            entry_point_method="embedding",
            max_hops=2,
            n_entry_points=1,
            embedder=embedder,
            vector_store=store,
        )

        result, ep = retriever.retrieve(
            issue_text="Fix the process method",
            repo="test/repo",
            k=20,
        )

        assert isinstance(result, RetrievalResult)
        assert result.strategy == "graph_bfs"
        assert len(result.entities) > 0
        assert all(e.rank >= 1 for e in result.entities)
        assert ep.method == "embedding"
        assert len(ep.matched_node_ids) > 0

    def test_bfs_hop_distance_ordering(self) -> None:
        graph = _build_test_graph()

        undirected = graph.to_undirected(as_view=True)
        distances = _bfs_multi_source(undirected, ["pkg.other.ClassB.run"], max_depth=2)

        assert distances["pkg.other.ClassB.run"] == 0
        assert distances["pkg.other.ClassB"] == 1
        assert distances["pkg.mod.helper"] == 1

    def test_bfs_no_entry_points_returns_empty(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        from token_sniper.retrieval.vectorstore import EntityVectorStore

        store = EntityVectorStore(persist_directory=tmp_path / "empty", collection_prefix="test")

        retriever = GraphBFSRetriever(
            graph=graph,
            entry_point_method="embedding",
            max_hops=2,
            embedder=embedder,
            vector_store=store,
        )

        result, ep = retriever.retrieve(issue_text="test", repo="test/repo", k=10)
        assert len(result.entities) == 0


class TestGraphPPRRetriever:
    def test_ppr_produces_ranked_results(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        retriever = GraphPPRRetriever(
            graph=graph,
            entry_point_method="embedding",
            alpha=0.15,
            n_entry_points=1,
            embedder=embedder,
            vector_store=store,
        )

        result, ep = retriever.retrieve(
            issue_text="Fix the process method",
            repo="test/repo",
            k=10,
        )

        assert isinstance(result, RetrievalResult)
        assert result.strategy == "graph_ppr"
        assert len(result.entities) > 0
        scores = [e.score for e in result.entities]
        assert scores == sorted(scores, reverse=True)

    def test_ppr_respects_alpha(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        retriever_high = GraphPPRRetriever(
            graph=graph,
            entry_point_method="embedding",
            alpha=0.5,
            n_entry_points=1,
            embedder=embedder,
            vector_store=store,
        )

        result_high, _ = retriever_high.retrieve(
            issue_text="Fix the process method",
            repo="test/repo",
            k=10,
        )

        retriever_low = GraphPPRRetriever(
            graph=graph,
            entry_point_method="embedding",
            alpha=0.05,
            n_entry_points=1,
            embedder=embedder,
            vector_store=store,
        )

        result_low, _ = retriever_low.retrieve(
            issue_text="Fix the process method",
            repo="test/repo",
            k=10,
        )

        assert len(result_high.entities) > 0
        assert len(result_low.entities) > 0


class TestEntryPoints:
    def test_embedding_entry_points(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        result = identify_entry_points_by_embedding(
            issue_text="Fix the helper function",
            repo="test/repo",
            embedder=embedder,
            vector_store=store,
            graph=graph,
            n_candidates=3,
        )

        assert isinstance(result, EntryPointResult)
        assert result.method == "embedding"
        assert len(result.matched_node_ids) > 0
        for node_id in result.matched_node_ids:
            assert node_id in graph.nodes()

    def test_identify_entry_points_dispatcher(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        result = identify_entry_points(
            method="embedding",
            issue_text="test",
            repo="test/repo",
            graph=graph,
            embedder=embedder,
            vector_store=store,
        )
        assert result.method == "embedding"

    def test_unknown_method_raises(self) -> None:
        graph = _build_test_graph()
        with pytest.raises(ValueError, match="Unknown"):
            identify_entry_points(
                method="invalid",
                issue_text="test",
                repo="test/repo",
                graph=graph,
            )

    def test_parse_fqn_response_json_object(self) -> None:
        text = '{"entry_points": ["pkg.mod.helper", "pkg.mod.ClassA.process"]}'
        result = _parse_fqn_response(text)
        assert result == ["pkg.mod.helper", "pkg.mod.ClassA.process"]

    def test_parse_fqn_response_json_array(self) -> None:
        text = 'The relevant entities are: ["pkg.mod.helper", "pkg.mod.ClassA.process"]'
        result = _parse_fqn_response(text)
        assert result == ["pkg.mod.helper", "pkg.mod.ClassA.process"]

    def test_parse_fqn_response_with_think_block(self) -> None:
        text = '<think>Let me analyze the issue...</think>{"entry_points": ["pkg.mod.helper"]}'
        result = _parse_fqn_response(text)
        assert result == ["pkg.mod.helper"]

    def test_parse_fqn_response_plain(self) -> None:
        text = 'I think "pkg.mod.helper" and "pkg.other.ClassB" are relevant.'
        result = _parse_fqn_response(text)
        assert "pkg.mod.helper" in result
        assert "pkg.other.ClassB" in result

    def test_parse_fqn_response_empty(self) -> None:
        result = _parse_fqn_response("No entities found.")
        assert result == []


class TestPhase1Evaluation:
    def test_compute_file_level_metrics(self) -> None:
        from token_sniper.evaluation.phase1 import compute_file_level_metrics

        prec, rec = compute_file_level_metrics(
            retrieved_files=["a.py", "b.py", "c.py", "d.py"],
            gold_files=["a.py", "c.py"],
            k_values=[2, 4],
        )
        assert prec[2] == 0.5
        assert prec[4] == 0.5
        assert rec[2] == 0.5
        assert rec[4] == 1.0

    def test_aggregate_condition_metrics(self) -> None:
        from token_sniper.evaluation.phase1 import (
            TaskRetrievalMetrics,
            aggregate_condition_metrics,
        )

        tasks = [
            TaskRetrievalMetrics(
                instance_id="task1",
                condition="test",
                repo="test/repo",
                difficulty="easy",
                precision_at_k={10: 0.5, 20: 0.3},
                recall_at_k={10: 0.8, 20: 1.0},
                latency_ms=100.0,
            ),
            TaskRetrievalMetrics(
                instance_id="task2",
                condition="test",
                repo="test/repo",
                difficulty="hard",
                precision_at_k={10: 0.3, 20: 0.2},
                recall_at_k={10: 0.5, 20: 0.7},
                latency_ms=200.0,
            ),
        ]

        summary = aggregate_condition_metrics(tasks, condition="test", k_values=[10, 20])
        assert summary.task_count == 2
        assert summary.mean_precision_at_k[10] == 0.4
        assert summary.mean_recall_at_k[10] == 0.65
        assert summary.mean_latency_ms == 150.0
        assert "easy" in summary.per_difficulty
        assert "hard" in summary.per_difficulty

    def test_kill_criterion_killed(self) -> None:
        from token_sniper.evaluation.phase1 import ConditionSummary, evaluate_kill_criterion

        summaries = {
            "embedding_knn": ConditionSummary(
                condition="embedding_knn",
                task_count=10,
                mean_precision_at_k={10: 0.5, 20: 0.4},
            ),
            "graph_bfs": ConditionSummary(
                condition="graph_bfs",
                task_count=10,
                mean_precision_at_k={10: 0.3, 20: 0.2},
            ),
            "graph_ppr": ConditionSummary(
                condition="graph_ppr",
                task_count=10,
                mean_precision_at_k={10: 0.4, 20: 0.3},
            ),
        }

        killed, detail = evaluate_kill_criterion(summaries, [10, 20])
        assert killed is True
        assert "Neither" in detail

    def test_kill_criterion_go(self) -> None:
        from token_sniper.evaluation.phase1 import ConditionSummary, evaluate_kill_criterion

        summaries = {
            "embedding_knn": ConditionSummary(
                condition="embedding_knn",
                task_count=10,
                mean_precision_at_k={10: 0.3, 20: 0.2},
            ),
            "graph_bfs": ConditionSummary(
                condition="graph_bfs",
                task_count=10,
                mean_precision_at_k={10: 0.5, 20: 0.4},
            ),
            "graph_ppr": ConditionSummary(
                condition="graph_ppr",
                task_count=10,
                mean_precision_at_k={10: 0.2, 20: 0.1},
            ),
        }

        killed, detail = evaluate_kill_criterion(summaries, [10, 20])
        assert killed is False
        assert "graph_bfs beats" in detail

    def test_build_summary_table(self) -> None:
        from token_sniper.evaluation.phase1 import ConditionSummary, build_summary_table

        summaries = {
            "embedding_knn": ConditionSummary(
                condition="embedding_knn",
                task_count=10,
                mean_precision_at_k={10: 0.5},
                mean_recall_at_k={10: 0.8},
                mean_latency_ms=50.0,
            ),
        }

        table = build_summary_table(summaries, [10])
        assert "Precision@k" in table
        assert "Recall@k" in table
        assert "embedding_knn" in table
        assert "0.5000" in table

    def test_write_decision(self, tmp_path: Path) -> None:
        from token_sniper.evaluation.phase1 import (
            ConditionSummary,
            Phase1Summary,
            write_phase1_decision,
        )

        summary = Phase1Summary(
            run_id="test_run",
            task_count=10,
            k_values=[10, 20],
            conditions={
                "embedding_knn": ConditionSummary(
                    condition="embedding_knn",
                    task_count=10,
                    mean_precision_at_k={10: 0.5, 20: 0.4},
                    mean_recall_at_k={10: 0.8, 20: 0.9},
                ),
            },
            kill_criterion_met=False,
            kill_criterion_detail="Test detail",
        )

        path = write_phase1_decision(output_dir=tmp_path, summary=summary)
        assert path.exists()
        content = path.read_text()
        assert "Phase 1" in content
        assert "Proceed to Phase 2" in content


class TestSharedInterface:
    def test_all_strategies_produce_retrieval_result(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        strategies = [
            ("graph_bfs", GraphBFSRetriever(
                graph=graph, entry_point_method="embedding",
                max_hops=2, embedder=embedder, vector_store=store,
            )),
            ("graph_ppr", GraphPPRRetriever(
                graph=graph, entry_point_method="embedding",
                alpha=0.15, embedder=embedder, vector_store=store,
            )),
        ]

        for name, retriever in strategies:
            result, ep = retriever.retrieve(
                issue_text="Fix the helper function",
                repo="test/repo",
                k=5,
            )
            assert isinstance(result, RetrievalResult), f"{name} didn't return RetrievalResult"
            assert result.strategy == name
            assert isinstance(result.entity_ids, list)
            assert isinstance(result.file_paths, list)

        from token_sniper.retrieval.retriever import DenseRetriever

        dense = DenseRetriever(embedder=embedder, vector_store=store)
        result = dense.retrieve(issue_text="Fix it", repo="test/repo", k=5)
        assert isinstance(result, RetrievalResult)
        assert result.strategy == "dense_rag"

    def test_retrieval_result_file_paths_deduplication(self, tmp_path: Path) -> None:
        graph = _build_test_graph()
        embedder = FakeEmbedder(dim=64)
        store = _build_indexed_store(graph, embedder, tmp_path)

        retriever = GraphBFSRetriever(
            graph=graph, entry_point_method="embedding",
            max_hops=2, embedder=embedder, vector_store=store,
        )

        result, _ = retriever.retrieve(issue_text="test", repo="test/repo", k=20)
        assert len(result.file_paths) == len(set(result.file_paths))


def _build_indexed_store(
    graph: nx.MultiDiGraph[Any],
    embedder: FakeEmbedder,
    tmp_path: Path,
) -> Any:
    from token_sniper.graph.schema import is_file_node_id
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

    entities_data = []
    texts = []
    for node_id, data in graph.nodes(data=True):
        if is_file_node_id(node_id):
            continue
        if data.get("node_type") == "module_preamble":
            continue
        text = f"def {node_id.split('.')[-1]}(): pass"
        entities_data.append({
            "entity_id": node_id,
            "entity_type": data.get("node_type", ""),
            "file_path": data.get("file_path", ""),
            "fully_qualified_name": node_id,
            "start_line": data.get("start_line", 0),
            "end_line": data.get("end_line", 0),
            "estimated_tokens": 10,
            "document_text": text,
        })
        texts.append(text)

    embeddings = embedder.embed_documents(texts)
    store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)
    return store
