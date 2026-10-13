from __future__ import annotations

import random
from typing import Any

import networkx as nx
import pytest
import torch

from token_sniper.graph.schema import file_node_id


def _make_ssg() -> nx.MultiDiGraph[Any]:
    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()

    files = ["pkg/mod.py", "pkg/utils.py", "pkg/sub/deep.py"]
    entities = [
        ("pkg.mod.ClassA", "class", "pkg/mod.py"),
        ("pkg.mod.ClassA.run", "method", "pkg/mod.py"),
        ("pkg.mod.helper", "function", "pkg/mod.py"),
        ("pkg.utils.util_a", "function", "pkg/utils.py"),
        ("pkg.utils.util_b", "function", "pkg/utils.py"),
        ("pkg.sub.deep.deep_fn", "function", "pkg/sub/deep.py"),
    ]

    for fp in files:
        fid = file_node_id(fp)
        g.add_node(fid, node_type="file", file_path=fp)

    for fqn, ntype, fp in entities:
        g.add_node(fqn, node_type=ntype, file_path=fp, start_line=1, end_line=10)
        g.add_edge(file_node_id(fp), fqn, edge_type="contains")

    g.add_edge("pkg.mod.ClassA", "pkg.mod.ClassA.run", edge_type="contains")
    g.add_edge("pkg.mod.ClassA.run", "pkg.mod.helper", edge_type="calls", confidence="high")
    g.add_edge(file_node_id("pkg/mod.py"), file_node_id("pkg/utils.py"), edge_type="imports")

    return g


@pytest.fixture
def small_ssg() -> nx.MultiDiGraph[Any]:
    return _make_ssg()


class TestCandidates:
    def test_generate_candidates_nonempty(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.candidates import generate_candidates

        from token_sniper.graph.schema import is_file_node_id
        node_ids = [n for n in small_ssg.nodes() if not is_file_node_id(n)]
        cands = generate_candidates(small_ssg, node_ids, max_fs_hops=3, require_shared_import=False)
        assert len(cands) > 0
        for s, d in cands.pairs:
            assert s != d
            assert 0 <= s < len(node_ids)
            assert 0 <= d < len(node_ids)

    def test_excludes_existing_edges(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.candidates import generate_candidates

        from token_sniper.graph.schema import is_file_node_id
        node_ids = [n for n in small_ssg.nodes() if not is_file_node_id(n)]
        name_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        cands = generate_candidates(small_ssg, node_ids, max_fs_hops=10, require_shared_import=False)

        existing = set()
        for u, v in small_ssg.edges():
            if u in name_to_idx and v in name_to_idx:
                existing.add((name_to_idx[u], name_to_idx[v]))

        for pair in cands.pairs:
            assert pair not in existing

    def test_edge_index_from_graph(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.candidates import edge_index_from_graph

        node_ids = list(small_ssg.nodes())
        ei = edge_index_from_graph(small_ssg, node_ids)
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0
        assert ei.dtype == torch.long

    def test_fs_distance(self) -> None:
        from token_sniper.edge_pred.candidates import _fs_distance

        assert _fs_distance("a/b/c.py", "a/b/d.py") == 2
        assert _fs_distance("a/b/c.py", "a/b/c.py") == 0
        assert _fs_distance("a/b/c.py", "x/y/z.py") == 6


try:
    import torch_geometric  # noqa: F401
    _has_pyg = True
except ImportError:
    _has_pyg = False

_skip_no_pyg = pytest.mark.skipif(not _has_pyg, reason="torch_geometric not installed")


@_skip_no_pyg
class TestEdgePredictor:
    def test_forward_pass(self) -> None:
        from token_sniper.edge_pred.model import EdgePredictor, EdgePredictorConfig

        cfg = EdgePredictorConfig(num_files=5, hidden_dim=32, out_dim=16, mlp_hidden=32)
        model = EdgePredictor(cfg)
        model.eval()

        n_nodes = 10
        type_onehot = torch.zeros(n_nodes, 5)
        type_onehot[:, 0] = 1
        sig_emb = torch.randn(n_nodes, 1024)
        file_idx = torch.randint(0, 5, (n_nodes,))
        degree_log = torch.randn(n_nodes)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        h = model.encode(type_onehot, sig_emb, file_idx, degree_log, edge_index)
        assert h.shape == (n_nodes, cfg.out_dim)

        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
        scores = model.score_pairs(h, pairs)
        assert scores.shape == (3,)

    def test_gradient_flows(self) -> None:
        from token_sniper.edge_pred.model import EdgePredictor, EdgePredictorConfig

        cfg = EdgePredictorConfig(num_files=3, hidden_dim=16, out_dim=8, mlp_hidden=16)
        model = EdgePredictor(cfg)

        type_onehot = torch.zeros(4, 5)
        type_onehot[:, 0] = 1
        sig_emb = torch.randn(4, 1024)
        file_idx = torch.randint(0, 3, (4,))
        degree_log = torch.randn(4)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        h = model.encode(type_onehot, sig_emb, file_idx, degree_log, edge_index)
        logits = model.score_pairs(h, pairs)
        loss = logits.sum()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


@_skip_no_pyg
class TestFocalLoss:
    def test_focal_loss_zero_when_correct(self) -> None:
        from token_sniper.edge_pred.train import FocalLossWithLogits

        loss_fn = FocalLossWithLogits(gamma=2.0)
        logits = torch.tensor([10.0])
        targets = torch.tensor([1.0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_focal_loss_high_when_wrong(self) -> None:
        from token_sniper.edge_pred.train import FocalLossWithLogits

        loss_fn = FocalLossWithLogits(gamma=2.0)
        logits = torch.tensor([-10.0])
        targets = torch.tensor([1.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 1.0

    def test_focal_loss_gradient(self) -> None:
        from token_sniper.edge_pred.train import FocalLossWithLogits

        loss_fn = FocalLossWithLogits(gamma=2.0)
        logits = torch.tensor([0.5], requires_grad=True)
        targets = torch.tensor([1.0])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


def _dummy_features(node_ids: list[str]) -> Any:
    from token_sniper.edge_pred.features import NodeFeatureBundle

    n = len(node_ids)
    return NodeFeatureBundle(
        node_ids=node_ids,
        type_onehot=torch.zeros(n, 5),
        signature_emb=torch.zeros(n, 1024),
        file_idx=torch.zeros(n, dtype=torch.long),
        degree_log=torch.zeros(n),
        file_vocab={"dummy": 0},
    )


@_skip_no_pyg
class TestAugment:
    def test_augment_adds_edges(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.augment import augment_graph_with_top_m
        from token_sniper.edge_pred.dataset import TaskGraph

        from token_sniper.graph.schema import is_file_node_id

        node_ids = [n for n in small_ssg.nodes() if not is_file_node_id(n)]

        dummy_feats = _dummy_features(node_ids)

        candidates = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long)
        tg = TaskGraph(
            instance_id="test",
            repo="test/repo",
            node_ids=node_ids,
            features=dummy_feats,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            positive_pairs=torch.empty(0, 2, dtype=torch.long),
            candidate_pairs=candidates,
        )

        scores = torch.tensor([0.9, 0.1, 0.5])
        augmented = augment_graph_with_top_m(small_ssg, tg, scores, m=2)

        original_edges = small_ssg.number_of_edges()
        assert augmented.number_of_edges() > original_edges

        for u, v, data in augmented.edges(data=True):
            if data.get("learned"):
                assert data["edge_type"] == "calls"
                assert data["confidence"] == "low"
                assert "predicted_score" in data

    def test_augment_preserves_original(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.augment import augment_graph_with_top_m
        from token_sniper.edge_pred.dataset import TaskGraph

        from token_sniper.graph.schema import is_file_node_id

        node_ids = [n for n in small_ssg.nodes() if not is_file_node_id(n)]
        dummy_feats = _dummy_features(node_ids)
        tg = TaskGraph(
            instance_id="test",
            repo="test/repo",
            node_ids=node_ids,
            features=dummy_feats,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            positive_pairs=torch.empty(0, 2, dtype=torch.long),
            candidate_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        )
        scores = torch.tensor([0.9])
        orig_edge_count = small_ssg.number_of_edges()

        _ = augment_graph_with_top_m(small_ssg, tg, scores, m=1)
        assert small_ssg.number_of_edges() == orig_edge_count

    def test_augment_empty_scores(self, small_ssg: nx.MultiDiGraph[Any]) -> None:
        from token_sniper.edge_pred.augment import augment_graph_with_top_m
        from token_sniper.edge_pred.dataset import TaskGraph

        from token_sniper.graph.schema import is_file_node_id

        node_ids = [n for n in small_ssg.nodes() if not is_file_node_id(n)]
        dummy_feats = _dummy_features(node_ids)
        tg = TaskGraph(
            instance_id="test",
            repo="test/repo",
            node_ids=node_ids,
            features=dummy_feats,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            positive_pairs=torch.empty(0, 2, dtype=torch.long),
            candidate_pairs=torch.empty(0, 2, dtype=torch.long),
        )
        result = augment_graph_with_top_m(small_ssg, tg, torch.empty(0), m=10)
        assert result.number_of_edges() == small_ssg.number_of_edges()


class TestSampleNegatives:
    def test_sample_negatives_basic(self) -> None:
        from token_sniper.edge_pred.dataset import TaskGraph

        node_ids = ["a", "b", "c", "d", "e"]
        dummy_feats = _dummy_features(node_ids)
        candidates = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]], dtype=torch.long)
        positives = torch.tensor([[0, 1]], dtype=torch.long)

        tg = TaskGraph(
            instance_id="test",
            repo="test/repo",
            node_ids=node_ids,
            features=dummy_feats,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            positive_pairs=positives,
            candidate_pairs=candidates,
        )

        negs = tg.sample_negatives(3, random.Random(42))
        assert negs.shape[1] == 2
        assert negs.shape[0] <= 3
        neg_set = {tuple(row.tolist()) for row in negs}
        assert (0, 1) not in neg_set

    def test_sample_negatives_empty_candidates(self) -> None:
        from token_sniper.edge_pred.dataset import TaskGraph

        node_ids = ["a", "b"]
        dummy_feats = _dummy_features(node_ids)
        tg = TaskGraph(
            instance_id="test",
            repo="test/repo",
            node_ids=node_ids,
            features=dummy_feats,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            positive_pairs=torch.empty(0, 2, dtype=torch.long),
            candidate_pairs=torch.empty(0, 2, dtype=torch.long),
        )
        negs = tg.sample_negatives(5, random.Random(42))
        assert negs.shape == (0, 2)
