from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING

import networkx as nx
import pytest
import torch

from scripts.submit_vgib_k8s import _build_runner_script, _model_flags
from token_sniper.vgib_selector.backbone import (
    LEXICAL_FEATURE_NAMES,
    LEXICAL_FEATURE_VERSION,
    GraphSAGEConfig,
)
from token_sniper.vgib_selector.dataset import load_reranked_vgib_task, structural_prior_inputs
from token_sniper.vgib_selector.features import (
    BUNDLE_VERSION,
    VGIBFeatureBundle,
    build_vgib_features,
    lexical_node_features,
)
from token_sniper.vgib_selector.selector import VGIBSelector
from token_sniper.vgib_selector.train import VGIBTrainConfig, train_vgib_compression

if TYPE_CHECKING:
    from pathlib import Path

COL = {name: i for i, name in enumerate(LEXICAL_FEATURE_NAMES)}
DIM = 8

ISSUE = """QuerySet.filter() crashes on an empty list.

Traceback (most recent call last):
  File "/usr/lib/python3/site-packages/pkg/query.py", line 12, in filter
    return self._chain(args)
ValueError: empty
"""

SOURCES = {
    "file:pkg/query.py": "",
    "pkg.query.QuerySet": "class QuerySet:\n    def filter(self, args): ...\n    def exclude(self): ...",
    "pkg.query.QuerySet.filter": "def filter(self, args):\n    if not args:\n        raise ValueError\n    return self._chain(args)",
    "pkg.query.QuerySet.exclude": "def exclude(self):\n    return self._negate()",
    "file:pkg/other.py": "",
    "pkg.other.helper": "def helper(x):\n    return x + 1",
}


def _graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("file:pkg/query.py", node_type="file", file_path="pkg/query.py", start_line=1, end_line=30)
    g.add_node("pkg.query.QuerySet", node_type="class", file_path="pkg/query.py", start_line=3, end_line=30,
               fully_qualified_name="pkg.query.QuerySet", module_qualname="pkg.query")
    g.add_node("pkg.query.QuerySet.filter", node_type="method", file_path="pkg/query.py", start_line=10,
               end_line=15, fully_qualified_name="pkg.query.QuerySet.filter", module_qualname="pkg.query")
    g.add_node("pkg.query.QuerySet.exclude", node_type="method", file_path="pkg/query.py", start_line=17,
               end_line=20, fully_qualified_name="pkg.query.QuerySet.exclude", module_qualname="pkg.query")
    g.add_node("file:pkg/other.py", node_type="file", file_path="pkg/other.py", start_line=1, end_line=5)
    g.add_node("pkg.other.helper", node_type="function", file_path="pkg/other.py", start_line=1, end_line=2,
               fully_qualified_name="pkg.other.helper", module_qualname="pkg.other")
    g.add_edge("file:pkg/query.py", "pkg.query.QuerySet", edge_type="contains")
    g.add_edge("pkg.query.QuerySet", "pkg.query.QuerySet.filter", edge_type="contains")
    g.add_edge("pkg.query.QuerySet", "pkg.query.QuerySet.exclude", edge_type="contains")
    g.add_edge("file:pkg/other.py", "pkg.other.helper", edge_type="contains")
    g.add_edge("pkg.query.QuerySet.filter", "pkg.other.helper", edge_type="calls")
    return g


class TextEncoder:
    model_name = "test-encoder"

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), DIM)
        for i, text in enumerate(texts):
            for ch in text:
                out[i, ord(ch) % DIM] += 1.0
        return out


class TextEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return TextEncoder().encode_batch([text])[0].tolist()


def _lexical(issue: str = ISSUE) -> tuple[list[str], torch.Tensor]:
    g = _graph()
    node_ids = list(g.nodes)
    texts = [SOURCES[n] for n in node_ids]
    emb = TextEncoder().encode_batch(texts)
    issue_emb = torch.tensor(TextEmbedder().embed_query(issue))
    feats = lexical_node_features(issue_text=issue, graph=g, node_ids=node_ids, source_texts=texts,
                                  signature_emb=emb, issue_emb=issue_emb)
    return node_ids, feats


def _row(node_ids: list[str], feats: torch.Tensor, nid: str) -> dict[str, float]:
    return {name: feats[node_ids.index(nid), i].item() for name, i in COL.items()}


def test_lexical_features_point_at_the_function_the_issue_names():
    node_ids, feats = _lexical()
    assert feats.shape == (len(node_ids), len(LEXICAL_FEATURE_NAMES))

    target = _row(node_ids, feats, "pkg.query.QuerySet.filter")
    sibling = _row(node_ids, feats, "pkg.query.QuerySet.exclude")
    other = _row(node_ids, feats, "pkg.other.helper")

    assert target["name_in_issue"] == 1.0 and sibling["name_in_issue"] == 0.0
    assert target["qualname_in_issue"] == 1.0 and sibling["qualname_in_issue"] == 0.0
    assert target["qualname_part_frac"] > sibling["qualname_part_frac"]
    assert target["file_in_traceback"] == sibling["file_in_traceback"] == 1.0
    assert other["file_in_traceback"] == 0.0
    assert target["line_in_traceback"] == 1.0 and sibling["line_in_traceback"] == 0.0
    assert other["line_in_traceback"] == 0.0
    assert target["bm25_rel"] == pytest.approx(1.0)
    assert target["bm25_log"] > _row(node_ids, feats, "pkg.query.QuerySet")["bm25_log"] > 0.0
    assert sibling["bm25_log"] == other["bm25_log"] == 0.0


def test_lexical_file_mentions_and_cosine():
    node_ids, feats = _lexical("pkg.other is slow, see pkg/other.py")
    assert _row(node_ids, feats, "pkg.other.helper")["file_in_issue"] == 1.0
    assert _row(node_ids, feats, "file:pkg/other.py")["file_in_issue"] == 1.0
    assert _row(node_ids, feats, "pkg.query.QuerySet")["file_in_issue"] == 0.0

    cos = feats[:, COL["issue_cosine"]]
    assert cos[node_ids.index("file:pkg/query.py")].item() == 0.0
    assert ((cos >= -1.0) & (cos <= 1.0 + 1e-6)).all()


def test_lexical_features_are_deterministic_and_zero_on_an_empty_issue():
    assert torch.equal(_lexical()[1], _lexical()[1])
    _, feats = _lexical("")
    assert feats[:, : COL["issue_cosine"]].abs().sum().item() == 0.0


def test_lexical_features_reject_a_bundle_without_source_texts():
    g = _graph()
    with pytest.raises(ValueError, match="source texts"):
        lexical_node_features(issue_text=ISSUE, graph=g, node_ids=list(g.nodes), source_texts=[],
                              signature_emb=torch.zeros(len(g), DIM), issue_emb=torch.zeros(DIM))


def test_feature_cache_is_versioned(tmp_path: Path):
    g = _graph()
    bundle = build_vgib_features(g, encoder=TextEncoder(), cache_dir=tmp_path)
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].name.startswith(f"vgib_features_v{BUNDLE_VERSION}_")
    assert len(bundle.source_texts) == len(bundle.node_ids)

    cached = build_vgib_features(g, encoder=TextEncoder(), cache_dir=tmp_path)
    assert cached.source_texts == bundle.source_texts

    payload = torch.load(files[0], weights_only=False)
    payload["bundle_version"] = BUNDLE_VERSION - 1
    torch.save(payload, files[0])
    with pytest.raises(ValueError, match="bundle v1"):
        VGIBFeatureBundle.load(files[0])


def _task(gold: str = "pkg.query.QuerySet.filter"):
    g = _graph()
    task = load_reranked_vgib_task(
        instance_id="t1", repo="o/r", issue_text=ISSUE,
        candidate_files=["pkg/query.py", "pkg/other.py"], gold_files=["pkg/query.py"], graph=g,
        encoder=TextEncoder(), embedder=TextEmbedder(),
        node_tokens={n: 10 for n in g.nodes}, gold_node_ids={gold},
    )
    return task


def _selector(**cfg) -> VGIBSelector:
    torch.manual_seed(0)
    base = {"num_files": 4, "source_emb_dim": DIM, "task_emb_dim": DIM, "hidden_dim": 16, "out_dim": 8}
    return VGIBSelector(GraphSAGEConfig(**{**base, **cfg})).eval()


LEXICAL = {"lexical_dim": len(LEXICAL_FEATURE_NAMES), "lexical_version": LEXICAL_FEATURE_VERSION}


def test_loader_attaches_lexical_features_and_forward_task_uses_them():
    task = _task()
    assert task.lexical_features.shape == (len(task.features.node_ids), len(LEXICAL_FEATURE_NAMES))

    model = _selector(**LEXICAL)
    out = model.forward_task(task, torch.device("cpu"))
    assert out["retention_probs"].shape == (len(task.features.node_ids),)

    feats = task.features
    n = len(feats.node_ids)
    with pytest.raises(ValueError, match="lexical"):
        model(feats.type_onehot, feats.signature_emb, feats.file_idx, feats.degree_log, feats.clustering,
              task.task_embedding.expand(n, -1), task.edge_index)

    assert _selector().forward_task(task, torch.device("cpu"))["retention_probs"].shape == (n,)


def test_checkpoint_with_another_feature_definition_refuses_to_load(tmp_path: Path):
    from token_sniper.rl.trainer import load_warm_start_policy

    model = _selector(**LEXICAL)
    good = tmp_path / "good.pt"
    torch.save({"model_state": model.state_dict(), "model_cfg": model.cfg.__dict__}, good)
    assert load_warm_start_policy(good, device="cpu").selector.cfg.lexical_version == LEXICAL_FEATURE_VERSION

    stale = tmp_path / "stale.pt"
    torch.save({"model_state": model.state_dict(),
                "model_cfg": {**model.cfg.__dict__, "lexical_version": LEXICAL_FEATURE_VERSION + 1}}, stale)
    with pytest.raises(ValueError, match="lexical features"):
        load_warm_start_policy(stale, device="cpu")

    with pytest.raises(ValueError, match="node features"):
        GraphSAGEConfig(num_files=1, node_feature_version=99)

    legacy = {k: v for k, v in _selector().cfg.__dict__.items()
              if k not in {"node_feature_version", "lexical_dim", "lexical_version", "use_edges", "pooling"}}
    assert GraphSAGEConfig(**legacy).lexical_dim == 0


def test_no_graph_edges_ignores_the_edges_it_is_given():
    task = _task()
    no_edges = _selector(**LEXICAL, use_edges=False)
    with_edges = _selector(**LEXICAL)
    assert task.edge_index.shape[1] > 0

    stripped = replace(task, edge_index=torch.empty((2, 0), dtype=torch.long))
    dev = torch.device("cpu")
    a = no_edges.forward_task(task, dev)["retention_probs"]
    b = no_edges.forward_task(stripped, dev)["retention_probs"]
    assert torch.equal(a, b)
    assert not torch.equal(with_edges.forward_task(task, dev)["retention_probs"],
                           with_edges.forward_task(stripped, dev)["retention_probs"])


def test_mean_pooling_only_changes_the_pooled_prediction():
    task = _task()
    dev = torch.device("cpu")
    attn = _selector(**LEXICAL).forward_task(task, dev)
    mean = _selector(**LEXICAL, pooling="mean").forward_task(task, dev)
    assert torch.equal(attn["retention_probs"], mean["retention_probs"])
    assert attn["prediction_logit"].item() != mean["prediction_logit"].item()
    with pytest.raises(ValueError, match="pooling"):
        GraphSAGEConfig(num_files=1, pooling="max")


def test_structural_prior_inputs_on_the_full_graph():
    g = _graph()
    node_ids = ["pkg.query.QuerySet.exclude", "pkg.query.QuerySet.filter", "pkg.other.helper", "file:pkg/other.py"]
    ppr, near = structural_prior_inputs(g, ["pkg.query.QuerySet.filter", "not.in.graph"], node_ids)
    assert near.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert ppr[1] == ppr.max() and (ppr > 0).all()

    ppr, near = structural_prior_inputs(g, ["not.in.graph"], node_ids)
    assert ppr.sum().item() == 0.0 and near.sum().item() == 0.0


@pytest.mark.parametrize(
    "switches",
    [
        {},
        {"use_edges": False, "gamma": 0.0, "pooling": "mean", "lexical_features": False},
        {"prior_weight": 1.0, "prior_alpha": 2.0, "prior_gamma_s": 1.0, "prior_base_rate": 0.1},
    ],
)
def test_compression_training_records_what_it_trained_on(tmp_path: Path, monkeypatch, switches):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    tasks = [_task(), _task("pkg.query.QuerySet.exclude")]
    for i, t in enumerate(tasks):
        t.instance_id = f"t{i}"
        t.prior_ppr, t.prior_seed_mask = structural_prior_inputs(
            _graph(), ["pkg.query.QuerySet.filter"], t.features.node_ids,
        )
    cfg = VGIBTrainConfig(output_dir=tmp_path / "out", epochs=2, beta=0.1, gamma=switches.pop("gamma", 0.001),
                          max_val_token_ratio=1.0, hidden_dim=16, out_dim=8, device="cpu", **switches)
    best = train_vgib_compression(tasks[:1], cfg, val_tasks=tasks[1:])
    ckpt = torch.load(best, weights_only=False)

    assert ckpt["feature_versions"]["bundle"] == BUNDLE_VERSION
    assert ckpt["feature_versions"]["lexical"] == (LEXICAL_FEATURE_VERSION if cfg.lexical_features else 0)
    assert ckpt["first_epoch_within_budget"] == 1
    model_cfg = GraphSAGEConfig(**ckpt["model_cfg"])
    assert (model_cfg.use_edges, model_cfg.pooling) == (cfg.use_edges, cfg.pooling)
    assert model_cfg.lexical_dim == (len(LEXICAL_FEATURE_NAMES) if cfg.lexical_features else 0)


def test_prior_run_needs_prior_inputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    cfg = VGIBTrainConfig(output_dir=tmp_path, epochs=1, prior_weight=1.0, hidden_dim=16, out_dim=8, device="cpu")
    with pytest.raises(ValueError, match="without prior inputs"):
        train_vgib_compression([_task()], cfg, val_tasks=[_task()])


def _submit_args(**overrides) -> argparse.Namespace:
    base = {"no_graph_edges": False, "pooling": "cross_attention", "no_lexical_features": False,
            "prior_weight": 0.0, "prior_alpha": 1.0, "prior_gamma_s": 1.0, "prior_base_rate": 0.5}
    return argparse.Namespace(**{**base, **overrides})


def test_submit_passes_model_flags_to_train_vgib():
    assert _model_flags(_submit_args()) == []
    flags = _model_flags(_submit_args(no_graph_edges=True, pooling="mean", no_lexical_features=True,
                                      prior_weight=1.0, prior_alpha=2.0))
    assert flags == ["--no-graph-edges", "--pooling mean", "--no-lexical-features", "--prior-weight 1.0",
                     "--prior-alpha 2.0", "--prior-gamma-s 1.0", "--prior-base-rate 0.5"]

    script = _build_runner_script(job_name="j", embedding_model="m", embedding_dim=256, reranker_dir="r",
                                  epochs=1, lr=1e-4, beta=0.1, gamma=0.0, patience=1, model_flags=flags)
    train_call = script[script.index("python scripts/train_vgib.py"):script.index("VGIB Training Complete")]
    for flag in flags:
        assert flag in train_call
    assert "REPO=/workspace/runs/j/repo" in script
