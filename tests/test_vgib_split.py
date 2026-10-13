from __future__ import annotations

import json
from typing import TYPE_CHECKING

import networkx as nx
import pytest
import torch

from scripts.train_vgib import check_loaded_tasks, load_split_records
from token_sniper.vgib_selector.dataset import load_reranked_vgib_task

if TYPE_CHECKING:
    from pathlib import Path


class ZeroEncoder:
    model_name = "zero"

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), 8)


class ZeroEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8


def _split(tmp_path: Path, train: list[str], val: list[str]) -> Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": train, "val": val}))
    return path


def _records(*ids: str) -> list[dict]:
    return [{"instance_id": i, "candidate_files": [], "gold_files": []} for i in ids]


def test_split_partitions_records_in_split_order(tmp_path: Path) -> None:
    train, val = load_split_records(_split(tmp_path, ["b", "a"], ["c"]), _records("a", "b", "c"))
    assert [r["instance_id"] for r in train] == ["b", "a"]
    assert [r["instance_id"] for r in val] == ["c"]


@pytest.mark.parametrize(
    ("train", "val", "records"),
    [
        (["a", "b"], ["c"], ("a", "c")),
        (["a"], ["c"], ("a", "b", "c")),
        (["a", "c"], ["c"], ("a", "c")),
    ],
)
def test_split_mismatch_raises(tmp_path: Path, train: list[str], val: list[str], records: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        load_split_records(_split(tmp_path, train, val), _records(*records))


def _graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("file:a.py", node_type="file", file_path="a.py", start_line=1, end_line=1)
    g.add_node("a.f", node_type="function", file_path="a.py", start_line=1, end_line=2)
    g.add_node("a.g", node_type="function", file_path="a.py", start_line=4, end_line=5)
    g.add_edge("file:a.py", "a.f", edge_type="contains")
    g.add_edge("file:a.py", "a.g", edge_type="contains")
    return g


def _load(node_tokens: dict[str, int] | None):
    return load_reranked_vgib_task(
        instance_id="t1",
        repo="o/r",
        issue_text="fix f",
        candidate_files=["a.py"],
        gold_files=["a.py"],
        graph=_graph(),
        encoder=ZeroEncoder(),
        embedder=ZeroEmbedder(),
        node_tokens=node_tokens,
    )


def test_loader_takes_token_counts_from_sidecar() -> None:
    counts = {"file:a.py": 3, "a.f": 11, "a.g": 7}
    task = _load(counts)
    assert task.token_counts.tolist() == [float(counts[n]) for n in task.features.node_ids]


def test_loader_raises_on_node_without_count() -> None:
    with pytest.raises(KeyError, match="a.g"):
        _load({"file:a.py": 3, "a.f": 11})


def test_inference_load_has_no_token_counts() -> None:
    assert _load(None).token_counts is None


def test_check_loaded_tasks_rejects_lost_or_uncounted_tasks() -> None:
    task = _load({"file:a.py": 3, "a.f": 11, "a.g": 7})
    check_loaded_tasks(_records("t1"), [task], {}, "TRAIN")
    with pytest.raises(RuntimeError, match="failed to load"):
        check_loaded_tasks(_records("t1", "t2"), [task], {}, "TRAIN")
    with pytest.raises(RuntimeError, match="no token counts"):
        check_loaded_tasks(_records("t1"), [_load(None)], {}, "TRAIN")


def test_check_loaded_tasks_accepts_instances_dropped_for_a_stated_reason() -> None:
    task = _load({"file:a.py": 3, "a.f": 11, "a.g": 7})
    check_loaded_tasks(_records("t1", "t2"), [task], {"t2": "no gold node"}, "TRAIN")
    with pytest.raises(RuntimeError, match="failed to load"):
        check_loaded_tasks(_records("t1", "t2", "t3"), [task], {"t2": "no gold node"}, "TRAIN")
