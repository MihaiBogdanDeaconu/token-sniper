from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from token_sniper.evaluation.eval_stages import (
    check_expand_matches,
    check_ssg_matches,
    rank_within_files,
    read_record,
    selection,
    sha256_file,
    summarize_selection,
    write_record,
)
from token_sniper.evaluation.locagent_gold import extract_gold_labels
from token_sniper.evaluation.locagent_ranking import entity_keys
from token_sniper.retrieval.retriever import BM25Retriever
from token_sniper.vgib_selector.features import ChromaSourceEncoder

if TYPE_CHECKING:
    from pathlib import Path


def node(file: str, qualname: str | None, p_v: float, tokens: int = 10) -> dict:
    if qualname is None:
        return {"file": file, "module_key": None, "entity_key": None, "p_v": p_v, "tokens": tokens}
    return {"file": file, "module_key": f"{file}:{qualname.split('.')[0]}", "entity_key": f"{file}:{qualname}",
            "p_v": p_v, "tokens": tokens}


def test_rank_within_files_keeps_file_order_and_sorts_by_p_v_inside():
    nodes = [
        node("b.py", "g", 0.9),
        node("a.py", "A", 0.1),
        node("a.py", "A.run", 0.7),
        node("a.py", "f", 0.3),
        node("a.py", None, 0.99),
        node("c.py", "h", 1.0),
    ]
    modules, entities = rank_within_files(["a.py", "b.py"], nodes)
    assert entities == ["a.py:A.run", "a.py:f", "a.py:A", "b.py:g"]
    assert modules == ["a.py:A", "a.py:f", "b.py:g"]


def test_selection_counts_kept_gold_and_tokens():
    gold = extract_gold_labels("x", ["a.py:A.run", "a.py:f", "z.py:missing"])
    nodes = [node("a.py", "A.run", 0.8, 5), node("a.py", "f", 0.2, 7), node("a.py", None, 0.6, 3)]
    row = selection(nodes, gold, pool_tokens=150)
    assert row["gold_files_kept"] is False
    assert row["gold_functions_in_topk"] is False
    assert row["gold_functions_kept"] is False
    assert row["function_recall"] == pytest.approx(1 / 3)
    assert (row["kept_tokens"], row["topk_tokens"], row["pool_tokens"]) == (8, 15, 150)

    summary = summarize_selection([row])
    assert summary["mean_kept_ratio_of_topk"] == pytest.approx(round(8 / 15, 4))


def test_init_is_merged_into_its_class_like_locagent_gold():
    gold = extract_gold_labels("x", ["a.py:Box.__init__"])
    keys = entity_keys("pkg.a.Box.__init__", {"file_path": "a.py", "module_qualname": "pkg.a",
                                             "fully_qualified_name": "pkg.a.Box.__init__"})
    assert keys == ("a.py:Box", "a.py:Box")
    assert keys[1] in gold.gold_entities
    assert entity_keys("file:a.py", {"file_path": "a.py"}) is None


def test_cached_records_are_rechecked_against_the_ssg_only(tmp_path: Path):
    ssg = tmp_path / "ssg.json"
    ssg.write_text("{}")
    record = tmp_path / "candidates.json"
    header = {"ssg_sha256": sha256_file(ssg), "collection_count": 123}

    check_ssg_matches(record, header, ssg)

    ssg.write_text('{"nodes": []}')
    with pytest.raises(ValueError, match="SSG sha256"):
        check_ssg_matches(record, header, ssg)


def test_expanded_and_plain_caches_cannot_be_mixed(tmp_path: Path):
    plain = {"candidate_files": ["a.py"], "expand_depth": 0}
    expanded = {"candidate_files": ["a.py"], "expand_depth": 3}
    path = tmp_path / "x.json"

    check_expand_matches(path, plain, 0)
    check_expand_matches(path, expanded, 3)
    check_expand_matches(path, {"excluded": "no SSG"}, 3)

    with pytest.raises(ValueError, match="--cache-name"):
        check_expand_matches(path, plain, 3)
    with pytest.raises(ValueError, match="--cache-name"):
        check_expand_matches(path, expanded, 0)


def test_read_record_raises_on_header_mismatch(tmp_path: Path):
    path = tmp_path / "r.json"
    write_record(path, {"checkpoint_sha256": "aa"}, {"nodes": []})
    assert read_record(path, {"checkpoint_sha256": "aa"}) == ({"checkpoint_sha256": "aa"}, {"nodes": []})
    with pytest.raises(ValueError, match="expected"):
        read_record(path, {"checkpoint_sha256": "bb"})


class FakeCollection:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs

    def count(self) -> int:
        return len(self.docs)

    def get(self, include, limit, offset) -> dict:
        batch = self.docs[offset:offset + limit]
        return {"documents": batch, "metadatas": [{} for _ in batch], "ids": [str(i) for i in range(len(batch))]}


class FakeStore:
    @staticmethod
    def repo_slug(repo: str) -> str:
        return repo

    def _get_or_create_collection(self, slug: str) -> FakeCollection:
        return FakeCollection([f"{slug} alpha", f"{slug} beta"])


class FakeEmbeddingStore:
    @staticmethod
    def repo_slug(repo: str) -> str:
        return repo

    def _get_or_create_collection(self, slug: str) -> FakeCollection:
        collection = FakeCollection([f"{slug}.f", f"{slug}.g"])
        collection.get = lambda include, limit, offset: {
            "ids": collection.docs[offset:offset + limit],
            "embeddings": [[1.0, 0.0] for _ in collection.docs[offset:offset + limit]],
            "documents": [f"def {eid}(): pass" for eid in collection.docs[offset:offset + limit]],
        }
        return collection


def test_source_encoder_keeps_only_the_latest_collection():
    encoder = ChromaSourceEncoder.__new__(ChromaSourceEncoder)
    encoder._store = FakeEmbeddingStore()
    encoder._dim = 2
    encoder._embedding_cache = {}
    encoder._document_cache = {}

    first = encoder._load_collection("r1")
    assert encoder._load_collection("r1") is first
    encoder._load_collection("r2")
    assert list(encoder._embedding_cache) == ["r2"]
    assert list(encoder._document_cache) == ["r2"]


def test_bm25_keeps_only_the_latest_repo_index():
    bm25 = BM25Retriever(vector_store=FakeStore())
    first = bm25._build_index("r1")
    assert bm25._build_index("r1") is first
    bm25._build_index("r2")
    assert list(bm25._indices) == ["r2"]
