from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from token_sniper.parse.entities import CodeEntity
from token_sniper.retrieval.config import EmbeddingConfig, VectorStoreConfig
from token_sniper.retrieval.embeddings import (
    VoyageCodeEmbedder,
    _estimate_tokens,
    _token_aware_batches,
    entity_text_for_embedding,
)
from token_sniper.retrieval.retriever import (
    DenseRetriever,
    RetrievalResult,
    RetrievedEntity,
    build_entity_index,
    prepare_entities_for_indexing,
)
from token_sniper.retrieval.vectorstore import (
    EntityVectorStore,
    StoredEntityMetadata,
    VectorSearchResult,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "minimal_pkg"

SAMPLE_ENTITIES = [
    CodeEntity(
        entity_type="function",
        fully_qualified_name="minimal_pkg.core.greet",
        file_path="minimal_pkg/core.py",
        module_qualname="minimal_pkg.core",
        start_line=1,
        end_line=3,
        source_text='def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello, {name}!"',
        referenced_identifiers=["name"],
        estimated_tokens=20,
    ),
    CodeEntity(
        entity_type="class",
        fully_qualified_name="minimal_pkg.core.Greeter",
        file_path="minimal_pkg/core.py",
        module_qualname="minimal_pkg.core",
        start_line=5,
        end_line=12,
        source_text=(
            'class Greeter:\n    """A greeter class."""\n\n'
            "    def __init__(self, prefix: str) -> None:\n"
            "        self.prefix = prefix\n\n"
            "    def say(self, name: str) -> str:\n"
            '        return f"{self.prefix}, {name}!"'
        ),
        referenced_identifiers=["prefix", "name"],
        estimated_tokens=40,
    ),
    CodeEntity(
        entity_type="method",
        fully_qualified_name="minimal_pkg.core.Greeter.say",
        file_path="minimal_pkg/core.py",
        module_qualname="minimal_pkg.core",
        start_line=11,
        end_line=12,
        source_text='    def say(self, name: str) -> str:\n        return f"{self.prefix}, {name}!"',
        referenced_identifiers=["prefix", "name"],
        estimated_tokens=15,
    ),
]

LARGE_ENTITY = CodeEntity(
    entity_type="function",
    fully_qualified_name="minimal_pkg.large.big_function",
    file_path="minimal_pkg/large.py",
    module_qualname="minimal_pkg.large",
    start_line=1,
    end_line=100,
    source_text="def big_function():\n" + "    x = 1\n" * 200,
    signature_summary="def big_function() -> None: ...",
    referenced_identifiers=[],
    estimated_tokens=600,
)


def _fake_embeddings(n: int, dim: int = 1024) -> list[list[float]]:
    return [[float(i + j) / (dim * n) for j in range(dim)] for i in range(n)]


class FakeEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(texts)
        return [[hash(t) % 100 / 100.0 + i / 1000.0 for i in range(self._dim)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return [hash(text) % 100 / 100.0 + i / 1000.0 for i in range(self._dim)]


class TestEmbeddingConfig:
    def test_default_config(self) -> None:
        config = EmbeddingConfig()
        assert config.name == "voyage-code-3"
        assert config.provider == "voyageai"
        assert config.dimension == 1024
        assert config.api_key_env == "VOYAGE_API_KEY"
        assert config.batch_size == 32

    def test_resolve_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key-123")
        config = EmbeddingConfig()
        assert config.resolve_api_key() == "test-key-123"

    def test_resolve_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        config = EmbeddingConfig()
        assert config.resolve_api_key() is None


class TestVectorStoreConfig:
    def test_default_config(self) -> None:
        config = VectorStoreConfig()
        assert config.backend == "chromadb"
        assert config.persist_directory == "data/chromadb/"
        assert config.collection_prefix == "token_sniper"


class TestEntityTextSelection:
    def test_small_entity_uses_source_text(self) -> None:
        result = entity_text_for_embedding(
            source_text="def hello(): pass",
            signature_summary="def hello() -> None: ...",
            estimated_tokens=10,
        )
        assert result == "def hello(): pass"

    def test_large_entity_with_summary_uses_summary(self) -> None:
        result = entity_text_for_embedding(
            source_text="def big():\n" + "    x = 1\n" * 200,
            signature_summary="def big() -> None: ...",
            estimated_tokens=600,
        )
        assert result == "def big() -> None: ..."

    def test_large_entity_without_summary_uses_source(self) -> None:
        source = "def big():\n" + "    x = 1\n" * 200
        result = entity_text_for_embedding(
            source_text=source,
            signature_summary=None,
            estimated_tokens=600,
        )
        assert result == source

    def test_custom_threshold(self) -> None:
        result = entity_text_for_embedding(
            source_text="def f(): pass",
            signature_summary="def f(): ...",
            estimated_tokens=50,
            token_threshold=30,
        )
        assert result == "def f(): ..."


class TestTokenEstimation:
    def test_basic_estimation(self) -> None:
        assert _estimate_tokens("hello world") >= 1

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 1

    def test_longer_text(self) -> None:
        text = "x" * 400
        tokens = _estimate_tokens(text)
        assert tokens == 100


class TestTokenAwareBatches:
    def test_single_batch_within_limits(self) -> None:
        texts = ["hello", "world"]
        batches = _token_aware_batches(texts, max_batch_size=10, max_tokens=10000)
        assert len(batches) == 1
        assert batches[0] == texts

    def test_splits_by_count(self) -> None:
        texts = ["a", "b", "c", "d", "e"]
        batches = _token_aware_batches(texts, max_batch_size=2, max_tokens=10000)
        assert len(batches) == 3
        assert batches[0] == ["a", "b"]
        assert batches[1] == ["c", "d"]
        assert batches[2] == ["e"]

    def test_splits_by_tokens(self) -> None:
        texts = ["x" * 100, "y" * 100, "z" * 100]
        batches = _token_aware_batches(texts, max_batch_size=100, max_tokens=30)
        assert len(batches) == 3

    def test_empty_input(self) -> None:
        batches = _token_aware_batches([], max_batch_size=10, max_tokens=10000)
        assert batches == []


class TestVoyageCodeEmbedder:
    def test_embed_documents_calls_api(self) -> None:
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        with patch("voyageai.Client", return_value=mock_client):
            embedder = VoyageCodeEmbedder(api_key="test-key")
            result = embedder.embed_documents(["code1", "code2"])

        assert len(result) == 2
        assert len(result[0]) == 1024
        mock_client.embed.assert_called_once_with(
            ["code1", "code2"],
            model="voyage-code-3",
            input_type="document",
        )

    def test_embed_query_uses_query_type(self) -> None:
        mock_result = MagicMock()
        mock_result.embeddings = [[0.5] * 1024]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        with patch("voyageai.Client", return_value=mock_client):
            embedder = VoyageCodeEmbedder(api_key="test-key")
            result = embedder.embed_query("fix the import error")

        assert len(result) == 1024
        mock_client.embed.assert_called_once_with(
            ["fix the import error"],
            model="voyage-code-3",
            input_type="query",
        )

    def test_dimension_property(self) -> None:
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        with patch("voyageai.Client", return_value=mock_client):
            embedder = VoyageCodeEmbedder(api_key="test-key")
            assert embedder.dimension == 1024
            embedder.embed_documents(["test"])
            assert embedder.dimension == 1024

    def test_empty_input(self) -> None:
        with patch("voyageai.Client"):
            embedder = VoyageCodeEmbedder(api_key="test-key")
            result = embedder.embed_documents([])
        assert result == []

    def test_batching_for_large_input(self) -> None:
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        with patch("voyageai.Client", return_value=mock_client):
            embedder = VoyageCodeEmbedder(api_key="test-key", batch_size=2)
            texts = [f"code_{i}" for i in range(5)]
            embedder.embed_documents(texts)

        assert mock_client.embed.call_count == 3


class TestEntityVectorStore:
    def test_index_and_query(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        dim = 64
        entities_data = [
            {
                "entity_id": "pkg.func_a",
                "entity_type": "function",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.func_a",
                "start_line": 1,
                "end_line": 5,
                "estimated_tokens": 20,
                "document_text": "def func_a(): pass",
            },
            {
                "entity_id": "pkg.func_b",
                "entity_type": "function",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.func_b",
                "start_line": 7,
                "end_line": 10,
                "estimated_tokens": 15,
                "document_text": "def func_b(): return 1",
            },
        ]
        embeddings = [[1.0] * dim, [0.5] * dim]

        count = store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)
        assert count == 2

        results = store.query(
            repo="test/repo",
            query_embedding=[1.0] * dim,
            k=2,
        )
        assert len(results.results) == 2
        assert results.total_indexed == 2
        assert results.results[0].metadata.entity_type == "function"

    def test_persistence_across_restarts(self, tmp_path: Path) -> None:
        persist_dir = tmp_path / "chroma"
        dim = 64

        store1 = EntityVectorStore(persist_directory=persist_dir, collection_prefix="test")
        entities_data = [
            {
                "entity_id": "pkg.func",
                "entity_type": "function",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.func",
                "start_line": 1,
                "end_line": 3,
                "estimated_tokens": 10,
                "document_text": "def func(): pass",
            },
        ]
        store1.index_entities(entities_data=entities_data, repo="test/repo", embeddings=[[1.0] * dim])
        del store1

        store2 = EntityVectorStore(persist_directory=persist_dir, collection_prefix="test")
        stats = store2.get_collection_stats("test/repo")
        assert stats["entity_count"] == 1

        results = store2.query(repo="test/repo", query_embedding=[1.0] * dim, k=1)
        assert len(results.results) == 1
        assert results.results[0].entity_id == "pkg.func"

    def test_metadata_filtering(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        dim = 64
        entities_data = [
            {
                "entity_id": "pkg.ClassA",
                "entity_type": "class",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.ClassA",
                "start_line": 1,
                "end_line": 10,
                "estimated_tokens": 30,
                "document_text": "class ClassA: pass",
            },
            {
                "entity_id": "pkg.func_a",
                "entity_type": "function",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.func_a",
                "start_line": 12,
                "end_line": 15,
                "estimated_tokens": 10,
                "document_text": "def func_a(): pass",
            },
        ]
        embeddings = [[1.0] * dim, [0.8] * dim]
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        results = store.query(
            repo="test/repo",
            query_embedding=[1.0] * dim,
            k=10,
            where_filter={"entity_type": "class"},
        )
        assert len(results.results) == 1
        assert results.results[0].metadata.entity_type == "class"

    def test_collection_stats(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        stats = store.get_collection_stats("empty/repo")
        assert stats["entity_count"] == 0

    def test_delete_collection(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        dim = 64
        entities_data = [
            {
                "entity_id": "pkg.func",
                "entity_type": "function",
                "file_path": "pkg/mod.py",
                "fully_qualified_name": "pkg.func",
                "document_text": "def func(): pass",
            },
        ]
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=[[1.0] * dim])
        assert store.get_collection_stats("test/repo")["entity_count"] == 1

        store.delete_collection("test/repo")
        stats = store.get_collection_stats("test/repo")
        assert stats["entity_count"] == 0

    def test_repo_slug(self) -> None:
        assert EntityVectorStore.repo_slug("django/django") == "django__django"
        assert EntityVectorStore.repo_slug("scikit-learn/scikit-learn") == "scikit-learn__scikit-learn"
        slug = EntityVectorStore.repo_slug("astropy/astropy@298ccb478e6b")
        assert "@" not in slug
        assert slug == "astropy__astropy_at_298ccb478e6b"

    def test_mismatched_entities_embeddings_raises(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        with pytest.raises(ValueError, match="Mismatch"):
            store.index_entities(
                entities_data=[{"entity_id": "a", "entity_type": "function", "file_path": "x.py", "fully_qualified_name": "a", "document_text": ""}],
                repo="test/repo",
                embeddings=[],
            )

    def test_empty_entities(self, tmp_path: Path) -> None:
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")
        count = store.index_entities(entities_data=[], repo="test/repo", embeddings=[])
        assert count == 0


class TestPrepareEntities:
    def test_basic_preparation(self) -> None:
        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        assert len(entities_data) == 3
        assert len(texts) == 3
        assert entities_data[0]["entity_id"] == "minimal_pkg.core.greet"
        assert entities_data[0]["entity_type"] == "function"
        assert "def greet" in texts[0]

    def test_large_entity_uses_summary(self) -> None:
        entities_data, texts = prepare_entities_for_indexing([LARGE_ENTITY], repo="test/repo")
        assert len(texts) == 1
        assert texts[0] == "def big_function() -> None: ..."

    def test_empty_input(self) -> None:
        entities_data, texts = prepare_entities_for_indexing([], repo="test/repo")
        assert entities_data == []
        assert texts == []


class TestDenseRetriever:
    def test_end_to_end_retrieve(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        embeddings = embedder.embed_documents(texts)
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(
            issue_text="Fix the greeting function",
            repo="test/repo",
            k=3,
        )

        assert isinstance(result, RetrievalResult)
        assert result.strategy == "dense_rag"
        assert result.repo == "test/repo"
        assert len(result.entities) == 3
        assert result.entities[0].rank == 1
        assert result.latency_ms > 0

    def test_entity_ids_property(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        embeddings = embedder.embed_documents(texts)
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(issue_text="test", repo="test/repo", k=3)

        assert len(result.entity_ids) == 3
        assert all(isinstance(eid, str) for eid in result.entity_ids)

    def test_file_paths_property(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        embeddings = embedder.embed_documents(texts)
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(issue_text="test", repo="test/repo", k=3)

        assert result.file_paths == ["minimal_pkg/core.py"]

    def test_metadata_filter(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        embeddings = embedder.embed_documents(texts)
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(
            issue_text="test",
            repo="test/repo",
            k=10,
            where_filter={"entity_type": "class"},
        )

        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "class"

    def test_query_latency_under_500ms(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        entities_data, texts = prepare_entities_for_indexing(SAMPLE_ENTITIES, repo="test/repo")
        embeddings = embedder.embed_documents(texts)
        store.index_entities(entities_data=entities_data, repo="test/repo", embeddings=embeddings)

        retriever = DenseRetriever(embedder=embedder, vector_store=store)

        t0 = time.perf_counter()
        result = retriever.retrieve(issue_text="Fix the greeting bug", repo="test/repo", k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 500, f"Query took {elapsed_ms:.1f}ms, expected < 500ms"
        assert result.latency_ms < 500

    def test_empty_index(self, tmp_path: Path) -> None:
        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(issue_text="test", repo="empty/repo", k=10)

        assert len(result.entities) == 0
        assert result.total_indexed == 0


class TestBuildEntityIndex:
    def test_index_minimal_pkg(self, tmp_path: Path) -> None:
        if not _FIXTURES_DIR.is_dir():
            pytest.skip("Fixture directory not found")

        embedder = FakeEmbedder(dim=64)
        store = EntityVectorStore(persist_directory=tmp_path / "chroma", collection_prefix="test")

        stats = build_entity_index(
            repo_root=_FIXTURES_DIR,
            repo_name="test/minimal_pkg",
            embedder=embedder,
            vector_store=store,
        )

        assert stats["repo"] == "test/minimal_pkg"
        assert stats["entity_count"] > 0
        assert stats["embed_time_s"] >= 0
        assert stats["index_time_s"] >= 0

        retriever = DenseRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve(
            issue_text="call function",
            repo="test/minimal_pkg",
            k=5,
        )
        assert len(result.entities) > 0


class TestRetrievalModels:
    def test_retrieved_entity_frozen(self) -> None:
        entity = RetrievedEntity(
            entity_id="pkg.func",
            score=0.95,
            rank=1,
            entity_type="function",
            file_path="pkg/mod.py",
            fully_qualified_name="pkg.func",
        )
        with pytest.raises(ValidationError):
            entity.score = 0.5  # type: ignore[misc]

    def test_retrieval_result_entity_ids(self) -> None:
        result = RetrievalResult(
            strategy="dense_rag",
            repo="test/repo",
            query="test",
            k=2,
            entities=[
                RetrievedEntity(
                    entity_id="a.func1",
                    score=0.9,
                    rank=1,
                    entity_type="function",
                    file_path="a.py",
                    fully_qualified_name="a.func1",
                ),
                RetrievedEntity(
                    entity_id="b.func2",
                    score=0.8,
                    rank=2,
                    entity_type="function",
                    file_path="b.py",
                    fully_qualified_name="b.func2",
                ),
            ],
        )
        assert result.entity_ids == ["a.func1", "b.func2"]
        assert result.file_paths == ["a.py", "b.py"]

    def test_stored_entity_metadata(self) -> None:
        meta = StoredEntityMetadata(
            entity_type="function",
            file_path="pkg/mod.py",
            repository="test/repo",
            fully_qualified_name="pkg.func",
            start_line=1,
            end_line=5,
        )
        assert meta.entity_type == "function"
        assert meta.repository == "test/repo"

    def test_vector_search_result_score(self) -> None:
        result = VectorSearchResult(
            entity_id="pkg.func",
            distance=0.2,
            score=0.8,
            metadata=StoredEntityMetadata(
                entity_type="function",
                file_path="pkg/mod.py",
                repository="test/repo",
                fully_qualified_name="pkg.func",
            ),
        )
        assert result.score == 0.8
        assert result.distance == 0.2


class TestCreateEmbedder:
    def test_voyage_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        config = EmbeddingConfig(provider="voyageai")
        from token_sniper.retrieval.embeddings import create_embedder

        with pytest.raises(RuntimeError, match="API key"):
            create_embedder(config)

    def test_voyage_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        config = EmbeddingConfig(provider="voyageai")

        with patch("voyageai.Client"):
            from token_sniper.retrieval.embeddings import create_embedder

            embedder = create_embedder(config)
            assert isinstance(embedder, VoyageCodeEmbedder)

    def test_unknown_provider_raises(self) -> None:
        config = EmbeddingConfig(provider="voyageai")  # type: ignore[arg-type]
        config_dict = config.model_dump()
        config_dict["provider"] = "unknown"
        from token_sniper.retrieval.embeddings import create_embedder

        with pytest.raises((ValueError, Exception)):
            create_embedder(EmbeddingConfig.model_validate(config_dict))
