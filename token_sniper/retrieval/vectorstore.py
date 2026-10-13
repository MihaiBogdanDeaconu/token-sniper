from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StoredEntityMetadata(BaseModel):
    entity_type: str
    file_path: str
    repository: str
    fully_qualified_name: str
    start_line: int = 0
    end_line: int = 0
    estimated_tokens: int = 0

    model_config = {"frozen": True}


class VectorSearchResult(BaseModel):
    entity_id: str
    distance: float
    score: float = Field(description="Similarity score (1 - distance for cosine)")
    metadata: StoredEntityMetadata
    document: str | None = None

    model_config = {"frozen": True}


class VectorSearchResults(BaseModel):
    results: list[VectorSearchResult] = Field(default_factory=list)
    total_indexed: int = 0

    model_config = {"frozen": True}


class EntityVectorStore:
    def __init__(
        self,
        *,
        persist_directory: str | Path = "data/chromadb/",
        collection_prefix: str = "token_sniper",
    ) -> None:
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'chromadb' package is required for the vector store. "
                "Install the project with the 'retrieval' extra."
            ) from exc

        persist_path = Path(persist_directory).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection_prefix = collection_prefix
        self._collections: dict[str, Any] = {}

    def _collection_name(self, repo_slug: str) -> str:
        return f"{self._collection_prefix}_{repo_slug}"

    def _get_or_create_collection(self, repo_slug: str) -> Any:
        if repo_slug in self._collections:
            return self._collections[repo_slug]

        collection_name = self._collection_name(repo_slug)
        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collections[repo_slug] = collection
        return collection

    @staticmethod
    def repo_slug(repo_name: str) -> str:
        import re

        slug = repo_name.replace("/", "__").replace("@", "_at_")
        slug = re.sub(r"[^a-zA-Z0-9._-]", "_", slug)
        slug = slug.strip("._-") or "default"
        return slug

    def index_entities(
        self,
        *,
        entities_data: list[dict[str, Any]],
        repo: str,
        embeddings: list[list[float]],
    ) -> int:
        if len(entities_data) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(entities_data)} entities but {len(embeddings)} embeddings."
            )
        if not entities_data:
            return 0

        slug = self.repo_slug(repo)
        collection = self._get_or_create_collection(slug)

        batch_size = 5000
        total = 0
        for i in range(0, len(entities_data), batch_size):
            batch_entities = entities_data[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [ent["entity_id"] for ent in batch_entities]
            documents = [ent.get("document_text", "") for ent in batch_entities]
            metadatas = [
                {
                    "entity_type": ent["entity_type"],
                    "file_path": ent["file_path"],
                    "repository": repo,
                    "fully_qualified_name": ent["fully_qualified_name"],
                    "start_line": ent.get("start_line", 0),
                    "end_line": ent.get("end_line", 0),
                    "estimated_tokens": ent.get("estimated_tokens", 0),
                }
                for ent in batch_entities
            ]

            collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(ids)

        logger.info("Indexed %d entities for repo %s (collection=%s)", total, repo, slug)
        return total

    def query(
        self,
        *,
        repo: str,
        query_embedding: list[float],
        k: int = 20,
        where_filter: dict[str, Any] | None = None,
        include_documents: bool = True,
    ) -> VectorSearchResults:
        slug = self.repo_slug(repo)
        collection = self._get_or_create_collection(slug)

        include = ["metadatas", "distances"]
        if include_documents:
            include.append("documents")

        actual_k = min(k, collection.count())
        if actual_k <= 0:
            return VectorSearchResults(total_indexed=0)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": include,
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        raw = collection.query(**query_kwargs)

        results: list[VectorSearchResult] = []
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        documents = raw.get("documents", [[]])[0] if include_documents else [None] * len(ids)

        for entity_id, dist, meta, doc in zip(ids, distances, metadatas, documents, strict=True):
            results.append(
                VectorSearchResult(
                    entity_id=entity_id,
                    distance=dist,
                    score=max(0.0, 1.0 - dist),
                    metadata=StoredEntityMetadata(
                        entity_type=str(meta.get("entity_type", "")),
                        file_path=str(meta.get("file_path", "")),
                        repository=str(meta.get("repository", "")),
                        fully_qualified_name=str(meta.get("fully_qualified_name", "")),
                        start_line=int(meta.get("start_line", 0)),
                        end_line=int(meta.get("end_line", 0)),
                        estimated_tokens=int(meta.get("estimated_tokens", 0)),
                    ),
                    document=doc,
                )
            )

        return VectorSearchResults(results=results, total_indexed=collection.count())

    def get_collection_stats(self, repo: str) -> dict[str, Any]:
        slug = self.repo_slug(repo)
        collection = self._get_or_create_collection(slug)
        count = collection.count()
        return {
            "collection_name": self._collection_name(slug),
            "repo": repo,
            "entity_count": count,
        }

    def delete_collection(self, repo: str) -> None:
        slug = self.repo_slug(repo)
        name = self._collection_name(slug)
        try:
            self._client.delete_collection(name)
        except Exception:
            logger.warning("Collection %s did not exist or could not be deleted.", name)
        self._collections.pop(slug, None)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]
