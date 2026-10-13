from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from token_sniper.parse import CodeEntity, extract_python_entities
from token_sniper.retrieval.embeddings import Embedder, entity_text_for_embedding
from token_sniper.retrieval.vectorstore import EntityVectorStore, VectorSearchResults  # noqa: TC001

logger = logging.getLogger(__name__)


class RetrievedEntity(BaseModel):
    entity_id: str = Field(description="Unique identifier (typically the FQN)")
    score: float = Field(description="Relevance score (higher is better)")
    rank: int = Field(ge=1, description="1-based rank in the result list")
    entity_type: str
    file_path: str
    fully_qualified_name: str
    start_line: int = 0
    end_line: int = 0
    source_text: str = ""
    estimated_tokens: int = 0

    model_config = {"frozen": True}


class RetrievalResult(BaseModel):
    strategy: str
    repo: str
    query: str
    k: int
    entities: list[RetrievedEntity] = Field(default_factory=list)
    total_indexed: int = 0
    latency_ms: float = 0.0

    model_config = {"frozen": True}

    @property
    def entity_ids(self) -> list[str]:
        return [e.entity_id for e in self.entities]

    @property
    def file_paths(self) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for e in self.entities:
            if e.file_path not in seen:
                seen.add(e.file_path)
                paths.append(e.file_path)
        return paths


def prepare_entities_for_indexing(
    entities: list[CodeEntity],
    *,
    repo: str,
    token_threshold: int = 512,
) -> tuple[list[dict[str, Any]], list[str]]:
    deduped: dict[str, CodeEntity] = {}
    duplicates_found = 0
    for entity in entities:
        fqn = entity.fully_qualified_name
        if fqn in deduped:
            duplicates_found += 1
            existing = deduped[fqn]
            if len(entity.source_text) > len(existing.source_text):
                deduped[fqn] = entity
        else:
            deduped[fqn] = entity

    if duplicates_found > 0:
        logger.info(
            "Deduplicated %d entities with duplicate FQNs (kept longest source text)",
            duplicates_found,
        )

    entities_data: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []

    for entity in deduped.values():
        result = entity_text_for_embedding(
            entity.source_text,
            entity.signature_summary,
            entity.estimated_tokens,
            token_threshold=token_threshold,
            file_path=entity.file_path,
        )

        if isinstance(result, list):
            for i, chunk_text in enumerate(result):
                chunk_id = f"{entity.fully_qualified_name}#chunk{i}"
                entities_data.append(
                    {
                        "entity_id": chunk_id,
                        "entity_type": entity.entity_type,
                        "file_path": entity.file_path,
                        "fully_qualified_name": entity.fully_qualified_name,
                        "start_line": entity.start_line,
                        "end_line": entity.end_line,
                        "estimated_tokens": entity.estimated_tokens,
                        "document_text": chunk_text,
                    }
                )
                texts_to_embed.append(chunk_text)
        else:
            entities_data.append(
                {
                    "entity_id": entity.fully_qualified_name,
                    "entity_type": entity.entity_type,
                    "file_path": entity.file_path,
                    "fully_qualified_name": entity.fully_qualified_name,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "estimated_tokens": entity.estimated_tokens,
                    "document_text": result,
                }
            )
            texts_to_embed.append(result)

    return entities_data, texts_to_embed


def build_entity_index(
    *,
    repo_root: Path | str,
    repo_name: str,
    embedder: Embedder,
    vector_store: EntityVectorStore,
    include_glob: str = "**/*.py",
    extra_globs: list[str] | None = None,
    token_threshold: int = 512,
) -> dict[str, Any]:
    from token_sniper.graph.ssg import _NON_PY_EXTENSIONS, _read_file_blob

    root = Path(repo_root).resolve()
    py_files = sorted(p for p in root.glob(include_glob) if p.is_file())

    logger.info("Parsing %d Python files from %s", len(py_files), repo_name)

    all_entities: list[CodeEntity] = []
    seen_paths: set[str] = set()
    for abs_path in py_files:
        rel = abs_path.relative_to(root)
        rel_posix = rel.as_posix()
        seen_paths.add(rel_posix)
        try:
            raw_ents = extract_python_entities(abs_path, project_root=root)
            for e in raw_ents:
                all_entities.append(e.model_copy(update={"file_path": rel_posix}))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.debug("Skipping %s: %s", rel, exc)
            continue

    if extra_globs is None:
        extra_globs = [f"**/*{ext}" for ext in sorted(_NON_PY_EXTENSIONS)]
    n_blobs = 0
    for glob_pat in extra_globs:
        for abs_path in sorted(root.glob(glob_pat)):
            if not abs_path.is_file():
                continue
            rel_posix = abs_path.relative_to(root).as_posix()
            if rel_posix in seen_paths:
                continue
            seen_paths.add(rel_posix)
            blob = _read_file_blob(abs_path, rel_posix)
            if blob is not None:
                all_entities.append(blob)
                n_blobs += 1

    logger.info(
        "Extracted %d entities (%d file blobs) from %s",
        len(all_entities), n_blobs, repo_name,
    )

    if not all_entities:
        return {"repo": repo_name, "entity_count": 0, "embed_time_s": 0, "index_time_s": 0}

    entities_data, texts = prepare_entities_for_indexing(
        all_entities,
        repo=repo_name,
        token_threshold=token_threshold,
    )

    t0 = time.perf_counter()
    embeddings = embedder.embed_documents(texts)
    embed_time = time.perf_counter() - t0
    logger.info("Embedded %d entities in %.1fs", len(embeddings), embed_time)

    t1 = time.perf_counter()
    count = vector_store.index_entities(
        entities_data=entities_data,
        repo=repo_name,
        embeddings=embeddings,
    )
    index_time = time.perf_counter() - t1
    logger.info("Indexed %d entities in %.1fs", count, index_time)

    return {
        "repo": repo_name,
        "entity_count": count,
        "embed_time_s": round(embed_time, 2),
        "index_time_s": round(index_time, 2),
    }


class DenseRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: EntityVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        *,
        issue_text: str,
        repo: str,
        k: int = 20,
        where_filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        query_embedding = self._embedder.embed_query(issue_text)
        raw: VectorSearchResults = self._vector_store.query(
            repo=repo,
            query_embedding=query_embedding,
            k=k,
            where_filter=where_filter,
            include_documents=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        entities = [
            RetrievedEntity(
                entity_id=r.entity_id,
                score=r.score,
                rank=rank,
                entity_type=r.metadata.entity_type,
                file_path=r.metadata.file_path,
                fully_qualified_name=r.metadata.fully_qualified_name,
                start_line=r.metadata.start_line,
                end_line=r.metadata.end_line,
                source_text=r.document or "",
                estimated_tokens=r.metadata.estimated_tokens,
            )
            for rank, r in enumerate(raw.results, start=1)
        ]

        return RetrievalResult(
            strategy="dense_rag",
            repo=repo,
            query=issue_text[:500],
            k=k,
            entities=entities,
            total_indexed=raw.total_indexed,
            latency_ms=round(latency_ms, 1),
        )


class BM25Retriever:
    def __init__(
        self,
        *,
        vector_store: EntityVectorStore,
    ) -> None:
        self._vector_store = vector_store
        self._indices: dict[str, Any] = {}

    def _build_index(self, repo: str) -> Any:
        if repo in self._indices:
            return self._indices[repo]
        self._indices.clear()

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("pip install rank-bm25") from exc

        slug = self._vector_store.repo_slug(repo)
        collection = self._vector_store._get_or_create_collection(slug)
        count = collection.count()

        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        ids: list[str] = []
        batch_size = 5000
        for offset in range(0, count, batch_size):
            batch = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            docs.extend(batch.get("documents", []) or [])
            metas.extend(batch.get("metadatas", []) or [])
            ids.extend(batch.get("ids", []) or [])

        if not docs:
            self._indices[repo] = None
            return None

        tokenized = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)

        self._indices[repo] = {
            "bm25": bm25,
            "docs": docs,
            "metas": metas,
            "ids": ids,
        }
        return self._indices[repo]

    def retrieve(
        self,
        *,
        issue_text: str,
        repo: str,
        k: int = 50,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        idx = self._build_index(repo)

        if idx is None:
            return RetrievalResult(
                strategy="bm25", repo=repo, query=issue_text[:500],
                k=k, entities=[], total_indexed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

        query_tokens = issue_text.lower().split()
        scores = idx["bm25"].get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:k]

        entities = []
        for rank_i, doc_idx in enumerate(top_indices):
            score = float(scores[doc_idx])
            if score <= 0:
                break
            meta = idx["metas"][doc_idx]
            entities.append(RetrievedEntity(
                entity_id=meta.get("fully_qualified_name", idx["ids"][doc_idx]),
                score=score,
                rank=rank_i + 1,
                entity_type=meta.get("entity_type", "unknown"),
                file_path=meta.get("file_path", ""),
                fully_qualified_name=meta.get("fully_qualified_name", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                source_text=idx["docs"][doc_idx][:500],
                estimated_tokens=meta.get("estimated_tokens", 0),
            ))

        return RetrievalResult(
            strategy="bm25",
            repo=repo,
            query=issue_text[:500],
            k=k,
            entities=entities,
            total_indexed=len(idx["docs"]),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


def merge_retrieval_results(
    dense: RetrievalResult,
    sparse: RetrievalResult,
    *,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> RetrievalResult:
    RRF_K = 60

    file_scores: dict[str, float] = {}
    entity_map: dict[str, RetrievedEntity] = {}

    for entity in dense.entities:
        fid = entity.entity_id
        file_scores[fid] = file_scores.get(fid, 0.0) + dense_weight / (RRF_K + entity.rank)
        if fid not in entity_map:
            entity_map[fid] = entity

    for entity in sparse.entities:
        fid = entity.entity_id
        file_scores[fid] = file_scores.get(fid, 0.0) + sparse_weight / (RRF_K + entity.rank)
        if fid not in entity_map:
            entity_map[fid] = entity

    ranked = sorted(file_scores.items(), key=lambda x: -x[1])

    merged_entities = []
    for new_rank, (eid, score) in enumerate(ranked, start=1):
        orig = entity_map[eid]
        merged_entities.append(orig.model_copy(update={
            "score": score,
            "rank": new_rank,
        }))

    return RetrievalResult(
        strategy="hybrid_rrf",
        repo=dense.repo,
        query=dense.query,
        k=len(merged_entities),
        entities=merged_entities,
        total_indexed=max(dense.total_indexed, sparse.total_indexed),
        latency_ms=round(dense.latency_ms + sparse.latency_ms, 1),
    )


def expand_with_graph_neighbors(
    result: RetrievalResult,
    graph: Any,
    *,
    n_seeds: int = 5,
    max_depth: int = 3,
    max_neighbors: int = 200,
) -> RetrievalResult:
    import networkx as nx
    from token_sniper.graph.schema import is_file_node_id

    if not result.entities or graph.number_of_nodes() == 0:
        return result

    existing_files: set[str] = set()
    for e in result.entities:
        existing_files.add(e.file_path)

    seed_fqns: list[str] = []
    for e in result.entities[:n_seeds]:
        if e.fully_qualified_name and e.fully_qualified_name in graph:
            seed_fqns.append(e.fully_qualified_name)
        fid = f"file:{e.file_path}"
        if fid in graph:
            seed_fqns.append(fid)

    if not seed_fqns:
        return result

    neighbor_files: set[str] = set()
    visited: set[str] = set(seed_fqns)
    frontier = list(seed_fqns)

    for _depth in range(max_depth):
        next_frontier: list[str] = []
        for node in frontier:
            for _, target, data in graph.out_edges(node, data=True):
                if target in visited:
                    continue
                edge_type = data.get("edge_type", "")
                if edge_type not in ("calls", "imports", "inherits", "contains"):
                    continue
                visited.add(target)
                next_frontier.append(target)
                if not is_file_node_id(target):
                    fp = graph.nodes[target].get("file_path", "")
                    if fp and fp not in existing_files:
                        neighbor_files.add(fp)
                else:
                    fp = target[len("file:"):]
                    if fp not in existing_files:
                        neighbor_files.add(fp)

            for source, _, data in graph.in_edges(node, data=True):
                if source in visited:
                    continue
                edge_type = data.get("edge_type", "")
                if edge_type not in ("calls", "imports", "inherits"):
                    continue
                visited.add(source)
                next_frontier.append(source)
                if not is_file_node_id(source):
                    fp = graph.nodes[source].get("file_path", "")
                    if fp and fp not in existing_files:
                        neighbor_files.add(fp)

        frontier = next_frontier
        if not frontier:
            break

    new_entities = list(result.entities)
    base_rank = len(new_entities)
    added = 0
    for fp in sorted(neighbor_files):
        if added >= max_neighbors:
            break
        base_rank += 1
        new_entities.append(RetrievedEntity(
            entity_id=f"graph_neighbor:{fp}",
            score=0.0,
            rank=base_rank,
            entity_type="graph_neighbor",
            file_path=fp,
            fully_qualified_name="",
            start_line=0,
            end_line=0,
            source_text="",
            estimated_tokens=0,
        ))
        added += 1

    return RetrievalResult(
        strategy=f"{result.strategy}+graph_expand",
        repo=result.repo,
        query=result.query,
        k=len(new_entities),
        entities=new_entities,
        total_indexed=result.total_indexed,
        latency_ms=result.latency_ms,
    )
