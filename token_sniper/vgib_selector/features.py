from __future__ import annotations

import hashlib
import keyword
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import networkx as nx
import torch

from token_sniper.vgib_selector.backbone import (
    EMBEDDING_DIM,
    LEXICAL_FEATURE_NAMES,
    NODE_FEATURE_VERSION,
    NODE_TYPE_TO_IDX,
    NODE_TYPES,
)

FILE_NODE_PREFIX = "file:"


def is_file_node_id(node_id: str) -> bool:
    return node_id.startswith(FILE_NODE_PREFIX)


logger = logging.getLogger(__name__)

SIG_MAX_CHARS = 8192

BUNDLE_VERSION = 2
"""Layout of a cached bundle; part of the cache key, so bumping it orphans every old file.

v1: the model inputs.  v2: adds ``source_texts`` for the lexical features.
A change to what a model input means bumps ``backbone.NODE_FEATURE_VERSION``
instead, which is also part of the key."""


@dataclass(frozen=True)
class VGIBFeatureBundle:
    node_ids: list[str]
    type_onehot: torch.Tensor
    signature_emb: torch.Tensor
    degree_log: torch.Tensor
    clustering: torch.Tensor
    file_idx: torch.Tensor
    file_vocab: dict[str, int]
    source_texts: list[str] = field(default_factory=list)
    """Per-node source (at most ``SIG_MAX_CHARS``), read by the lexical features; "" for file nodes."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bundle_version": BUNDLE_VERSION,
                "node_feature_version": NODE_FEATURE_VERSION,
                "node_ids": self.node_ids,
                "type_onehot": self.type_onehot,
                "signature_emb": self.signature_emb,
                "degree_log": self.degree_log,
                "clustering": self.clustering,
                "file_idx": self.file_idx,
                "file_vocab": self.file_vocab,
                "source_texts": self.source_texts,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> VGIBFeatureBundle:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        versions = (payload.pop("bundle_version", 1), payload.pop("node_feature_version", 1))
        if versions != (BUNDLE_VERSION, NODE_FEATURE_VERSION):
            raise ValueError(
                f"{path} holds bundle v{versions[0]} / node features v{versions[1]}, "
                f"this code builds v{BUNDLE_VERSION} / v{NODE_FEATURE_VERSION}"
            )
        return cls(**payload)


class SentenceTransformerSourceEncoder:
    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-code-embeddings-0.5b",
        device: str = "cuda",
        batch_size: int = 64,
        target_dimension: int = EMBEDDING_DIM,
        max_seq_length: int = 8192,
    ) -> None:
        from token_sniper.retrieval.embeddings import SentenceTransformerEmbedder

        self._embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            target_dimension=target_dimension,
            max_seq_length=max_seq_length,
        )
        self.model_name = model_name
        self._dim = target_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self._dim, dtype=torch.float32)

        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i][:SIG_MAX_CHARS] for i in non_empty_indices]

        result = torch.zeros(len(texts), self._dim, dtype=torch.float32)

        if non_empty_texts:
            embeddings = self._embedder.embed_documents(non_empty_texts)
            for row, orig_idx in enumerate(non_empty_indices):
                result[orig_idx] = torch.tensor(embeddings[row], dtype=torch.float32)

        return result


class SourceTextEncoder:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "voyage-code-3",
        batch_size: int = 32,
    ) -> None:
        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not resolved_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is required for VGIB feature encoding. Set it in your .env file."
            )

        from token_sniper.retrieval.embeddings import VoyageCodeEmbedder

        self._embedder = VoyageCodeEmbedder(
            api_key=resolved_key,
            model=model,
            batch_size=batch_size,
            max_retries=5,
            retry_base_delay=2.0,
            max_concurrent_batches=4,
        )
        self.model_name = model
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self._dim, dtype=torch.float32)

        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i][:SIG_MAX_CHARS] for i in non_empty_indices]

        result = torch.zeros(len(texts), self._dim, dtype=torch.float32)

        if non_empty_texts:
            embeddings = self._embedder.embed_documents(non_empty_texts)
            for row, orig_idx in enumerate(non_empty_indices):
                result[orig_idx] = torch.tensor(embeddings[row], dtype=torch.float32)

        return result


class ChromaSourceEncoder:
    def __init__(
        self,
        *,
        chromadb_dir: str | Path = "data/chromadb/",
        collection_prefix: str = "token_sniper",
        model_name: str = "jinaai/jina-code-embeddings-0.5b",
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        from token_sniper.retrieval.vectorstore import EntityVectorStore

        self._store = EntityVectorStore(
            persist_directory=str(chromadb_dir),
            collection_prefix=collection_prefix,
        )
        self.model_name = model_name
        self._dim = embedding_dim
        self._embedding_cache: dict[str, dict[str, list[float]]] = {}
        self._document_cache: dict[str, dict[str, str]] = {}

    @property
    def dimension(self) -> int:
        return self._dim

    def _load_collection(self, collection_repo: str) -> dict[str, list[float]]:
        if collection_repo in self._embedding_cache:
            return self._embedding_cache[collection_repo]
        self._embedding_cache.clear()
        self._document_cache.clear()

        slug = self._store.repo_slug(collection_repo)
        collection = self._store._get_or_create_collection(slug)
        count = collection.count()
        if count == 0:
            logger.warning("ChromaDB collection %r is empty", collection_repo)
            self._embedding_cache[collection_repo] = {}
            self._document_cache[collection_repo] = {}
            return {}

        try:
            lookup: dict[str, list[float]] = {}
            chunks: dict[str, list[tuple[int, str]]] = {}
            batch_size = 5000
            for offset in range(0, count, batch_size):
                result = collection.get(
                    include=["embeddings", "documents"],
                    limit=batch_size,
                    offset=offset,
                )
                ids: list[str] = result.get("ids", [])
                embeddings: list[list[float]] = result.get("embeddings", [])
                documents: list[str] = result.get("documents", [])
                for eid, emb, doc in zip(ids, embeddings, documents, strict=True):
                    chunk = -1
                    if "#chunk" in eid:
                        eid, chunk_str = eid.split("#chunk", 1)
                        chunk = int(chunk_str)
                    chunks.setdefault(eid, []).append((chunk, _strip_file_header(doc or "")))
                    if chunk >= 0 and eid in lookup:
                        continue
                    lookup[eid] = emb
        except Exception:
            logger.warning(
                "Failed to read ChromaDB collection %r — falling back to zero vectors",
                collection_repo,
                exc_info=True,
            )
            self._embedding_cache[collection_repo] = {}
            self._document_cache[collection_repo] = {}
            return {}
        self._document_cache[collection_repo] = {
            eid: "\n".join(text for _, text in sorted(parts)) for eid, parts in chunks.items()
        }

        if lookup:
            sample_dim = len(next(iter(lookup.values())))
            if sample_dim != self._dim:
                logger.warning(
                    "ChromaDB dim %d != expected %d for %r — adjusting",
                    sample_dim, self._dim, collection_repo,
                )
                self._dim = sample_dim

        self._embedding_cache[collection_repo] = lookup
        logger.info(
            "Loaded %d cached embeddings from ChromaDB collection %r",
            len(lookup),
            collection_repo,
        )
        return lookup

    def evict_collection(self, collection_repo: str) -> None:
        if collection_repo in self._embedding_cache:
            del self._embedding_cache[collection_repo]
            self._document_cache.pop(collection_repo, None)
            return
        to_drop = [k for k in self._embedding_cache if k.startswith(collection_repo + "@")]
        for k in to_drop:
            del self._embedding_cache[k]
            self._document_cache.pop(k, None)

    def encode_from_chromadb(
        self,
        node_ids: list[str],
        graph: nx.MultiDiGraph[Any],
        collection_repo: str,
    ) -> torch.Tensor:
        lookup = self._load_collection(collection_repo)
        result = torch.zeros(len(node_ids), self._dim, dtype=torch.float32)
        hits = 0
        for i, nid in enumerate(node_ids):
            if is_file_node_id(nid):
                continue
            data = graph.nodes[nid]
            if data.get("node_type") == "module_preamble":
                continue
            fqn = data.get("fully_qualified_name", nid)
            emb = lookup.get(fqn)
            if emb is not None:
                result[i] = torch.tensor(emb, dtype=torch.float32)
                hits += 1

        logger.info(
            "ChromaDB lookup: %d/%d nodes matched (%.0f%%)",
            hits,
            len(node_ids),
            100.0 * hits / max(len(node_ids), 1),
        )
        return result

    def documents_from_chromadb(
        self,
        node_ids: list[str],
        graph: nx.MultiDiGraph[Any],
        collection_repo: str,
    ) -> list[str]:
        self._load_collection(collection_repo)
        lookup = self._document_cache[collection_repo]
        texts: list[str] = []
        for nid in node_ids:
            data = graph.nodes[nid]
            if is_file_node_id(nid) or data.get("node_type") == "module_preamble":
                texts.append("")
                continue
            texts.append(lookup.get(data.get("fully_qualified_name", nid), "")[:SIG_MAX_CHARS])
        return texts


def _strip_file_header(document: str) -> str:
    if document.startswith("# file: "):
        return document.split("\n", 1)[1] if "\n" in document else ""
    return document


def _extract_source_text(
    data: dict[str, Any],
    node_id: str,
    repo_root: Path | None,
) -> str:
    if is_file_node_id(node_id) or data.get("node_type") == "module_preamble":
        return ""

    if repo_root is not None:
        file_path = data.get("file_path", "")
        start_line = int(data.get("start_line", 0))
        end_line = int(data.get("end_line", 0))
        if file_path and start_line >= 1 and end_line >= start_line:
            abs_path = repo_root / file_path
            try:
                lines = abs_path.read_text(encoding="utf-8").splitlines()
                source = "\n".join(lines[start_line - 1 : end_line])
                if source.strip():
                    return source[:SIG_MAX_CHARS]
            except (OSError, UnicodeDecodeError):
                pass

    fqn = data.get("fully_qualified_name", "")
    node_type = data.get("node_type", "")
    file_path = data.get("file_path", "")
    lines = f"L{data.get('start_line', 0)}-{data.get('end_line', 0)}"
    return f"{node_type} {fqn} ({file_path}:{lines})"


def _content_hash(node_ids: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(f"bundle-v{BUNDLE_VERSION}/nodes-v{NODE_FEATURE_VERSION}\x00".encode())
    h.update(model_name.encode())
    for nid in node_ids:
        h.update(nid.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def edge_index_from_graph(
    graph: nx.MultiDiGraph[Any],
    node_ids: list[str],
) -> torch.Tensor:
    name_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    src_list: list[int] = []
    dst_list: list[int] = []
    for u, v in graph.edges():
        if u in name_to_idx and v in name_to_idx:
            src_list.append(name_to_idx[u])
            dst_list.append(name_to_idx[v])
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


VGIBEncoder = SentenceTransformerSourceEncoder | ChromaSourceEncoder | SourceTextEncoder


def build_vgib_features(
    graph: nx.MultiDiGraph[Any],
    *,
    encoder: VGIBEncoder,
    repo_root: Path | None = None,
    cache_dir: Path | None = None,
    collection_repo: str | None = None,
) -> VGIBFeatureBundle:
    undirected = nx.Graph(graph.to_undirected(as_view=True))
    clustering_map = nx.clustering(undirected)

    node_ids: list[str] = []
    texts: list[str] = []
    types: list[int] = []
    degrees: list[float] = []
    clusts: list[float] = []
    file_paths: list[str] = []

    for nid, data in graph.nodes(data=True):
        node_ids.append(nid)
        node_type = data.get("node_type", "function")
        types.append(NODE_TYPE_TO_IDX.get(node_type, NODE_TYPE_TO_IDX["function"]))
        degrees.append(float(undirected.degree(nid)))
        clusts.append(float(clustering_map.get(nid, 0.0)))
        file_paths.append(data.get("file_path", ""))
        texts.append(_extract_source_text(data, nid, repo_root))

    cache_path: Path | None = None
    if cache_dir is not None:
        key = _content_hash(node_ids, encoder.model_name)
        cache_path = cache_dir / f"vgib_features_v{BUNDLE_VERSION}_{key}.pt"
        if cache_path.exists():
            logger.info("Loading cached VGIB features from %s", cache_path)
            return VGIBFeatureBundle.load(cache_path)

    if isinstance(encoder, ChromaSourceEncoder):
        if collection_repo is None:
            raise ValueError("collection_repo is required when using ChromaSourceEncoder")
        logger.info("Loading %d node embeddings from ChromaDB...", len(node_ids))
        sig_emb = encoder.encode_from_chromadb(node_ids, graph, collection_repo)
        texts = encoder.documents_from_chromadb(node_ids, graph, collection_repo)
    else:
        non_empty = sum(1 for t in texts if t.strip())
        logger.info(
            "Encoding %d nodes with %s (%d with source text)...",
            len(texts), encoder.model_name, non_empty,
        )
        sig_emb = encoder.encode_batch(texts)

    file_vocab: dict[str, int] = {}
    file_idx = torch.zeros(len(node_ids), dtype=torch.long)
    for i, fp in enumerate(file_paths):
        idx = file_vocab.setdefault(fp, len(file_vocab))
        file_idx[i] = idx

    type_onehot = torch.zeros(len(node_ids), len(NODE_TYPES), dtype=torch.float32)
    for i, t in enumerate(types):
        type_onehot[i, t] = 1.0

    bundle = VGIBFeatureBundle(
        node_ids=node_ids,
        type_onehot=type_onehot,
        signature_emb=sig_emb,
        degree_log=torch.log1p(torch.tensor(degrees, dtype=torch.float32)),
        clustering=torch.tensor(clusts, dtype=torch.float32),
        file_idx=file_idx,
        file_vocab=file_vocab,
        source_texts=texts,
    )
    if cache_path is not None:
        bundle.save(cache_path)
    return bundle


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SUBTOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_PY_PATH = re.compile(r"[\w./\\-]+\.py\b")
_TRACE_FRAMES = (
    re.compile(r'File "([^"]+\.py)", line (\d+)'),
    re.compile(r"File '([^']+\.py)', line (\d+)"),
    re.compile(r"([\w./\\-]+\.py):(\d+)"),
)
_STOP_TERMS = frozenset(k.lower() for k in keyword.kwlist) | {"self", "cls", "none", "true", "false"}
_BM25_K1 = 1.2
_BM25_B = 0.75


def _terms(text: str) -> list[str]:
    out: list[str] = []
    for ident in _IDENT.findall(text):
        pieces = [p.lower() for part in ident.split("_") for p in _SUBTOKEN.findall(part)]
        for term in {ident.lower(), *pieces}:
            if len(term) >= 2 and term not in _STOP_TERMS:
                out.append(term)
    return out


def _norm_path(path: str) -> list[str]:
    return [p for p in path.replace("\\", "/").split("/") if p and p != "."]


def _path_matches(mentioned: str, file_path: str) -> bool:
    m, f = _norm_path(mentioned), _norm_path(file_path)
    shared = 0
    while shared < min(len(m), len(f)) and m[-1 - shared] == f[-1 - shared]:
        shared += 1
    return shared >= min(2, len(m), len(f)) and shared > 0


def _dotted_mention(dotted: str, issue_text: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(dotted) + r"(?!\w)", issue_text) is not None


def lexical_node_features(
    *,
    issue_text: str,
    graph: nx.MultiDiGraph[Any],
    node_ids: list[str],
    source_texts: list[str],
    signature_emb: torch.Tensor,
    issue_emb: torch.Tensor,
) -> torch.Tensor:
    n = len(node_ids)
    if len(source_texts) != n or signature_emb.shape[0] != n:
        raise ValueError(
            f"{n} nodes but {len(source_texts)} source texts and {signature_emb.shape[0]} embeddings: "
            f"a feature bundle from before v{BUNDLE_VERSION}?"
        )
    if n and signature_emb.shape[1] != issue_emb.shape[-1]:
        raise ValueError(
            f"source embeddings are {signature_emb.shape[1]}-d, the issue embedding "
            f"{issue_emb.shape[-1]}-d: the cosine needs one embedding space"
        )
    out = torch.zeros(n, len(LEXICAL_FEATURE_NAMES), dtype=torch.float32)
    if n == 0:
        return out

    issue_idents = set(_IDENT.findall(issue_text))
    mentioned_paths = _PY_PATH.findall(issue_text)
    frames = [(path, int(line)) for pat in _TRACE_FRAMES for path, line in pat.findall(issue_text)]

    file_cache: dict[str, tuple[float, float]] = {}
    for i, nid in enumerate(node_ids):
        data = graph.nodes[nid]
        file_path = data.get("file_path", "")
        if file_path not in file_cache:
            module = file_path[:-3].replace("/", ".") if file_path.endswith(".py") else ""
            module = module.removesuffix(".__init__")
            dotted = {m for m in (module, data.get("module_qualname", "")) if "." in m}
            in_issue = any(_path_matches(p, file_path) for p in mentioned_paths) or any(
                _dotted_mention(m, issue_text) for m in dotted
            )
            in_trace = any(_path_matches(p, file_path) for p, _ in frames)
            file_cache[file_path] = (float(in_issue), float(in_trace))
        out[i, 3], out[i, 4] = file_cache[file_path]

        if is_file_node_id(nid):
            continue
        parts = data.get("fully_qualified_name", nid).split(".")
        out[i, 0] = float(parts[-1] in issue_idents)
        out[i, 1] = sum(p in issue_idents for p in parts) / len(parts)
        out[i, 2] = float(len(parts) >= 2 and _dotted_mention(".".join(parts[-2:]), issue_text))
        start, end = int(data.get("start_line", 0)), int(data.get("end_line", 0))
        out[i, 5] = float(any(start <= line <= end and _path_matches(p, file_path) for p, line in frames))

    query = set(_terms(issue_text))
    docs = [Counter(_terms(text)) for text in source_texts]
    lengths = [sum(d.values()) for d in docs]
    n_docs = sum(1 for length in lengths if length)
    if query and n_docs:
        avg_len = sum(lengths) / n_docs
        df = Counter(t for d in docs for t in d.keys() & query)
        idf = {t: math.log(1.0 + (n_docs - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        for i, (doc, length) in enumerate(zip(docs, lengths, strict=True)):
            score = 0.0
            for t, w in idf.items():
                tf = doc.get(t, 0)
                if tf:
                    score += w * tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * length / avg_len))
            out[i, 6] = math.log1p(score)
        best = out[:, 6].max()
        if best > 0:
            out[:, 7] = torch.expm1(out[:, 6]) / torch.expm1(best)

    norms = signature_emb.norm(dim=1) * issue_emb.norm()
    dots = signature_emb @ issue_emb.to(signature_emb.dtype)
    out[:, 8] = torch.where(norms > 0, dots / norms.clamp(min=1e-12), torch.zeros_like(dots))
    return out
