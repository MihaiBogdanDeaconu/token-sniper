from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from token_sniper.graph.schema import is_file_node_id
from token_sniper.vgib_selector.features import (
    ChromaSourceEncoder,
    SentenceTransformerSourceEncoder,
)

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


NODE_TYPES = ("file", "class", "function", "method", "module_preamble")
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
EMBEDDING_DIM = 256

EdgePredEncoder = ChromaSourceEncoder | SentenceTransformerSourceEncoder


@dataclass(frozen=True)
class NodeFeatureBundle:
    node_ids: list[str]
    type_onehot: torch.Tensor
    signature_emb: torch.Tensor
    degree_log: torch.Tensor
    file_idx: torch.Tensor
    file_vocab: dict[str, int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "node_ids": self.node_ids,
                "type_onehot": self.type_onehot,
                "signature_emb": self.signature_emb,
                "degree_log": self.degree_log,
                "file_idx": self.file_idx,
                "file_vocab": self.file_vocab,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "NodeFeatureBundle":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("clustering", None)
        return cls(**payload)


SignatureEncoder = EdgePredEncoder


def _signature_text(data: dict[str, Any]) -> str:
    fqn = data.get("fully_qualified_name", "")
    node_type = data.get("node_type", "")
    file_path = data.get("file_path", "")
    lines = f"L{data.get('start_line', 0)}-{data.get('end_line', 0)}"
    return f"{node_type} {fqn} ({file_path}:{lines})"


def _content_hash(node_ids: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    for nid in node_ids:
        h.update(nid.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def build_node_features(
    graph: nx.MultiDiGraph[Any],
    *,
    encoder: EdgePredEncoder,
    cache_dir: Path | None = None,
    collection_repo: str | None = None,
) -> NodeFeatureBundle:
    undirected = nx.Graph(graph.to_undirected(as_view=True))

    node_ids: list[str] = []
    texts: list[str] = []
    types: list[int] = []
    degrees: list[float] = []
    file_paths: list[str] = []

    has_text: list[bool] = []
    for nid, data in graph.nodes(data=True):
        node_ids.append(nid)
        node_type = data.get("node_type", "function")
        types.append(NODE_TYPE_TO_IDX.get(node_type, NODE_TYPE_TO_IDX["function"]))
        degrees.append(float(undirected.degree(nid)))
        file_paths.append(data.get("file_path", ""))
        if is_file_node_id(nid) or node_type == "module_preamble":
            texts.append("")
            has_text.append(False)
        else:
            texts.append(_signature_text(data))
            has_text.append(True)

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = cache_dir / f"features_{_content_hash(node_ids, encoder.model_name)}.pt"
        if cache_path.exists():
            logger.info("Loading cached features from %s", cache_path)
            return NodeFeatureBundle.load(cache_path)

    if isinstance(encoder, ChromaSourceEncoder):
        if collection_repo is None:
            raise ValueError("collection_repo is required when using ChromaSourceEncoder")
        logger.info("Loading %d node embeddings from ChromaDB...", len(node_ids))
        sig_emb = encoder.encode_from_chromadb(node_ids, graph, collection_repo)
    else:
        logger.info(
            "Encoding %d signatures with %s (%d non-empty)...",
            len(texts), encoder.model_name, sum(has_text),
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

    bundle = NodeFeatureBundle(
        node_ids=node_ids,
        type_onehot=type_onehot,
        signature_emb=sig_emb,
        degree_log=torch.log1p(torch.tensor(degrees, dtype=torch.float32)),
        file_idx=file_idx,
        file_vocab=file_vocab,
    )
    if cache_path is not None:
        bundle.save(cache_path)
    return bundle
