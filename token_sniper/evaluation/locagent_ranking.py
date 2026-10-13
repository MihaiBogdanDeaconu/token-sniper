from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import torch

from token_sniper.graph.schema import is_file_node_id

if TYPE_CHECKING:
    from collections.abc import Mapping


def _within_file_qualname(fully_qualified_name: str, module_qualname: str) -> str:
    prefix = f"{module_qualname}." if module_qualname else ""
    if prefix and fully_qualified_name.startswith(prefix):
        return fully_qualified_name[len(prefix):]
    return fully_qualified_name


def _strip_init(qualname: str) -> str:
    suffix = ".__init__"
    return qualname[: -len(suffix)] if qualname.endswith(suffix) else qualname


@dataclass(frozen=True)
class EntityScore:
    node_id: str
    file_path: str
    entity_key: str
    """LocAgent-format function-level key: "file_path:qualname", .__init__ stripped."""
    module_key: str
    """LocAgent-format module-level key: "file_path:top_level_qualname_segment"."""
    score: float


def entity_keys(node_id: str, node_data: Mapping[str, Any]) -> tuple[str, str] | None:
    if is_file_node_id(node_id):
        return None
    file_path = node_data.get("file_path", "")
    fqn = node_data.get("fully_qualified_name", "") or node_id
    within_file = _within_file_qualname(fqn, node_data.get("module_qualname", ""))
    if not within_file or not file_path:
        return None
    return f"{file_path}:{within_file.split('.')[0]}", f"{file_path}:{_strip_init(within_file)}"


def rank_entities(
    node_ids: list[str],
    p_v: torch.Tensor,
    graph: nx.MultiDiGraph,
) -> list[EntityScore]:
    scores: list[EntityScore] = []
    for nid, prob in zip(node_ids, p_v.tolist(), strict=True):
        node_data = graph.nodes.get(nid)
        if node_data is None:
            continue
        keys = entity_keys(nid, node_data)
        if keys is None:
            continue
        module_key, entity_key = keys

        scores.append(EntityScore(
            node_id=nid,
            file_path=node_data["file_path"],
            entity_key=entity_key,
            module_key=module_key,
            score=float(prob),
        ))
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def _maxpool(entity_scores: list[EntityScore], key_fn) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for s in entity_scores:
        key = key_fn(s)
        if not key:
            continue
        if s.score > best.get(key, float("-inf")):
            best[key] = s.score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def maxpool_to_files(entity_scores: list[EntityScore]) -> list[tuple[str, float]]:
    return _maxpool(entity_scores, lambda s: s.file_path)


def maxpool_to_modules(entity_scores: list[EntityScore]) -> list[tuple[str, float]]:
    return _maxpool(entity_scores, lambda s: s.module_key)


def maxpool_to_entities(entity_scores: list[EntityScore]) -> list[tuple[str, float]]:
    return _maxpool(entity_scores, lambda s: s.entity_key)
