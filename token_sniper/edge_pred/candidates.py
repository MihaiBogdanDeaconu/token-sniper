from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from token_sniper.graph.schema import is_file_node_id

logger = logging.getLogger(__name__)

FS_MAX_HOPS = 5


@dataclass(frozen=True)
class CandidateSet:
    pairs: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.pairs)


def _file_of(graph: nx.MultiDiGraph[Any], nid: str) -> str:
    return graph.nodes[nid].get("file_path", "")


def _fs_distance(path_a: str, path_b: str) -> int:
    if path_a == path_b:
        return 0
    parts_a = path_a.split("/")
    parts_b = path_b.split("/")
    common = 0
    for x, y in zip(parts_a, parts_b):
        if x == y:
            common += 1
        else:
            break
    return (len(parts_a) - common) + (len(parts_b) - common)


def _build_fs_buckets(files: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for f in files:
        top = f.split("/", 1)[0] if "/" in f else "."
        buckets[top].append(f)
    return buckets


def _imports_per_file(graph: nx.MultiDiGraph[Any]) -> dict[str, set[str]]:
    imp: dict[str, set[str]] = defaultdict(set)
    for src, dst, data in graph.edges(data=True):
        if data.get("edge_type") != "imports":
            continue
        src_file = _file_of(graph, src) if src in graph.nodes else ""
        target = data.get("target_name") or dst
        if src_file:
            imp[src_file].add(target)
    return imp


def generate_candidates(
    graph: nx.MultiDiGraph[Any],
    node_ids: list[str],
    *,
    max_fs_hops: int = FS_MAX_HOPS,
    require_shared_import: bool = True,
    max_per_node: int = 500,
) -> CandidateSet:
    name_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    file_to_nodes: dict[str, list[int]] = defaultdict(list)
    for i, nid in enumerate(node_ids):
        if is_file_node_id(nid):
            continue
        if graph.nodes[nid].get("node_type") == "module_preamble":
            continue
        f = _file_of(graph, nid)
        if f:
            file_to_nodes[f].append(i)

    files = list(file_to_nodes.keys())
    imports = _imports_per_file(graph)

    neighbors_by_file: dict[str, list[str]] = {}
    for f in files:
        nearby = [g for g in files if g != f and _fs_distance(f, g) <= max_fs_hops]
        if require_shared_import:
            nearby_plus = [
                g for g in files if g != f and (imports.get(f, set()) & imports.get(g, set()))
            ]
            nearby = sorted(set(nearby) | set(nearby_plus))
        neighbors_by_file[f] = nearby

    existing: set[tuple[int, int]] = set()
    for u, v in graph.edges():
        if u in name_to_idx and v in name_to_idx:
            existing.add((name_to_idx[u], name_to_idx[v]))

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for f, src_nodes in file_to_nodes.items():
        partner_files = neighbors_by_file.get(f, [])
        dst_nodes: list[int] = []
        for g in partner_files:
            dst_nodes.extend(file_to_nodes.get(g, []))
        dst_nodes.extend(src_nodes)

        for s in src_nodes:
            added_for_s = 0
            for d in dst_nodes:
                if s == d:
                    continue
                key = (s, d)
                if key in seen or key in existing:
                    continue
                seen.add(key)
                pairs.append(key)
                added_for_s += 1
                if added_for_s >= max_per_node:
                    break

    logger.info(
        "Generated %d candidate edges over %d nodes (avg %.1f / src-node)",
        len(pairs),
        len(node_ids),
        len(pairs) / max(1, sum(len(v) for v in file_to_nodes.values())),
    )
    return CandidateSet(pairs=pairs)


def edge_index_from_graph(graph: nx.MultiDiGraph[Any], node_ids: list[str]) -> Any:
    import torch

    name_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    srcs: list[int] = []
    dsts: list[int] = []
    for u, v in graph.edges():
        if u not in name_to_idx or v not in name_to_idx:
            continue
        srcs.append(name_to_idx[u])
        dsts.append(name_to_idx[v])
        srcs.append(name_to_idx[v])
        dsts.append(name_to_idx[u])
    return torch.tensor([srcs, dsts], dtype=torch.long)
