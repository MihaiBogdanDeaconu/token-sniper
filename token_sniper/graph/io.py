from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from token_sniper.graph.schema import (
    SSG_META_UNRESOLVED_KEY,
    SSG_META_VERSION_KEY,
    SSG_SCHEMA_VERSION,
)


def write_ssg_json(graph: nx.MultiDiGraph[Any], path: Path | str) -> None:
    p = Path(path)
    data = nx.node_link_data(graph)
    graph_block = dict(data.get("graph") or {})
    graph_block.setdefault(
        SSG_META_VERSION_KEY,
        graph.graph.get(SSG_META_VERSION_KEY, SSG_SCHEMA_VERSION),
    )
    if SSG_META_UNRESOLVED_KEY in graph.graph:
        graph_block[SSG_META_UNRESOLVED_KEY] = list(graph.graph[SSG_META_UNRESOLVED_KEY])
    data["graph"] = graph_block
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_ssg_json(path: Path | str) -> nx.MultiDiGraph[Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    g = nx.node_link_graph(payload, multigraph=True, directed=True)
    gb = payload.get("graph") or {}
    if SSG_META_UNRESOLVED_KEY in gb:
        g.graph[SSG_META_UNRESOLVED_KEY] = list(gb[SSG_META_UNRESOLVED_KEY])
    if SSG_META_VERSION_KEY in gb:
        g.graph[SSG_META_VERSION_KEY] = gb[SSG_META_VERSION_KEY]
    return g
