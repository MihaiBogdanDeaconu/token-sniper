from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx

from token_sniper.graph.schema import SSG_META_UNRESOLVED_KEY


def ssg_quality_snapshot(
    graph: nx.MultiDiGraph[Any],
    *,
    expensive: bool = False,
) -> dict[str, Any]:
    node_types = Counter(str(graph.nodes[n].get("node_type", "")) for n in graph)
    edge_types: Counter[str] = Counter()
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        et = str(data.get("edge_type", ""))
        edge_types[et] += 1

    degrees = [graph.degree(n) for n in graph]
    out_degrees = [graph.out_degree(n) for n in graph]
    in_degrees = [graph.in_degree(n) for n in graph]

    snap: dict[str, Any] = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "unresolved_edges": len(graph.graph.get(SSG_META_UNRESOLVED_KEY, [])),
        "degree": _basic_stats(degrees),
        "out_degree": _basic_stats(out_degrees),
        "in_degree": _basic_stats(in_degrees),
        "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
    }

    if expensive and graph.number_of_nodes() > 0:
        wccs = list(nx.weakly_connected_components(graph))
        largest = max(wccs, key=len)
        sub = graph.subgraph(largest).copy()
        if sub.number_of_nodes() > 1 and nx.is_weakly_connected(sub):
            snap["largest_wcc_nodes"] = sub.number_of_nodes()
            snap["largest_wcc_diameter"] = nx.diameter(sub.to_undirected())
        else:
            snap["largest_wcc_nodes"] = sub.number_of_nodes()
            snap["largest_wcc_diameter"] = 0

    return snap


def _basic_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
