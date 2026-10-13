from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from token_sniper.traces.ssg_mapping import SSGNodeIndex
from token_sniper.traces.tracer import TraceResult

logger = logging.getLogger(__name__)


EdgeCategory = Literal[
    "polymorphic_dispatch",
    "decorator",
    "callback",
    "metaclass",
    "unresolved_dynamic",
]


@dataclass(frozen=True)
class MissingEdge:
    src: str
    dst: str
    category: EdgeCategory
    call_site_lineno: int


class MissingEdgeReport(BaseModel):
    total_trace_edges: int = 0
    resolved_to_ssg: int = 0
    missing_edges: int = 0
    category_counts: dict[str, int] = Field(default_factory=dict)
    dropped_unresolved_callers: int = 0
    dropped_unresolved_callees: int = 0

    model_config = {"frozen": True}


@dataclass
class _Classifier:
    index: SSGNodeIndex

    def classify(self, src: str, dst: str) -> EdgeCategory:
        graph = self.index.graph
        src_out = {v for _, v, d in graph.out_edges(src, data=True) if d.get("edge_type") == "calls"}
        same_last_segment = [c for c in src_out if c.split(".")[-1] == dst.split(".")[-1]]
        if same_last_segment:
            return "polymorphic_dispatch"

        dst_data = graph.nodes.get(dst, {})
        if dst_data.get("node_type") in {"function", "method"}:
            predecessors = list(graph.predecessors(dst))
            if any(
                d.get("edge_type") == "imports" or (d.get("target_name") or "").endswith("wraps")
                for _p in predecessors
                for d in (graph.get_edge_data(_p, dst) or {}).values()
            ):
                return "decorator"

        dst_last = dst.split(".")[-1]
        if dst_last in {"__new__", "__call__", "__init_subclass__", "__class_getitem__"}:
            return "metaclass"

        return "unresolved_dynamic"


def classify_missing_edges(
    trace: TraceResult,
    index: SSGNodeIndex,
) -> tuple[list[MissingEdge], MissingEdgeReport]:
    clf = _Classifier(index=index)
    missing: list[MissingEdge] = []
    resolved = 0
    dropped_src = 0
    dropped_dst = 0

    for src_raw, dst_raw, lineno in trace.edges:
        src_file = _guess_file_from_fqn(src_raw, index)
        dst_file = _guess_file_from_fqn(dst_raw, index)
        src = index.resolve(src_raw, hint_file=src_file)
        dst = index.resolve(dst_raw, hint_file=dst_file)
        if src is None:
            dropped_src += 1
            continue
        if dst is None:
            dropped_dst += 1
            continue
        if src == dst:
            continue
        if index.is_static_edge(src, dst):
            resolved += 1
            continue
        missing.append(
            MissingEdge(
                src=src,
                dst=dst,
                category=clf.classify(src, dst),
                call_site_lineno=lineno,
            )
        )

    index.log_resolution_stats()

    cats = Counter(m.category for m in missing)
    report = MissingEdgeReport(
        total_trace_edges=len(trace.edges),
        resolved_to_ssg=resolved,
        missing_edges=len(missing),
        category_counts=dict(cats),
        dropped_unresolved_callers=dropped_src,
        dropped_unresolved_callees=dropped_dst,
    )
    return missing, report


def _guess_file_from_fqn(fqn: str, index: SSGNodeIndex) -> str | None:
    parts = fqn.split(".")
    for end in range(len(parts), 0, -1):
        prefix = "/".join(parts[:end]) + ".py"
        if index.by_file.get(prefix):
            return prefix
        pkg = "/".join(parts[:end]) + "/__init__.py"
        if index.by_file.get(pkg):
            return pkg
    return None
