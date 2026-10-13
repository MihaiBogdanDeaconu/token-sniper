from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

_LOCALS_RE = re.compile(r"\.<locals>\.[^.]+")


def _strip_all_locals(fqn: str) -> str:
    return _LOCALS_RE.sub("", fqn)


@dataclass
class SSGNodeIndex:
    graph: nx.MultiDiGraph[Any]
    by_fqn: set[str]
    by_last_segment: dict[str, list[str]]
    by_suffix: dict[str, list[str]]
    by_file: dict[str, list[str]] = field(default_factory=dict)

    _hits_exact: int = field(default=0, repr=False)
    _hits_locals_strip: int = field(default=0, repr=False)
    _hits_suffix: int = field(default=0, repr=False)
    _hits_last_seg: int = field(default=0, repr=False)
    _hits_file_scoped: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def resolve(self, fqn: str, *, hint_file: str | None = None) -> str | None:
        if fqn in self.by_fqn:
            self._hits_exact += 1
            return fqn

        if "<locals>" in fqn:
            stripped = _strip_all_locals(fqn)
            if stripped in self.by_fqn:
                self._hits_locals_strip += 1
                return stripped

        parts = fqn.split(".")
        for start in range(len(parts) - 1):
            suffix = ".".join(parts[start:])
            candidates = self.by_suffix.get(suffix, [])
            if len(candidates) == 1:
                self._hits_suffix += 1
                return candidates[0]

        last = parts[-1]
        candidates = self.by_last_segment.get(last, [])
        if len(candidates) == 1:
            self._hits_last_seg += 1
            return candidates[0]

        if hint_file and candidates:
            same_file = [c for c in candidates if self.graph.nodes.get(c, {}).get("file_path") == hint_file]
            if len(same_file) == 1:
                self._hits_file_scoped += 1
                return same_file[0]

        self._misses += 1
        return None

    def is_static_edge(self, src: str, dst: str) -> bool:
        if not self.graph.has_edge(src, dst):
            return False
        edata = self.graph.get_edge_data(src, dst) or {}
        return any(d.get("edge_type") == "calls" for d in edata.values())

    def log_resolution_stats(self) -> None:
        total = (
            self._hits_exact + self._hits_locals_strip + self._hits_suffix
            + self._hits_last_seg + self._hits_file_scoped + self._misses
        )
        if total == 0:
            return
        logger.info(
            "SSGNodeIndex resolution: %d total, %d exact, %d locals-strip, "
            "%d suffix, %d last-seg, %d file-scoped, %d miss (%.1f%% resolved)",
            total,
            self._hits_exact,
            self._hits_locals_strip,
            self._hits_suffix,
            self._hits_last_seg,
            self._hits_file_scoped,
            self._misses,
            100 * (total - self._misses) / total,
        )


def build_ssg_node_index(graph: nx.MultiDiGraph[Any]) -> SSGNodeIndex:
    from token_sniper.graph.schema import is_file_node_id

    by_fqn: set[str] = set()
    by_last: dict[str, list[str]] = {}
    by_suffix: dict[str, list[str]] = {}
    by_file: dict[str, list[str]] = {}
    for node_id, data in graph.nodes(data=True):
        if is_file_node_id(node_id):
            continue
        if data.get("node_type") == "module_preamble":
            continue
        by_fqn.add(node_id)
        parts = node_id.split(".")
        last = parts[-1]
        by_last.setdefault(last, []).append(node_id)
        fp = data.get("file_path", "")
        if fp:
            by_file.setdefault(fp, []).append(node_id)
        for start in range(1, len(parts)):
            suffix = ".".join(parts[start:])
            by_suffix.setdefault(suffix, []).append(node_id)
    return SSGNodeIndex(
        graph=graph, by_fqn=by_fqn, by_last_segment=by_last,
        by_suffix=by_suffix, by_file=by_file,
    )
