from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from token_sniper.graph.schema import file_node_id, is_file_node_id

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@")

Span = tuple[str, int, int]


@dataclass
class PatchEdits:
    changed: dict[str, set[int]] = field(default_factory=dict)
    """Rows removed or rewritten."""
    inserted_before: dict[str, set[int]] = field(default_factory=dict)
    """Row r: new lines go between old rows r - 1 and r."""


@dataclass(frozen=True)
class GoldNodes:
    nodes: frozenset[str]
    missing_files: tuple[str, ...]


def parse_patch(patch: str) -> PatchEdits:
    edits = PatchEdits()
    path: str | None = None
    row = old_left = new_left = 0
    after_removal = False
    for line in patch.splitlines():
        if old_left == 0 and new_left == 0:
            if line.startswith("--- "):
                old = line[4:].split("\t")[0].strip()
                path = None if old == "/dev/null" else old.removeprefix("a/")
            elif m := _HUNK.match(line):
                start = int(m.group(1))
                old_left = int(m.group(2) or 1)
                new_left = int(m.group(3) or 1)
                row = start if old_left else start + 1
                after_removal = False
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            if path is not None:
                edits.changed.setdefault(path, set()).add(row)
            row += 1
            old_left -= 1
            after_removal = True
        elif line.startswith("+"):
            if path is not None and not after_removal:
                edits.inserted_before.setdefault(path, set()).add(row)
            new_left -= 1
        else:
            row += 1
            old_left -= 1
            new_left -= 1
            after_removal = False
        if old_left < 0 or new_left < 0:
            raise ValueError(f"hunk line counts do not match its header near {line!r}")
    if old_left or new_left:
        raise ValueError("patch ends inside a hunk")
    return edits


def _innermost(spans: list[Span], first: int, last: int, file_node: str) -> str:
    best, best_length = file_node, None
    for node_id, start, end in spans:
        if start <= first and last <= end:
            length = end - start
            if best_length is not None and length == best_length:
                raise ValueError(f"ambiguous owner of rows {first}-{last}: {best} and {node_id}")
            if best_length is None or length < best_length:
                best, best_length = node_id, length
    return best


def gold_nodes(graph: Any, edits: PatchEdits) -> GoldNodes:
    spans: dict[str, list[Span]] = {}
    for node_id, data in graph.nodes(data=True):
        if not is_file_node_id(node_id):
            spans.setdefault(data["file_path"], []).append((node_id, data["start_line"], data["end_line"]))

    nodes: set[str] = set()
    missing: list[str] = []
    for path in sorted(edits.changed.keys() | edits.inserted_before.keys()):
        file_node = file_node_id(path)
        if file_node not in graph:
            missing.append(path)
            continue
        file_spans = spans.get(path, [])
        nodes.update(_innermost(file_spans, row, row, file_node) for row in edits.changed.get(path, ()))
        nodes.update(_innermost(file_spans, row - 1, row, file_node) for row in edits.inserted_before.get(path, ()))
    return GoldNodes(frozenset(nodes), tuple(missing))
