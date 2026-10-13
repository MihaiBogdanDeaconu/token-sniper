from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from token_sniper.graph.schema import is_file_node_id
from token_sniper.vgib_selector.gold_nodes import PatchEdits, gold_nodes, parse_patch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

CONTEXT_HEADER = (
    "Retrieval-augmented mode: the code below was selected from the repository as relevant "
    "to the issue. You still have full access to the repository via tools."
)
MERGE_GAP = 3
"""Spans at most this many rows apart are shown as one group (as before this module).

``sources`` arguments map a file path to its rows (``source_rows``) at the base commit.
"""


@dataclass(frozen=True)
class Span:
    file_path: str
    start_line: int
    end_line: int
    node_type: str
    """"file" for a whole file, the SSG node type for a node, "lines" for other ranges."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def whole_file(path: str, sources: Mapping[str, list[str]]) -> Span:
    return Span(path, 1, len(sources[path]), "file")


def source_rows(blob: bytes) -> list[str]:
    return blob.decode("utf-8", errors="replace").split("\n")


def normalize(spans: Sequence[Span], sources: Mapping[str, list[str]]) -> list[Span]:
    order: list[str] = []
    by_file: dict[str, list[Span]] = {}
    for span in spans:
        n_rows = len(sources[span.file_path])
        if not 1 <= span.start_line <= span.end_line <= n_rows:
            raise ValueError(f"{span} is outside {span.file_path}'s {n_rows} rows")
        if span.file_path not in by_file:
            order.append(span.file_path)
        by_file.setdefault(span.file_path, []).append(span)

    groups: list[Span] = []
    for path in order:
        merged: list[tuple[int, int, set[str]]] = []
        for span in sorted(by_file[path], key=lambda s: (s.start_line, s.end_line)):
            if merged and span.start_line <= merged[-1][1] + MERGE_GAP:
                start, end, types = merged[-1]
                merged[-1] = (start, max(end, span.end_line), types | {span.node_type})
            else:
                merged.append((span.start_line, span.end_line, {span.node_type}))
        n_rows = len(sources[path])
        for start, end, types in merged:
            node_type = "file" if (start, end) == (1, n_rows) else ",".join(sorted(types - {"file"}) or ["lines"])
            groups.append(Span(path, start, end, node_type))
    return groups


def render(groups: Sequence[Span], sources: Mapping[str, list[str]]) -> str:
    sections = [CONTEXT_HEADER]
    for group in groups:
        rows = sources[group.file_path][group.start_line - 1 : group.end_line]
        if group.node_type == "file":
            sections.append(f"\n--- {group.file_path} ---\n" + "\n".join(rows))
        else:
            sections.append(f"\n--- {group.file_path} (lines {group.start_line}-{group.end_line}) ---\n"
                            + "\n".join(rows))
    return "\n".join(sections)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def line_prefix(groups: Sequence[Span], n_lines: int) -> list[Span]:
    out: list[Span] = []
    left = n_lines
    for group in groups:
        if left <= 0:
            break
        size = group.end_line - group.start_line + 1
        if size <= left:
            out.append(group)
        else:
            out.append(Span(group.file_path, group.start_line, group.start_line + left - 1,
                            "lines" if group.node_type == "file" else group.node_type))
        left -= size
    return out


def truncate_to_budget(groups: Sequence[Span], sources: Mapping[str, list[str]], tokenizer: Any,
                       budget: int) -> list[Span]:
    if count_tokens(tokenizer, render([], sources)) > budget:
        raise ValueError(f"a budget of {budget} tokens does not fit the context header")
    total = sum(g.end_line - g.start_line + 1 for g in groups)
    if count_tokens(tokenizer, render(groups, sources)) <= budget:
        return list(groups)
    lo, hi = 0, total
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count_tokens(tokenizer, render(line_prefix(groups, mid), sources)) <= budget:
            lo = mid
        else:
            hi = mid
    return line_prefix(groups, lo)


def pack_to_budget(ranked: Sequence[Span], sources: Mapping[str, list[str]], tokenizer: Any,
                   budget: int) -> tuple[list[Span], list[Span]]:
    kept: list[Span] = []
    groups = normalize(kept, sources)
    for span in ranked:
        trial = normalize([*kept, span], sources)
        if count_tokens(tokenizer, render(trial, sources)) <= budget:
            kept.append(span)
            groups = trial
    return kept, groups


def max_under_budget(make_text: Callable[[float], str], count: Callable[[str], int], budget: int, *,
                     lo: float, hi: float, rounds: int = 8) -> tuple[float, str]:
    text = make_text(hi)
    if count(text) <= budget:
        return hi, text
    best_x, best = lo, make_text(lo)
    if count(best) > budget:
        raise ValueError(f"even the most aggressive setting ({lo}) exceeds the {budget}-token budget")
    for _ in range(rounds):
        mid = (best_x + hi) / 2
        text = make_text(mid)
        if count(text) <= budget:
            best_x, best = mid, text
        else:
            hi = mid
    return best_x, best


def shown_rows(groups: Sequence[Span]) -> dict[str, set[int]]:
    rows: dict[str, set[int]] = {}
    for group in groups:
        rows.setdefault(group.file_path, set()).update(range(group.start_line, group.end_line + 1))
    return rows


def patch_files(patch: str) -> list[str]:
    files: list[str] = []
    old: str | None = None
    for line in patch.splitlines():
        if line.startswith("--- "):
            old = line[4:].split("\t")[0].strip()
        elif line.startswith("+++ ") and old is not None:
            new = line[4:].split("\t")[0].strip()
            path = new.removeprefix("b/") if old == "/dev/null" else old.removeprefix("a/")
            if path not in files:
                files.append(path)
            old = None
    return files


def gold_units(graph: Any, patch: str) -> dict[str, frozenset[int]]:
    edits = parse_patch(patch)
    units: dict[str, set[int]] = {}
    for path in sorted(edits.changed.keys() | edits.inserted_before.keys()):
        points = [(r, r, PatchEdits(changed={path: {r}})) for r in sorted(edits.changed.get(path, ()))]
        points += [(max(r - 1, 1), r, PatchEdits(inserted_before={path: {r}}))
                   for r in sorted(edits.inserted_before.get(path, ()))]
        for first, last, single in points:
            found = gold_nodes(graph, single)
            if found.missing_files:
                units.setdefault(f"missing:{path}", set()).update(range(first, last + 1))
                continue
            (node,) = found.nodes
            if is_file_node_id(node) or graph.nodes[node]["node_type"] == "module_preamble":
                units.setdefault(node, set()).update(range(first, last + 1))
            else:
                data = graph.nodes[node]
                units[node] = set(range(data["start_line"], data["end_line"] + 1))
    return {unit: frozenset(rows) for unit, rows in units.items()}


def unit_file(unit: str, graph: Any) -> str:
    if unit.startswith("missing:"):
        return unit.removeprefix("missing:")
    return graph.nodes[unit]["file_path"]


def recall(groups: Sequence[Span], gold_files: Sequence[str], units: Mapping[str, frozenset[int]],
           graph: Any) -> tuple[float, float]:
    if not gold_files:
        raise ValueError("the gold patch touches no file")
    if not units:
        raise ValueError("the gold patch has no base-commit location (it only creates files)")
    rows = shown_rows(groups)
    file_recall = sum(1 for f in gold_files if f in rows) / len(gold_files)
    covered = sum(1 for unit, need in units.items() if need <= rows.get(unit_file(unit, graph), set()))
    return file_recall, covered / len(units)
