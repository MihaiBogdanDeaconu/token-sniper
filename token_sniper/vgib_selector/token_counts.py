from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from token_sniper.graph.schema import file_node_id, is_file_node_id

TOKENIZER_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
TOKENIZER_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
COUNTING = "exclusive_lines_v1"
SIDECAR_NAME = "token_counts.json"

_ROW_BREAK = re.compile(rb"(?<=\n)")

Span = tuple[str, int, int]


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)


def split_rows(source: bytes) -> list[bytes]:
    return _ROW_BREAK.split(source)


def line_owners(spans: list[Span], n_rows: int, file_node: str) -> list[str]:
    owners = [file_node] * n_rows
    stack: list[Span] = []
    for span in sorted(spans, key=lambda s: (s[1], -s[2])):
        node_id, start, end = span
        if start < 1 or end < start or end > n_rows:
            raise ValueError(f"{node_id}: span {start}-{end} is outside the file's {n_rows} rows")
        while stack and stack[-1][2] < start:
            stack.pop()
        if stack and (end > stack[-1][2] or (start, end) == stack[-1][1:]):
            parent, p_start, p_end = stack[-1]
            raise ValueError(f"{node_id}: span {start}-{end} is not nested inside {parent} {p_start}-{p_end}")
        stack.append(span)
        owners[start - 1 : end] = [node_id] * (end - start + 1)
    return owners


def count_file_tokens(tokenizer: Any, source: bytes, file_path: str, spans: list[Span]) -> dict[str, int]:
    rows = split_rows(source)
    file_node = file_node_id(file_path)
    owners = line_owners(spans, len(rows), file_node)

    runs: list[tuple[str, bytes]] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or owners[i] != owners[start]:
            runs.append((owners[start], b"".join(rows[start:i])))
            start = i

    counts = {file_node: 0}
    counts.update((node_id, 0) for node_id, _, _ in spans)
    texts = [chunk.decode("utf-8", errors="replace") for _, chunk in runs]
    for (owner, _), ids in zip(runs, tokenizer(texts, add_special_tokens=False)["input_ids"], strict=True):
        counts[owner] += len(ids)
    return counts


def read_blobs(git_dir: Path, commit: str, paths: list[str]) -> dict[str, bytes]:
    request = "".join(f"{commit}:{path}\n" for path in paths).encode()
    out = subprocess.run(
        ["git", "--git-dir", str(git_dir), "cat-file", "--batch", "--follow-symlinks"],
        input=request,
        capture_output=True,
        check=True,
    ).stdout

    blobs: dict[str, bytes] = {}
    pos = 0
    for path in paths:
        header_end = out.index(b"\n", pos)
        header = out[pos:header_end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(
                f"{commit}:{path} is not a blob in {git_dir}: {out[pos:header_end].decode(errors='replace')}"
            )
        size = int(header[2])
        blobs[path] = out[header_end + 1 : header_end + 1 + size]
        pos = header_end + 1 + size + 1
    return blobs


def count_ssg_tokens(graph: Any, git_dir: Path, commit: str, tokenizer: Any) -> dict[str, int]:
    spans: dict[str, list[Span]] = {}
    for node_id, data in graph.nodes(data=True):
        file_spans = spans.setdefault(data["file_path"], [])
        if not is_file_node_id(node_id):
            file_spans.append((node_id, data["start_line"], data["end_line"]))

    sources = read_blobs(git_dir, commit, sorted(spans))
    counts: dict[str, int] = {}
    for file_path, file_spans in spans.items():
        counts.update(count_file_tokens(tokenizer, sources[file_path], file_path, file_spans))

    extra = sorted(counts.keys() - set(graph.nodes))
    if extra:
        raise ValueError(f"entities without a file node: {extra[:5]}")
    return counts


def sidecar_path(ssg_path: Path) -> Path:
    return ssg_path.with_name(SIDECAR_NAME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(ssg_path: Path) -> dict[str, str]:
    return {
        "tokenizer": TOKENIZER_NAME,
        "revision": TOKENIZER_REVISION,
        "counting": COUNTING,
        "ssg_sha256": _sha256(ssg_path),
    }


def write_token_sidecar(ssg_path: Path, counts: dict[str, int]) -> Path:
    path = sidecar_path(ssg_path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**_header(ssg_path), "counts": counts}), encoding="utf-8")
    tmp.replace(path)
    return path


def read_token_sidecar(ssg_path: Path) -> dict[str, int]:
    path = sidecar_path(ssg_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.pop("counts")
    expected = _header(ssg_path)
    if payload != expected:
        raise ValueError(f"{path} was built for {payload}, expected {expected}")
    return counts
