from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

_REPO_LOCK_GUARD = Lock()
_REPO_LOCKS: dict[str, Lock] = {}
_HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class Phase0Task(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    difficulty: str | None = None
    gold_patch_files: list[str] = Field(default_factory=list)
    gold_patch_file_count: int
    bucket: Literal["single_file", "multi_file_2_3", "multi_file_4_plus"]


class Phase0Manifest(BaseModel):
    tasks: list[Phase0Task]


def load_phase0_manifest(path: Path | str) -> Phase0Manifest:
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return Phase0Manifest(tasks=[Phase0Task(**task) for task in data])


def build_oracle_collapsed_context(
    *,
    repo_root: Path,
    gold_patch: str,
    oracle_files: list[str],
    context_lines: int = 15,
) -> str:
    repo_root = repo_root.resolve()
    windows_by_path = _extract_patch_windows(gold_patch, context_lines=context_lines)
    sections = [
        "Oracle mode is enabled. These files are known relevant.",
        (
            "The preloaded context is collapsed to edited regions from the gold patch with a "
            f"+/-{context_lines} line buffer around each edit."
        ),
        "If more surrounding context is needed, inspect the listed files with shell tools.",
    ]
    missing_files: list[str] = []

    for relative_path in oracle_files:
        abs_path = (repo_root / relative_path).resolve()
        if repo_root not in abs_path.parents and abs_path != repo_root:
            raise ValueError(f"Oracle file escapes repository root: {relative_path}")
        if not abs_path.is_file():
            missing_files.append(relative_path)
            continue

        file_lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        sections.append(
            _format_oracle_file_section(
                relative_path=relative_path,
                file_lines=file_lines,
                windows=windows_by_path.get(relative_path, []),
            )
        )

    if missing_files:
        sections.append(
            "\n".join(
                [
                    "These oracle files are part of the gold patch but do not exist at the base commit:",
                    *[f"- {relative_path}" for relative_path in missing_files],
                ]
            )
        )
    return "\n\n".join(sections)


def materialize_repo_snapshot(
    *,
    repo: str,
    base_commit: str,
    cache_dir: Path,
    work_dir: Path,
) -> Path:
    cache_dir = cache_dir.resolve()
    work_dir = work_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cached_repo = ensure_repo_cached(repo=repo, base_commit=base_commit, cache_dir=cache_dir)

    if work_dir.exists() and any(work_dir.iterdir()):
        import shutil

        shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    _run_git(["clone", str(cached_repo), str(work_dir)])
    _run_git(["checkout", base_commit], cwd=work_dir)

    return work_dir


def ensure_repo_cached(
    *,
    repo: str,
    base_commit: str,
    cache_dir: Path,
) -> Path:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir_name = repo.replace("/", "__")
    cached_repo = cache_dir / repo_dir_name

    with _repo_lock_for(repo):
        if not cached_repo.exists():
            _run_git(["clone", "--bare", f"https://github.com/{repo}.git", str(cached_repo)])

        try:
            _run_git(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=cached_repo)
        except RuntimeError:
            _run_git(["fetch", "origin", "+refs/heads/*:refs/heads/*"], cwd=cached_repo)
            try:
                _run_git(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=cached_repo)
            except RuntimeError:
                _run_git(["fetch", "origin", base_commit], cwd=cached_repo)
                _run_git(["rev-parse", "--verify", f"{base_commit}^{{commit}}"], cwd=cached_repo)

    return cached_repo


def write_decision_document(
    *,
    summary_path: Path,
    decision_path: Path,
    gap_threshold: float = 0.08,
) -> None:
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    blind_pass = _extract_condition_metric(summary_data, "blind_agent", "pass_at_1")
    oracle_pass = _extract_condition_metric(summary_data, "oracle_retrieval", "pass_at_1")
    gap = oracle_pass - blind_pass

    lines = [
        "# Phase 0 Decision",
        "",
        f"**Blind Agent Pass@1:** {blind_pass:.1%}",
        f"**Oracle Retrieval Pass@1:** {oracle_pass:.1%}",
        f"**Retrieval Ceiling Gap:** {gap:.1%} ({gap * 100:.1f} percentage points)",
        "",
    ]

    if gap < gap_threshold:
        lines.extend(
            [
                "## DECISION: NO-GO / PIVOT",
                "",
                f"The retrieval ceiling gap ({gap:.1%}) is below the {gap_threshold:.1%} threshold. "
                "This suggests that providing oracle-collapsed gold context does not significantly improve the agent's performance. "
                "We should reconsider the evaluation axis or benchmark before building complex retrieval infrastructure.",
            ]
        )
    else:
        lines.extend(
            [
                "## DECISION: GO",
                "",
                f"The retrieval ceiling gap ({gap:.1%}) exceeds the {gap_threshold:.1%} threshold. "
                "This confirms that stronger context selection has meaningful upside, and building improved retrieval or localization infrastructure is justified. "
                "Proceed to Phase 1.",
            ]
        )

    lines.append("")
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def _extract_condition_metric(
    summary_data: dict[str, object],
    condition: str,
    metric_name: str,
) -> float:
    conditions = summary_data.get("conditions")
    if isinstance(conditions, dict):
        condition_summary = conditions.get(condition)
        if isinstance(condition_summary, dict):
            metrics = condition_summary.get("metrics")
            if isinstance(metrics, dict):
                value = metrics.get(metric_name)
                if isinstance(value, int | float):
                    return float(value)

    metrics = summary_data.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(f"{condition}.{metric_name}")
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{exc.stderr}") from exc


def _repo_lock_for(repo: str) -> Lock:
    with _REPO_LOCK_GUARD:
        return _REPO_LOCKS.setdefault(repo, Lock())


def _extract_patch_windows(
    gold_patch: str, *, context_lines: int
) -> dict[str, list[tuple[int, int]]]:
    windows_by_path: dict[str, list[tuple[int, int]]] = {}
    current_old_path: str | None = None
    current_new_path: str | None = None

    for line in gold_patch.splitlines():
        if line.startswith("diff --git "):
            current_old_path = None
            current_new_path = None
            continue
        if line.startswith("--- "):
            current_old_path = _normalize_patch_path(line[4:].strip())
            continue
        if line.startswith("+++ "):
            current_new_path = _normalize_patch_path(line[4:].strip())
            continue

        match = _HUNK_HEADER_PATTERN.match(line)
        if match is None:
            continue

        relative_path = current_new_path or current_old_path
        if relative_path is None:
            continue

        old_start = int(match.group("old_start"))
        old_count = int(match.group("old_count") or "1")
        if old_count == 0:
            start_line = max(1, old_start - context_lines)
            end_line = old_start + context_lines
        else:
            start_line = max(1, old_start - context_lines)
            end_line = old_start + old_count - 1 + context_lines
        windows_by_path.setdefault(relative_path, []).append((start_line, end_line))

    return windows_by_path


def _normalize_patch_path(raw_path: str) -> str | None:
    if raw_path == "/dev/null":
        return None
    if raw_path.startswith("a/") or raw_path.startswith("b/"):
        return raw_path[2:]
    return raw_path


def _format_oracle_file_section(
    *,
    relative_path: str,
    file_lines: list[str],
    windows: list[tuple[int, int]],
) -> str:
    section_lines = [f"Path: {relative_path}"]
    if not file_lines:
        section_lines.append("File is empty at the base commit.")
        return "\n".join(section_lines)

    if not windows:
        section_lines.append(
            "No precise edit hunk could be parsed from the gold patch. Inspect this file with tools if needed."
        )
        return "\n".join(section_lines)

    merged_windows = _merge_line_windows(windows, max_line=len(file_lines))
    language = _fence_language(relative_path)
    for index, (start_line, end_line) in enumerate(merged_windows, start=1):
        snippet = _format_numbered_snippet(file_lines, start_line=start_line, end_line=end_line)
        section_lines.extend(
            [
                f"Window {index} (lines {start_line}-{end_line}):",
                f"```{language}",
                snippet,
                "```",
            ]
        )
    return "\n".join(section_lines)


def _merge_line_windows(
    windows: list[tuple[int, int]],
    *,
    max_line: int,
) -> list[tuple[int, int]]:
    if not windows:
        return []

    merged: list[tuple[int, int]] = []
    for start_line, end_line in sorted(windows):
        clipped_start = max(1, min(start_line, max_line))
        clipped_end = max(clipped_start, min(end_line, max_line))
        if not merged or clipped_start > merged[-1][1] + 1:
            merged.append((clipped_start, clipped_end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], clipped_end))
    return merged


def _format_numbered_snippet(
    file_lines: list[str],
    *,
    start_line: int,
    end_line: int,
) -> str:
    return "\n".join(
        f"{line_number:>4}: {file_lines[line_number - 1]}"
        for line_number in range(start_line, end_line + 1)
    )


def _fence_language(relative_path: str) -> str:
    return "python" if relative_path.endswith(".py") else ""
