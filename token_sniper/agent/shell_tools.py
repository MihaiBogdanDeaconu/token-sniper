from __future__ import annotations

import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def build_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Run a bash command in the repository.\n"
                    "* State is persistent across calls.\n"
                    "* You don't have internet access.\n"
                    "* To inspect a line range, use 'sed -n 10,25p /path/to/file'.\n"
                    "* Avoid commands that produce very large output.\n"
                    "* Run long-lived commands in the background with '&'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": (
                    "View, create, and edit files. Commands: view, create, "
                    "str_replace, insert, undo_edit.\n"
                    "* Use absolute paths (e.g. /repo/src/file.py).\n"
                    "* 'view' on a file shows it with line numbers. "
                    "On a directory, lists files up to 2 levels.\n"
                    "* 'str_replace': old_str must match EXACTLY one occurrence "
                    "in the file (include enough context). "
                    "new_str replaces it.\n"
                    "* 'create': creates a new file (fails if it exists).\n"
                    "* 'insert': inserts new_str AFTER the given insert_line.\n"
                    "* 'undo_edit': reverts the last edit to the file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file or directory.",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "For str_replace: exact string to find.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "For str_replace/insert: replacement or inserted text.",
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "For insert: line number after which to insert.",
                        },
                        "file_text": {
                            "type": "string",
                            "description": "For create: file contents.",
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "For view: [start_line, end_line]. Use [start, -1] for rest of file.",
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "complete",
                "description": (
                    "Signal that you have finished editing. "
                    "The system will run 'git diff' to collect your patch."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Brief summary of the changes made.",
                        },
                    },
                    "required": ["message"],
                },
            },
        },
    ]


class ShellToolExecutor:
    def __init__(self, repo_root: Path | str) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._undo_stack: dict[str, str] = {}
        self._completed = False
        self._complete_message: str | None = None

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def complete_message(self) -> str | None:
        return self._complete_message

    def generate_patch(self) -> str | None:
        try:
            tracked = subprocess.run(
                ["git", "diff"],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            parts = []
            if tracked.stdout.strip():
                parts.append(tracked.stdout.rstrip("\r\n"))
            for new_file in untracked.stdout.strip().splitlines():
                new_file = new_file.strip()
                if not new_file or new_file.endswith(".bak"):
                    continue
                content = (self._repo_root / new_file).read_text(
                    encoding="utf-8", errors="replace",
                )
                parts.append(
                    f"diff --git a/{new_file} b/{new_file}\n"
                    f"new file mode 100644\n"
                    f"--- /dev/null\n"
                    f"+++ b/{new_file}\n"
                    f"@@ -0,0 +1,{len(content.splitlines())} @@\n"
                    + "\n".join("+" + line for line in content.splitlines())
                )
            diff = "\n".join(parts)
            return diff if diff else None
        except Exception:
            return None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "bash":
            return self._bash(command=str(arguments["command"]))
        if name == "str_replace_editor":
            return self._str_replace_editor(arguments)
        if name == "complete":
            self._completed = True
            self._complete_message = str(arguments.get("message", ""))
            return {"status": "completed", "message": self._complete_message}
        if name == "grep":
            return self._grep(
                pattern=str(arguments["pattern"]),
                path=str(arguments.get("path", ".")),
                glob=str(arguments.get("glob", "*")),
                case_sensitive=bool(arguments.get("case_sensitive", False)),
                max_matches=int(arguments.get("max_matches", 20)),
            )
        if name == "find":
            return self._find(
                path=str(arguments.get("path", ".")),
                pattern=str(arguments.get("pattern", "*")),
                kind=str(arguments.get("kind", "files")),
                limit=int(arguments.get("limit", 50)),
            )
        if name == "cat":
            return self._cat(
                path=str(arguments["path"]),
                start_line=_optional_int(arguments.get("start_line")),
                end_line=_optional_int(arguments.get("end_line")),
            )
        if name == "head":
            line_count = int(arguments.get("lines", 20))
            return self._cat(path=str(arguments["path"]), start_line=1, end_line=line_count)
        if name == "tail":
            return self._tail(path=str(arguments["path"]), lines=int(arguments.get("lines", 20)))
        if name == "sed":
            return self._cat(
                path=str(arguments["path"]),
                start_line=int(arguments["start_line"]),
                end_line=int(arguments["end_line"]),
            )
        raise ValueError(f"Unsupported tool: {name}")

    def _bash(self, *, command: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "GIT_PAGER": "cat"},
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output.strip():
                output = "Command ran successfully (no output)."
            else:
                output = _smart_truncate_bash(command, output)
            return {
                "tool": "bash",
                "exit_code": result.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {"tool": "bash", "exit_code": -1, "output": "Command timed out (120s)"}
        except Exception as e:
            return {"tool": "bash", "exit_code": -1, "output": f"Error: {e}"}

    def _str_replace_editor(self, args: dict[str, Any]) -> dict[str, Any]:
        cmd = args["command"]
        path_str = str(args["path"])

        if not os.path.isabs(path_str):
            path_str = str(self._repo_root / path_str)
        file_path = Path(path_str).resolve()

        if cmd == "view":
            return self._editor_view(file_path, args.get("view_range"))
        if cmd == "create":
            return self._editor_create(file_path, str(args.get("file_text", "")))
        if cmd == "str_replace":
            return self._editor_str_replace(
                file_path,
                str(args.get("old_str", "")),
                str(args.get("new_str", "")),
            )
        if cmd == "insert":
            return self._editor_insert(
                file_path,
                int(args.get("insert_line", 0)),
                str(args.get("new_str", "")),
            )
        if cmd == "undo_edit":
            return self._editor_undo(file_path)
        return {"error": f"Unknown editor command: {cmd}"}

    _VIEW_LINE_LIMIT = 100

    def _editor_view(self, path: Path, view_range: list[int] | None) -> dict[str, Any]:
        if path.is_dir():
            entries: list[str] = []
            for item in sorted(path.iterdir()):
                if item.name.startswith("."):
                    continue
                rel = item.relative_to(path)
                entries.append(str(rel) + ("/" if item.is_dir() else ""))
                if item.is_dir():
                    for sub in sorted(item.iterdir()):
                        if sub.name.startswith("."):
                            continue
                        entries.append("  " + str(sub.relative_to(path)))
            return {"tool": "str_replace_editor", "command": "view", "content": "\n".join(entries)}

        if not path.is_file():
            return {"error": f"File not found: {path}"}

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        if view_range:
            start = max(view_range[0], 1)
            end = total_lines if (len(view_range) > 1 and view_range[1] == -1) else (
                view_range[1] if len(view_range) > 1 else total_lines
            )
            end = min(end, total_lines)
        else:
            start, end = 1, min(self._VIEW_LINE_LIMIT, total_lines)

        if end - start + 1 > self._VIEW_LINE_LIMIT:
            end = start + self._VIEW_LINE_LIMIT - 1

        numbered = [f"{i:6d}\t{lines[i - 1]}" for i in range(start, end + 1)]
        content = "\n".join(numbered)

        context_msg = ""
        above = start - 1
        below = total_lines - end
        if above > 0 or below > 0:
            parts = []
            if above > 0:
                parts.append(f"{above} lines above")
            if below > 0:
                parts.append(f"{below} lines below")
            context_msg = f"\n[{', '.join(parts)} | total: {total_lines} lines | use view_range to see more]"

        return {
            "tool": "str_replace_editor",
            "command": "view",
            "path": str(path),
            "lines": total_lines,
            "content": content + context_msg,
        }

    def _editor_create(self, path: Path, content: str) -> dict[str, Any]:
        if path.exists():
            return {"error": f"File already exists: {path}. Use str_replace to edit it."}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"tool": "str_replace_editor", "command": "create", "path": str(path)}

    def _editor_str_replace(
        self, path: Path, old_str: str, new_str: str,
    ) -> dict[str, Any]:
        if not path.is_file():
            return {"error": f"File not found: {path}"}

        content = path.read_text(encoding="utf-8")
        count = content.count(old_str)

        if count == 0:
            return {
                "error": (
                    f"old_str not found in {path}. "
                    "Make sure it matches the file content exactly, including whitespace."
                ),
            }
        if count > 1:
            return {
                "error": (
                    f"old_str found {count} times in {path}. "
                    "Include more surrounding context to make it unique."
                ),
            }

        self._undo_stack[str(path)] = content
        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content, encoding="utf-8")

        return {
            "tool": "str_replace_editor",
            "command": "str_replace",
            "path": str(path),
            "status": "ok",
        }

    def _editor_insert(self, path: Path, line_num: int, new_str: str) -> dict[str, Any]:
        if not path.is_file():
            return {"error": f"File not found: {path}"}

        content = path.read_text(encoding="utf-8")
        self._undo_stack[str(path)] = content
        lines = content.splitlines(keepends=True)
        insert_idx = min(line_num, len(lines))
        new_lines = new_str.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[insert_idx:insert_idx] = new_lines
        path.write_text("".join(lines), encoding="utf-8")

        return {
            "tool": "str_replace_editor",
            "command": "insert",
            "path": str(path),
            "status": "ok",
        }

    def _editor_undo(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key not in self._undo_stack:
            return {"error": f"No edit to undo for {path}"}
        path.write_text(self._undo_stack.pop(key), encoding="utf-8")
        return {"tool": "str_replace_editor", "command": "undo_edit", "path": str(path), "status": "ok"}

    def _find(self, *, path: str, pattern: str, kind: str, limit: int) -> dict[str, Any]:
        search_root = self._resolve_path(path)
        matches: list[str] = []
        for candidate in search_root.rglob(pattern):
            if kind == "files" and not candidate.is_file():
                continue
            if kind == "dirs" and not candidate.is_dir():
                continue
            matches.append(candidate.relative_to(self._repo_root).as_posix())
            if len(matches) >= limit:
                break
        return {
            "tool": "find",
            "path": search_root.relative_to(self._repo_root).as_posix()
            if search_root != self._repo_root
            else ".",
            "pattern": pattern,
            "kind": kind,
            "matches": matches,
            "truncated": len(matches) >= limit,
        }

    def _grep(
        self,
        *,
        pattern: str,
        path: str,
        glob: str,
        case_sensitive: bool,
        max_matches: int,
    ) -> dict[str, Any]:
        search_root = self._resolve_path(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        results: list[dict[str, Any]] = []
        for file_path in self._iter_files(search_root, glob):
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    results.append(
                        {
                            "path": file_path.relative_to(self._repo_root).as_posix(),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
                    if len(results) >= max_matches:
                        return {
                            "tool": "grep",
                            "pattern": pattern,
                            "matches": results,
                            "truncated": True,
                        }
        return {
            "tool": "grep",
            "pattern": pattern,
            "matches": results,
            "truncated": False,
        }

    def _cat(
        self,
        *,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        file_path = self._resolve_file(path)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        start_index = 0 if start_line is None else start_line - 1
        end_index = len(lines) if end_line is None else end_line
        excerpt = lines[start_index:end_index]
        return {
            "tool": "cat" if start_line is None and end_line is None else "sed",
            "path": file_path.relative_to(self._repo_root).as_posix(),
            "start_line": 1 if start_line is None else start_line,
            "end_line": len(lines) if end_line is None else min(end_line, len(lines)),
            "content": "\n".join(excerpt),
        }

    def _tail(self, *, path: str, lines: int) -> dict[str, Any]:
        file_path = self._resolve_file(path)
        all_lines = file_path.read_text(encoding="utf-8").splitlines()
        excerpt = all_lines[-lines:]
        start_line = max(len(all_lines) - len(excerpt) + 1, 1)
        return {
            "tool": "tail",
            "path": file_path.relative_to(self._repo_root).as_posix(),
            "start_line": start_line,
            "end_line": len(all_lines),
            "content": "\n".join(excerpt),
        }

    def _iter_files(self, search_root: Path, pattern: str) -> Iterable[Path]:
        if search_root.is_file():
            yield search_root
            return
        for candidate in search_root.rglob(pattern):
            if candidate.is_file():
                yield candidate

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = (self._repo_root / raw_path).resolve()
        if candidate != self._repo_root and self._repo_root not in candidate.parents:
            raise ValueError(f"Path escapes repository root: {raw_path}")
        return candidate

    def _resolve_file(self, raw_path: str) -> Path:
        candidate = self._resolve_path(raw_path)
        if not candidate.is_file():
            raise ValueError(f"Expected a file path inside the repository: {raw_path}")
        return candidate


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")[0].strip()
    return int(value)


_BASH_MAX_CHARS = 5_000
_BASH_MAX_LINES = 120


def _smart_truncate_bash(command: str, output: str) -> str:
    cmd_lower = command.strip().lower()

    if _is_test_command(cmd_lower):
        return _truncate_test_output(output)

    if _is_grep_command(cmd_lower):
        return _truncate_grep_output(output)

    if _is_find_or_ls(cmd_lower):
        return _truncate_list_output(output)

    return _truncate_generic(output)


def _is_test_command(cmd: str) -> bool:
    return any(kw in cmd for kw in ("pytest", "runtests", "python -m test", "unittest"))


def _is_grep_command(cmd: str) -> bool:
    return any(kw in cmd for kw in ("grep ", "rg ", "ag "))


def _is_find_or_ls(cmd: str) -> bool:
    parts = cmd.split("|")[0]
    return any(kw in parts for kw in ("find ", "ls ", "ls\n"))


def _truncate_test_output(output: str) -> str:
    lines = output.splitlines()
    if len(lines) <= _BASH_MAX_LINES:
        return _truncate_by_chars(output)

    head = lines[:20]
    for i, line in enumerate(lines):
        if any(marker in line for marker in ("FAILED", "PASSED", "ERROR", "= FAILURES =", "= ERRORS =", "short test summary")):
            tail_start = max(i - 5, len(lines) - 80)
            tail = lines[tail_start:]
            omitted = len(lines) - len(head) - len(tail)
            return "\n".join(
                head
                + [f"\n[... {omitted} lines of test output omitted ...]\n"]
                + tail
            )

    tail = lines[-60:]
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join(
        head
        + [f"\n[... {omitted} lines of test output omitted ...]\n"]
        + tail
    )


def _truncate_grep_output(output: str) -> str:
    lines = output.splitlines()
    if len(lines) <= 30:
        return _truncate_by_chars(output)

    kept = lines[:25]
    rest = lines[25:]

    files_in_rest: set[str] = set()
    for line in rest:
        if ":" in line:
            files_in_rest.add(line.split(":")[0])

    summary = f"\n[... {len(rest)} more matches"
    if files_in_rest:
        summary += f" in {len(files_in_rest)} files: {', '.join(sorted(files_in_rest)[:10])}"
        if len(files_in_rest) > 10:
            summary += f" and {len(files_in_rest) - 10} more"
    summary += " ...]"

    return "\n".join(kept) + summary


def _truncate_list_output(output: str) -> str:
    lines = output.splitlines()
    if len(lines) <= 50:
        return _truncate_by_chars(output)
    kept = lines[:50]
    return "\n".join(kept) + f"\n[... {len(lines) - 50} more entries omitted ...]"


def _truncate_generic(output: str) -> str:
    lines = output.splitlines()
    if len(lines) <= _BASH_MAX_LINES and len(output) <= _BASH_MAX_CHARS:
        return output

    if len(lines) > _BASH_MAX_LINES:
        head = lines[:40]
        tail = lines[-40:]
        omitted = len(lines) - 80
        return "\n".join(
            head
            + [f"\n[... {omitted} lines omitted ...]\n"]
            + tail
        )

    return _truncate_by_chars(output)


def _truncate_by_chars(output: str) -> str:
    if len(output) <= _BASH_MAX_CHARS:
        return output
    half = _BASH_MAX_CHARS // 2
    return output[:half] + f"\n\n[... {len(output) - _BASH_MAX_CHARS} chars omitted ...]\n\n" + output[-half:]
