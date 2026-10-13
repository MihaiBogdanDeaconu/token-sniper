from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections import deque
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from token_sniper.agent.shell_tools import ShellToolExecutor, build_tool_schemas

if TYPE_CHECKING:
    from token_sniper.agent.providers import ChatModelClient, LLMResponse
    from token_sniper.guardrails.halt_scorer import HALTScorer

AgentMode = Literal["blind", "oracle"]


_LOOP_WINDOW = 6
_LOOP_THRESHOLD = 3


def _tool_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    if name == "bash":
        cmd = arguments.get("command", "")
        normalized = re.sub(r"\s+", " ", cmd).strip()
        return f"bash:{normalized}"
    if name == "str_replace_editor":
        command = arguments.get("command", "")
        path = arguments.get("path", "")
        vr = arguments.get("view_range", "")
        return f"editor:{command}:{path}:{vr}"
    return f"{name}:{json.dumps(arguments, sort_keys=True)}"


def _detect_loop(recent: deque[str]) -> bool:
    if len(recent) < _LOOP_THRESHOLD:
        return False
    last = recent[-1]
    count = sum(1 for fp in list(recent)[-_LOOP_WINDOW:] if fp == last)
    return count >= _LOOP_THRESHOLD


def _build_loop_nudge(recent: deque[str], turn: int, max_turns: int) -> str:
    last_fp = recent[-1]
    if last_fp.startswith("bash:"):
        return (
            f"[Turn {turn}/{max_turns}] LOOP DETECTED: You have repeated the same "
            "or very similar bash command multiple times with the same result. "
            "Stop searching and take action:\n"
            "- If a function or feature doesn't exist yet, CREATE it with str_replace_editor.\n"
            "- If you've found the relevant code, make your edit now.\n"
            "- If you're stuck, try a completely different approach."
        )
    if last_fp.startswith("editor:view"):
        return (
            f"[Turn {turn}/{max_turns}] LOOP DETECTED: You keep viewing the same "
            "file sections repeatedly. You have enough context — make your edit "
            "now with str_replace_editor (str_replace command), then test it."
        )
    return (
        f"[Turn {turn}/{max_turns}] LOOP DETECTED: You are repeating the same "
        "action without making progress. Try a different approach: make an edit, "
        "run a test, or call complete if you've already fixed the issue."
    )


def _trim_repeated_tool_results(
    messages: list[dict[str, Any]],
    n_recent_to_trim: int = 4,
) -> None:
    trimmed = 0
    for msg in reversed(messages):
        if trimmed >= n_recent_to_trim:
            break
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                msg["content"] = content[:200] + "\n[Truncated — loop detected. Try a different approach.]"
                trimmed += 1


class AgentRunSettings(BaseModel):
    repo_root: str
    issue_description: str
    output_dir: str
    mode: AgentMode = "blind"
    oracle_files: list[str] = Field(default_factory=list)
    oracle_context: str | None = None
    max_turns: int = Field(default=100, gt=0)
    context_window: int = Field(default=65_536, gt=0)
    max_output_tokens: int = Field(default=16_384, gt=0)
    disable_tools: bool = False
    fail_to_pass_tests: list[str] = Field(default_factory=list)
    test_cmd: str | None = None

    model_config = {"frozen": True}


class TurnToolLog(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TurnLog(BaseModel):
    turn_index: int
    prompt_tokens: int
    completion_tokens: int
    wall_time_seconds: float
    finish_reason: str | None = None
    assistant_text: str = ""
    tool_calls: list[TurnToolLog] = Field(default_factory=list)
    patch_validation: PatchValidationLog | None = None
    halt_score: float | None = None

    model_config = {"frozen": True}


class PatchValidationLog(BaseModel):
    source: Literal["assistant_text", "submit_patch_tool"]
    patch_sha256: str
    validation_status: Literal["accepted", "rejected"]
    validation_stage: Literal["parse", "path", "apply"] | None = None
    validation_message: str | None = None
    modified_files: list[str] = Field(default_factory=list)
    repair_prompt_sent: bool = False

    model_config = {"frozen": True}


class AgentRunResult(BaseModel):
    mode: AgentMode
    repo_root: str
    output_dir: str
    patch_path: str | None = None
    session_log_path: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_wall_time_seconds: float
    patch_attempt_count: int = 0
    valid_patch_attempt_count: int = 0
    invalid_patch_attempt_count: int = 0
    repair_turn_count: int = 0
    turns: list[TurnLog]
    status: Literal["success", "failed"] = "success"
    error_message: str | None = None
    halt_mean_score: float | None = None
    halt_max_score: float | None = None
    halt_submit_score: float | None = None

    model_config = {"frozen": True}


def run_agent_session(
    settings: AgentRunSettings,
    client: ChatModelClient,
    *,
    tool_executor: ShellToolExecutor | Any | None = None,
    halt_scorer: HALTScorer | None = None,
) -> AgentRunResult:
    if tool_executor is not None:
        repo_root = Path(settings.repo_root)
    else:
        repo_root = Path(settings.repo_root).resolve()
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = []
    turns: list[TurnLog] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_start = time.perf_counter()
    patch_path: Path | None = None
    error: BaseException | None = None
    patch_attempt_count = 0
    valid_patch_attempt_count = 0
    invalid_patch_attempt_count = 0
    repair_turn_count = 0
    max_repair_attempts = 3
    final_halt_score: float | None = None
    logprobs_path = output_dir / "edit_logprobs.jsonl"

    try:
        executor = tool_executor or ShellToolExecutor(repo_root)
        tools = [] if settings.disable_tools else build_tool_schemas()
        messages = _build_initial_messages(settings, repo_root)
        edit_nudge_turn = min(20, settings.max_turns // 4)
        submit_nudge_turn = max(settings.max_turns - 5, int(settings.max_turns * 0.85))
        has_made_edit = False
        recent_fingerprints: deque[str] = deque(maxlen=_LOOP_WINDOW)
        loop_nudge_cooldown = 0
        for turn_index in range(1, settings.max_turns + 1):
            if loop_nudge_cooldown > 0:
                loop_nudge_cooldown -= 1

            if turn_index == edit_nudge_turn and not has_made_edit and not executor.completed:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Turn {turn_index}/{settings.max_turns}] "
                        "You have not made any edits yet. "
                        "Based on what you have read so far, identify the root cause "
                        "and make your fix with str_replace_editor. "
                        "You can iterate if the first attempt is wrong."
                    ),
                })
            elif turn_index == submit_nudge_turn and not executor.completed:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Turn {turn_index}/{settings.max_turns}] "
                        "You are running low on turns. "
                        "If you have made edits, run the failing tests and call complete. "
                        "If not, make your best fix now, test it, and call complete."
                    ),
                })
            turn_start = time.perf_counter()
            _trim_conversation_history(
                messages,
                context_window=settings.context_window,
                max_output_tokens=settings.max_output_tokens,
            )
            try:
                response = client.complete(messages, tools)
            except Exception as ctx_err:
                if "maximum context length" in str(ctx_err):
                    _trim_conversation_history(
                        messages,
                        context_window=settings.context_window - 16_384,
                        max_output_tokens=settings.max_output_tokens,
                    )
                    response = client.complete(messages, tools)
                else:
                    raise
            wall_time = time.perf_counter() - turn_start
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            turn_halt_score: float | None = None

            if response.tool_calls:
                messages.append(_assistant_tool_message(response))
                tool_logs: list[TurnToolLog] = []
                turn_has_edit = False
                for tool_call in response.tool_calls:
                    try:
                        tool_result = executor.execute(tool_call.name, tool_call.arguments)
                    except Exception as exc:
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                    if tool_call.name == "str_replace_editor" and tool_call.arguments.get("command") in ("str_replace", "create", "insert"):
                        has_made_edit = True
                        turn_has_edit = True
                    tool_result = _truncate_tool_result(tool_result)
                    tool_logs.append(
                        TurnToolLog(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            result=tool_result,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": json.dumps(tool_result, indent=2, sort_keys=True),
                        }
                    )
                    recent_fingerprints.append(
                        _tool_fingerprint(tool_call.name, tool_call.arguments)
                    )

                if turn_has_edit:
                    turn_halt_score = _score_turn_halt(response, halt_scorer)
                    if response.token_logprobs is not None:
                        _save_edit_logprobs(
                            logprobs_path, turn_index, response.token_logprobs,
                        )

                turns.append(
                    TurnLog(
                        turn_index=turn_index,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        wall_time_seconds=wall_time,
                        finish_reason=response.finish_reason,
                        assistant_text=response.content,
                        tool_calls=tool_logs,
                        halt_score=turn_halt_score,
                    )
                )
                if executor.completed:
                    break

                if _detect_loop(recent_fingerprints) and loop_nudge_cooldown == 0:
                    _trim_repeated_tool_results(messages)
                    messages.append({
                        "role": "user",
                        "content": _build_loop_nudge(
                            recent_fingerprints, turn_index, settings.max_turns,
                        ),
                    })
                    loop_nudge_cooldown = 5
                    recent_fingerprints.clear()

                continue

            messages.append({"role": "assistant", "content": response.content})
            turns.append(
                TurnLog(
                    turn_index=turn_index,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    wall_time_seconds=wall_time,
                    finish_reason=response.finish_reason,
                    assistant_text=response.content,
                    halt_score=turn_halt_score,
                )
            )

            if not response.content:
                continue

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue working on the fix. Use str_replace_editor to "
                        "edit files, then call complete when done."
                    ),
                }
            )
        else:
            error = RuntimeError(
                "Agent reached the maximum number of turns without completing."
            )

        diff_text = executor.generate_patch()
        edit_halt_scores = [t.halt_score for t in turns if t.halt_score is not None]
        final_halt_score: float | None = (
            edit_halt_scores[-1] if edit_halt_scores else None
        )
        if diff_text:
            patch_attempt_count = 1
            valid_patch_attempt_count = 1
            patch_path = output_dir / "patch.diff"
            patch_path.write_text(diff_text + "\n", encoding="utf-8")
            if isinstance(error, RuntimeError) and "maximum number of turns" in str(error):
                error = None
        elif not error:
            error = RuntimeError("Agent completed but produced no changes (empty git diff).")
    except BaseException as exc:
        error = exc
    finally:
        halt_scores = [t.halt_score for t in turns if t.halt_score is not None]
        run_result = AgentRunResult(
            mode=settings.mode,
            repo_root=str(repo_root),
            output_dir=str(output_dir),
            patch_path=str(patch_path) if patch_path is not None else None,
            session_log_path=str(output_dir / "session.json"),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_wall_time_seconds=time.perf_counter() - total_start,
            patch_attempt_count=patch_attempt_count,
            valid_patch_attempt_count=valid_patch_attempt_count,
            invalid_patch_attempt_count=invalid_patch_attempt_count,
            repair_turn_count=repair_turn_count,
            turns=turns,
            status="failed" if error is not None else "success",
            error_message=_format_error_message(error),
            halt_mean_score=(
                sum(halt_scores) / len(halt_scores) if halt_scores else None
            ),
            halt_max_score=max(halt_scores) if halt_scores else None,
            halt_submit_score=final_halt_score,
        )
        Path(run_result.session_log_path).write_text(
            run_result.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if error is not None:
        raise error
    return run_result


def extract_patch(text: str) -> str | None:
    candidate = text.strip()
    fenced_match = re.search(r"```(?:diff)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced_match is not None:
        candidate = fenced_match.group(1).strip()

    if not _looks_like_unified_diff(candidate):
        return None
    if not candidate.endswith("\n"):
        candidate += "\n"
    return candidate


def _build_initial_messages(
    settings: AgentRunSettings,
    repo_root: Path,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _build_system_prompt(repo_root, settings.mode, settings),
        },
        {
            "role": "user",
            "content": (
                "<issue>\n"
                f"{settings.issue_description.strip()}\n"
                "</issue>\n\n"
                "Fix this issue. Make minimal changes to non-test files only."
            ),
        },
    ]
    if settings.mode == "oracle":
        if settings.oracle_context is not None:
            oracle_context = settings.oracle_context
        elif settings.oracle_files:
            oracle_context = _build_oracle_context(repo_root, settings.oracle_files)
        else:
            raise ValueError("Oracle mode requires at least one file to preload.")

        preamble_tokens = sum(_estimate_message_tokens(m.get("content")) for m in messages)
        reserved = max(1_024, settings.max_output_tokens) + 4_096
        oracle_budget_tokens = settings.context_window - reserved - preamble_tokens
        oracle_budget_chars = max(oracle_budget_tokens * 4, 8_000)
        if len(oracle_context) > oracle_budget_chars:
            oracle_context = oracle_context[:oracle_budget_chars] + (
                "\n\n[Oracle context truncated to fit context window. "
                "Focus on the files shown above to produce your patch.]"
            )

        messages.append(
            {
                "role": "user",
                "content": oracle_context,
            }
        )
    return messages


def _build_system_prompt(repo_root: Path, mode: AgentMode, settings: AgentRunSettings) -> str:
    if settings.disable_tools:
        return (
            "You are an expert software engineer fixing a bug in a Python repository. "
            f"The repository is at {repo_root.as_posix()}. "
            "You will be given the issue description and the content of relevant source files. "
            "Your task is to produce a unified diff patch that fixes the issue.\n\n"
            "WORKFLOW:\n"
            "1. Analyze the issue description carefully.\n"
            "2. Study the provided file contents to understand the relevant code.\n"
            "3. Produce a unified diff patch as your response.\n\n"
            "CRITICAL RULES:\n"
            "- Your response MUST contain a unified diff inside a ```diff code block.\n"
            "- Context lines in your diff MUST match the file content EXACTLY.\n"
            "- Include 3+ lines of unchanged context before and after each change.\n"
            "- Be surgical: only change what is necessary to fix the bug.\n"
            "- If multiple files need changes, include all hunks in a single diff.\n\n"
            "PATCH FORMAT:\n"
            "```diff\n"
            "--- a/path/to/file.py\n"
            "+++ b/path/to/file.py\n"
            "@@ -10,8 +10,10 @@\n"
            " unchanged line 1\n"
            " unchanged line 2\n"
            " unchanged line 3\n"
            "-removed line\n"
            "+added line 1\n"
            "+added line 2\n"
            " unchanged line 4\n"
            " unchanged line 5\n"
            " unchanged line 6\n"
            "```\n"
        )

    test_section = ""
    if settings.fail_to_pass_tests:
        test_names = "\n".join(f"  - {t}" for t in settings.fail_to_pass_tests)
        test_section = (
            "\n\nFAILING TESTS (these must pass after your fix):\n"
            f"{test_names}\n"
        )
        if settings.test_cmd:
            test_section += f"\nTest command: {settings.test_cmd}\n"

    return (
        f"You are an expert software engineer. "
        f"Fix the bug in the repository at {repo_root.as_posix()}. "
        "All dependencies are pre-installed. You have no internet access.\n\n"
        "STRATEGY — follow this order:\n"
        "1. Run the failing tests to see the exact error.\n"
        "2. Locate the buggy code (grep/find, then read the relevant source).\n"
        "3. Make your edit with str_replace_editor. Be surgical — change only "
        "what is needed.\n"
        "4. Re-run the tests. If they pass, call complete.\n\n"
        f"You have {settings.max_turns} turns. "
        "Aim to make your first edit by turn 15. "
        "Do not spend all turns reading — make an edit, test it, iterate. "
        "If a function or method you are searching for does not exist, you need to CREATE it.\n\n"
        "RULES:\n"
        "- Use absolute paths (e.g. " + repo_root.as_posix() + "/path/to/file.py).\n"
        "- str_replace: old_str must match EXACTLY one location. "
        "Include surrounding context.\n"
        "- Do NOT modify test files.\n"
        "- Call complete only after the tests pass."
        + test_section
    )


def _build_oracle_context(repo_root: Path, oracle_files: list[str]) -> str:
    sections = [
        "Oracle mode is enabled. These files are known relevant and are preloaded into context.",
    ]
    missing_files: list[str] = []
    for relative_path in oracle_files:
        abs_path = (repo_root / relative_path).resolve()
        if repo_root not in abs_path.parents and abs_path != repo_root:
            raise ValueError(f"Oracle file escapes repository root: {relative_path}")
        if not abs_path.is_file():
            missing_files.append(relative_path)
            continue
        content = abs_path.read_text(encoding="utf-8")
        sections.append(
            "\n".join(
                [
                    f"Path: {relative_path}",
                    "```",
                    content,
                    "```",
                ]
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


_TOOL_TRUNCATION_MSG = "[Output truncated. Re-read the file or re-run the command if needed.]"
_MAX_TOOL_RESULT_CHARS = 2_000
_MAX_TOOL_OUTPUT_CHARS = 5_000


def _truncate_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result

    truncated = {}
    for key, value in result.items():
        if isinstance(value, str) and len(value) > _MAX_TOOL_OUTPUT_CHARS:
            half = _MAX_TOOL_OUTPUT_CHARS // 2
            truncated[key] = (
                value[:half]
                + f"\n\n[... {len(value) - _MAX_TOOL_OUTPUT_CHARS} chars truncated ...]\n\n"
                + value[-half:]
            )
        else:
            truncated[key] = value
    return truncated


def _trim_conversation_history(
    messages: list[dict[str, Any]],
    *,
    context_window: int,
    max_output_tokens: int,
) -> None:
    budget = context_window - max(1_024, max_output_tokens) - 8_192
    total = sum(_estimate_message_tokens(m.get("content")) for m in messages)
    if total <= budget:
        return
    protected_tail = 6
    _AGGRESSIVE_TRIM = 500
    for i in range(len(messages) - protected_tail):
        if total <= budget:
            break
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        threshold = _AGGRESSIVE_TRIM if total > budget * 1.3 else _MAX_TOOL_RESULT_CHARS
        if isinstance(content, str) and len(content) > threshold:
            old_tokens = _estimate_message_tokens(content)
            msg["content"] = content[:threshold] + "\n" + _TOOL_TRUNCATION_MSG
            new_tokens = _estimate_message_tokens(msg["content"])
            total -= (old_tokens - new_tokens)

    if total <= budget:
        return
    _ORACLE_TRIM_LIMIT = 60_000
    for i in range(1, len(messages) - protected_tail):
        if total <= budget:
            break
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > _ORACLE_TRIM_LIMIT:
            old_tokens = _estimate_message_tokens(content)
            msg["content"] = content[:_ORACLE_TRIM_LIMIT] + (
                "\n\n[Oracle context truncated to fit context window. "
                "Use shell tools to read files directly.]"
            )
            new_tokens = _estimate_message_tokens(msg["content"])
            total -= (old_tokens - new_tokens)


def _ensure_context_budget(
    messages: list[dict[str, Any]],
    *,
    context_window: int,
    max_output_tokens: int,
) -> None:
    estimated_tokens = sum(_estimate_message_tokens(message.get("content")) for message in messages)
    reserved_tokens = max(1_024, max_output_tokens)
    if estimated_tokens > context_window - reserved_tokens:
        raise ValueError(
            "Oracle context exceeds the estimated model budget. "
            f"Estimated prompt tokens={estimated_tokens}, context_window={context_window}, "
            f"reserved_tokens={reserved_tokens}, max_output_tokens={max_output_tokens}."
        )


def _estimate_message_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return max(1, ceil(len(content) / 3))
    if isinstance(content, list):
        return sum(_estimate_message_tokens(item) for item in content)
    return max(1, ceil(len(str(content)) / 3))


def _format_error_message(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}"


def _score_turn_halt(
    response: LLMResponse,
    halt_scorer: HALTScorer | None,
) -> float | None:
    if halt_scorer is None:
        return None
    lps = response.token_logprobs
    if lps is None or len(lps) < 3:
        return None
    try:
        return halt_scorer.score_logprobs(lps)
    except Exception:
        return None


def _save_edit_logprobs(
    path: Path,
    turn_index: int,
    logprobs: list[list[float]],
) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "turn_index": turn_index,
                "logprobs": logprobs,
                "n_tokens": len(logprobs),
            }) + "\n")
    except Exception:
        pass


def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in response.tool_calls
        ],
    }


def _looks_like_unified_diff(text: str) -> bool:
    has_file_header = ("--- " in text and "+++ " in text) or ("diff --git " in text)
    has_hunk = "@@" in text or "new file mode" in text or "deleted file mode" in text
    return has_file_header and has_hunk


def _compute_patch_sha256(patch_text: str) -> str:
    return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


def _extract_modified_files_from_patch(patch_text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for raw_line in patch_text.splitlines():
        line = raw_line.strip()
        candidate: str | None = None
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[3]
        elif line.startswith("+++ "):
            candidate = line[4:]
        if candidate is None or candidate == "/dev/null":
            continue
        if candidate.startswith("a/") or candidate.startswith("b/"):
            candidate = candidate[2:]
        if candidate and candidate not in seen:
            seen.add(candidate)
            files.append(candidate)
    return files


def _looks_like_patch_attempt(text: str) -> bool:
    candidate = text.strip()
    return any(
        marker in candidate
        for marker in ("diff --git ", "--- ", "+++ ", "@@", "new file mode", "deleted file mode")
    )


def _build_patch_repair_prompt(
    validation: PatchValidationLog,
    *,
    attempt: int = 1,
    max_attempts: int = 3,
) -> str:
    lines = [
        f"Patch rejected (repair attempt {attempt}/{max_attempts}).",
        f"Error: {validation.validation_message or 'Unknown validation failure.'}",
    ]
    if validation.validation_stage == "apply":
        lines.extend([
            "The diff failed `git apply --check`. Common causes:",
            "- Context lines don't match the actual file content",
            "- Wrong line numbers in @@ hunk headers",
            "- Missing or extra blank lines in context",
            "Re-read the relevant file sections with `cat -n` to verify exact content,",
            "then submit a corrected patch via the submit_patch tool.",
        ])
    elif validation.validation_stage == "parse":
        lines.extend([
            "The output was not a valid unified diff.",
            "Use standard `--- a/` and `+++ b/` headers with `@@ ... @@` hunks.",
        ])
    else:
        lines.extend([
            "Fix the issue and call submit_patch with a corrected unified diff.",
        ])
    return "\n".join(lines)


def _validate_patch_candidate(
    *,
    repo_root: Path,
    patch_text: str,
    source: Literal["assistant_text", "submit_patch_tool"],
) -> PatchValidationLog:
    normalized_patch = _normalize_patch_text(patch_text)
    patch_sha256 = _compute_patch_sha256(normalized_patch)
    modified_files = _extract_modified_files_from_patch(normalized_patch)
    if not modified_files:
        return PatchValidationLog(
            source=source,
            patch_sha256=patch_sha256,
            validation_status="rejected",
            validation_stage="parse",
            validation_message="Patch did not contain any modified files.",
            modified_files=[],
        )

    invalid_path = _first_invalid_patch_path(repo_root=repo_root, modified_files=modified_files)
    if invalid_path is not None:
        return PatchValidationLog(
            source=source,
            patch_sha256=patch_sha256,
            validation_status="rejected",
            validation_stage="path",
            validation_message=f"Patch targets invalid path outside the repository: {invalid_path}",
            modified_files=modified_files,
        )

    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, dir=repo_root) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(normalized_patch)

    try:
        completed = subprocess.run(
            ["git", "apply", "--check", "--verbose", temp_path.name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        validation_message = completed.stderr.strip() or completed.stdout.strip()
        if not validation_message:
            validation_message = "git apply --check rejected the patch."
        return PatchValidationLog(
            source=source,
            patch_sha256=patch_sha256,
            validation_status="rejected",
            validation_stage="apply",
            validation_message=validation_message,
            modified_files=modified_files,
        )

    return PatchValidationLog(
        source=source,
        patch_sha256=patch_sha256,
        validation_status="accepted",
        modified_files=modified_files,
    )


def _first_invalid_patch_path(*, repo_root: Path, modified_files: list[str]) -> str | None:
    for relative_path in modified_files:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return relative_path
        resolved = (repo_root / candidate).resolve()
        if repo_root not in resolved.parents and resolved != repo_root:
            return relative_path
    return None


def _normalize_patch_text(patch_text: str) -> str:
    text = patch_text if patch_text.endswith("\n") else f"{patch_text}\n"
    text = _fix_empty_context_lines(text)
    return _fix_hunk_counts(text)


def _fix_empty_context_lines(patch_text: str) -> str:
    lines = patch_text.split("\n")
    result: list[str] = []
    in_hunk = False

    for i, line in enumerate(lines):
        if _HUNK_RE.match(line):
            in_hunk = True
            result.append(line)
        elif line.startswith("diff ") or line.startswith("--- ") or line.startswith("+++ "):
            in_hunk = False
            result.append(line)
        elif in_hunk and line == "" and i < len(lines) - 1:
            result.append(" ")
        else:
            result.append(line)

    return "\n".join(result)


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _fix_hunk_counts(patch_text: str) -> str:
    lines = patch_text.split("\n")
    result: list[str] = []
    hunk_start: int | None = None
    old_start = new_start = 0

    def _flush_hunk() -> None:
        nonlocal hunk_start
        if hunk_start is None:
            return
        old_count = new_count = 0
        for ln in result[hunk_start + 1 :]:
            if ln.startswith("+"):
                new_count += 1
            elif ln.startswith("-"):
                old_count += 1
            elif ln.startswith(" ") or ln == "":
                old_count += 1
                new_count += 1
            elif ln.startswith("\\"):
                pass
        m = _HUNK_RE.match(result[hunk_start])
        if m:
            trailer = m.group(5) or ""
            result[hunk_start] = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{trailer}"
        hunk_start = None

    for line in lines:
        m = _HUNK_RE.match(line)
        if m:
            _flush_hunk()
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            hunk_start = len(result)
            result.append(line)
        elif line.startswith("diff ") or line.startswith("--- ") or line.startswith("+++ "):
            _flush_hunk()
            result.append(line)
        else:
            result.append(line)

    _flush_hunk()
    return "\n".join(result)
