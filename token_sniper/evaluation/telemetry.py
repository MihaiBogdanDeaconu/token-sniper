from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentRunRecord:
    instance_id: str
    dataset: str
    """e.g. "swebench_verified", "swebench_live"."""
    arm: str
    """The condition name from eval_ablation_swebench.VALID_CONDITIONS."""
    seed: int
    model: str
    """The agent LLM."""
    git_ref: str
    selector_checkpoint_sha256: str | None
    """None for arms without a learned selector."""
    beta: float | None

    retrieved_files: list[str]
    retrieved_spans: list[dict[str, Any]]
    """{"file_path", "start_line", "end_line", "node_type"} per span; empty for whole-file arms."""
    retained_span_count: int
    fallback_triggered: bool
    """True if the arm did not use its own selection (e.g. the selector kept nothing)."""
    gold_files: list[str]
    file_recall: float
    function_recall: float
    selector_latency_ms: float | None
    """None for arms without a selector (e.g. shell-only)."""

    first_turn_context_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    turns: int
    tool_calls: int
    wall_seconds: float
    usd_cost: float | None
    """None until a pricing model is supplied explicitly; never guessed."""

    applied: bool
    resolved: bool
    error_class: str | None

    schema_version: int = SCHEMA_VERSION


_FIELDS = {f.name for f in fields(AgentRunRecord)}


def append_record(path: Path, record: AgentRunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def read_records(path: Path) -> list[AgentRunRecord]:
    records = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing, unknown = _FIELDS - raw.keys(), raw.keys() - _FIELDS
        if missing or unknown:
            raise ValueError(f"{path}:{n}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{path}:{n}: schema_version {raw['schema_version']}, expected {SCHEMA_VERSION}")
        records.append(AgentRunRecord(**raw))
    return records
