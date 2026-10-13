from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest

from token_sniper.evaluation.telemetry import AgentRunRecord, append_record, read_records

if TYPE_CHECKING:
    from pathlib import Path


def _record(**overrides) -> AgentRunRecord:
    base = AgentRunRecord(
        instance_id="django__django-1", dataset="swebench_verified", arm="dense_reranker_vgib", seed=0,
        model="Qwen/Qwen3.5-35B-A3B-FP8", git_ref="abc", selector_checkpoint_sha256="ff", beta=0.1,
        retrieved_files=["a.py"], retrieved_spans=[{"file_path": "a.py", "start_line": 1, "end_line": 9,
                                                    "node_type": "function"}],
        retained_span_count=1, fallback_triggered=False, gold_files=["a.py"], file_recall=1.0,
        function_recall=1.0, selector_latency_ms=12.5, first_turn_context_tokens=900, prompt_tokens=5000,
        completion_tokens=400, cached_tokens=0, turns=7, tool_calls=6, wall_seconds=120.0, usd_cost=None,
        applied=True, resolved=True, error_class=None,
    )
    return replace(base, **overrides)


def test_records_round_trip(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    first, second = _record(), _record(seed=1, resolved=False, error_class="tests_failed")
    append_record(path, first)
    append_record(path, second)
    assert read_records(path) == [first, second]


def test_missing_or_unknown_fields_raise(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    raw = asdict(_record())
    del raw["fallback_triggered"]
    raw["surprise"] = 1
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="fallback_triggered"):
        read_records(path)


def test_schema_version_mismatch_raises(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({**asdict(_record()), "schema_version": 999}) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        read_records(path)
