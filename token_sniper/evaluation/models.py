from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenUsageBreakdown(BaseModel):
    navigation_prompt_tokens: int = 0
    navigation_completion_tokens: int = 0
    reasoning_prompt_tokens: int = 0
    reasoning_completion_tokens: int = 0

    model_config = {"frozen": True}

    @property
    def navigation_tokens(self) -> int:
        return self.navigation_prompt_tokens + self.navigation_completion_tokens

    @property
    def reasoning_tokens(self) -> int:
        return self.reasoning_prompt_tokens + self.reasoning_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.navigation_tokens + self.reasoning_tokens


class SWEbenchTaskMetadata(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_patch: str
    test_patch: str | None = None
    fail_to_pass_tests: list[str] = Field(default_factory=list)
    pass_to_pass_tests: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    hints_text: str | None = None
    version: str | None = None
    environment_setup_commit: str | None = None

    model_config = {"frozen": True}


class SWEbenchEvaluationAttempt(BaseModel):
    attempt_index: int
    model_name: str
    patch_path: str
    patch_sha256: str
    run_id: str
    resolved: bool
    patch_exists: bool
    patch_successfully_applied: bool
    summary_report_path: str
    instance_report_path: str
    harness_stdout_path: str
    harness_stderr_path: str
    fail_to_pass_success: list[str] = Field(default_factory=list)
    fail_to_pass_failure: list[str] = Field(default_factory=list)
    pass_to_pass_success: list[str] = Field(default_factory=list)
    pass_to_pass_failure: list[str] = Field(default_factory=list)
    token_usage: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    session_log_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TaskEvaluationResult(BaseModel):
    instance_id: str
    phase: str
    condition: str
    dataset_name: str
    split: str
    resolved: bool
    repo: str | None = None
    base_commit: str | None = None
    difficulty: str | None = None
    fail_to_pass_tests: list[str] = Field(default_factory=list)
    pass_to_pass_tests: list[str] = Field(default_factory=list)
    gold_patch_files: list[str] = Field(default_factory=list)
    retrieved_files: list[str] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    attempts: list[SWEbenchEvaluationAttempt] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class ConditionRunSummary(BaseModel):
    schema_version: int = 1
    phase: str
    condition: str
    run_id: str
    dataset_name: str
    split: str
    run_dir: str
    config_path: str
    metrics_path: str
    task_results_path: str
    total_tasks: int
    metrics: dict[str, float | int] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}


class PairedComparisonResult(BaseModel):
    baseline_condition: str
    candidate_condition: str
    paired_tasks: int
    baseline_only_correct: int
    candidate_only_correct: int
    discordant_pairs: int
    statistic: float
    p_value: float

    model_config = {"frozen": True}
