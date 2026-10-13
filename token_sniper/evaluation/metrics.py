from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from scipy.stats import binomtest

from token_sniper.evaluation.models import (
    PairedComparisonResult,
    TaskEvaluationResult,
    TokenUsageBreakdown,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence


def compute_patch_sha256(patch_text: str) -> str:
    return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


def extract_modified_files_from_patch(patch_text: str) -> list[str]:
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


def pass_at_k(task_attempt_outcomes: Sequence[Sequence[bool]], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not task_attempt_outcomes:
        return 0.0
    successes = sum(1 for attempts in task_attempt_outcomes if any(attempts[:k]))
    return successes / len(task_attempt_outcomes)


def precision_at_k(retrieved: Sequence[str], gold: Collection[str], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not retrieved:
        return 0.0
    top_k = list(retrieved[:k])
    if not top_k:
        return 0.0
    gold_set = set(gold)
    hits = sum(1 for item in top_k if item in gold_set)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], gold: Collection[str], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    hits = sum(1 for item in retrieved[:k] if item in gold_set)
    return hits / len(gold_set)


def average_precision(retrieved: Sequence[str], gold: Collection[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0

    hits = 0
    precision_sum = 0.0
    seen_relevant: set[str] = set()
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set and item not in seen_relevant:
            seen_relevant.add(item)
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / len(gold_set)


def mean_average_precision(samples: Iterable[tuple[Sequence[str], Collection[str]]]) -> float:
    values = [average_precision(retrieved, gold) for retrieved, gold in samples]
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_retrieval_metrics(
    retrieved: Sequence[str],
    gold: Collection[str],
    *,
    k_values: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"retrieval_precision_at_{k}"] = precision_at_k(retrieved, gold, k=k)
        metrics[f"retrieval_recall_at_{k}"] = recall_at_k(retrieved, gold, k=k)
    metrics["retrieval_map"] = average_precision(retrieved, gold)
    return metrics


def token_usage_from_session_log(session_log_path: Path | str) -> TokenUsageBreakdown:
    from token_sniper.agent.runner import AgentRunResult

    path = Path(session_log_path).resolve()
    result = AgentRunResult.model_validate_json(path.read_text(encoding="utf-8"))

    nav_prompt = 0
    nav_completion = 0
    reasoning_prompt = 0
    reasoning_completion = 0
    saw_tool_outputs = False
    for turn in result.turns:
        if saw_tool_outputs:
            nav_prompt += turn.prompt_tokens
        else:
            reasoning_prompt += turn.prompt_tokens
        if turn.tool_calls:
            nav_completion += turn.completion_tokens
            saw_tool_outputs = True
        else:
            reasoning_completion += turn.completion_tokens
            saw_tool_outputs = False
    return TokenUsageBreakdown(
        navigation_prompt_tokens=nav_prompt,
        navigation_completion_tokens=nav_completion,
        reasoning_prompt_tokens=reasoning_prompt,
        reasoning_completion_tokens=reasoning_completion,
    )


def compute_token_efficiency(
    task_results: Sequence[TaskEvaluationResult],
) -> dict[str, float | int]:
    resolved_tasks = sum(1 for task in task_results if task.resolved)
    usage_by_task = [_token_usage_for_task_result(task) for task in task_results]
    navigation_tokens = sum(usage.navigation_tokens for usage in usage_by_task)
    reasoning_tokens = sum(usage.reasoning_tokens for usage in usage_by_task)
    total_tokens = navigation_tokens + reasoning_tokens

    def resolved_per_1k(tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        return (resolved_tasks * 1000.0) / tokens

    navigation_fraction = navigation_tokens / total_tokens if total_tokens else 0.0
    reasoning_fraction = reasoning_tokens / total_tokens if total_tokens else 0.0

    return {
        "resolved_tasks": resolved_tasks,
        "total_tokens": total_tokens,
        "navigation_tokens": navigation_tokens,
        "reasoning_tokens": reasoning_tokens,
        "navigation_token_fraction": navigation_fraction,
        "reasoning_token_fraction": reasoning_fraction,
        "token_efficiency_total": resolved_per_1k(total_tokens),
        "token_efficiency_navigation": resolved_per_1k(navigation_tokens),
        "token_efficiency_reasoning": resolved_per_1k(reasoning_tokens),
        "tokens_per_resolved_task_total": (total_tokens / resolved_tasks)
        if resolved_tasks
        else 0.0,
        "tokens_per_resolved_task_navigation": (
            navigation_tokens / resolved_tasks if resolved_tasks else 0.0
        ),
        "tokens_per_resolved_task_reasoning": (
            reasoning_tokens / resolved_tasks if resolved_tasks else 0.0
        ),
    }


def _token_usage_for_task_result(task_result: TaskEvaluationResult) -> TokenUsageBreakdown:
    if task_result.attempts:
        return TokenUsageBreakdown(
            navigation_prompt_tokens=sum(
                attempt.token_usage.navigation_prompt_tokens for attempt in task_result.attempts
            ),
            navigation_completion_tokens=sum(
                attempt.token_usage.navigation_completion_tokens for attempt in task_result.attempts
            ),
            reasoning_prompt_tokens=sum(
                attempt.token_usage.reasoning_prompt_tokens for attempt in task_result.attempts
            ),
            reasoning_completion_tokens=sum(
                attempt.token_usage.reasoning_completion_tokens for attempt in task_result.attempts
            ),
        )

    session_log_path = task_result.metadata.get("session_log_path")
    if isinstance(session_log_path, str) and session_log_path:
        path = Path(session_log_path)
        if path.is_file():
            return token_usage_from_session_log(path)
    return TokenUsageBreakdown()


def mcnemar_test(first: Sequence[bool], second: Sequence[bool]) -> dict[str, float | int]:
    if len(first) != len(second):
        raise ValueError("Inputs must have the same length for a paired comparison.")

    baseline_only_correct = sum(1 for a, b in zip(first, second, strict=True) if a and not b)
    candidate_only_correct = sum(1 for a, b in zip(first, second, strict=True) if not a and b)
    discordant_pairs = baseline_only_correct + candidate_only_correct
    if discordant_pairs == 0:
        return {
            "baseline_only_correct": baseline_only_correct,
            "candidate_only_correct": candidate_only_correct,
            "discordant_pairs": discordant_pairs,
            "statistic": 0.0,
            "p_value": 1.0,
        }

    statistic = (
        (abs(candidate_only_correct - baseline_only_correct) - 1.0) ** 2
    ) / discordant_pairs
    p_value = float(
        binomtest(
            min(baseline_only_correct, candidate_only_correct),
            n=discordant_pairs,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return {
        "baseline_only_correct": baseline_only_correct,
        "candidate_only_correct": candidate_only_correct,
        "discordant_pairs": discordant_pairs,
        "statistic": statistic,
        "p_value": p_value,
    }


def compare_task_resolution(
    baseline_condition: str,
    candidate_condition: str,
    baseline_results: Sequence[TaskEvaluationResult],
    candidate_results: Sequence[TaskEvaluationResult],
) -> PairedComparisonResult:
    baseline_map = {task.instance_id: task.resolved for task in baseline_results}
    candidate_map = {task.instance_id: task.resolved for task in candidate_results}
    baseline_ids = set(baseline_map)
    candidate_ids = set(candidate_map)
    if baseline_ids != candidate_ids:
        missing_from_candidate = sorted(baseline_ids - candidate_ids)
        missing_from_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "Paired comparison requires the same task set in both conditions. "
            f"Missing from candidate={missing_from_candidate or []}, "
            f"missing from baseline={missing_from_baseline or []}."
        )

    paired_ids = sorted(baseline_ids)
    baseline = [baseline_map[instance_id] for instance_id in paired_ids]
    candidate = [candidate_map[instance_id] for instance_id in paired_ids]
    stats = mcnemar_test(baseline, candidate)
    return PairedComparisonResult(
        baseline_condition=baseline_condition,
        candidate_condition=candidate_condition,
        paired_tasks=len(paired_ids),
        baseline_only_correct=int(stats["baseline_only_correct"]),
        candidate_only_correct=int(stats["candidate_only_correct"]),
        discordant_pairs=int(stats["discordant_pairs"]),
        statistic=float(stats["statistic"]),
        p_value=float(stats["p_value"]),
    )


def summarize_condition_metrics(
    task_results: Sequence[TaskEvaluationResult],
    *,
    retrieval_k_values: Sequence[int],
) -> dict[str, float | int]:
    attempt_outcomes = [[attempt.resolved for attempt in task.attempts] for task in task_results]
    metrics: dict[str, float | int] = {
        "num_tasks": len(task_results),
        "pass_at_1": pass_at_k(attempt_outcomes, k=1) if attempt_outcomes else 0.0,
        "pass_at_3": pass_at_k(attempt_outcomes, k=3) if attempt_outcomes else 0.0,
    }

    retrieval_values: dict[str, list[float]] = defaultdict(list)
    fail_to_pass_rates: list[float] = []
    pass_to_pass_rates: list[float] = []
    for task in task_results:
        for key, value in task.retrieval_metrics.items():
            retrieval_values[key].append(value)
        for attempt in task.attempts:
            if attempt.metadata.get("evaluation_error"):
                continue
            ftp_total = len(attempt.fail_to_pass_success) + len(attempt.fail_to_pass_failure)
            ptp_total = len(attempt.pass_to_pass_success) + len(attempt.pass_to_pass_failure)
            fail_to_pass_rates.append(
                len(attempt.fail_to_pass_success) / ftp_total if ftp_total else 1.0
            )
            pass_to_pass_rates.append(
                len(attempt.pass_to_pass_success) / ptp_total if ptp_total else 1.0
            )

    for key in (
        [f"retrieval_precision_at_{k}" for k in retrieval_k_values]
        + [f"retrieval_recall_at_{k}" for k in retrieval_k_values]
        + ["retrieval_map"]
    ):
        values = retrieval_values.get(key, [])
        metrics[key] = sum(values) / len(values) if values else 0.0

    metrics["mean_fail_to_pass_rate"] = (
        sum(fail_to_pass_rates) / len(fail_to_pass_rates) if fail_to_pass_rates else 0.0
    )
    metrics["mean_pass_to_pass_rate"] = (
        sum(pass_to_pass_rates) / len(pass_to_pass_rates) if pass_to_pass_rates else 0.0
    )
    metrics.update(compute_token_efficiency(task_results))
    return metrics


def to_jsonl(records: Iterable[dict[str, object]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
