from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_sniper.evaluation.locagent_scoring import LITE_K_GROUPS, LOCBENCH_K_GROUPS
from token_sniper.evaluation.statistics import bootstrap_ci, mcnemar_holm
from token_sniper.evaluation.telemetry import AgentRunRecord, read_records

CiFn = Callable[[Sequence[float]], tuple[float, float]]
McnemarFn = Callable[[Mapping[str, tuple[Sequence[bool], Sequence[bool]]]], dict[str, dict[str, Any]]]

LEVELS = ("file", "module", "function")
LEVEL_HEADER = {"file": "File", "module": "Mod", "function": "Func"}
LITE_COLUMNS = [(level, k) for level, ks in zip(LEVELS, LITE_K_GROUPS, strict=True) for k in ks]
LOCBENCH_COLUMNS = [(level, k) for level, ks in zip(LEVELS, LOCBENCH_K_GROUPS, strict=True) for k in ks]
EXPECTED_INSTANCES = {"lite_274": 274, "locbench_560": 560}

MCNEMAR_MAIN_ONLY = "baseline_only_correct"
MCNEMAR_ARM_ONLY = "candidate_only_correct"
MCNEMAR_P_HOLM = "p_holm"
MCNEMAR_SIGNIFICANT = "significant"

RETENTION_KEY = "mean_kept_ratio_of_pool"

DASH = "---"

T1_PUBLISHED: list[tuple[str, str, str | None, tuple[float, ...]]] = [
    ("Agent", "MoatlessTools", "GPT-4o", (73.36, 84.31, 85.04, 74.82, 76.28, 57.30, 59.49)),
    ("Agent", "MoatlessTools", "Claude-3.5", (72.63, 85.77, 86.13, 76.28, 76.28, 64.60, 64.96)),
    ("Agent", "SWE-agent", "GPT-4o", (57.30, 64.96, 68.98, 58.03, 58.03, 45.99, 46.35)),
    ("Agent", "SWE-agent", "Claude-3.5", (77.37, 87.23, 90.15, 77.74, 78.10, 64.23, 64.60)),
    ("Agent", "OpenHands", "GPT-4o", (60.95, 71.90, 73.72, 62.41, 63.87, 49.64, 50.36)),
    ("Agent", "OpenHands", "Claude-3.5", (76.28, 89.78, 90.15, 83.21, 83.58, 68.25, 70.07)),
    ("Agent", "LocAgent", "Qwen2.5-7B(ft)", (70.80, 84.67, 88.32, 81.02, 82.85, 64.23, 71.53)),
    ("Agent", "LocAgent", "Qwen2.5-32B(ft)", (75.91, 90.51, 92.70, 85.77, 87.23, 71.90, 77.01)),
    ("Agent", "LocAgent", "Claude-3.5", (77.74, 91.97, 94.16, 86.50, 87.59, 73.36, 77.37)),
    ("Retriever", "BM25", None, (38.69, 51.82, 61.68, 45.26, 52.92, 31.75, 36.86)),
    ("Retriever", "Jina-Code-v2 (161M)", None, (43.43, 71.17, 80.29, 63.50, 72.63, 42.34, 52.19)),
    ("Retriever", "Codesage-large-v2 (1.3B)", None, (47.81, 69.34, 78.10, 60.58, 69.71, 33.94, 44.53)),
    ("Retriever", "CodeRankEmbed (137M)", None, (52.55, 77.74, 84.67, 71.90, 78.83, 51.82, 58.76)),
    ("Retriever", "SFR-Embedding-2 (7B)", None, (58.03, 80.29, 83.94, 70.07, 79.20, 56.20, 64.23)),
    ("Retriever", "GTE-Qwen2-7B-Instruct (7B)", None, (65.33, 82.85, 89.78, 76.28, 83.58, 63.14, 70.44)),
    ("Retriever", "SweRankEmbed-Small (137M)", None, (66.42, 86.50, 90.88, 79.56, 85.04, 63.14, 74.45)),
    ("Retriever", "SweRankEmbed-Large (7B)", None, (72.63, 91.24, 94.16, 84.31, 89.78, 71.90, 82.12)),
    ("+Rerank", "CodeRankLLM (7B)", None, (72.99, 89.78, 93.80, 85.04, 90.88, 71.90, 83.58)),
    ("+Rerank", "GPT-4.1", None, (82.12, 95.62, 97.08, 93.07, 93.43, 81.75, 87.96)),
    ("+Rerank", "SweRankLLM-Small (7B)", None, (78.10, 92.34, 94.53, 89.05, 92.70, 79.56, 86.13)),
    ("+Rerank", "SweRankLLM-Large (32B)", None, (83.21, 94.89, 95.99, 90.88, 93.43, 81.39, 88.69)),
]
T1_OURS_TYPE = "Graph"

T2_PUBLISHED: list[tuple[str, str | None, tuple[float, ...]]] = [
    ("Agentless (FSE'25)", "Claude-3.5", (67.50, 67.50, 53.39, 53.39, 42.68, 42.68)),
    ("OpenHands", "Claude-3.5", (79.82, 80.00, 68.93, 69.11, 59.11, 59.29)),
    ("SWE-agent", "Claude-3.5", (77.68, 77.68, 63.57, 63.75, 51.96, 51.96)),
    ("LocAgent (ACL'25)", "Qwen2.5-7B(ft)", (78.57, 79.64, 63.04, 63.04, 51.43, 51.79)),
    ("LocAgent", "Claude-3.5", (83.39, 86.07, 70.89, 71.07, 59.29, 60.71)),
    ("CodeRankEmbed (137M)", None, (74.29, 80.36, 63.93, 67.86, 47.86, 50.89)),
    ("GTE-Qwen2-7B-Instruct", None, (75.54, 82.50, 67.14, 71.61, 51.79, 57.14)),
    ("SweRankEmbed-Small (137M)", None, (80.36, 84.82, 71.43, 75.00, 58.57, 63.39)),
    ("SweRankEmbed-Large (7B)", None, (82.14, 86.96, 75.54, 78.93, 63.21, 67.32)),
    ("CodeRankLLM (7B)", None, (83.93, 88.21, 76.96, 80.89, 64.64, 69.29)),
    ("GPT-4.1", None, (85.89, 88.75, 79.64, 82.50, 71.61, 74.64)),
    ("SweRankLLM-Small (7B)", None, (85.54, 88.39, 79.11, 82.14, 69.46, 74.46)),
    ("SweRankLLM-Large (32B)", None, (86.61, 89.82, 81.07, 83.21, 71.25, 76.25)),
]
T2_DAGGER_METHOD = "CodeRankEmbed (137M)"
T2_DAGGER_NOTE = (r"$^\dagger$SweRank Table 2 values; LocAgent Table 7 reports "
                  r"74.29 / 80.89 / 63.21 / 67.50 / 43.39 / 46.61.")

T3_PUBLISHED: list[tuple[str, str, str, str, str, str]] = [
    ("LocAgent (Claude-3.5)", "agentic LLM", "$0.66", "7", "77.37", "1.2"),
    ("LocAgent (Qwen2.5-7B ft)", "7B", "$0.05", "6", "71.53", "13.2"),
    ("LocAgent (Qwen2.5-32B ft)", "32B", "$0.09", "9", "77.01", "8.6"),
    ("SWE-agent (Claude-3.5)", "agentic LLM", "$0.67", "9", "64.60", "1.0"),
    ("SWE-agent (GPT-4o)", "agentic LLM", "$0.56", "8", "46.35", "0.8"),
    ("OpenHands (Claude-3.5)", "agentic LLM", "$0.79", "13", "70.07", "0.9"),
    ("OpenHands (GPT-4o)", "agentic LLM", "$0.83", "15", "50.36", "0.6"),
    ("CodeRankEmbed", "137M", "~$0 (precomputable)", "—", "58.76", "—"),
    ("SweRankEmbed-Small", "137M", "~$0 (precomputable)", "—", "74.45", "—"),
    ("SweRankLLM-Large", "32B", "LLM rerank per query", "—", "88.69", "—"),
]
T3_OURS_ROUNDS = "1"
T3_EMBEDDER_NOTE = ("The query-time embedder (jina-code-embeddings-0.5b, $\\sim$0.5B parameters) is not "
                    "included in the selector parameter count.")
COST_KEYS = {"selector_params", "reranker_trees", "latency_ms_per_instance",
             "amortized_build_seconds_per_instance", "gpu_hour_usd", "cpu_hour_usd",
             "usd_per_instance", "instances", "source"}
LATENCY_KEYS = {"mean", "p50", "p95"}


_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "—": DASH}


def tex(text: str) -> str:
    return "".join(_TEX_ESCAPES.get(c, c) for c in text)


def latex_table(*, caption: str, label: str, colspec: str, header: Sequence[str],
                rows: Sequence[Sequence[str] | None], notes: Sequence[str] = ()) -> str:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"]
    lines += [r"\midrule" if row is None else " & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [rf"\par\footnotesize {note}" for note in notes]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def fmt_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", "{,}")


def fmt_usd(value: float) -> str:
    return rf"\${value:.2f}" if value >= 0.01 or value == 0 else rf"\${value:.2g}"


def fmt_p(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def bold_max(values: Sequence[Sequence[float]]) -> list[list[str]]:
    rounded = [[round(v, 2) for v in row] for row in values]
    best = [max(col) for col in zip(*rounded, strict=True)]
    return [[rf"\textbf{{{v:.2f}}}" if v == b else f"{v:.2f}" for v, b in zip(row, best, strict=True)]
            for row in rounded]


@dataclass(frozen=True)
class LocRef:
    results: Path
    checkpoint: str


def parse_labeled(spec: str, flag: str) -> tuple[str, str]:
    label, sep, value = spec.partition("=")
    if not sep or not label or not value:
        raise ValueError(f"{flag} {spec!r}: expected LABEL=VALUE")
    return label, value


def parse_loc_spec(spec: str, flag: str) -> tuple[str, LocRef]:
    label, value = parse_labeled(spec, flag)
    results, sep, checkpoint = value.rpartition(":")
    if not sep or not results or not checkpoint:
        raise ValueError(f"{flag} {spec!r}: expected LABEL=RESULTS_DIR:CHECKPOINT")
    return label, LocRef(Path(results), checkpoint)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def read_accuracy(ref: LocRef, columns: Sequence[tuple[str, int]],
                  expected_instances: int | None = None) -> list[float]:
    if expected_instances is not None:
        inputs = read_json(ref.results / "inputs.json")
        if inputs["manifest_instances"] != expected_instances:
            raise ValueError(f"{ref.results}: manifest has {inputs['manifest_instances']} instances, "
                             f"the published protocol uses {expected_instances}")
    scores = read_json(ref.results / "scores.json")
    if ref.checkpoint not in scores:
        raise KeyError(f"{ref.results}/scores.json has no checkpoint {ref.checkpoint!r}; "
                       f"it has {sorted(scores)}")
    table = scores[ref.checkpoint]["all_instances"]
    values = []
    for level, k in columns:
        if f"Acc@{k}" not in table.get(level, {}):
            raise KeyError(f"{ref.results}/scores.json[{ref.checkpoint}] has no {level} Acc@{k}")
        values.append(100 * table[level][f"Acc@{k}"])
    return values


def read_selection(ref: LocRef) -> dict[str, Any]:
    selection = read_json(ref.results / "selection.json")
    if ref.checkpoint not in selection:
        raise KeyError(f"{ref.results}/selection.json has no checkpoint {ref.checkpoint!r}; "
                       f"it has {sorted(k for k in selection if k != 'keep_threshold')}")
    return selection[ref.checkpoint]


def build_t1(rows: Sequence[tuple[str, LocRef]]) -> str:
    ours = [read_accuracy(ref, LITE_COLUMNS, EXPECTED_INSTANCES["lite_274"]) for _, ref in rows]
    cells = bold_max([list(v) for *_, v in T1_PUBLISHED] + ours)
    body: list[list[str] | None] = []
    previous_type = None
    labels = ([(kind, tex(method), tex(model) if model else DASH) for kind, method, model, _ in T1_PUBLISHED]
              + [(T1_OURS_TYPE, label, DASH) for label, _ in rows])
    for (kind, method, model), values in zip(labels, cells, strict=True):
        if previous_type is not None and kind != previous_type:
            body.append(None)
        previous_type = kind
        body.append([tex(kind), method, model, *values])
    return latex_table(
        caption="Localization on SWE-Bench-Lite (274 instances), strict Acc@k (\\%). Baselines from "
                "SweRank Table 1 (cross-checked against LocAgent Table 4).",
        label="tab:t1-lite", colspec="lll" + "r" * len(LITE_COLUMNS),
        header=["Type", "Method", "Model", *(f"{LEVEL_HEADER[lv]}@{k}" for lv, k in LITE_COLUMNS)],
        rows=body,
    )


def build_t2(rows: Sequence[tuple[str, LocRef]]) -> str:
    ours = [read_accuracy(ref, LOCBENCH_COLUMNS, EXPECTED_INSTANCES["locbench_560"]) for _, ref in rows]
    cells = bold_max([list(v) for *_, v in T2_PUBLISHED] + ours)
    body: list[list[str] | None] = []
    for (method, model, _), values in zip(T2_PUBLISHED, cells[:len(T2_PUBLISHED)], strict=True):
        name = tex(method) + (r"$^\dagger$" if method == T2_DAGGER_METHOD else "")
        body.append([name, tex(model) if model else DASH, *values])
    body.append(None)
    body += [[label, DASH, *values] for (label, _), values in zip(rows, cells[len(T2_PUBLISHED):], strict=True)]
    return latex_table(
        caption="Localization on LocBench (560 instances), strict Acc@k (\\%). Baselines from SweRank Table 2.",
        label="tab:t2-locbench", colspec="ll" + "r" * len(LOCBENCH_COLUMNS),
        header=["Method", "Loc Model", *(f"{LEVEL_HEADER[lv]}@{k}" for lv, k in LOCBENCH_COLUMNS)],
        rows=body, notes=[T2_DAGGER_NOTE],
    )


def read_cost(path: Path) -> dict[str, Any]:
    cost = read_json(path)
    missing, unknown = COST_KEYS - cost.keys(), cost.keys() - COST_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    latency = cost["latency_ms_per_instance"]
    if set(latency) != LATENCY_KEYS:
        raise ValueError(f"{path}: latency_ms_per_instance must have exactly {sorted(LATENCY_KEYS)}")
    if cost["usd_per_instance"] <= 0:
        raise ValueError(f"{path}: usd_per_instance must be positive, got {cost['usd_per_instance']}")
    return cost


def build_t3(costs: Sequence[tuple[str, Path]], lite_rows: Sequence[tuple[str, LocRef]]) -> str:
    lite = dict(lite_rows)
    func10 = LITE_COLUMNS.index(("function", 10))
    body: list[list[str] | None] = [[tex(c) for c in row[:4]] + [row[4], tex(row[5]), DASH]
                                    for row in T3_PUBLISHED]
    body.append(None)
    notes = []
    for label, path in costs:
        if label not in lite:
            raise ValueError(f"--cost {label!r} has no --lite row with the same label to join Func Acc@10")
        cost = read_cost(path)
        acc = read_accuracy(lite[label], LITE_COLUMNS, EXPECTED_INSTANCES["lite_274"])[func10]
        latency = cost["latency_ms_per_instance"]
        body.append([
            label, f"{cost['selector_params'] / 1e6:.1f}M + {cost['reranker_trees']}-tree XGB",
            fmt_usd(cost["usd_per_instance"]), T3_OURS_ROUNDS, f"{acc:.2f}",
            f"{acc / 100 / cost['usd_per_instance']:.1f}", f"{latency['p50']:.0f}",
        ])
        notes.append(
            f"{label}: latency p95 {latency['p95']:.0f}\\,ms, mean {latency['mean']:.0f}\\,ms; cost includes "
            f"the amortised SSG + index build ({cost['amortized_build_seconds_per_instance']:.1f}\\,s per "
            f"instance) at \\${cost['gpu_hour_usd']:.2f}/GPU-h and \\${cost['cpu_hour_usd']:.2f}/CPU-h, "
            f"measured over {cost['instances']} instances ({tex(cost['source'])})."
        )
    return latex_table(
        caption="Cost--accuracy on SWE-Bench-Lite. Baseline cost and rounds from LocAgent Table 5, "
                "Func Acc@10 joined from LocAgent Table 4. Acc per \\$ = Func Acc@10 (fraction) / cost. "
                f"{T3_EMBEDDER_NOTE} " + " ".join(notes),
        label="tab:t3-cost", colspec="llrrrrr",
        header=["Method", "Selector params", r"Cost / inst.", "Rounds", "Func Acc@10", r"Acc per \$",
                "Latency p50 (ms)"],
        rows=body,
    )


RunIndex = dict[tuple[str, int], AgentRunRecord]


def index_runs(records: Sequence[AgentRunRecord], what: str) -> RunIndex:
    index: RunIndex = {}
    for r in records:
        key = (r.instance_id, r.seed)
        if key in index:
            raise ValueError(f"{what}: duplicate record for task {r.instance_id} seed {r.seed}")
        index[key] = r
    return index


@dataclass(frozen=True)
class Uniform:
    tasks: list[str]
    """Tasks kept on every arm, sorted."""
    dropped: list[str]
    seeds: list[int]


def uniform_tasks(groups: Mapping[str, RunIndex]) -> Uniform:
    seed_sets = {name: {seed for _, seed in index} for name, index in groups.items()}
    seeds = next(iter(seed_sets.values()))
    if any(s != seeds for s in seed_sets.values()):
        raise ValueError(f"conditions were run with different seeds: {({k: sorted(v) for k, v in seed_sets.items()})}")
    all_tasks = sorted({task for index in groups.values() for task, _ in index})
    kept = [t for t in all_tasks
            if all((r := index.get((t, s))) is not None and r.error_class is None
                   for index in groups.values() for s in seeds)]
    if not kept:
        raise ValueError(f"no task succeeded on every condition and seed (of {len(all_tasks)} tasks)")
    return Uniform(kept, sorted(set(all_tasks) - set(kept)), sorted(seeds))


@dataclass(frozen=True)
class ArmSummary:
    resolve_mean: float
    resolve_std: float | None
    """None with a single seed."""
    ci: tuple[float, float]
    first_turn_tokens: float
    total_tokens: float
    usd_per_task: float | None
    usd_per_resolved: float | None
    """None without usd_cost, or when nothing was resolved."""
    wall_seconds: float
    n_tasks: int
    n_seeds: int


def summarize_arm(index: RunIndex, uniform: Uniform, ci_fn: CiFn, what: str) -> ArmSummary:
    runs = [index[(t, s)] for t in uniform.tasks for s in uniform.seeds]
    per_seed = [statistics.fmean(index[(t, s)].resolved for t in uniform.tasks) for s in uniform.seeds]
    per_task = [statistics.fmean(index[(t, s)].resolved for s in uniform.seeds) for t in uniform.tasks]
    costs = [r.usd_cost for r in runs]
    if all(c is None for c in costs):
        usd_per_task = usd_per_resolved = None
    elif any(c is None for c in costs):
        raise ValueError(f"{what}: usd_cost is set on some runs and missing on others")
    else:
        total_usd = sum(c for c in costs if c is not None)
        n_resolved = sum(r.resolved for r in runs)
        usd_per_task = total_usd / len(runs)
        usd_per_resolved = total_usd / n_resolved if n_resolved else None
    return ArmSummary(
        resolve_mean=statistics.fmean(per_seed),
        resolve_std=statistics.stdev(per_seed) if len(per_seed) > 1 else None,
        ci=ci_fn(per_task),
        first_turn_tokens=statistics.fmean(r.first_turn_context_tokens for r in runs),
        total_tokens=statistics.fmean(r.prompt_tokens + r.completion_tokens for r in runs),
        usd_per_task=usd_per_task,
        usd_per_resolved=usd_per_resolved,
        wall_seconds=statistics.fmean(r.wall_seconds for r in runs),
        n_tasks=len(uniform.tasks),
        n_seeds=len(uniform.seeds),
    )


def majority_resolved(index: RunIndex, uniform: Uniform, what: str) -> list[bool]:
    flags = []
    for task in uniform.tasks:
        n = sum(index[(task, s)].resolved for s in uniform.seeds)
        if 2 * n == len(uniform.seeds):
            raise ValueError(f"{what}: task {task} is resolved on exactly half of its "
                             f"{len(uniform.seeds)} seeds, so its majority flag is undefined")
        flags.append(2 * n > len(uniform.seeds))
    return flags


def fmt_resolve(s: ArmSummary) -> str:
    return f"{100 * s.resolve_mean:.1f}" + (f" $\\pm$ {100 * s.resolve_std:.1f}" if s.resolve_std is not None else "")


def single_value(records: Sequence[AgentRunRecord], field: str, what: str) -> Any:
    values = {getattr(r, field) for r in records}
    if len(values) != 1:
        raise ValueError(f"{what}: expected one {field}, found {sorted(values, key=str)}")
    return values.pop()


def load_t4(records: Sequence[AgentRunRecord], arms: Sequence[tuple[str, str]],
            main_arm: str) -> dict[str, dict[str, RunIndex]]:
    labels = dict(arms)
    if main_arm not in labels:
        raise ValueError(f"--main-arm {main_arm!r} is not among --arm {list(labels)}")
    unlabeled = {r.arm for r in records} - labels.keys()
    if unlabeled:
        raise ValueError(f"telemetry has arms without an --arm label: {sorted(unlabeled)}")
    out = {}
    for dataset in sorted({r.dataset for r in records}):
        by_arm = {arm: [r for r in records if r.dataset == dataset and r.arm == arm] for arm in labels}
        empty = [arm for arm, rs in by_arm.items() if not rs]
        if empty:
            raise ValueError(f"{dataset}: no telemetry for arms {empty}")
        for arm, rs in by_arm.items():
            single_value(rs, "beta", f"{dataset}/{arm}")
        out[dataset] = {arm: index_runs(rs, f"{dataset}/{arm}") for arm, rs in by_arm.items()}
    return out


def drop_note(uniform: Uniform) -> str:
    n_all = len(uniform.tasks) + len(uniform.dropped)
    return (f"{len(uniform.tasks)} tasks $\\times$ {len(uniform.seeds)} seeds; {len(uniform.dropped)} of {n_all} "
            "tasks are dropped from every downstream exhibit because they errored or have no record on at "
            "least one arm, ablation row or $\\beta$ and seed.")


def build_t4(dataset: str, indexes: Mapping[str, RunIndex], arms: Sequence[tuple[str, str]], main_arm: str,
             uniform: Uniform, ci_fn: CiFn, mcnemar_fn: McnemarFn,
             ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    labels = dict(arms)
    summaries = {arm: summarize_arm(idx, uniform, ci_fn, f"{dataset}/{arm}") for arm, idx in indexes.items()}
    main_flags = majority_resolved(indexes[main_arm], uniform, f"{dataset}/{main_arm}")
    comparisons = {arm: (main_flags, majority_resolved(idx, uniform, f"{dataset}/{arm}"))
                   for arm, idx in indexes.items() if arm != main_arm}
    tests = mcnemar_fn(comparisons) if comparisons else {}
    body = []
    for arm, label in arms:
        s = summaries[arm]
        if arm == main_arm:
            test_cells = ["(main)", DASH]
        else:
            t = tests[arm]
            keys = (MCNEMAR_MAIN_ONLY, MCNEMAR_ARM_ONLY, MCNEMAR_P_HOLM, MCNEMAR_SIGNIFICANT)
            if any(k not in t for k in keys):
                raise KeyError(f"mcnemar_holm result for {arm} lacks {keys}; it has {sorted(t)}")
            p = fmt_p(t[MCNEMAR_P_HOLM]) + ("$^*$" if t[MCNEMAR_SIGNIFICANT] else "")
            test_cells = [f"{t[MCNEMAR_ARM_ONLY]} / {t[MCNEMAR_MAIN_ONLY]}", p]
        body.append([
            label, fmt_resolve(s), f"[{100 * s.ci[0]:.1f}, {100 * s.ci[1]:.1f}]",
            fmt_int(s.first_turn_tokens), fmt_int(s.total_tokens),
            DASH if s.usd_per_task is None else fmt_usd(s.usd_per_task),
            DASH if s.usd_per_resolved is None else fmt_usd(s.usd_per_resolved),
            f"{s.wall_seconds:.0f}", *test_cells,
        ])
    caption = (
        f"Downstream resolution on {tex(dataset)}: {drop_note(uniform)} Resolve@1 (\\%): mean $\\pm$ std over "
        "seeds, 95\\% bootstrap CI over tasks. Tokens are means per run; total = prompt + completion. "
        "\\$ columns are blank (---) where the run has no priced cost. McNemar over tasks (each arm's "
        f"majority resolved flag across seeds) against {labels[main_arm]}, Holm-corrected; $n_+ / n_-$ = "
        "tasks only this arm / only the main arm resolved; $^*$ significant at 0.05."
    )
    latex = latex_table(
        caption=caption, label=f"tab:t4-{dataset.replace('_', '-')}", colspec="lrrrrrrrrr",
        header=["Arm", "Resolve@1", "95\\% CI", "Ctx tok. (turn 1)", "Total tok.", r"\$/task",
                r"\$/resolved", "Wall (s)", "$n_+ / n_-$", "$p_\\mathrm{Holm}$"],
        rows=body,
    )
    audit = {
        "main_arm": main_arm, "seeds": uniform.seeds, "kept_tasks": uniform.tasks,
        "dropped_tasks": uniform.dropped,
        "arms": {arm: {"label": labels[arm], **vars(summaries[arm])} for arm in labels},
        "mcnemar_holm": tests,
    }
    f2 = [f2_row(dataset, "arm", label, None, summaries[arm]) for arm, label in arms]
    return latex, audit, f2


@dataclass(frozen=True)
class TelemetryRef:
    path: Path
    arm: str
    beta: float | None


def parse_loc_ref(raw: Any, base: Path, where: str) -> LocRef | None:
    if raw is None:
        return None
    if set(raw) != {"results", "checkpoint"}:
        raise ValueError(f"{where}: localization must have exactly 'results' and 'checkpoint'")
    return LocRef(base / raw["results"], raw["checkpoint"])


def parse_telemetry_ref(raw: Any, base: Path, where: str) -> TelemetryRef | None:
    if raw is None:
        return None
    if not {"path", "arm"} <= set(raw) <= {"path", "arm", "beta"}:
        raise ValueError(f"{where}: telemetry must have 'path' and 'arm', optionally 'beta'")
    return TelemetryRef(base / raw["path"], raw["arm"], raw.get("beta"))


def read_manifest(path: Path, list_key: str, entry_keys: set[str]) -> list[dict[str, Any]]:
    manifest = read_json(path)
    if set(manifest) != {list_key}:
        raise ValueError(f"{path}: expected exactly one key {list_key!r}")
    for n, entry in enumerate(manifest[list_key]):
        if set(entry) != entry_keys:
            raise ValueError(f"{path}: {list_key}[{n}] must have exactly {sorted(entry_keys)}, has {sorted(entry)}")
    return manifest[list_key]


@dataclass(frozen=True)
class Runs:
    dataset: str
    indexes: dict[str, RunIndex]
    betas: dict[str, float | None]


def resolve_refs(refs: Mapping[str, TelemetryRef]) -> Runs | None:
    if not refs:
        return None
    cache: dict[Path, list[AgentRunRecord]] = {}
    indexes, betas, datasets = {}, {}, {}
    for name, ref in refs.items():
        if ref.path not in cache:
            cache[ref.path] = read_records(ref.path)
        records = [r for r in cache[ref.path]
                   if r.arm == ref.arm and (ref.beta is None or r.beta == ref.beta)]
        if not records:
            raise ValueError(f"{name}: no records for arm {ref.arm!r} (beta {ref.beta}) in {ref.path}")
        betas[name] = single_value(records, "beta", name)
        datasets[name] = single_value(records, "dataset", name)
        indexes[name] = index_runs(records, name)
    if len(set(datasets.values())) != 1:
        raise ValueError(f"rows mix datasets: {datasets}")
    return Runs(next(iter(datasets.values())), indexes, betas)


@dataclass(frozen=True)
class T5Input:
    rows: list[dict[str, Any]]
    locs: dict[str, LocRef | None]
    runs: Runs | None


def load_t5(manifest: Path) -> T5Input:
    rows = read_manifest(manifest, "rows", {"label", "localization", "telemetry"})
    base = manifest.parent
    locs = {row["label"]: parse_loc_ref(row["localization"], base, row["label"]) for row in rows}
    if len(locs) != len(rows):
        raise ValueError(f"{manifest}: row labels must be unique")
    refs = {row["label"]: ref for row in rows
            if (ref := parse_telemetry_ref(row["telemetry"], base, row["label"])) is not None}
    return T5Input(rows, locs, resolve_refs(refs))


def build_t5(t5: T5Input, uniforms: Mapping[str, Uniform], ci_fn: CiFn) -> str:
    summaries, note = {}, "No row has downstream runs yet."
    if t5.runs is not None:
        uniform = uniforms[t5.runs.dataset]
        summaries = {name: summarize_arm(idx, uniform, ci_fn, name) for name, idx in t5.runs.indexes.items()}
        note = f"Resolve@1 (\\%) on {tex(t5.runs.dataset)}, mean $\\pm$ std over seeds: {drop_note(uniform)}"
    body = []
    for row in t5.rows:
        loc = t5.locs[row["label"]]
        func10 = DASH if loc is None else f"{read_accuracy(loc, [('function', 10)])[0]:.2f}"
        retention = DASH if loc is None else f"{100 * read_selection(loc)[RETENTION_KEY]:.1f}"
        resolve = fmt_resolve(summaries[row["label"]]) if row["label"] in summaries else DASH
        body.append([row["label"], func10, retention, resolve])
    return latex_table(
        caption="Ablation. Func Acc@10 (\\%) over all manifest instances; token retention (\\%) = tokens "
                f"kept by the selector over the candidate pool. {note} --- = not run.",
        label="tab:t5-ablation", colspec="lrrr",
        header=["Variant", "Func Acc@10", "Token retention", "Resolve@1"], rows=body,
    )


F2_FIELDS = ["dataset", "series", "label", "beta", "first_turn_context_tokens", "total_tokens",
             "resolve_mean", "resolve_std", "resolve_ci_low", "resolve_ci_high", "n_tasks", "n_seeds"]
F3_FIELDS = ["beta", "token_retention_pool", "token_retention_topk", "mean_function_recall",
             "gold_files_kept_rate", "func_acc_at_10", "resolve_mean", "resolve_std", "resolve_ci_low",
             "resolve_ci_high", "first_turn_context_tokens", "n_tasks", "n_seeds"]


def f2_row(dataset: str, series: str, label: str, beta: float | None, s: ArmSummary) -> dict[str, Any]:
    return {"dataset": dataset, "series": series, "label": label, "beta": beta,
            "first_turn_context_tokens": s.first_turn_tokens, "total_tokens": s.total_tokens,
            "resolve_mean": s.resolve_mean, "resolve_std": s.resolve_std, "resolve_ci_low": s.ci[0],
            "resolve_ci_high": s.ci[1], "n_tasks": s.n_tasks, "n_seeds": s.n_seeds}


@dataclass(frozen=True)
class F3Input:
    points: list[dict[str, Any]]
    names: list[str]
    locs: dict[str, LocRef | None]
    runs: Runs | None


def load_f3(manifest: Path) -> F3Input:
    points = read_manifest(manifest, "points", {"beta", "localization", "telemetry"})
    base = manifest.parent
    names = [f"beta={p['beta']}" for p in points]
    if len(set(names)) != len(names):
        raise ValueError(f"{manifest}: betas must be unique")
    refs = {name: ref for name, p in zip(names, points, strict=True)
            if (ref := parse_telemetry_ref(p["telemetry"], base, name)) is not None}
    runs = resolve_refs(refs)
    if runs is not None:
        for name, p in zip(names, points, strict=True):
            if name in runs.betas and runs.betas[name] != p["beta"]:
                raise ValueError(f"{name}: telemetry records have beta {runs.betas[name]}")
    locs = {name: parse_loc_ref(p["localization"], base, name) for name, p in zip(names, points, strict=True)}
    return F3Input(points, names, locs, runs)


def build_f3(f3: F3Input, uniforms: Mapping[str, Uniform],
             ci_fn: CiFn) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: dict[str, ArmSummary] = {}
    f2 = []
    if f3.runs is not None:
        uniform = uniforms[f3.runs.dataset]
        summaries = {name: summarize_arm(idx, uniform, ci_fn, name) for name, idx in f3.runs.indexes.items()}
        f2 = [f2_row(f3.runs.dataset, "beta", name, p["beta"], summaries[name])
              for name, p in zip(f3.names, f3.points, strict=True) if name in summaries]
    rows = []
    for name, p in zip(f3.names, f3.points, strict=True):
        row: dict[str, Any] = {"beta": p["beta"]}
        loc = f3.locs[name]
        if loc is not None:
            sel = read_selection(loc)
            row |= {"token_retention_pool": sel["mean_kept_ratio_of_pool"],
                    "token_retention_topk": sel["mean_kept_ratio_of_topk"],
                    "mean_function_recall": sel["mean_function_recall"],
                    "gold_files_kept_rate": sel["gold_files_kept_rate"],
                    "func_acc_at_10": read_accuracy(loc, [("function", 10)])[0] / 100}
        if name in summaries:
            s = summaries[name]
            row |= {"resolve_mean": s.resolve_mean, "resolve_std": s.resolve_std, "resolve_ci_low": s.ci[0],
                    "resolve_ci_high": s.ci[1], "first_turn_context_tokens": s.first_turn_tokens,
                    "n_tasks": s.n_tasks, "n_seeds": s.n_seeds}
        rows.append(row)
    return rows, f2


@dataclass(frozen=True)
class Downstream:
    t4: dict[str, tuple[str, dict[str, Any], list[dict[str, Any]]]]
    """Per dataset: (LaTeX, audit JSON, F2 rows)."""
    t5: str | None
    f3: list[dict[str, Any]] | None
    f2: list[dict[str, Any]]
    uniforms: dict[str, Uniform]


def build_downstream(*, records: Sequence[AgentRunRecord] = (), arms: Sequence[tuple[str, str]] = (),
                     main_arm: str | None = None, t5_manifest: Path | None = None,
                     f3_manifest: Path | None = None, ci_fn: CiFn, mcnemar_fn: McnemarFn) -> Downstream:
    if records and main_arm is None:
        raise ValueError("telemetry records need a main arm")
    t4_runs = load_t4(records, arms, main_arm) if records and main_arm is not None else {}
    t5 = load_t5(t5_manifest) if t5_manifest else None
    f3 = load_f3(f3_manifest) if f3_manifest else None

    groups: dict[str, dict[str, RunIndex]] = {}
    for dataset, indexes in t4_runs.items():
        groups.setdefault(dataset, {}).update({f"T4 arm {arm}": idx for arm, idx in indexes.items()})
    for kind, loaded in (("T5 row", t5), ("F3", f3)):
        if loaded is not None and loaded.runs is not None:
            groups.setdefault(loaded.runs.dataset, {}).update(
                {f"{kind} {name}": idx for name, idx in loaded.runs.indexes.items()})
    uniforms = {dataset: uniform_tasks(g) for dataset, g in groups.items()}

    t4 = {dataset: build_t4(dataset, indexes, arms, main_arm, uniforms[dataset], ci_fn, mcnemar_fn)
          for dataset, indexes in t4_runs.items() if main_arm is not None}
    f2 = [row for _, _, rows in t4.values() for row in rows]
    f3_rows = None
    if f3 is not None:
        f3_rows, f2_beta = build_f3(f3, uniforms, ci_fn)
        f2 += f2_beta
    return Downstream(t4=t4, t5=build_t5(t5, uniforms, ci_fn) if t5 else None, f3=f3_rows, f2=f2,
                      uniforms=uniforms)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper tables T1-T5 (LaTeX) and F2/F3 series (CSV)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lite", action="append", default=[], metavar="LABEL=DIR:CHECKPOINT")
    parser.add_argument("--locbench", action="append", default=[], metavar="LABEL=DIR:CHECKPOINT")
    parser.add_argument("--cost", action="append", default=[], metavar="LABEL=COST_JSON")
    parser.add_argument("--telemetry", action="append", default=[], type=Path, metavar="TELEMETRY_JSONL")
    parser.add_argument("--arm", action="append", default=[], metavar="ARM=LABEL",
                        help="T4 row label (LaTeX) per arm, in row order; every telemetry arm needs one")
    parser.add_argument("--main-arm", default=None, help="T4 arm every other arm is tested against")
    parser.add_argument("--t5-manifest", type=Path, default=None)
    parser.add_argument("--f3-manifest", type=Path, default=None)
    return parser.parse_args(argv)


def build(args: argparse.Namespace, *, ci_fn: CiFn = bootstrap_ci,
          mcnemar_fn: McnemarFn = mcnemar_holm) -> list[Path]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(name: str, text: str) -> None:
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)

    lite = [parse_loc_spec(s, "--lite") for s in args.lite]
    locbench = [parse_loc_spec(s, "--locbench") for s in args.locbench]
    for flag, rows in (("--lite", lite), ("--locbench", locbench)):
        if len({label for label, _ in rows}) != len(rows):
            raise ValueError(f"{flag} labels must be unique")
    if lite:
        write("t1_lite_274.tex", build_t1(lite))
    if locbench:
        write("t2_locbench_560.tex", build_t2(locbench))
    if args.cost:
        costs = [parse_labeled(s, "--cost") for s in args.cost]
        write("t3_cost.tex", build_t3([(label, Path(p)) for label, p in costs], lite))

    records: list[AgentRunRecord] = []
    arms: list[tuple[str, str]] = []
    if args.telemetry:
        if not args.arm or args.main_arm is None:
            raise ValueError("--telemetry needs --arm ARM=LABEL for every arm and --main-arm")
        arms = [parse_labeled(s, "--arm") for s in args.arm]
        if len({arm for arm, _ in arms}) != len(arms):
            raise ValueError("--arm names must be unique")
        records = [r for path in args.telemetry for r in read_records(path)]
    if records or args.t5_manifest or args.f3_manifest:
        down = build_downstream(records=records, arms=arms, main_arm=args.main_arm,
                                t5_manifest=args.t5_manifest, f3_manifest=args.f3_manifest,
                                ci_fn=ci_fn, mcnemar_fn=mcnemar_fn)
        for dataset, (latex, audit, _) in down.t4.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset):
                raise ValueError(f"dataset name {dataset!r} is not usable in a file name")
            write(f"t4_{dataset}.tex", latex)
            write(f"t4_{dataset}.json", json.dumps(audit, indent=2) + "\n")
        if down.t5 is not None:
            write("t5_ablation.tex", down.t5)
        if down.f3 is not None:
            write_csv(out / "f3_beta.csv", F3_FIELDS, down.f3)
            written.append(out / "f3_beta.csv")
        if down.f2:
            write_csv(out / "f2_pareto.csv", F2_FIELDS, down.f2)
            written.append(out / "f2_pareto.csv")
    if not written:
        raise ValueError("no inputs given: pass at least one of --lite, --locbench, --telemetry, "
                         "--t5-manifest, --f3-manifest")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    for path in build(parse_args(argv)):
        print(path)


if __name__ == "__main__":
    main()
