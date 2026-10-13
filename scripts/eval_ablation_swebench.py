from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from token_sniper.evaluation.arms.selection import (
    ARMS,
    BUDGET_ARMS,
    CHECKPOINT_ARMS,
    Selection,
    SelectionConfig,
    Selector,
    TaskInput,
    checkpoint_identity,
    task_input,
)
from token_sniper.evaluation.phase0 import Phase0Task, load_phase0_manifest
from token_sniper.evaluation.telemetry import AgentRunRecord, append_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_CONDITIONS = list(ARMS)
TESTBED = "/testbed"
CONTEXT_WINDOW = 65_536
MAX_OUTPUT_TOKENS = 16_384
AGENT_OUTCOME_ERRORS = (
    "Agent reached the maximum number of turns without completing.",
    "Agent completed but produced no changes (empty git diff).",
)
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    split: str
    namespace: str


DATASETS = {
    "swebench_verified": DatasetSpec("princeton-nlp/SWE-bench_Verified", "test", "swebench"),
    "swebench_live": DatasetSpec("SWE-bench-Live/SWE-bench-Live", "verified", "starryzhang"),
}


def load_task_input(task: Phase0Task, spec: DatasetSpec, ssg_dir: Path) -> TaskInput:
    from token_sniper.evaluation.swebench import load_swebench_task_metadata
    from token_sniper.graph.io import read_ssg_json

    metadata = load_swebench_task_metadata(task.instance_id, dataset_name=spec.hf_name, split=spec.split)
    ssg = ssg_dir / task.repo.replace("/", "__") / task.base_commit[:12] / "ssg.json"
    if not ssg.exists():
        raise FileNotFoundError(f"no SSG at {ssg}")
    return task_input(task, issue=metadata.problem_statement, gold_patch=metadata.gold_patch,
                      graph=read_ssg_json(ssg))


def select_all(tasks: list[Phase0Task], arms: list[str], selector: Selector, spec: DatasetSpec,
               ssg_dir: Path) -> dict[str, dict[str, Selection | str]]:
    out: dict[str, dict[str, Selection | str]] = {arm: {} for arm in arms}
    for n, task in enumerate(tasks, 1):
        try:
            ti = load_task_input(task, spec, ssg_dir)
        except Exception:
            error = traceback.format_exc()
            logger.error("[%d/%d] %s: task inputs failed:\n%s", n, len(tasks), task.instance_id, error)
            for arm in arms:
                out[arm][task.instance_id] = error
            continue
        for arm in arms:
            try:
                out[arm][task.instance_id] = selector.get(arm, ti)
            except Exception:
                out[arm][task.instance_id] = traceback.format_exc()
                logger.error("[%d/%d] %s %s: selection failed:\n%s", n, len(tasks), task.instance_id, arm,
                             out[arm][task.instance_id])
        logger.info("[%d/%d] %s selected", n, len(tasks), task.instance_id)
    for arm, by_task in out.items():
        failed = sum(1 for s in by_task.values() if isinstance(s, str))
        fallbacks = sum(1 for s in by_task.values() if isinstance(s, Selection) and s.fallback_triggered)
        logger.info("%s: %d selections, %d failed, %d fallback_triggered", arm, len(by_task), failed, fallbacks)
        if by_task and failed == len(by_task):
            raise RuntimeError(f"every selection of arm {arm} failed; see the log above")
    return out


def harness_test_spec(row: dict[str, Any], namespace: str) -> Any:
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(row, namespace=namespace)


def harness_test_command(spec: Any) -> str:
    lines = spec.eval_script.splitlines()
    for i, line in enumerate(lines[:-1]):
        if "Start Test Output" in line:
            return lines[i + 1].strip()
    raise ValueError(f"no test command in the eval script of {spec.instance_id}")


def check_context_sent_whole(settings: Any, selection: Selection) -> None:
    from token_sniper.agent.runner import _build_initial_messages

    if selection.mode != "oracle":
        return
    sent = _build_initial_messages(settings, Path(settings.repo_root))[-1]["content"]
    if sent != selection.context:
        raise RuntimeError(
            f"the agent runner truncates the {selection.context_tokens}-token context of "
            f"{selection.arm}; lower --max-context-tokens"
        )


def run_agent_task(task: Phase0Task, selection: Selection, *, row: dict[str, Any], issue: str,
                   fail_to_pass: list[str], run_dir: Path, namespace: str, model_config: Any, seed: int,
                   max_turns: int) -> dict[str, Any]:
    from token_sniper.agent.docker_tools import (
        DockerStartError,
        DockerToolExecutor,
        start_task_container,
        stop_task_container,
    )
    from token_sniper.agent.runner import AgentRunSettings, run_agent_session
    from token_sniper.evaluation.arms.client import SeededUsageClient

    out = run_dir / "agent_outputs" / task.instance_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    spec = harness_test_spec(row, namespace)
    settings = AgentRunSettings(
        repo_root=TESTBED, issue_description=issue, output_dir=str(out), mode=selection.mode,
        oracle_context=selection.context, max_turns=max_turns, context_window=CONTEXT_WINDOW,
        max_output_tokens=MAX_OUTPUT_TOKENS, fail_to_pass_tests=fail_to_pass, test_cmd=harness_test_command(spec),
    )
    check_context_sent_whole(settings, selection)

    state: dict[str, Any] = {"error_class": None, "error": None, "grade": None}
    try:
        container = start_task_container(task.instance_id, image_tag=spec.instance_image_key)
    except DockerStartError as exc:
        logger.error("%s: container failed to start: %s %s", task.instance_id, exc, exc.stderr.strip())
        return {**state, "error_class": "docker_start", "error": f"{exc}: {exc.stderr.strip()}",
                "patch": False, **_session_usage(out)}
    try:
        client = SeededUsageClient(model_config, seed=seed, usage_log=out / "usage.jsonl")
        run_agent_session(settings, client, tool_executor=DockerToolExecutor(container, testbed=TESTBED))
    except Exception as exc:
        if str(exc) not in AGENT_OUTCOME_ERRORS:
            logger.error("%s: agent failed: %s", task.instance_id, exc)
            state = {**state, "error_class": "agent_exception", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        stop_task_container(container)
    patch = out / "patch.diff"
    return {**state, "patch": patch.is_file() and bool(patch.read_text(encoding="utf-8").strip()),
            **_session_usage(out)}


def _session_usage(out: Path) -> dict[str, Any]:
    from token_sniper.evaluation.arms.client import read_usage

    usage = read_usage(out / "usage.jsonl")
    session_path = out / "session.json"
    if not session_path.exists():
        if usage["calls"]:
            raise RuntimeError(f"{out}: usage was logged but the runner wrote no session.json")
        return {"turns": 0, "tool_calls": 0, "wall_seconds": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "cached_tokens": 0}
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if (session["total_prompt_tokens"], session["total_completion_tokens"]) != (
            usage["prompt_tokens"], usage["completion_tokens"]):
        raise RuntimeError(f"{out}: session.json token totals disagree with usage.jsonl")
    return {
        "turns": len(session["turns"]),
        "tool_calls": sum(len(turn["tool_calls"]) for turn in session["turns"]),
        "wall_seconds": float(session["total_wall_time_seconds"]),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": usage["cached_tokens"],
    }


def preflight_provider(model_config: Any, seed: int, scratch: Path) -> None:
    from token_sniper.evaluation.arms.client import SeededUsageClient

    scratch.mkdir(parents=True, exist_ok=True)
    client = SeededUsageClient(model_config.model_copy(update={"max_tokens": 1}), seed=seed,
                               usage_log=scratch / "preflight_usage.jsonl")
    client.complete([{"role": "user", "content": "ping"}], [])
    logger.info("Provider preflight passed (seeded request, cached tokens reported)")


def grade(attempt: Any) -> tuple[dict[str, Any], str | None]:
    status = attempt.metadata["harness_status"]
    if status == "reported":
        return {"resolved": attempt.resolved, "applied": attempt.patch_successfully_applied,
                "harness_status": status}, None
    log = Path(attempt.instance_report_path).with_name("run_instance.log")
    if log.exists() and APPLY_PATCH_FAIL in log.read_text(encoding="utf-8", errors="replace"):
        return {"resolved": False, "applied": False, "harness_status": status}, None
    return {"resolved": False, "applied": False, "harness_status": status}, f"harness_{status}"


def grade_pending(run_dir: Path, states: dict[str, dict[str, Any]], *, spec: DatasetSpec, base_config: dict,
                  model_name: str, arm: str, seed: int, batch_size: int, max_workers: int) -> None:
    from token_sniper.evaluation.config import HarnessConfig
    from token_sniper.evaluation.swebench import HarnessPatchSubmission, SWEbenchHarnessRunner

    pending = sorted(iid for iid, s in states.items()
                     if s["patch"] and s["grade"] is None and s["error_class"] is None)
    if not pending:
        return
    harness_raw = base_config.get("evaluation", {}).get("harness", {})
    harness = SWEbenchHarnessRunner(HarnessConfig(**{
        **{k: v for k, v in harness_raw.items() if k in HarnessConfig.model_fields},
        "dataset_name": spec.hf_name, "split": spec.split, "namespace": spec.namespace, "max_workers": max_workers,
    }))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        run_id = f"{arm}_s{seed}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        logger.info("Grading %d patches (%d/%d) as %s", len(batch), start + len(batch), len(pending), run_id)
        attempts = harness.evaluate_patches(
            work_dir=run_dir / "harness" / run_id,
            submissions=[HarnessPatchSubmission(instance_id=iid, model_name=model_name,
                                                patch_file=run_dir / "agent_outputs" / iid / "patch.diff")
                         for iid in batch],
            run_id=run_id,
        )
        for iid in batch:
            states[iid]["grade"], states[iid]["error_class"] = grade(attempts[iid])
            write_state(run_dir, iid, states[iid])


def remove_eval_images() -> None:
    listed = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], check=True,
                            capture_output=True, text=True).stdout.split()
    images = [image for image in listed if "sweb.eval." in image]
    if images:
        subprocess.run(["docker", "rmi", "-f", *images], check=True, capture_output=True)
        logger.info("Removed %d SWE-bench images", len(images))


def state_path(run_dir: Path, instance_id: str) -> Path:
    return run_dir / "tasks" / f"{instance_id}.json"


def write_state(run_dir: Path, instance_id: str, state: dict[str, Any]) -> None:
    path = state_path(run_dir, instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_states(run_dir: Path) -> dict[str, dict[str, Any]]:
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in tasks_dir.glob("*.json")}


def is_final(state: dict[str, Any]) -> bool:
    return state["error_class"] is not None or not state["patch"] or state["grade"] is not None


@dataclass(frozen=True)
class Pricing:
    prompt: float
    cached: float
    completion: float

    def cost(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
        return ((prompt_tokens - cached_tokens) * self.prompt + cached_tokens * self.cached
                + completion_tokens * self.completion) / 1e6


def build_record(state: dict[str, Any], selection: Selection | str, *, instance_id: str, gold_files: list[str],
                 dataset: str, arm: str, seed: int, model: str, git_ref: str,
                 arm_checkpoint: tuple[str | None, float | None], pricing: Pricing | None) -> AgentRunRecord:
    if isinstance(selection, Selection):
        sel: dict[str, Any] = {
            "retrieved_files": selection.retrieved_files, "retrieved_spans": selection.retrieved_spans,
            "retained_span_count": selection.retained_span_count,
            "fallback_triggered": selection.fallback_triggered, "gold_files": selection.gold_files,
            "file_recall": selection.file_recall, "function_recall": selection.function_recall,
            "selector_latency_ms": selection.selector_latency_ms,
            "first_turn_context_tokens": selection.context_tokens,
        }
        if (selection.selector_checkpoint_sha256, selection.beta) != arm_checkpoint:
            raise ValueError(f"{instance_id}: selection checkpoint {selection.selector_checkpoint_sha256} "
                             f"differs from the arm's {arm_checkpoint[0]}")
    else:
        sel = {"retrieved_files": [], "retrieved_spans": [], "retained_span_count": 0,
               "fallback_triggered": False, "gold_files": gold_files, "file_recall": 0.0,
               "function_recall": 0.0, "selector_latency_ms": None, "first_turn_context_tokens": 0}
    graded = state["grade"] or {"resolved": False, "applied": False}
    return AgentRunRecord(
        instance_id=instance_id, dataset=dataset, arm=arm, seed=seed, model=model, git_ref=git_ref,
        selector_checkpoint_sha256=arm_checkpoint[0], beta=arm_checkpoint[1], **sel,
        prompt_tokens=state["prompt_tokens"], completion_tokens=state["completion_tokens"],
        cached_tokens=state["cached_tokens"], turns=state["turns"], tool_calls=state["tool_calls"],
        wall_seconds=state["wall_seconds"],
        usd_cost=None if pricing is None else pricing.cost(state["prompt_tokens"], state["cached_tokens"],
                                                            state["completion_tokens"]),
        applied=graded["applied"], resolved=graded["resolved"], error_class=state["error_class"],
    )


def write_telemetry(path: Path, records: list[AgentRunRecord]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    for record in records:
        append_record(tmp, record)
    tmp.touch()
    tmp.replace(path)


def git_ref() -> str:
    repo = Path(__file__).resolve().parent.parent
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream agent run: one arm and seed per invocation")
    parser.add_argument("--arm", nargs="+", required=True, choices=ARMS,
                        help="One arm; several only with --select-only")
    parser.add_argument("--seed", type=int, default=None, help="Agent sampling seed (sent with every request)")
    parser.add_argument("--select-only", action="store_true", help="Build and cache selections, run no agent")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None, help="First N manifest tasks (smoke tests)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue the run in the existing run directory")

    data = parser.add_argument_group("selection inputs")
    data.add_argument("--ssg-dir", type=Path, required=True)
    data.add_argument("--repos-dir", type=Path, required=True, help="Bare repo cache: <repos-dir>/owner__name")
    data.add_argument("--chromadb-dir", type=Path, default=None, help="bm25, dense_knn, ppr_topk")
    data.add_argument("--cache-root", type=Path, default=None, help="Eval stage cache (stage A/B records)")
    data.add_argument("--cache-name", default=None, help="Stage cache namespace, e.g. verified_500")
    data.add_argument("--reranker-model", type=Path, default=None)
    data.add_argument("--vgib-checkpoint", type=Path, default=None)
    data.add_argument("--grpo-checkpoint", type=Path, default=None)
    data.add_argument("--swe-pruner-model", type=Path, default=None, help="SWE-Pruner weights directory")
    data.add_argument("--top-k-files", type=int, default=10, help="Files shown by the file-level arms")
    data.add_argument("--subgraph-top-k", type=int, default=20, help="Reranker files VGIB/PPR select nodes from")
    data.add_argument("--fallback-top-k", type=int, default=5,
                      help="Files shown when VGIB keeps nothing (recorded as fallback_triggered)")
    data.add_argument("--max-context-tokens", type=int, default=32_000,
                      help="Cap on every first-turn context (agent tokenizer); keeps the runner from truncating")
    data.add_argument("--budget-tokens", type=int, default=None,
                      help="Fixed budget for the matched-budget arms (default: match --budget-arm per task)")
    data.add_argument("--budget-arm", default="dense_reranker_vgib", choices=ARMS,
                      help="Matched-budget arms get this arm's first-turn tokens, task by task")
    data.add_argument("--device", default="cpu", help="Jina embedder device")
    data.add_argument("--compressor-device", default="cuda", help="LLMLingua-2 device")

    agent = parser.add_argument_group("agent")
    agent.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    agent.add_argument("--qwen-base-url", default="http://qwen-vllm:8077/v1")
    agent.add_argument("--model", default=None)
    agent.add_argument("--max-turns", type=int, default=100)
    agent.add_argument("--agent-workers", type=int, default=4)
    agent.add_argument("--eval-batch-size", type=int, default=50, help="Patches per harness invocation")
    agent.add_argument("--max-harness-workers", type=int, default=4)
    agent.add_argument("--usd-per-mtok-prompt", type=float, default=None,
                       help="Uncached prompt price; give all three prices or none (usd_cost stays None)")
    agent.add_argument("--usd-per-mtok-cached", type=float, default=None)
    agent.add_argument("--usd-per-mtok-completion", type=float, default=None)
    args = parser.parse_args(argv)

    if args.select_only:
        if args.seed is not None:
            parser.error("--select-only takes no --seed")
    else:
        if len(args.arm) != 1:
            parser.error("an agent run takes exactly one --arm (several only with --select-only)")
        if args.seed is None:
            parser.error("--seed is required for an agent run")
        if args.agent_workers <= 0:
            parser.error("--agent-workers must be positive")
    prices = (args.usd_per_mtok_prompt, args.usd_per_mtok_cached, args.usd_per_mtok_completion)
    if any(p is not None for p in prices) and not all(p is not None for p in prices):
        parser.error("give all three --usd-per-mtok-* prices or none")
    if args.budget_tokens is not None and args.budget_tokens <= 0:
        parser.error("--budget-tokens must be positive")
    if args.budget_arm in BUDGET_ARMS:
        parser.error("--budget-arm must not itself be a matched-budget arm")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = DATASETS[args.dataset]
    tasks = load_phase0_manifest(args.manifest).tasks[:args.max_tasks]
    dataset_dir = args.output_root / args.dataset
    selector = Selector(SelectionConfig(
        ssg_dir=args.ssg_dir, repos_dir=args.repos_dir, cache_root=args.cache_root, cache_name=args.cache_name,
        chromadb_dir=args.chromadb_dir, reranker_model=args.reranker_model, vgib_checkpoint=args.vgib_checkpoint,
        grpo_checkpoint=args.grpo_checkpoint, swe_pruner_model=args.swe_pruner_model,
        top_k_files=args.top_k_files, subgraph_top_k=args.subgraph_top_k, fallback_top_k=args.fallback_top_k,
        max_context_tokens=args.max_context_tokens, budget_tokens=args.budget_tokens, budget_arm=args.budget_arm,
        device=args.device, compressor_device=args.compressor_device,
    ), cache_dir=dataset_dir / "selections")

    if args.select_only:
        select_all(tasks, args.arm, selector, spec, args.ssg_dir)
        return
    (arm,) = args.arm
    run_agents(args, arm, spec, tasks, selector, dataset_dir / arm / f"seed{args.seed}")


def run_agents(args: argparse.Namespace, arm: str, spec: DatasetSpec, tasks: list[Phase0Task],
               selector: Selector, run_dir: Path) -> None:
    from token_sniper.agent.config import AgentModelConfig, load_base_config, load_yaml_config
    from token_sniper.evaluation.arms.context import patch_files
    from token_sniper.evaluation.swebench import _load_dataset_rows, load_swebench_task_metadata

    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise SystemExit(f"{run_dir} already holds a run: pass --resume to continue it")
    run_dir.mkdir(parents=True, exist_ok=True)
    ref = git_ref()
    base_config = load_yaml_config(args.base_config)
    base_agent = load_base_config(args.base_config).models.agent
    model_config = AgentModelConfig.model_validate({
        **base_agent.model_dump(), "name": args.model or base_agent.name, "base_url": args.qwen_base_url,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "context_window": CONTEXT_WINDOW,
    })
    pricing = None if args.usd_per_mtok_prompt is None else Pricing(
        args.usd_per_mtok_prompt, args.usd_per_mtok_cached, args.usd_per_mtok_completion)
    arm_checkpoint: tuple[str | None, float | None] = (None, None)
    if arm in CHECKPOINT_ARMS:
        path = getattr(args, CHECKPOINT_ARMS[arm])
        if path is None:
            raise SystemExit(f"arm {arm} needs --{CHECKPOINT_ARMS[arm].replace('_', '-')}")
        arm_checkpoint = checkpoint_identity(path, require_beta=arm == "dense_reranker_vgib")
    logger.info("Arm %s | seed %d | %s | %d tasks | run dir %s | git %s",
                arm, args.seed, args.dataset, len(tasks), run_dir, ref)

    selections = select_all(tasks, [arm], selector, spec, args.ssg_dir)[arm]
    states = load_states(run_dir)
    for task in tasks:
        iid = task.instance_id
        state, failed = states.get(iid), isinstance(selections[iid], str)
        if state is not None and state["error_class"] == "selection_error" and not failed:
            del states[iid]
        elif state is not None and state["error_class"] != "selection_error" and failed:
            raise RuntimeError(f"{iid} already ran, but its selection now fails:\n{selections[iid]}")
        elif state is None and failed:
            states[iid] = {
                "error_class": "selection_error", "error": selections[iid], "patch": False, "grade": None,
                "turns": 0, "tool_calls": 0, "wall_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "cached_tokens": 0,
            }
            write_state(run_dir, iid, states[iid])
    pending = [t for t in tasks if t.instance_id not in states]
    logger.info("%d tasks to run, %d already done", len(pending), len(tasks) - len(pending))

    rows = _load_dataset_rows(spec.hf_name, spec.split)
    if pending:
        preflight_provider(model_config, args.seed, run_dir / "preflight")

    def execute(task: Phase0Task) -> None:
        metadata = load_swebench_task_metadata(task.instance_id, dataset_name=spec.hf_name, split=spec.split)
        t0 = time.monotonic()
        state = run_agent_task(
            task, selections[task.instance_id], row=rows[task.instance_id], issue=metadata.problem_statement,
            fail_to_pass=metadata.fail_to_pass_tests, run_dir=run_dir, namespace=spec.namespace,
            model_config=model_config, seed=args.seed, max_turns=args.max_turns,
        )
        write_state(run_dir, task.instance_id, state)
        states[task.instance_id] = state
        logger.info("%s done in %.0fs (patch=%s, error_class=%s)", task.instance_id, time.monotonic() - t0,
                    state["patch"], state["error_class"])

    for start in range(0, max(len(pending), 1), args.eval_batch_size):
        chunk = pending[start:start + args.eval_batch_size]
        with ThreadPoolExecutor(max_workers=min(args.agent_workers, max(len(chunk), 1))) as pool:
            for future in [pool.submit(execute, task) for task in chunk]:
                future.result()
        grade_pending(run_dir, states, spec=spec, base_config=base_config, model_name=model_config.name,
                      arm=arm, seed=args.seed, batch_size=args.eval_batch_size,
                      max_workers=args.max_harness_workers)
        remove_eval_images()

    records = []
    for task in tasks:
        state = states[task.instance_id]
        if not is_final(state):
            continue
        gold = patch_files(load_swebench_task_metadata(task.instance_id, dataset_name=spec.hf_name,
                                                       split=spec.split).gold_patch)
        records.append(build_record(
            state, selections[task.instance_id], instance_id=task.instance_id, gold_files=gold,
            dataset=args.dataset, arm=arm, seed=args.seed, model=model_config.name, git_ref=ref,
            arm_checkpoint=arm_checkpoint, pricing=pricing,
        ))
    write_telemetry(run_dir / "telemetry.jsonl", records)
    resolved = sum(r.resolved for r in records)
    errors = sum(r.error_class is not None for r in records)
    fallbacks = sum(r.fallback_triggered for r in records)
    logger.info("%s seed %d: %d records, %d resolved, %d errors, %d fallback_triggered -> %s",
                arm, args.seed, len(records), resolved, errors, fallbacks, run_dir / "telemetry.jsonl")
    if len(records) < len(tasks):
        logger.error("%d tasks are not finished: rerun with --resume", len(tasks) - len(records))
        sys.exit(1)


if __name__ == "__main__":
    main()
