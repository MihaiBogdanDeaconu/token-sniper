from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

TESTBED = "/testbed"


def load_swebench_tasks(
    dataset_name: str,
    split: str,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    tasks = [dict(row) for row in ds]
    if max_tasks:
        tasks = tasks[:max_tasks]
    log.info("Loaded %d SWE-bench tasks from %s:%s", len(tasks), dataset_name, split)
    return tasks


def _extract_test_cmd(task: dict[str, Any]) -> str | None:
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec

        spec = make_test_spec(task)
        script = spec.eval_script or ""
        lines = script.splitlines()
        for i, line in enumerate(lines):
            if "Start Test Output" in line:
                if i + 1 < len(lines):
                    cmd = lines[i + 1].strip()
                    if cmd and not cmd.startswith(":") and not cmd.startswith("#"):
                        return cmd
    except Exception:
        pass
    return None


def run_agent_on_task(
    task: dict[str, Any],
    client: Any,
    halt_scorer: Any,
    output_base: Path,
    *,
    max_turns: int = 100,
    context_window: int = 65_536,
    max_output_tokens: int = 16_384,
    use_docker: bool = False,
) -> dict[str, Any]:
    import subprocess

    from token_sniper.agent.runner import AgentRunSettings, run_agent_session

    instance_id = task["instance_id"]
    repo = task["repo"]
    base_commit = task["base_commit"]
    task_dir = output_base / instance_id.replace("/", "__")
    task_dir.mkdir(parents=True, exist_ok=True)

    container_name: str | None = None
    executor = None

    if use_docker:
        from token_sniper.agent.docker_tools import (
            DockerToolExecutor,
            start_task_container,
            stop_task_container,
        )
        try:
            container_name = start_task_container(instance_id)
            executor = DockerToolExecutor(container_name, testbed=TESTBED)
            repo_root = TESTBED
        except Exception as e:
            log.error("[%s] Failed to start container: %s", instance_id, e)
            return {
                "instance_id": instance_id,
                "repo": repo,
                "status": "failed",
                "n_turns": 0,
                "wall_time_seconds": 0.0,
                "halt_submit_score": None,
                "halt_mean_score": None,
                "halt_max_score": None,
                "has_patch": False,
                "error": f"Container start failed: {e}",
            }
    else:
        repo_dir = task_dir / "repo"
        if not repo_dir.exists():
            log.info("[%s] Cloning %s @ %s", instance_id, repo, base_commit[:8])
            subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(repo_dir)],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", "-q", base_commit],
                cwd=repo_dir, check=True, capture_output=True,
            )
        repo_root = str(repo_dir)

    fail_to_pass = task.get("FAIL_TO_PASS", "")
    if isinstance(fail_to_pass, str):
        import json as _json
        try:
            fail_to_pass_tests = _json.loads(fail_to_pass)
        except (ValueError, TypeError):
            fail_to_pass_tests = [t.strip() for t in fail_to_pass.split("\n") if t.strip()]
    else:
        fail_to_pass_tests = list(fail_to_pass) if fail_to_pass else []

    test_cmd = _extract_test_cmd(task)

    settings = AgentRunSettings(
        repo_root=repo_root,
        issue_description=task.get("problem_statement", ""),
        output_dir=str(task_dir / "agent_output"),
        mode="blind",
        max_turns=max_turns,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        fail_to_pass_tests=fail_to_pass_tests,
        test_cmd=test_cmd,
    )

    t0 = time.perf_counter()
    try:
        result = run_agent_session(
            settings, client,
            halt_scorer=halt_scorer,
            tool_executor=executor,
        )
        status = result.status
        patch_path = result.patch_path
        halt_submit = result.halt_submit_score
        halt_mean = result.halt_mean_score
        halt_max = result.halt_max_score
        n_turns = len(result.turns)
        error_msg = result.error_message
    except Exception as e:
        log.warning("[%s] Agent failed: %s", instance_id, e)
        status = "failed"
        patch_path = None
        halt_submit = halt_mean = halt_max = None
        n_turns = 0
        error_msg = str(e)
        session_path = Path(task_dir / "agent_output" / "session.json")
        if session_path.exists():
            try:
                session_data = json.loads(session_path.read_text(encoding="utf-8"))
                n_turns = len(session_data.get("turns", []))
                if session_data.get("patch_path"):
                    patch_path = session_data["patch_path"]
                    halt_submit = session_data.get("halt_submit_score")
                    halt_mean = session_data.get("halt_mean_score")
                    halt_max = session_data.get("halt_max_score")
            except Exception:
                pass
    finally:
        if container_name:
            from token_sniper.agent.docker_tools import stop_task_container
            stop_task_container(container_name)

    elapsed = time.perf_counter() - t0

    record = {
        "instance_id": instance_id,
        "repo": repo,
        "status": status,
        "n_turns": n_turns,
        "wall_time_seconds": round(elapsed, 2),
        "halt_submit_score": halt_submit,
        "halt_mean_score": halt_mean,
        "halt_max_score": halt_max,
        "has_patch": patch_path is not None,
        "error": error_msg,
    }

    if patch_path and Path(patch_path).exists():
        record["patch_text"] = Path(patch_path).read_text(encoding="utf-8")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="HALT SWE-bench evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halt_swebench"))
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--exclude-ids-file", type=Path, default=None,
                        help="JSON file with list of instance_ids to skip (leakage prevention)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks that already have results")
    parser.add_argument("--docker", action="store_true",
                        help="Run agent tools inside SWE-bench Docker containers")
    parser.add_argument("--build-images", action="store_true",
                        help="Build SWE-bench Docker images before running")
    parser.add_argument("--image-workers", type=int, default=4,
                        help="Parallel workers for image building")
    parser.add_argument("--task-workers", type=int, default=1,
                        help="Number of agent tasks to run concurrently (each in its own Docker container)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    log.info("Loading HALT scorer from %s", args.checkpoint)
    from token_sniper.guardrails.halt_scorer import HALTScorer
    halt_scorer = HALTScorer.from_checkpoint(args.checkpoint, device="cpu")

    log.info("Setting up agent LLM client with logprob collection")
    from token_sniper.agent.config import AgentModelConfig
    from token_sniper.agent.providers import OpenAIChatModelClient

    model_config = AgentModelConfig(
        name="Qwen/Qwen3.5-35B-A3B-FP8",
        provider="local",
        base_url="http://qwen-vllm:8077/v1",
        api_key_env="LOCAL_OPENAI_API_KEY",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_tokens=16_384,
        context_window=65_536,
        max_turns=args.max_turns,
    )
    client = OpenAIChatModelClient(model_config, collect_logprobs=True)

    excluded_ids: set[str] = set()
    if args.exclude_ids_file is not None:
        excluded_ids = set(json.loads(args.exclude_ids_file.read_text(encoding="utf-8")))
        log.info("Loaded %d instance_ids to exclude from %s", len(excluded_ids), args.exclude_ids_file)

    tasks = load_swebench_tasks(args.dataset, args.split, max_tasks=None)
    if excluded_ids:
        before = len(tasks)
        tasks = [t for t in tasks if t["instance_id"] not in excluded_ids]
        log.info("Excluded %d tasks (overlap with held-out set)", before - len(tasks))
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    failed_image_ids: set[str] = set()
    if args.docker and args.build_images:
        log.info("Building SWE-bench Docker images for %d tasks...", len(tasks))
        from token_sniper.agent.docker_tools import build_swebench_images
        ok, failed = build_swebench_images(tasks, max_workers=args.image_workers)
        log.info("Images built: %d succeeded, %d failed", len(ok), len(failed))
        if failed:
            log.warning("Failed images (will skip): %s", failed)
            failed_image_ids = set(failed)

    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info("Resuming: %d tasks already completed", len(done_ids))

    pending = [t for t in tasks if t["instance_id"] not in done_ids]
    if args.docker and failed_image_ids:
        skipped = [t for t in pending if t["instance_id"] in failed_image_ids]
        pending = [t for t in pending if t["instance_id"] not in failed_image_ids]
        log.info("Skipping %d tasks with failed Docker images", len(skipped))
    mode_str = "Docker-isolated" if args.docker else "bare-metal"
    log.info("Running agent on %d tasks (%s, max_turns=%d)", len(pending), mode_str, args.max_turns)

    results: list[dict[str, Any]] = []
    results_lock = __import__("threading").Lock()

    def _process(idx_task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, task = idx_task
        log.info("[%d/%d] START %s", i + 1, len(pending), task["instance_id"])
        record = run_agent_on_task(
            task, client, halt_scorer, args.output_dir,
            max_turns=args.max_turns,
            use_docker=args.docker,
        )
        with results_lock:
            results.append(record)
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if record["halt_submit_score"] is not None:
            log.info(
                "[%d/%d] DONE %s | HALT=%.3f | status=%s | turns=%d",
                i + 1, len(pending), task["instance_id"],
                record["halt_submit_score"], record["status"], record["n_turns"],
            )
        else:
            log.info(
                "[%d/%d] DONE %s | no patch | status=%s | turns=%d",
                i + 1, len(pending), task["instance_id"],
                record["status"], record["n_turns"],
            )
        return record

    if args.task_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        log.info("Parallel execution: %d concurrent agent tasks", args.task_workers)
        with ThreadPoolExecutor(max_workers=args.task_workers) as pool:
            list(pool.map(_process, enumerate(pending)))
    else:
        for item in enumerate(pending):
            _process(item)

    _print_summary(results, args.output_dir)


def _print_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    scores = [r["halt_submit_score"] for r in results if r["halt_submit_score"] is not None]
    n_total = len(results)
    n_patched = sum(1 for r in results if r["has_patch"])
    n_scored = len(scores)

    summary = {
        "total_tasks": n_total,
        "tasks_with_patch": n_patched,
        "tasks_with_halt_score": n_scored,
        "halt_submit_mean": sum(scores) / len(scores) if scores else None,
        "halt_submit_median": sorted(scores)[len(scores) // 2] if scores else None,
        "halt_submit_min": min(scores) if scores else None,
        "halt_submit_max": max(scores) if scores else None,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("=" * 50)
    log.info("HALT SWE-bench Summary")
    log.info("=" * 50)
    log.info("  Tasks:       %d total, %d produced patches", n_total, n_patched)
    log.info("  HALT scored: %d", n_scored)
    if scores:
        log.info("  Submit HALT: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
                 summary["halt_submit_mean"], summary["halt_submit_median"],
                 summary["halt_submit_min"], summary["halt_submit_max"])
    log.info("Results -> %s", output_dir / "results.jsonl")
    log.info("Summary -> %s", summary_path)
    log.info("")
    log.info("Next: run SWE-bench harness on the generated patches,")
    log.info("then compute AUROC(halt_submit_score, resolved)")


if __name__ == "__main__":
    main()
