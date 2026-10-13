from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_repo_checkout(repo: str, checkouts_root: Path, bare_repos_dir: Path) -> Path:
    import shutil
    import subprocess

    slug = repo.replace("/", "__")
    target = checkouts_root / slug
    bare_path = bare_repos_dir / slug

    if target.exists() and (target / ".git").exists():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target, capture_output=True, check=False,
        )
        if result.returncode == 0:
            return target
        logger.warning("Broken checkout at %s — removing", target)
        shutil.rmtree(target)
    elif target.exists():
        shutil.rmtree(target)

    if not bare_path.exists():
        raise FileNotFoundError(
            f"Bare repo not found at {bare_path}. "
            f"Run Phase 1 setup first, or pass --bare-repos-dir."
        )

    checkouts_root.mkdir(parents=True, exist_ok=True)
    logger.info("Local-cloning %s from bare repo (instant, no network) ...", repo)
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", "--no-checkout", "--quiet", str(bare_path), str(target)],
        check=True,
    )
    return target


def _run_one_task(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from token_sniper.evaluation.swebench import load_swebench_task_metadata
    from token_sniper.graph.io import read_ssg_json
    from token_sniper.traces.missing_edges import classify_missing_edges
    from token_sniper.traces.ssg_mapping import build_ssg_node_index
    from token_sniper.traces.trace_runner import TraceRunConfig, run_trace_for_task
    from token_sniper.traces.tracer import TraceResult

    instance_id = task["instance_id"]
    repo = task["repo"]
    commit = task["base_commit"]
    repo_slug = repo.replace("/", "__")
    commit_short = commit[:12]

    try:
        out_dir = args.output_dir
        trace_path = out_dir / f"{instance_id}.json"
        report_path = out_dir / f"{instance_id}.report.json"

        ssg_path = args.ssg_dir / repo_slug / commit_short / "ssg.json"
        if not ssg_path.exists():
            return {"instance_id": instance_id, "status": "skipped_no_ssg"}

        meta = load_swebench_task_metadata(instance_id)
        checkout = _ensure_repo_checkout(repo, args.checkouts_root, args.bare_repos_dir)

        cfg = TraceRunConfig(
            instance_id=instance_id,
            repo=repo,
            base_commit=commit,
            fail_to_pass_tests=meta.fail_to_pass_tests,
            gold_patch=meta.gold_patch,
            test_patch=meta.test_patch,
            ssg_path=ssg_path,
            output_path=trace_path,
            repo_checkout=checkout,
            python_executable=args.python_executable or sys.executable,
            timeout_seconds=args.timeout,
        )
        payload = run_trace_for_task(cfg)

        graph = read_ssg_json(ssg_path)
        index = build_ssg_node_index(graph)
        trace_obj = TraceResult(
            executed_nodes=set(payload.get("executed_nodes", [])),
            edges={(a, b, int(ln)) for a, b, ln in payload.get("edges", [])},
            call_events=int(payload.get("call_events", 0)),
            dropped_out_of_repo=int(payload.get("dropped_out_of_repo", 0)),
        )
        missing, report = classify_missing_edges(trace_obj, index)

        report_payload = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": commit,
            "report": report.model_dump(),
            "missing_edges": [
                {"src": m.src, "dst": m.dst, "category": m.category, "lineno": m.call_site_lineno}
                for m in missing
            ],
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        logger.info(
            "[%s] trace_edges=%d resolved=%d missing=%d cats=%s",
            instance_id,
            report.total_trace_edges,
            report.resolved_to_ssg,
            report.missing_edges,
            report.category_counts,
        )
        return {
            "instance_id": instance_id,
            "status": "ok",
            "missing_edges": report.missing_edges,
            "categories": report.category_counts,
        }
    except Exception:
        logger.error("[%s] failed\n%s", instance_id, traceback.format_exc())
        return {"instance_id": instance_id, "status": "failed"}



def _run_repo_group(repo_tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for t in repo_tasks:
        results.append(_run_one_task(t, args))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — execution trace collection")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument(
        "--bare-repos-dir",
        type=Path,
        default=Path("data/swebench/repos"),
        help="Directory containing bare git repos from Phase 1 setup.",
    )
    parser.add_argument(
        "--checkouts-root",
        type=Path,
        default=Path.home() / ".cache" / "token-sniper" / "workdirs",
        help="Working-tree clones are created here (outside workspace to avoid "
             "parent-git interference). Default: ~/.cache/token-sniper/workdirs/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/traces"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--python-executable", type=str, default=None,
                        help="Optional override of the interpreter used to run pytest under tracer")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Limit to these instance ids (for debugging)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tasks whose trace JSON already exists in output-dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.only:
        only = set(args.only)
        tasks = [t for t in tasks if t["instance_id"] in only]
    if args.skip_existing:
        before = len(tasks)
        tasks = [
            t for t in tasks
            if not (args.output_dir / f"{t['instance_id']}.json").exists()
        ]
        skipped = before - len(tasks)
        if skipped:
            logger.info("Skipped %d tasks with existing traces (%d remaining)", skipped, len(tasks))
    logger.info("Running traces for %d tasks (parallel=%d)", len(tasks), args.parallel)

    repos_needed = sorted(set(t["repo"] for t in tasks))
    logger.info("Pre-cloning %d repos from bare repos ...", len(repos_needed))
    for repo in repos_needed:
        try:
            _ensure_repo_checkout(repo, args.checkouts_root, args.bare_repos_dir)
        except Exception:
            logger.exception("Failed to clone %s", repo)
    logger.info("Pre-clone complete.")

    from collections import defaultdict
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        groups[t["repo"]].append(t)

    results: list[dict[str, Any]] = []
    if args.parallel <= 1 or len(groups) == 1:
        for repo, repo_tasks in groups.items():
            for t in repo_tasks:
                results.append(_run_one_task(t, args))
    else:
        with cf.ProcessPoolExecutor(max_workers=min(args.parallel, len(groups))) as pool:
            futures = {
                pool.submit(_run_repo_group, repo_tasks, args): repo
                for repo, repo_tasks in groups.items()
            }
            for fut in cf.as_completed(futures):
                repo = futures[fut]
                try:
                    results.extend(fut.result())
                except Exception:
                    logger.exception("Repo group %s failed", repo)

    status_counter = Counter(r.get("status", "unknown") for r in results)
    total_missing = sum(r.get("missing_edges", 0) for r in results if r.get("status") == "ok")
    all_cats: Counter[str] = Counter()
    for r in results:
        all_cats.update(r.get("categories") or {})

    summary = {
        "task_count": len(tasks),
        "status_counts": dict(status_counter),
        "total_missing_edges": total_missing,
        "aggregate_categories": dict(all_cats),
        "per_task": results,
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Done. %s missing edges across %d ok tasks. Breakdown: %s",
        total_missing,
        status_counter.get("ok", 0),
        dict(all_cats),
    )


if __name__ == "__main__":
    main()
