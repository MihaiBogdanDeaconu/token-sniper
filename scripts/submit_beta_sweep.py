from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.submit_eval_index_k8s import _git_output, _require_pushed

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_VGIB = REPO_ROOT / "scripts" / "submit_vgib_k8s.py"
PLAN_BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OWNED_FLAGS = ("--beta", "--job-name", "--wait", "--dry-run", "--output")
EVAL_JOB_PREFIX = "eval-"
_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def beta_tag(beta: float) -> str:
    return format(beta, "g").replace(".", "p")


def job_name(prefix: str, stamp: str, beta: float) -> str:
    name = f"{prefix}-{stamp}-b{beta_tag(beta)}"
    eval_name = EVAL_JOB_PREFIX + name
    if not _DNS_LABEL.match(name) or len(eval_name) > 63:
        raise ValueError(
            f"job name {eval_name!r} is not a valid Kubernetes name (lowercase alphanumerics and '-', "
            "at most 63 characters); shorten --name-prefix"
        )
    return name


def check_submit_args(submit_args: list[str]) -> argparse.Namespace:
    for token in submit_args:
        for flag in OWNED_FLAGS:
            if token == flag or token.startswith(flag + "="):
                raise ValueError(f"{flag} is set by submit_beta_sweep.py per run; remove it from the args after --")
    parser = argparse.ArgumentParser(prog="submit_vgib_k8s.py args", allow_abbrev=False)
    parser.add_argument("--kubeconfig", default="kubelet.yaml")
    parser.add_argument("--workspace-pvc", required=True)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--run-rl", action="store_true")
    known, _ = parser.parse_known_args(submit_args)
    return known


def run_entry(
    job: str, beta: float, *, kubeconfig: str, workspace_pvc: str, dataset: str, run_rl: bool,
) -> dict[str, Any]:
    eval_job = EVAL_JOB_PREFIX + job
    checkpoints = ["vgib", "grpo"] if run_rl else ["vgib"]
    models_dir = f"/models/runs/{job}/repo/experiments"
    return {
        "beta": beta,
        "job_name": job,
        "checkpoint_dir": f"/workspace/runs/{job}/repo/experiments/vgib",
        "models_dir": models_dir,
        "eval_job_name": eval_job,
        "eval_command": shlex.join([
            "./venv/bin/python", "scripts/submit_eval_retrieval_k8s.py",
            "--kubeconfig", kubeconfig, "--models-pvc", workspace_pvc,
            "--job-name", eval_job, "--datasets", dataset, "--models-dir", models_dir,
            "--checkpoints", *checkpoints,
        ]),
    }


def submit_sweep(
    betas: list[float],
    submit_args: list[str],
    *,
    name_prefix: str,
    stamp: str,
    dataset: str,
    manifest_out: Path | None,
    dry_run: bool,
    git_ref: str,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if len(set(betas)) != len(betas):
        raise ValueError(f"duplicate betas: {betas}")
    if any(b < 0 for b in betas):
        raise ValueError(f"betas must be >= 0: {betas}")
    known = check_submit_args(submit_args)
    if known.git_ref is None:
        submit_args = [*submit_args, "--git-ref", git_ref]
    else:
        git_ref = known.git_ref
    jobs = [job_name(name_prefix, stamp, beta) for beta in betas]

    manifest: dict[str, Any] = {
        "sweep_id": f"{name_prefix}-{stamp}",
        "created_utc": stamp,
        "git_ref": git_ref,
        "workspace_pvc": known.workspace_pvc,
        "eval_dataset": dataset,
        "submit_args": submit_args,
        "runs": [],
    }
    for beta, job in zip(betas, jobs, strict=True):
        cmd = [sys.executable, str(SUBMIT_VGIB), *submit_args, "--job-name", job, "--beta", repr(beta)]
        if dry_run:
            cmd.append("--dry-run")
        print(f"=== beta={beta!r}: {shlex.join(cmd)}", flush=True)
        run(cmd, check=True, cwd=REPO_ROOT)
        manifest["runs"].append(run_entry(
            job, beta, kubeconfig=known.kubeconfig, workspace_pvc=known.workspace_pvc, dataset=dataset,
            run_rl=known.run_rl,
        ))
        if manifest_out is not None:
            manifest_out.parent.mkdir(parents=True, exist_ok=True)
            manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("give the scripts/submit_vgib_k8s.py args after --  (at least --workspace-pvc)")
    split = argv.index("--")
    own, submit_args = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(description="Submit a VGIB beta sweep (one training job per beta)")
    parser.add_argument("--betas", type=float, nargs="+", default=PLAN_BETAS)
    parser.add_argument("--name-prefix", default="vgib-beta",
                        help="Job names are <prefix>-<UTC stamp>-b<beta>, e.g. vgib-beta-20260918120000-b0p001")
    parser.add_argument("--dataset", default="lite_274", help="Dataset in the printed stage B+C eval commands")
    parser.add_argument("--manifest-out", type=Path, default=None,
                        help="Default: configs/beta_sweeps/<sweep-id>.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render every job with submit_vgib_k8s.py --dry-run; write no manifest")
    args = parser.parse_args(own)

    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    manifest_out = None if args.dry_run else (
        args.manifest_out or REPO_ROOT / "configs" / "beta_sweeps" / f"{args.name_prefix}-{stamp}.json"
    )
    git_ref = _git_output(["git", "rev-parse", "HEAD"])
    if not args.dry_run:
        _require_pushed(check_submit_args(submit_args).git_ref or git_ref)

    manifest = submit_sweep(
        args.betas, submit_args, name_prefix=args.name_prefix, stamp=stamp, dataset=args.dataset,
        manifest_out=manifest_out, dry_run=args.dry_run, git_ref=git_ref,
    )
    if manifest_out is None:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"\nSweep manifest: {manifest_out}")
    print("\nAfter each training job completes, commit + push the manifest, then run its stage B+C eval:")
    for entry in manifest["runs"]:
        print(f"  {entry['eval_command']}")


if __name__ == "__main__":
    main()
