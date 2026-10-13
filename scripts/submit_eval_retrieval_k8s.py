from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import yaml

from scripts.submit_eval_index_k8s import (
    _git_output,
    _kubectl_apply,
    _namespace_from_kubeconfig,
    _require_pushed,
)

if TYPE_CHECKING:
    from collections.abc import Callable


CHECKPOINT_FILES = {
    "vgib": ("vgib/best.pt", "vgib_best.pt"),
    "grpo": ("rl_grpo/best_rl.pt", "grpo_best_rl.pt"),
}
"""Checkpoint name -> (path under --models-dir, file name in the run's models/ dir)."""


def add_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--eval-pvc", default="token-sniper-eval-cache")
    parser.add_argument("--reranker-pvc", default="reranker-20260518221350-workspace")
    parser.add_argument("--models-pvc", default="vgib-20260519-workspace",
                        help="PVC holding experiments/vgib/best.pt and experiments/rl_grpo/best_rl.pt")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU worker node without a GPU")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--node-pool", default="a100")
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="64Gi")
    parser.add_argument("--memory-limit", default="128Gi")
    parser.add_argument("--dry-run", action="store_true")


def submit_eval_job(args: argparse.Namespace, *, component: str, body: Callable[[str], str]) -> None:
    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"{component}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])
    if not args.dry_run:
        _require_pushed(git_ref)

    torch_index = "cpu" if args.cpu else "cu124"
    runner_script = dedent(f"""\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*
        git config --global --add safe.directory '*'

        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH=/workspace/repo

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/{torch_index}
        python -m pip install -e ".[retrieval,data,experiments]"
        python -m pip install xgboost rank-bm25 torch_geometric "sentence-transformers>=5.0.0" "transformers>=4.53.0"

    """) + body(job_name)

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": component,
        "runai/node-pool": "cpu-worker" if args.cpu else args.node_pool,
    }
    node_selector = {"node-role.kubernetes.io/runai-cpu-worker": "true"} if args.cpu else {}
    gpu = {} if args.cpu else {"nvidia.com/gpu": "1"}
    job: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "schedulerName": "runai-scheduler",
                    "nodeSelector": node_selector,
                    "volumes": [
                        {"name": "eval", "persistentVolumeClaim": {"claimName": args.eval_pvc}},
                        {"name": "reranker", "persistentVolumeClaim": {"claimName": args.reranker_pvc, "readOnly": True}},
                        {"name": "models", "persistentVolumeClaim": {"claimName": args.models_pvc, "readOnly": True}},
                    ],
                    "containers": [{
                        "name": "eval",
                        "image": args.runner_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-lc", runner_script],
                        "env": [
                            {"name": "PYTHONUNBUFFERED", "value": "1"},
                            {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
                            {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
                            {"name": "HF_HOME", "value": "/eval/hf"},
                            {
                                "name": "GITHUB_TOKEN",
                                "valueFrom": {"secretKeyRef": {"name": args.github_secret_name, "key": args.github_secret_key}},
                            },
                        ],
                        "resources": {
                            "requests": {"cpu": args.cpu_request, "memory": args.memory_request, **gpu},
                            "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit, **gpu},
                        },
                        "volumeMounts": [
                            {"name": "eval", "mountPath": "/eval"},
                            {"name": "reranker", "mountPath": "/reranker", "readOnly": True},
                            {"name": "models", "mountPath": "/models", "readOnly": True},
                        ],
                    }],
                },
            },
        },
    }

    manifest_yaml = yaml.safe_dump(job, sort_keys=False)
    if args.dry_run:
        sys.stdout.write(manifest_yaml)
        return

    _kubectl_apply(args.kubeconfig, manifest_yaml)
    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Follow logs: kubectl --kubeconfig {args.kubeconfig} logs -n {namespace} job/{job_name} -c eval -f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cached eval stages and LocAgent scoring")
    add_job_args(parser)
    parser.add_argument("--datasets", nargs="+", default=["lite_274", "locbench_560"])
    parser.add_argument("--top-k", type=int, default=20, help="Reranker top-k files VGIB runs on")
    parser.add_argument("--import-run", default=None, metavar="RUN",
                        help="Seed Lite stage A and the VGIB caches from /eval/results/RUN/lite_274")
    parser.add_argument("--models-dir", default="/models/repo/experiments",
                        help="Directory on the models PVC holding vgib/best.pt and rl_grpo/best_rl.pt. "
                             "Each training run clones into /workspace/runs/<job>/repo, so its "
                             "checkpoints are at /models/runs/<job>/repo/experiments.")
    parser.add_argument("--checkpoints", nargs="+", choices=sorted(CHECKPOINT_FILES), default=["vgib", "grpo"],
                        help="Which checkpoints under --models-dir to copy and score. A training run "
                             "without --run-rl has no GRPO checkpoint: pass --checkpoints vgib.")
    parser.add_argument("--expand-depth", type=int, default=0, metavar="HOPS",
                        help="Graph-expand the candidate pool by this many hops (0 = off, as in "
                             "every number reported so far). Expanded runs get their own cache "
                             "namespace, <dataset>_expand<hops>, so the two arms never mix.")
    args = parser.parse_args()
    if args.expand_depth and args.import_run:
        parser.error("--import-run holds unexpanded records; do not mix it with --expand-depth")
    if len(set(args.checkpoints)) != len(args.checkpoints):
        parser.error(f"--checkpoints lists a checkpoint twice: {args.checkpoints}")

    device = "cpu" if args.cpu else "cuda"
    expand_flags = ""
    common = "--manifest /eval/manifests/${KEY}_tasks.json --cache-root $CACHE"
    if args.expand_depth:
        expand_flags = f" --expand-depth {args.expand_depth}"
        common += " --cache-name ${KEY}_expand" + str(args.expand_depth)
    checkpoints = " ".join(f"--checkpoint {name}=$OUT/models/{CHECKPOINT_FILES[name][1]}" for name in args.checkpoints)
    reranker = "--reranker-model $OUT/models/xgb_reranker.json"

    def body(job_name: str) -> str:
        import_block = ""
        import_flag = ""
        if args.import_run:
            import_block = dedent(f"""\
                IMPORT_DIR=/eval/results/{args.import_run}/lite_274
                cp -rn $IMPORT_DIR/feature_cache/. $CACHE/vgib_features/
                cp -rn $IMPORT_DIR/issue_cache/. $CACHE/issue_emb/
            """)
            import_flag = f"--import-reranker-inputs /eval/results/{args.import_run}/lite_274/reranker_inputs.json"
        copies = "".join(
            f"cp {args.models_dir}/{CHECKPOINT_FILES[name][0]} $OUT/models/{CHECKPOINT_FILES[name][1]}\n"
            for name in args.checkpoints
        )
        hashed = " ".join(CHECKPOINT_FILES[name][1] for name in args.checkpoints)
        return dedent(f"""\
            CACHE=/eval/cache
            OUT=/eval/results/{job_name}
            mkdir -p $OUT/models $CACHE/vgib_features $CACHE/issue_emb
            cp /reranker/repo/experiments/reranker/xgb_reranker.json $OUT/models/xgb_reranker.json
        """) + copies + dedent(f"""\
            (cd $OUT/models && sha256sum xgb_reranker.json {hashed} | tee models.sha256)
            echo "$TOKEN_SNIPER_GIT_REF" > $OUT/git_ref
        """) + import_block + dedent(f"""\

            for KEY in {' '.join(args.datasets)}; do
                IMPORT=""
                if [ "$KEY" = lite_274 ]; then
                    IMPORT="{import_flag}"
                fi
                python scripts/eval_candidates.py --dataset-key $KEY {common}{expand_flags} \\
                    --ssg-dir /eval/ssg --chromadb-dir /eval/chromadb --device {device} $IMPORT
                python scripts/eval_retention.py --dataset-key $KEY {common} \\
                    --ssg-dir /eval/ssg --chromadb-dir /eval/chromadb --repos-dir /eval/repos \\
                    {reranker} {checkpoints} --top-k {args.top_k} --device {device}
                python scripts/eval_score.py --dataset-key $KEY {common} \\
                    {reranker} {checkpoints} --top-k {args.top_k} --output-dir $OUT/$KEY
            done
            echo "Results in $OUT"
        """)

    submit_eval_job(args, component="eval-stages", body=body)


if __name__ == "__main__":
    main()
