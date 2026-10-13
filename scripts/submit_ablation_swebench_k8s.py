from __future__ import annotations

import argparse
import shlex
import sys
from datetime import UTC, datetime
from textwrap import dedent
from typing import Any

import yaml

from scripts.eval_ablation_swebench import DATASETS
from scripts.submit_eval_index_k8s import (
    _git_output,
    _kubectl_apply,
    _namespace_from_kubeconfig,
    _require_pushed,
)
from scripts.submit_eval_retrieval_k8s import add_job_args, submit_eval_job
from token_sniper.evaluation.arms.selection import ARMS

HARNESS_INSTALL = {
    "swebench_verified": 'python -m pip install "swebench==4.1.0"',
    "swebench_live": dedent("""\
        if [ ! -f src/swebench/pyproject.toml ]; then
            rm -rf src/swebench
            git clone https://github.com/SWE-bench-Live/SWE-bench-Live.git src/swebench
        fi
        python -m pip install -e src/swebench"""),
}


def eval_command(args: argparse.Namespace, arms: list[str], seed: int | None) -> str:
    flags = [
        "--arm", *arms, "--dataset", args.dataset, "--manifest", args.manifest,
        "--output-root", args.output_root, "--ssg-dir", args.ssg_dir, "--repos-dir", args.repos_dir,
        "--chromadb-dir", args.chromadb_dir, "--cache-root", args.cache_root, "--cache-name", args.cache_name,
        "--reranker-model", args.reranker_model,
        "--vgib-checkpoint", f"{args.models_dir}/vgib/best.pt",
        "--grpo-checkpoint", f"{args.models_dir}/rl_grpo/best_rl.pt",
        "--top-k-files", str(args.top_k_files), "--subgraph-top-k", str(args.subgraph_top_k),
        "--max-context-tokens", str(args.max_context_tokens), "--budget-arm", args.budget_arm,
    ]
    if args.budget_tokens is not None:
        flags += ["--budget-tokens", str(args.budget_tokens)]
    if args.max_tasks is not None:
        flags += ["--max-tasks", str(args.max_tasks)]
    if "swe_pruner" in arms:
        flags += ["--swe-pruner-model", args.swe_pruner_model]
    if seed is None:
        flags += ["--select-only", "--device", "cpu" if args.cpu else "cuda",
                  "--compressor-device", "cpu" if args.cpu else "cuda"]
    else:
        flags += ["--seed", str(seed), "--qwen-base-url", args.qwen_base_url, "--max-turns", str(args.max_turns),
                  "--agent-workers", str(args.agent_workers), "--eval-batch-size", str(args.eval_batch_size),
                  "--max-harness-workers", str(args.max_harness_workers)]
        if args.model:
            flags += ["--model", args.model]
        if args.resume:
            flags.append("--resume")
    return "python scripts/eval_ablation_swebench.py " + " ".join(shlex.quote(f) for f in flags)


def submit_select_only(args: argparse.Namespace) -> None:
    extra = []
    if "llmlingua2" in args.arm:
        extra.append('python -m pip install "llmlingua==0.2.2"')
    if "swe_pruner" in args.arm:
        if args.cpu:
            raise SystemExit("swe_pruner needs a CUDA GPU (flash-attention 2); drop --cpu")
        if not args.flash_attn_wheel:
            raise SystemExit("swe_pruner needs --flash-attn-wheel URL (a flash-attn wheel for the pod's "
                             "torch/CUDA/Python 3.12; see the report for the pinned setup)")
        extra += [
            'python -m pip install "swe-pruner==0.1.1" "torch==2.8.0"',
            f"python -m pip install {shlex.quote(args.flash_attn_wheel)}",
            f"test -f {shlex.quote(args.swe_pruner_model)}/model.safetensors || "
            f"huggingface-cli download ayanami-kitasan/code-pruner --local-dir {shlex.quote(args.swe_pruner_model)}",
        ]

    def body(_job_name: str) -> str:
        return "\n".join([*extra, eval_command(args, args.arm, None), ""])

    submit_eval_job(args, component="downstream-select", body=body)


def agent_runner_script(args: argparse.Namespace, arm: str, seed: int) -> str:
    return dedent("""\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git curl
        rm -rf /var/lib/apt/lists/*
        DOCKER_VERSION="27.5.1"
        curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" | tar xz -C /tmp
        mv /tmp/docker/docker /usr/local/bin/docker
        rm -rf /tmp/docker
        git config --global --add safe.directory '*'
        until docker info >/dev/null 2>&1; do echo "Waiting for Docker daemon..."; sleep 2; done

        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH=/workspace/repo

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        python -m pip install -e ".[agent,experiments,data,retrieval]"
        python -m pip install xgboost
    """) + HARNESS_INSTALL[args.dataset] + "\n" + eval_command(args, [arm], seed) + "\n"


def submit_agent_run(args: argparse.Namespace, arm: str, seed: int) -> None:
    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    job_name = args.job_name or f"downstream-{arm.replace('_', '-')}-s{seed}-{stamp}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])
    if not args.dry_run:
        _require_pushed(git_ref)
    labels = {"app.kubernetes.io/name": "token-sniper", "app.kubernetes.io/component": "downstream-agent",
              "runai/node-pool": "cpu-worker"}
    job: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
        "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": labels}, "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"node-role.kubernetes.io/runai-cpu-worker": "true"},
            "volumes": [
                {"name": "eval", "persistentVolumeClaim": {"claimName": args.eval_pvc}},
                {"name": "reranker", "persistentVolumeClaim": {"claimName": args.reranker_pvc, "readOnly": True}},
                {"name": "models", "persistentVolumeClaim": {"claimName": args.models_pvc, "readOnly": True}},
                {"name": "workspace", "emptyDir": {}},
                {"name": "docker-graph-storage", "emptyDir": {}},
            ],
            "containers": [
                {
                    "name": "runner",
                    "image": args.runner_image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc", agent_runner_script(args, arm, seed)],
                    "env": [
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {"name": "DOCKER_HOST", "value": "tcp://127.0.0.1:2375"},
                        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
                        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
                        {"name": "HF_HOME", "value": "/eval/hf"},
                        {"name": "GITHUB_TOKEN", "valueFrom": {"secretKeyRef": {
                            "name": args.github_secret_name, "key": args.github_secret_key}}},
                    ],
                    "resources": {
                        "requests": {"cpu": args.cpu_request, "memory": args.memory_request},
                        "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit},
                    },
                    "volumeMounts": [
                        {"name": "eval", "mountPath": "/eval"},
                        {"name": "reranker", "mountPath": "/reranker", "readOnly": True},
                        {"name": "models", "mountPath": "/models", "readOnly": True},
                        {"name": "workspace", "mountPath": "/workspace"},
                    ],
                },
                {
                    "name": "docker-daemon",
                    "image": "docker:27-dind",
                    "imagePullPolicy": "IfNotPresent",
                    "args": ["--host=tcp://0.0.0.0:2375", "--host=unix:///var/run/docker.sock", "--tls=false",
                             "--storage-driver=overlay2",
                             *([f"--registry-mirror={args.registry_mirror}"] if args.registry_mirror else [])],
                    "env": [{"name": "DOCKER_TLS_CERTDIR", "value": ""}],
                    "securityContext": {"privileged": True},
                    "resources": {
                        "requests": {"cpu": args.dind_cpu_request, "memory": args.dind_memory_request},
                        "limits": {"cpu": args.dind_cpu_limit, "memory": args.dind_memory_limit},
                    },
                    "volumeMounts": [{"name": "docker-graph-storage", "mountPath": "/var/lib/docker"}],
                },
            ],
        }}},
    }
    manifest_yaml = yaml.safe_dump(job, sort_keys=False)
    if args.dry_run:
        sys.stdout.write(manifest_yaml + "---\n")
        return
    _kubectl_apply(args.kubeconfig, manifest_yaml)
    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Follow logs: kubectl --kubeconfig {args.kubeconfig} logs -n {namespace} job/{job_name} -c runner -f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit downstream agent runs or the selection pass")
    add_job_args(parser)
    parser.add_argument("--arm", nargs="+", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, nargs="+", default=None,
                        help="One job per (arm, seed); omit with --select-only")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-name", required=True, help="Stage A/B namespace under /eval/cache")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="/eval/results/downstream")
    parser.add_argument("--ssg-dir", default="/eval/ssg")
    parser.add_argument("--repos-dir", default="/eval/repos")
    parser.add_argument("--chromadb-dir", default="/eval/chromadb")
    parser.add_argument("--cache-root", default="/eval/cache")
    parser.add_argument("--reranker-model", default="/reranker/repo/experiments/reranker/xgb_reranker.json")
    parser.add_argument("--models-dir", default="/models/repo/experiments",
                        help="Holds vgib/best.pt and rl_grpo/best_rl.pt (a training run's checkpoints are at "
                             "/models/runs/<job>/repo/experiments)")
    parser.add_argument("--swe-pruner-model", default="/eval/models/code-pruner")
    parser.add_argument("--flash-attn-wheel", default=None, help="flash-attn wheel URL for the swe_pruner pass")
    parser.add_argument("--top-k-files", type=int, default=10)
    parser.add_argument("--subgraph-top-k", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=32_000)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument("--budget-arm", default="dense_reranker_vgib")
    parser.add_argument("--qwen-base-url", default="http://qwen-vllm:8077/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--agent-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--max-harness-workers", type=int, default=8)
    parser.add_argument("--dind-cpu-request", default="8")
    parser.add_argument("--dind-cpu-limit", default="16")
    parser.add_argument("--dind-memory-request", default="16Gi")
    parser.add_argument("--dind-memory-limit", default="32Gi")
    parser.add_argument("--registry-mirror", default=None,
                        help="Docker Hub pull-through cache for the DinD daemon: every (arm, seed) job pulls "
                             "all of its tasks' SWE-bench images")
    args = parser.parse_args()

    if args.select_only:
        if args.seed is not None:
            parser.error("--select-only takes no --seed")
        submit_select_only(args)
        return
    if args.seed is None:
        parser.error("give --seed (one job per arm and seed)")
    if args.job_name and len(args.arm) * len(args.seed) > 1:
        parser.error("--job-name names a single job")
    for arm in args.arm:
        for seed in args.seed:
            submit_agent_run(args, arm, seed)


if __name__ == "__main__":
    main()
