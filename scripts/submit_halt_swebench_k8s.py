from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit HALT SWE-bench eval to K8s")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--workspace-pvc", required=True)
    parser.add_argument("--cache-pvc", default=None)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--cpu-request", default="4")
    parser.add_argument("--cpu-limit", default="8")
    parser.add_argument("--memory-request", default="16Gi")
    parser.add_argument("--memory-limit", default="32Gi")
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="finetune_best.pt",
                        help="Checkpoint filename inside /workspace/cache/halt_checkpoints/")
    parser.add_argument("--docker", action="store_true",
                        help="Enable Docker isolation via DinD sidecar")
    parser.add_argument("--image-workers", type=int, default=4,
                        help="Parallel workers for SWE-bench image building")
    parser.add_argument("--docker-cache-pvc", default=None,
                        help="PVC for persistent Docker image cache (avoids rebuilds)")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="12h")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    suffix = "docker" if args.docker else "bare"
    job_name = args.job_name or f"halt-swebench-{suffix}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    runner_script = _build_runner_script(
        max_tasks=args.max_tasks,
        dataset=args.dataset,
        split=args.split,
        checkpoint=args.checkpoint,
        use_docker=args.docker,
        image_workers=args.image_workers,
    )

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "halt-swebench-eval",
    }

    env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": args.github_secret_name,
                    "key": args.github_secret_key,
                },
            },
        },
    ]
    if args.cache_pvc:
        env.extend([
            {"name": "PIP_CACHE_DIR", "value": "/cache/runner/pip"},
            {"name": "HF_HOME", "value": "/cache/runner/huggingface"},
        ])
    if args.docker:
        env.append({"name": "DOCKER_HOST", "value": "tcp://localhost:2375"})

    volumes: list[dict[str, Any]] = [
        {"name": "workspace", "persistentVolumeClaim": {"claimName": args.workspace_pvc}},
    ]
    mounts: list[dict[str, Any]] = [
        {"name": "workspace", "mountPath": "/workspace"},
    ]
    if args.cache_pvc:
        volumes.append(
            {"name": "shared-cache", "persistentVolumeClaim": {"claimName": args.cache_pvc}},
        )
        mounts.append({"name": "shared-cache", "mountPath": "/cache"})

    if args.docker:
        if args.docker_cache_pvc:
            volumes.append({
                "name": "docker-storage",
                "persistentVolumeClaim": {"claimName": args.docker_cache_pvc},
            })
        else:
            volumes.append({"name": "docker-storage", "emptyDir": {}})

    containers: list[dict[str, Any]] = [{
        "name": "evaluator",
        "image": args.runner_image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/bash", "-lc", runner_script],
        "env": env,
        "resources": {
            "requests": {"cpu": args.cpu_request, "memory": args.memory_request},
            "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit},
        },
        "volumeMounts": mounts,
    }]

    if args.docker:
        dind_mounts = [
            {"name": "docker-storage", "mountPath": "/var/lib/docker"},
        ]

        dind_env = [{"name": "DOCKER_TLS_CERTDIR", "value": ""}]

        containers.append({
            "name": "dind",
            "image": "docker:27-dind",
            "securityContext": {"privileged": True},
            "env": dind_env,
            "resources": {
                "requests": {"cpu": "2", "memory": "4Gi"},
                "limits": {"cpu": "4", "memory": "8Gi"},
            },
            "volumeMounts": dind_mounts,
        })

    job_manifest: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "volumes": volumes,
                    "containers": containers,
                },
            },
        },
    }

    manifest_yaml = yaml.safe_dump(job_manifest, sort_keys=False)

    if args.dry_run:
        sys.stdout.write(manifest_yaml)
        return

    _kubectl_apply(args.kubeconfig, manifest_yaml)
    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Mode: {'Docker-isolated (DinD sidecar)' if args.docker else 'bare-metal'}")
    print(f"Workspace PVC: {args.workspace_pvc}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c evaluator -f"
    )
    print(f"Results: /workspace/repo/experiments/halt_swebench/")

    if args.wait:
        subprocess.run(
            [
                "kubectl", "--kubeconfig", str(args.kubeconfig),
                "wait", "--for=condition=complete", f"--timeout={args.timeout}",
                "-n", namespace, f"job/{job_name}",
            ],
            check=True,
        )


def _build_runner_script(
    *,
    max_tasks: int,
    dataset: str,
    split: str,
    checkpoint: str,
    use_docker: bool = False,
    image_workers: int = 4,
) -> str:
    docker_setup = ""
    docker_flags = ""
    if use_docker:
        docker_setup = dedent("""\
            echo "=== Installing Docker CLI ==="
            apt-get install -y --no-install-recommends docker.io >/dev/null 2>&1 || \\
                (curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz | \\
                 tar xz --strip-components=1 -C /usr/local/bin docker/docker)
            echo "Waiting for DinD sidecar..."
            for i in $(seq 1 30); do
                docker info >/dev/null 2>&1 && break
                sleep 2
            done
            docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not available"; exit 1; }
            echo "Docker ready: $(docker version --format '{{.Server.Version}}')"

        """)
        docker_flags = f"    --docker \\\n    --build-images \\\n    --image-workers {image_workers} \\\n"

    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git curl
        rm -rf /var/lib/apt/lists/*

        git config --global --add safe.directory '*'

        {docker_setup}        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        if [ -d /workspace/repo/.git ]; then
            cd /workspace/repo
            git remote set-url origin "$auth_url"
            git fetch origin
            git checkout "$TOKEN_SNIPER_GIT_REF"
        else
            git clone "$auth_url" /workspace/repo
            cd /workspace/repo
            git checkout "$TOKEN_SNIPER_GIT_REF"
        fi
        unset auth_url
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        python -m pip install aiohttp datasets scikit-learn openai docker swebench
        python -m pip install -e ".[dev]"

        CKPT="/workspace/cache/halt_checkpoints/{checkpoint}"
        if [ ! -f "$CKPT" ]; then
            echo "ERROR: checkpoint not found at $CKPT"
            echo "Available checkpoints:"
            ls -lh /workspace/cache/halt_checkpoints/ 2>/dev/null || echo "  (none)"
            exit 1
        fi
        echo "Using checkpoint: $CKPT"

        echo "==========================================="
        echo "=== HALT SWE-bench Evaluation           ==="
        echo "=== Tasks: {max_tasks} | Dataset: {dataset}    ==="
        echo "=== Mode: {'Docker-isolated' if use_docker else 'bare-metal':20s}         ==="
        echo "==========================================="

        echo "Clearing stale results from previous runs..."
        rm -rf /workspace/repo/experiments/halt_swebench/
        mkdir -p /workspace/repo/experiments/halt_swebench

        python scripts/eval_halt_swebench.py \\
            --checkpoint "$CKPT" \\
            --output-dir /workspace/repo/experiments/halt_swebench \\
            --dataset {dataset} \\
            --split {split} \\
            --max-tasks {max_tasks} \\
{docker_flags}            --resume

        echo "==========================================="
        echo "=== Evaluation Complete                 ==="
        echo "==========================================="
        echo ""
        echo "=== Summary ==="
        cat /workspace/repo/experiments/halt_swebench/summary.json 2>/dev/null || echo "No summary"
        echo ""
        echo "=== Score Distribution ==="
        python3 -c "
import json, sys
scores = []
with open('/workspace/repo/experiments/halt_swebench/results.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        s = rec.get('halt_submit_score')
        if s is not None:
            scores.append(s)
if not scores:
    print('No scores found'); sys.exit(0)
import statistics
print(f'  N         = {{len(scores)}}')
print(f'  Mean      = {{statistics.mean(scores):.4f}}')
print(f'  Median    = {{statistics.median(scores):.4f}}')
print(f'  Stdev     = {{statistics.stdev(scores):.4f}}' if len(scores) > 1 else '')
print(f'  Min       = {{min(scores):.4f}}')
print(f'  Max       = {{max(scores):.4f}}')
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
hist = [0]*10
for s in scores:
    for i in range(10):
        if bins[i] <= s < bins[i+1]:
            hist[i] += 1; break
print('  Histogram:')
for i in range(10):
    bar = '#' * hist[i]
    print(f'    [{{bins[i]:.1f}}-{{bins[i+1]:.1f}})  {{hist[i]:3d}}  {{bar}}')
" 2>/dev/null || echo "Could not compute distribution"
    """)


def _namespace_from_kubeconfig(kubeconfig: Path) -> str:
    import yaml as _yaml
    cfg = _yaml.safe_load(kubeconfig.read_text(encoding="utf-8"))
    for ctx in cfg.get("contexts", []):
        ns = ctx.get("context", {}).get("namespace")
        if ns:
            return ns
    return "default"


def _git_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _kubectl_apply(kubeconfig: Path, manifest: str) -> None:
    proc = subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    print(proc.stdout.strip())


if __name__ == "__main__":
    main()
