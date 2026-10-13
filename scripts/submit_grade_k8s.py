from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit SWE-bench grading job")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--workspace-pvc", required=True)
    parser.add_argument("--cache-pvc", default=None)
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = f"grade-swebench-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = _git_output(["git", "remote", "get-url", "origin"])
    git_ref = _git_output(["git", "rev-parse", "HEAD"])

    runner_script = dedent(f"""\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git curl
        rm -rf /var/lib/apt/lists/*
        git config --global --add safe.directory '*'

        # Install Docker CLI
        curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz | \\
            tar xz --strip-components=1 -C /usr/local/bin docker/docker
        export DOCKER_HOST=tcp://localhost:2375
        echo "Waiting for DinD sidecar..."
        for i in $(seq 1 60); do
            docker info >/dev/null 2>&1 && break
            sleep 2
        done
        docker info >/dev/null 2>&1 || {{ echo "ERROR: Docker not available"; exit 1; }}
        echo "Docker ready: $(docker version --format '{{{{.Server.Version}}}}')"

        # Clone repo
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        if [ -d /workspace/repo/.git ]; then
            cd /workspace/repo
            git remote set-url origin "$auth_url"
            git fetch origin
            git checkout "{git_ref}"
        else
            git clone "$auth_url" /workspace/repo
            cd /workspace/repo
            git checkout "{git_ref}"
        fi
        unset auth_url
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        python -m pip install datasets docker swebench
        python -m pip install -e ".[dev]"

        echo "==========================================="
        echo "=== SWE-bench Patch Grading             ==="
        echo "==========================================="

        echo "--- Trace Analysis ---"
        python scripts/analyze_traces.py \\
            --results-dir /workspace/repo/experiments/halt_swebench

        echo ""
        echo "--- Grading ---"
        python scripts/grade_swebench.py \\
            --results-dir /workspace/repo/experiments/halt_swebench \\
            --max-workers {args.max_workers}

        echo "==========================================="
        echo "=== Grading Complete                    ==="
        echo "==========================================="
    """)

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "swebench-grading",
    }

    env = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
        {"name": "DOCKER_HOST", "value": "tcp://localhost:2375"},
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

    volumes = [
        {"name": "workspace", "persistentVolumeClaim": {"claimName": args.workspace_pvc}},
        {"name": "docker-storage", "emptyDir": {}},
    ]
    mounts = [
        {"name": "workspace", "mountPath": "/workspace"},
    ]
    if args.cache_pvc:
        volumes.append({"name": "shared-cache", "persistentVolumeClaim": {"claimName": args.cache_pvc}})
        mounts.append({"name": "shared-cache", "mountPath": "/cache"})

    job_manifest = {
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
                    "containers": [
                        {
                            "name": "grader",
                            "image": "python:3.12-slim",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", runner_script],
                            "env": env,
                            "resources": {
                                "requests": {"cpu": "40", "memory": "96Gi"},
                                "limits": {"cpu": "48", "memory": "128Gi"},
                            },
                            "volumeMounts": mounts,
                        },
                        {
                            "name": "dind",
                            "image": "docker:27-dind",
                            "securityContext": {"privileged": True},
                            "env": [{"name": "DOCKER_TLS_CERTDIR", "value": ""}],
                            "resources": {
                                "requests": {"cpu": "16", "memory": "32Gi"},
                                "limits": {"cpu": "16", "memory": "32Gi"},
                            },
                            "volumeMounts": [
                                {"name": "docker-storage", "mountPath": "/var/lib/docker"},
                            ],
                        },
                    ],
                },
            },
        },
    }

    manifest_yaml = yaml.safe_dump(job_manifest, sort_keys=False)
    if args.dry_run:
        sys.stdout.write(manifest_yaml)
        return

    proc = subprocess.run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "apply", "-f", "-"],
        input=manifest_yaml, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    print(proc.stdout.strip())
    print(f"Submitted grading job {job_name} in namespace {namespace}")
    print(f"Follow logs: kubectl --kubeconfig {args.kubeconfig} logs -n {namespace} job/{job_name} -c grader -f")


def _namespace_from_kubeconfig(kubeconfig: Path) -> str:
    cfg = yaml.safe_load(kubeconfig.read_text(encoding="utf-8"))
    for ctx in cfg.get("contexts", []):
        ns = ctx.get("context", {}).get("namespace")
        if ns:
            return ns
    return "default"


def _git_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


if __name__ == "__main__":
    main()
