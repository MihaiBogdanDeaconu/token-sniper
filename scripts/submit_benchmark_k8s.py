from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml

from token_sniper.agent.config import load_base_config, load_yaml_config


def build_runner_script(
    *,
    conditions: list[str],
    retrieval_k: int,
    agent_workers: int,
    eval_batch_size: int,
    harness_workers: int,
    base_config_path: str,
    manifest_path: str,
    ssg_dir: str,
    chromadb_dir: str,
    checkpoint_path: str | None,
) -> str:
    checkpoint_flag = f'  --checkpoint "{checkpoint_path}" \\' if checkpoint_path else ""
    conditions_str = " ".join(f'"{c}"' for c in conditions)

    return "\n\n".join([
        "set -euo pipefail",
        dedent("""
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends ca-certificates git
            rm -rf /var/lib/apt/lists/*
        """).strip(),
        dedent("""
            if [ -d /cache ]; then
              mkdir -p /cache/runner/pip /cache/runner/huggingface /cache/runner/xdg /cache/swebench
            fi
        """).strip(),
        dedent("""
            cleanup_docker_daemon() {
              python3 -c "
import os, signal
for e in os.scandir('/proc'):
    if not e.name.isdigit(): continue
    try:
        c = open(f'/proc/{e.name}/cmdline','rb').read().decode('utf-8','ignore')
    except OSError: continue
    if 'dockerd' in c:
        os.kill(int(e.name), signal.SIGTERM)
" 2>/dev/null || true
            }
            trap cleanup_docker_daemon EXIT
        """).strip(),
        dedent("""
            until python3 -c "
from urllib.request import urlopen
try:
    r = urlopen('http://127.0.0.1:2375/_ping', timeout=2)
    assert r.read().decode().strip() == 'OK'
except Exception:
    raise SystemExit(1)
"; do
              echo "waiting for docker sidecar"
              sleep 2
            done
        """).strip(),
        dedent(f"""
            git config --global --add safe.directory '*'
            rm -rf /workspace/repo
            auth_repo_url="$(python3 - <<'PY'
            import os
            repo_url = os.environ["TOKEN_SNIPER_REPO_URL"]
            token = os.environ["GITHUB_TOKEN"]
            print(f"https://x-access-token:{{token}}@{{repo_url[len('https://'):]}}")
            PY
            )"
            git clone "$auth_repo_url" /workspace/repo
            unset auth_repo_url
            cd /workspace/repo
            git checkout "$TOKEN_SNIPER_GIT_REF"
            export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"
        """).strip(),
        dedent("""
            python -m pip install --upgrade pip
            python -m pip install -e ".[agent,experiments,data,retrieval]"
        """).strip(),
        "\n".join([
            "python scripts/run_retrieval_benchmark.py \\",
            f'  --base-config "{base_config_path}" \\',
            f'  --manifest "{manifest_path}" \\',
            f'  --ssg-dir "{ssg_dir}" \\',
            f'  --chromadb-dir "{chromadb_dir}" \\',
            f"  --conditions {conditions_str} \\",
            checkpoint_flag,
            f'  --retrieval-k "{retrieval_k}" \\',
            f'  --agent-workers "{agent_workers}" \\',
            f'  --eval-batch-size "{eval_batch_size}" \\',
            f'  --max-workers "{harness_workers}"',
        ]),
        'echo "Benchmark artifacts in /workspace/repo/experiments/retrieval_benchmark"',
    ])


def build_k8s_resources(
    *,
    namespace: str,
    job_name: str,
    pvc_name: str,
    repo_url: str,
    git_ref: str,
    github_secret_name: str,
    github_secret_key: str,
    runner_image: str,
    model_name: str,
    qwen_base_url: str,
    cache_pvc_name: str,
    node_selector: dict[str, str],
    runner_script: str,
    storage_class: str,
    voyage_secret_name: str = "token-sniper-voyage",
    voyage_secret_key: str = "api-key",
) -> list[dict[str, Any]]:
    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "retrieval-benchmark",
        "runai/node-pool": "h100",
    }

    workspace_pvc: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace, "labels": labels},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage_class,
            "resources": {"requests": {"storage": "200Gi"}},
        },
    }

    runner_env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "DOCKER_HOST", "value": "tcp://127.0.0.1:2375"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
        {"name": "TOKEN_SNIPER_AGENT_PROVIDER", "value": "local"},
        {"name": "TOKEN_SNIPER_AGENT_BASE_URL", "value": qwen_base_url},
        {"name": "TOKEN_SNIPER_AGENT_MODEL", "value": model_name},
        {"name": "TOKEN_SNIPER_AGENT_API_KEY_ENV", "value": "LOCAL_OPENAI_API_KEY"},
        {"name": "LOCAL_OPENAI_API_KEY", "value": "local-placeholder"},
        {"name": "PIP_CACHE_DIR", "value": "/cache/runner/pip"},
        {"name": "HF_HOME", "value": "/cache/runner/huggingface"},
        {"name": "TOKEN_SNIPER_SWEBENCH_CACHE", "value": "/cache/swebench"},
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": github_secret_name, "key": github_secret_key},
            },
        },
        {
            "name": "VOYAGE_API_KEY",
            "valueFrom": {
                "secretKeyRef": {"name": voyage_secret_name, "key": voyage_secret_key},
            },
        },
    ]

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
                    "shareProcessNamespace": True,
                    "nodeSelector": node_selector,
                    "volumes": [
                        {"name": "workspace", "persistentVolumeClaim": {"claimName": pvc_name}},
                        {"name": "shared-cache", "persistentVolumeClaim": {"claimName": cache_pvc_name}},
                        {"name": "docker-graph", "emptyDir": {}},
                    ],
                    "containers": [
                        {
                            "name": "runner",
                            "image": runner_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", runner_script],
                            "env": runner_env,
                            "resources": {
                                "requests": {"cpu": "16", "memory": "64Gi"},
                                "limits": {"cpu": "48", "memory": "128Gi"},
                            },
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "shared-cache", "mountPath": "/cache"},
                            ],
                        },
                        {
                            "name": "docker-daemon",
                            "image": "docker:27-dind",
                            "imagePullPolicy": "IfNotPresent",
                            "args": [
                                "--host=tcp://0.0.0.0:2375",
                                "--host=unix:///var/run/docker.sock",
                                "--tls=false",
                                "--storage-driver=overlay2",
                            ],
                            "env": [{"name": "DOCKER_TLS_CERTDIR", "value": ""}],
                            "securityContext": {"privileged": True},
                            "resources": {
                                "requests": {"cpu": "4", "memory": "8Gi"},
                                "limits": {"cpu": "8", "memory": "16Gi"},
                            },
                            "volumeMounts": [
                                {"name": "docker-graph", "mountPath": "/var/lib/docker"},
                            ],
                        },
                    ],
                },
            },
        },
    }
    return [workspace_pvc, job]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--storage-class", default="nfs-client")
    parser.add_argument("--cache-pvc-name", default="token-sniper-shared-cache")
    parser.add_argument("--ssg-dir", default="/cache/cache/ssg")
    parser.add_argument("--chromadb-dir", default="/cache/cache/chromadb")
    parser.add_argument("--checkpoint", default=None, help="Path inside the pod to the edge predictor checkpoint")
    parser.add_argument("--conditions", nargs="+", choices=["ssg_ppr", "augmented_ppr"], default=["ssg_ppr"])
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--agent-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--qwen-base-url", default="http://qwen-vllm:8077/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--voyage-api-key", default=None)
    parser.add_argument("--voyage-secret-name", default="token-sniper-voyage")
    parser.add_argument("--voyage-secret-key", default="api-key")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agent_config = load_base_config(args.base_config)
    namespace = args.namespace or _load_namespace(args.kubeconfig)
    job_name = args.job_name or f"bench-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    pvc_name = f"{job_name}-workspace"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])
    model_name = args.model or agent_config.models.agent.name
    runner_script = build_runner_script(
        conditions=args.conditions,
        retrieval_k=args.retrieval_k,
        agent_workers=args.agent_workers,
        eval_batch_size=args.eval_batch_size,
        harness_workers=args.max_workers,
        base_config_path=args.base_config.as_posix(),
        manifest_path=args.manifest.as_posix(),
        ssg_dir=args.ssg_dir,
        chromadb_dir=args.chromadb_dir,
        checkpoint_path=args.checkpoint,
    )

    resources = build_k8s_resources(
        namespace=namespace,
        job_name=job_name,
        pvc_name=pvc_name,
        repo_url=repo_url,
        git_ref=git_ref,
        github_secret_name=args.github_secret_name,
        github_secret_key=args.github_secret_key,
        runner_image=args.runner_image,
        model_name=model_name,
        qwen_base_url=args.qwen_base_url,
        cache_pvc_name=args.cache_pvc_name,
        node_selector={"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
        runner_script=runner_script,
        storage_class=args.storage_class,
        voyage_secret_name=args.voyage_secret_name,
        voyage_secret_key=args.voyage_secret_key,
    )
    resources_yaml = yaml.safe_dump_all(resources, sort_keys=False)

    if args.dry_run:
        sys.stdout.write(resources_yaml)
        return

    github_token = args.github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not github_token:
        github_token = _git_output(["gh", "auth", "token"])
    secret_yaml = yaml.safe_dump({
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": args.github_secret_name, "namespace": namespace},
        "type": "Opaque",
        "stringData": {args.github_secret_key: github_token},
    }, sort_keys=False)
    _kubectl_apply(args.kubeconfig, secret_yaml)
    _kubectl_apply(args.kubeconfig, resources_yaml)

    print(f"Submitted benchmark job: {job_name}")
    print(f"Conditions: {', '.join(args.conditions)}")
    print(f"Manifest: {args.manifest}")
    print(f"Qwen: {args.qwen_base_url}")
    print(f"Follow logs: kubectl --kubeconfig {args.kubeconfig} logs -n {namespace} job/{job_name} -c runner -f")


def _load_namespace(kubeconfig: Path) -> str:
    raw = yaml.safe_load(kubeconfig.read_text(encoding="utf-8")) or {}
    current = raw.get("current-context")
    for ctx in raw.get("contexts", []):
        if ctx.get("name") == current:
            ns = ctx.get("context", {}).get("namespace")
            if isinstance(ns, str) and ns:
                return ns
    return "default"


def _kubectl_apply(kubeconfig: Path, manifest: str) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest, text=True, check=True,
    )


def _git_output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
