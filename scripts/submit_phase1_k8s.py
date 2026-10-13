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


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Phase 1 retrieval evaluation to Kubernetes")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--pvc-name", default=None)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--voyage-api-key", default=None, help="Voyage API key (or set VOYAGE_API_KEY)")
    parser.add_argument("--voyage-secret-name", default="token-sniper-voyage")
    parser.add_argument("--voyage-secret-key", default="api-key")
    parser.add_argument("--storage-class", default="nfs-client")
    parser.add_argument("--storage-size", default="500Gi")
    parser.add_argument("--cache-pvc-name", default=None)
    parser.add_argument("--create-cache-pvc", action="store_true")
    parser.add_argument("--cache-storage-class", default="nfs-client")
    parser.add_argument("--cache-storage-size", default="1Ti")
    parser.add_argument("--node-pool", default="default")
    parser.add_argument(
        "--entry-point-method",
        choices=["embedding", "llm_prompt"],
        default="embedding",
    )
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="32Gi")
    parser.add_argument("--memory-limit", default="64Gi")
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"phase1-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    pvc_name = args.pvc_name or f"{job_name}-workspace"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    node_selector: dict[str, str] = (
        {"kubernetes.io/hostname": args.node_name}
        if args.node_name
        else {"node-role.kubernetes.io/runai-cpu-worker": "true"}
    )

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "phase1-retrieval",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": args.node_pool,
    }

    runner_script = _build_runner_script(entry_point_method=args.entry_point_method)

    qwen_base_url = (
        "http://qwen-vllm.runai-romania-dev.svc.cluster.local:8077/v1"
    )

    runner_env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
        {"name": "TOKEN_SNIPER_AGENT_MODEL", "value": "Qwen/Qwen3.5-35B-A3B-FP8"},
        {"name": "TOKEN_SNIPER_AGENT_PROVIDER", "value": "local"},
        {"name": "TOKEN_SNIPER_AGENT_BASE_URL", "value": qwen_base_url},
        {"name": "TOKEN_SNIPER_AGENT_API_KEY_ENV", "value": "LOCAL_OPENAI_API_KEY"},
        {"name": "TOKEN_SNIPER_AGENT_CONTEXT_WINDOW", "value": "65536"},
        {"name": "TOKEN_SNIPER_AGENT_REQUEST_TIMEOUT", "value": "300"},
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": args.github_secret_name,
                    "key": args.github_secret_key,
                }
            },
        },
        {
            "name": "VOYAGE_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": args.voyage_secret_name,
                    "key": args.voyage_secret_key,
                }
            },
        },
    ]
    if args.cache_pvc_name:
        runner_env.extend([
            {"name": "PIP_CACHE_DIR", "value": "/cache/runner/pip"},
            {"name": "HF_HOME", "value": "/cache/runner/huggingface"},
            {"name": "HF_HUB_CACHE", "value": "/cache/runner/huggingface/hub"},
            {"name": "HF_DATASETS_CACHE", "value": "/cache/runner/huggingface/datasets"},
            {"name": "XDG_CACHE_HOME", "value": "/cache/runner/xdg"},
        ])

    volumes: list[dict[str, Any]] = [
        {"name": "workspace", "persistentVolumeClaim": {"claimName": pvc_name}},
    ]
    runner_mounts: list[dict[str, Any]] = [
        {"name": "workspace", "mountPath": "/workspace"},
    ]
    if args.cache_pvc_name:
        volumes.append(
            {"name": "shared-cache", "persistentVolumeClaim": {"claimName": args.cache_pvc_name}}
        )
        runner_mounts.append({"name": "shared-cache", "mountPath": "/cache"})

    resources: list[dict[str, Any]] = []

    resources.append({
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": namespace, "labels": labels},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": args.storage_class,
            "resources": {"requests": {"storage": args.storage_size}},
        },
    })

    if args.cache_pvc_name and args.create_cache_pvc:
        resources.append({
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": args.cache_pvc_name, "namespace": namespace, "labels": labels},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": args.cache_storage_class,
                "resources": {"requests": {"storage": args.cache_storage_size}},
            },
        })

    resources.append({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": node_selector,
                    "volumes": volumes,
                    "containers": [{
                        "name": "runner",
                        "image": args.runner_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-lc", runner_script],
                        "env": runner_env,
                        "resources": {
                            "requests": {"cpu": args.cpu_request, "memory": args.memory_request},
                            "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit},
                        },
                        "volumeMounts": runner_mounts,
                    }],
                },
            },
        },
    })

    resources_yaml = yaml.safe_dump_all(resources, sort_keys=False)

    if args.output is not None:
        args.output.write_text(resources_yaml, encoding="utf-8")

    if args.dry_run:
        sys.stdout.write(resources_yaml)
        return

    voyage_key = args.voyage_api_key or os.environ.get("VOYAGE_API_KEY", "")
    if not voyage_key:
        print("ERROR: --voyage-api-key or VOYAGE_API_KEY env var is required", file=sys.stderr)
        raise SystemExit(1)

    github_token = args.github_token or _resolve_github_token()

    _kubectl_apply(args.kubeconfig, yaml.safe_dump(
        _secret_manifest(namespace, args.github_secret_name, {args.github_secret_key: github_token}),
        sort_keys=False,
    ))
    _kubectl_apply(args.kubeconfig, yaml.safe_dump(
        _secret_manifest(namespace, args.voyage_secret_name, {args.voyage_secret_key: voyage_key}),
        sort_keys=False,
    ))
    _kubectl_apply(args.kubeconfig, resources_yaml)

    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Workspace PVC: {pvc_name}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c runner -f"
    )
    print("Results path: /workspace/repo/experiments/phase1/")

    if args.wait:
        subprocess.run(
            [
                "kubectl", "--kubeconfig", str(args.kubeconfig),
                "wait", "--for=condition=complete", f"--timeout={args.timeout}",
                "-n", namespace, f"job/{job_name}",
            ],
            check=True,
        )


def _build_runner_script(*, entry_point_method: str) -> str:
    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*

        # Persistent cache dirs survive repo re-clones
        mkdir -p /workspace/cache/ssg /workspace/cache/chromadb /workspace/cache/repos

        # Clone repo (code only — data lives in /workspace/cache/)
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        python -m pip install --upgrade pip
        python -m pip install -e ".[dev,retrieval,data,experiments,agent]"

        echo "=== Phase 1 Setup: clone repos, build SSGs, index entities ==="
        python scripts/setup_phase1.py --provider voyageai \\
            --cache-dir /workspace/cache/repos \\
            --ssg-dir /workspace/cache/ssg \\
            --persist-dir /workspace/cache/chromadb

        echo "=== Phase 1 Retrieval Evaluation ==="
        python scripts/run_phase1_retrieval.py \\
            --entry-point-method {entry_point_method} \\
            --conditions graph_bfs graph_ppr embedding_knn \\
            --ssg-dir /workspace/cache/ssg \\
            --persist-dir /workspace/cache/chromadb

        echo "=== Phase 1 complete — results in experiments/phase1/ ==="
        ls -la experiments/phase1/
    """)


def _namespace_from_kubeconfig(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ctx_name = raw.get("current-context")
    for ctx in raw.get("contexts", []):
        if ctx.get("name") == ctx_name:
            ns = ctx.get("context", {}).get("namespace")
            if isinstance(ns, str) and ns:
                return ns
    return "default"


def _secret_manifest(namespace: str, name: str, data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": "Opaque",
        "stringData": data,
    }


def _resolve_github_token() -> str:
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return _git_output(["gh", "auth", "token"])


def _git_output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _kubectl_apply(kubeconfig: Path, manifest_yaml: str) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest_yaml,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
