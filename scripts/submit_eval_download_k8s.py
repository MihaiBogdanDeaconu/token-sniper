from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml

DOWNLOAD_SCRIPT = Path(__file__).resolve().parent / "download_eval_datasets.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download eval datasets onto the eval PVC")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--eval-pvc", default="token-sniper-eval-cache")
    parser.add_argument("--storage-class", default="nfs-client")
    parser.add_argument("--storage-size", default="500Gi")
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--node-pool", default="default")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"eval-download-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    config_map_name = f"{job_name}-script"

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "eval-download",
        "runai/node-pool": args.node_pool,
    }

    runner_script = dedent("""\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates
        rm -rf /var/lib/apt/lists/*
        python -m pip install --upgrade pip
        python -m pip install datasets huggingface_hub
        mkdir -p /eval
        python /scripts/download_eval_datasets.py --output-dir /eval
        echo "=== download_manifest.json ==="
        cat /eval/raw/download_manifest.json
    """)

    resources: list[dict[str, Any]] = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": config_map_name, "namespace": namespace, "labels": labels},
            "data": {"download_eval_datasets.py": DOWNLOAD_SCRIPT.read_text(encoding="utf-8")},
        },
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": args.eval_pvc, "namespace": namespace, "labels": labels},
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": args.storage_class,
                "resources": {"requests": {"storage": args.storage_size}},
            },
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeSelector": {"node-role.kubernetes.io/runai-cpu-worker": "true"},
                        "volumes": [
                            {"name": "eval", "persistentVolumeClaim": {"claimName": args.eval_pvc}},
                            {"name": "script", "configMap": {"name": config_map_name}},
                        ],
                        "containers": [{
                            "name": "downloader",
                            "image": args.runner_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", runner_script],
                            "env": [
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "HF_HOME", "value": "/eval/hf"},
                            ],
                            "resources": {
                                "requests": {"cpu": "2", "memory": "8Gi"},
                                "limits": {"cpu": "4", "memory": "16Gi"},
                            },
                            "volumeMounts": [
                                {"name": "eval", "mountPath": "/eval"},
                                {"name": "script", "mountPath": "/scripts"},
                            ],
                        }],
                    },
                },
            },
        },
    ]

    manifests_yaml = yaml.safe_dump_all(resources, sort_keys=False)
    if args.dry_run:
        sys.stdout.write(manifests_yaml)
        return

    _kubectl_apply(args.kubeconfig, manifests_yaml)
    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Eval PVC: {args.eval_pvc}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c downloader -f"
    )
    if args.wait:
        subprocess.run(
            [
                "kubectl", "--kubeconfig", str(args.kubeconfig),
                "wait", "--for=condition=complete", f"--timeout={args.timeout}",
                "-n", namespace, f"job/{job_name}",
            ],
            check=True,
        )


def _namespace_from_kubeconfig(path: Path) -> str:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ctx_name = raw.get("current-context")
    for ctx in raw.get("contexts", []):
        if ctx.get("name") == ctx_name:
            ns = ctx.get("context", {}).get("namespace")
            if isinstance(ns, str) and ns:
                return ns
    return "default"


def _kubectl_apply(kubeconfig: Path, manifest_yaml: str) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest_yaml,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
