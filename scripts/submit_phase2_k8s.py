from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import json as _json

import yaml

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "configs" / "phases" / "phase0_tasks.json"


def _load_all_task_ids() -> list[str]:
    manifest = _json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return [t["instance_id"] for t in manifest]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Phase 2 pipeline to Kubernetes")
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
    parser.add_argument("--voyage-api-key", default=None,
                        help="(Legacy, no longer required) Voyage API key")
    parser.add_argument("--voyage-secret-name", default="token-sniper-voyage")
    parser.add_argument("--voyage-secret-key", default="api-key")
    parser.add_argument("--storage-class", default="nfs-client")
    parser.add_argument("--storage-size", default="200Gi")
    parser.add_argument("--cache-pvc-name", default=None,
                        help="Existing PVC with Phase 1 caches (SSGs, repos, ChromaDB)")
    parser.add_argument("--chromadb-pvc-name", default=None,
                        help="PVC containing Jina ChromaDB (e.g. phase2-20260426134409-workspace)")
    parser.add_argument("--create-cache-pvc", action="store_true")
    parser.add_argument("--cache-storage-class", default="nfs-client")
    parser.add_argument("--cache-storage-size", default="200Gi")
    parser.add_argument("--node-pool", default="h100")
    parser.add_argument("--gpu-product", default="NVIDIA-H100-80GB-HBM3",
                        help="GPU product label for node selector")
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="32Gi")
    parser.add_argument("--memory-limit", default="64Gi")
    parser.add_argument("--gpu", default="1", help="Number of GPUs to request (0 for CPU-only)")
    parser.add_argument("--device", default=None,
                        help="PyTorch device for training/eval (default: cuda if gpu>0, else cpu)")
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--trace-parallel", type=int, default=4,
                        help="Parallelism for trace collection (across repos)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train/test split ratio (repo-grouped)")
    parser.add_argument("--skip-trace-collection", action="store_true",
                        help="Skip Stage 1 trace collection (reuse traces from existing PVC)")
    parser.add_argument("--loader-workers", type=int, default=16,
                        help="Threads for parallel TaskGraph building (NFS-bound)")
    parser.add_argument("--exclude-repos", type=str,
                        default="aws-cloudformation/cfn-lint,pylint-dev/pylint",
                        help="Comma-separated repos to skip during training/evaluation")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"phase2-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    pvc_name = args.pvc_name or f"{job_name}-workspace"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    device = args.device or ("cuda" if args.gpu and args.gpu != "0" else "cpu")

    if args.node_name:
        node_selector: dict[str, str] = {"kubernetes.io/hostname": args.node_name}
    elif args.gpu and args.gpu != "0":
        node_selector = {"nvidia.com/gpu.product": args.gpu_product}
    else:
        node_selector = {"node-role.kubernetes.io/runai-cpu-worker": "true"}

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "phase2-pipeline",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": args.node_pool,
    }

    runner_script = _build_runner_script(
        trace_parallel=args.trace_parallel,
        epochs=args.epochs,
        patience=args.patience,
        train_ratio=args.train_ratio,
        device=device,
        has_cache_pvc=args.cache_pvc_name is not None,
        has_chromadb_pvc=args.chromadb_pvc_name is not None,
        skip_trace_collection=args.skip_trace_collection,
        loader_workers=args.loader_workers,
        exclude_repos=args.exclude_repos,
    )

    runner_env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
        {"name": "HF_HOME", "value": "/workspace/cache/hf"},
        {"name": "HF_DATASETS_CACHE", "value": "/workspace/cache/hf/datasets"},
        {"name": "PIP_CACHE_DIR", "value": "/tmp/pip-cache"},
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": args.github_secret_name,
                    "key": args.github_secret_key,
                }
            },
        },
    ]

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
    if args.chromadb_pvc_name:
        volumes.append(
            {"name": "chromadb-pvc", "persistentVolumeClaim": {"claimName": args.chromadb_pvc_name}}
        )
        runner_mounts.append({"name": "chromadb-pvc", "mountPath": "/chromadb-pvc"})

    gpu_resources: dict[str, str] = {}
    if args.gpu and args.gpu != "0":
        gpu_resources = {"nvidia.com/gpu": args.gpu}

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

    container_resources: dict[str, Any] = {
        "requests": {"cpu": args.cpu_request, "memory": args.memory_request, **gpu_resources},
        "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit, **gpu_resources},
    }

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
                        "resources": container_resources,
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

    github_token = args.github_token or _resolve_github_token()

    _kubectl_apply(args.kubeconfig, yaml.safe_dump(
        _secret_manifest(namespace, args.github_secret_name, {args.github_secret_key: github_token}),
        sort_keys=False,
    ))
    _kubectl_apply(args.kubeconfig, resources_yaml)

    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Workspace PVC: {pvc_name}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c runner -f"
    )
    print("Results path: /workspace/repo/experiments/phase2/")

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
    trace_parallel: int,
    epochs: int,
    patience: int,
    train_ratio: float,
    device: str,
    has_cache_pvc: bool,
    has_chromadb_pvc: bool = False,
    skip_trace_collection: bool = False,
    loader_workers: int = 1,
    exclude_repos: str = "aws-cloudformation/cfn-lint,pylint-dev/pylint",
) -> str:
    all_task_ids = _load_all_task_ids()
    task_ids = " ".join(all_task_ids)

    cache_setup = ""
    if has_cache_pvc:
        cache_setup = dedent("""\
            # Reuse shared NFS caches (PVC mounted at /cache)
            for d in ssg chromadb repos; do
                if [ -L "/workspace/cache/$d" ]; then
                    echo "Already linked (skipping): /workspace/cache/$d -> $(readlink "/workspace/cache/$d")"
                    continue
                fi
                if [ -d "/cache/cache/$d" ] && [ "$(ls -A /cache/cache/$d 2>/dev/null)" ]; then
                    rmdir "/workspace/cache/$d" 2>/dev/null || true
                    ln -s "/cache/cache/$d" "/workspace/cache/$d"
                    echo "Linked /cache/cache/$d -> /workspace/cache/$d"
                fi
            done
            # Link ChromaDB Jina embeddings for offline feature lookup
            # Priority: chromadb-pvc mount > shared cache variants
            if [ -L "/workspace/cache/chromadb_jina" ]; then
                echo "Already linked (skipping): /workspace/cache/chromadb_jina -> $(readlink /workspace/cache/chromadb_jina)"
                JINA_LINKED=1
            else
                JINA_LINKED=0
                for chroma_base in /chromadb-pvc/cache /cache/cache; do
                    for name in chromadb_jinaai_jina-code-embeddings-0.5b chromadb_jina_256; do
                        if [ -d "$chroma_base/$name" ] && [ "$(ls -A "$chroma_base/$name" 2>/dev/null)" ]; then
                            ln -s "$chroma_base/$name" "/workspace/cache/chromadb_jina"
                            echo "Linked ChromaDB Jina: $chroma_base/$name"
                            JINA_LINKED=1
                            break 2
                        fi
                    done
                done
            fi
            if [ "$JINA_LINKED" -eq 0 ]; then
                echo "WARNING: No Jina ChromaDB found — will fall back to Phase 1 setup"
            fi
            # Fallback: link ssg/repos from chromadb PVC if not linked from shared cache
            for d in ssg repos; do
                if [ -L "/workspace/cache/$d" ]; then
                    continue
                fi
                if [ -d "/chromadb-pvc/cache/$d" ] && [ "$(ls -A /chromadb-pvc/cache/$d 2>/dev/null)" ]; then
                    rm -rf "/workspace/cache/$d"
                    ln -s "/chromadb-pvc/cache/$d" "/workspace/cache/$d"
                    echo "Linked (chromadb-pvc) /chromadb-pvc/cache/$d -> /workspace/cache/$d"
                fi
            done
            # Copy existing traces from chromadb PVC to avoid re-collecting them
            if [ -d "/chromadb-pvc/cache/traces" ]; then
                cp -n /chromadb-pvc/cache/traces/*.json /workspace/cache/traces/ 2>/dev/null || true
                COPIED=$(ls /workspace/cache/traces/*.json 2>/dev/null | wc -l)
                echo "Copied $COPIED existing trace files from previous run"
            fi
        """)

    if skip_trace_collection:
        trace_collection_block = dedent("""\
            echo "==========================================="
            echo "=== Stage 1: SKIPPED (--skip-trace-collection) ==="
            echo "==========================================="
            EXISTING_TRACES=$(ls /workspace/cache/traces/*.json 2>/dev/null | wc -l)
            echo "Reusing $EXISTING_TRACES existing trace files from workspace PVC"
            if [ "$EXISTING_TRACES" -eq 0 ]; then
                echo "ERROR: --skip-trace-collection set but no traces found in /workspace/cache/traces/"
                exit 1
            fi
        """)
    else:
        trace_collection_block = dedent(f"""\
            echo "==========================================="
            echo "=== Stage 1: Trace Collection ({len(all_task_ids)} tasks) ==="
            echo "==========================================="

            # Skip Phase 1 setup if SSGs, repos, and ChromaDB are already cached.
            SSG_COUNT=$(find /workspace/cache/ssg -name 'ssg.json' 2>/dev/null | head -20 | wc -l)
            REPO_COUNT=$(ls /workspace/cache/repos 2>/dev/null | wc -l)
            CHROMA_OK=0
            for d in /workspace/cache/chromadb_jina /workspace/cache/chromadb; do
                if [ -d "$d" ] && [ "$(ls -A "$d" 2>/dev/null)" ]; then
                    CHROMA_OK=1; break
                fi
            done
            if [ "$SSG_COUNT" -ge 10 ] && [ "$REPO_COUNT" -ge 10 ] && [ "$CHROMA_OK" -eq 1 ]; then
                echo "Phase 1 caches found ($SSG_COUNT SSGs, $REPO_COUNT repos, ChromaDB OK) — skipping setup"
            else
                echo "Phase 1 caches incomplete — running setup_phase1.py"
                python scripts/setup_phase1.py --provider sentence-transformers \\
                    --cache-dir /workspace/cache/repos \\
                    --ssg-dir /workspace/cache/ssg \\
                    --persist-dir "$CHROMADB_DIR"
            fi

            python scripts/phase2_collect_traces.py \\
                --manifest configs/phases/phase0_tasks.json \\
                --ssg-dir /workspace/cache/ssg \\
                --bare-repos-dir /workspace/cache/repos \\
                --checkouts-root /workspace/workdirs \\
                --output-dir /workspace/cache/traces \\
                --parallel {trace_parallel} \\
                --skip-existing \\
                --only {task_ids}

            echo "Trace collection complete:"
            ls -la /workspace/cache/traces/ || true
        """)

    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates git build-essential \
            pkg-config gfortran libopenblas-dev libffi-dev \
            libfreetype-dev libpng-dev
        rm -rf /var/lib/apt/lists/*

        # NFS mounts have different ownership; allow git everywhere
        git config --global --add safe.directory '*'

        # Create workspace root and Phase 2-only dirs first
        mkdir -p /workspace/cache /workspace/cache/traces \\
                 /workspace/cache/edge_pred/features \\
                 /workspace/cache/edge_pred/candidates /workspace/cache/edge_pred/checkpoints \\
                 /workspace/cache/hf /workspace/workdirs /tmp/pip-cache
        # NFS mounts may have different ownership; fix for workdirs and caches
        chown -R root:root /workspace/cache/traces \\
                           /workspace/cache/edge_pred /workspace/cache/hf /workspace/workdirs

        {cache_setup}

        # Create remaining dirs only if not already symlinked from Phase 1
        mkdir -p /workspace/cache/ssg /workspace/cache/chromadb /workspace/cache/repos

        # Clone repo
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu124.html
        python -m pip install -e ".[dev,research,data]" scikit-learn python-dotenv

        # Resolve ChromaDB dir early: prefer linked Jina cache, fall back to default
        CHROMADB_DIR="/workspace/cache/chromadb_jina"
        if [ ! -d "$CHROMADB_DIR" ] || [ -z "$(ls -A "$CHROMADB_DIR" 2>/dev/null)" ]; then
            CHROMADB_DIR="/workspace/cache/chromadb"
        fi
        echo "Using ChromaDB at: $CHROMADB_DIR"

        {trace_collection_block}

        echo "========================================="
        echo "=== Stage 2: Edge Predictor Training ==="
        echo "========================================="

        # Clear stale split, candidate, and feature cache so fresh ones are generated
        rm -f /workspace/cache/edge_pred/splits.json
        rm -rf /workspace/cache/edge_pred/candidates
        rm -rf /workspace/cache/edge_pred/features
        mkdir -p /workspace/cache/edge_pred/candidates /workspace/cache/edge_pred/features

        python scripts/phase2_train_edge_predictor.py \\
            --manifest configs/phases/phase0_tasks.json \\
            --ssg-dir /workspace/cache/ssg \\
            --traces-dir /workspace/cache/traces \\
            --split-file /workspace/cache/edge_pred/splits.json \\
            --output-dir /workspace/cache/edge_pred/checkpoints \\
            --feature-cache /workspace/cache/edge_pred/features \\
            --candidate-cache /workspace/cache/edge_pred/candidates \\
            --encoder chromadb --chromadb-dir "$CHROMADB_DIR" \\
            --train-ratio {train_ratio} \\
            --loader-workers {loader_workers} \\
            --exclude-repos "{exclude_repos}" \\
            --epochs {epochs} --patience {patience} --device {device}

        echo "Training complete:"
        ls -la /workspace/cache/edge_pred/checkpoints/

        echo "============================================"
        echo "=== Stage 3: Experiments 2A + 2B         ==="
        echo "============================================"

        python scripts/phase2_run_experiments.py \\
            --experiment 2A 2B \\
            --checkpoint /workspace/cache/edge_pred/checkpoints/best.pt \\
            --manifest configs/phases/phase0_tasks.json \\
            --split-file /workspace/cache/edge_pred/splits.json \\
            --ssg-dir /workspace/cache/ssg \\
            --traces-dir /workspace/cache/traces \\
            --feature-cache /workspace/cache/edge_pred/features \\
            --candidate-cache /workspace/cache/edge_pred/candidates \\
            --output-dir experiments/phase2 \\
            --persist-dir "$CHROMADB_DIR" \\
            --chromadb-dir "$CHROMADB_DIR" \\
            --loader-workers {loader_workers} \\
            --exclude-repos "{exclude_repos}" \\
            --device {device}

        echo "============================================"
        echo "=== Phase 2 Complete                     ==="
        echo "============================================"
        echo "Results:"
        ls -la experiments/phase2/
        echo ""
        cat experiments/phase2/DECISION.md 2>/dev/null || echo "(no DECISION.md yet)"
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
