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
    parser = argparse.ArgumentParser(description="Submit reranker training to Kubernetes")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--workspace-pvc", required=True, help="Existing PVC with SSGs + ChromaDB")
    parser.add_argument("--cache-pvc", default=None, help="Shared cache PVC (pip, HF)")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--voyage-secret-name", default="token-sniper-voyage")
    parser.add_argument("--voyage-secret-key", default="api-key")
    parser.add_argument("--embedding-provider", default="sentence-transformers",
                        help="Embedding provider: sentence-transformers (free, GPU) or voyageai (paid, API)")
    parser.add_argument("--embedding-model", default="jinaai/jina-code-embeddings-0.5b")
    parser.add_argument("--node-pool", default="default")
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="32Gi")
    parser.add_argument("--memory-limit", default="64Gi")
    parser.add_argument("--gpu", action="store_true", help="Request 1 GPU for neural reranker")
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--neural-epochs", type=int, default=80)
    parser.add_argument("--edge-pred-checkpoint", type=str, default=None,
                        help="Path (inside container) to Phase 2 edge predictor best.pt. "
                             "Enables SSG augmentation. Use with --phase2-pvc to mount the PVC.")
    parser.add_argument("--edge-pred-top-m", type=int, default=200,
                        help="Number of top predicted edges to add per SSG")
    parser.add_argument("--phase2-pvc", default=None,
                        help="Phase 2 workspace PVC name (mounted at /phase2). "
                             "Required when --edge-pred-checkpoint references /phase2/...")
    parser.add_argument("--chromadb-pvc", default=None,
                        help="PVC containing Jina ChromaDB embeddings (mounted at /chromadb-pvc)")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"reranker-train-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    if args.node_name:
        node_selector: dict[str, str] = {"kubernetes.io/hostname": args.node_name}
    elif args.gpu:
        node_selector = {}
    else:
        node_selector = {"node-role.kubernetes.io/runai-cpu-worker": "true"}

    node_pool = args.node_pool if args.node_pool != "default" else ("h100" if args.gpu else "default")
    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "reranker-training",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": node_pool,
    }

    runner_script = _build_runner_script(
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        neural_epochs=args.neural_epochs,
        edge_pred_checkpoint=args.edge_pred_checkpoint,
        edge_pred_top_m=args.edge_pred_top_m,
        has_cache_pvc=args.cache_pvc is not None,
        has_chromadb_pvc=args.chromadb_pvc is not None,
    )

    runner_env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
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
    if args.cache_pvc:
        runner_env.extend([
            {"name": "PIP_CACHE_DIR", "value": "/cache/runner/pip"},
            {"name": "HF_HOME", "value": "/cache/runner/huggingface"},
            {"name": "HF_HUB_CACHE", "value": "/cache/runner/huggingface/hub"},
            {"name": "HF_DATASETS_CACHE", "value": "/cache/runner/huggingface/datasets"},
            {"name": "XDG_CACHE_HOME", "value": "/cache/runner/xdg"},
        ])

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
    if args.phase2_pvc:
        volumes.append(
            {"name": "phase2", "persistentVolumeClaim": {"claimName": args.phase2_pvc, "readOnly": True}},
        )
        mounts.append({"name": "phase2", "mountPath": "/phase2", "readOnly": True})
    if args.chromadb_pvc:
        volumes.append(
            {"name": "chromadb-pvc", "persistentVolumeClaim": {"claimName": args.chromadb_pvc, "readOnly": True}},
        )
        mounts.append({"name": "chromadb-pvc", "mountPath": "/chromadb-pvc", "readOnly": True})

    resources_spec: dict[str, Any] = {
        "requests": {"cpu": args.cpu_request, "memory": args.memory_request},
        "limits": {"cpu": args.cpu_limit, "memory": args.memory_limit},
    }
    if args.gpu:
        resources_spec["limits"]["nvidia.com/gpu"] = "1"

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
                    "schedulerName": "runai-scheduler",
                    "nodeSelector": node_selector,
                    "volumes": volumes,
                    "containers": [{
                        "name": "trainer",
                        "image": args.runner_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-lc", runner_script],
                        "env": runner_env,
                        "resources": resources_spec,
                        "volumeMounts": mounts,
                    }],
                },
            },
        },
    }

    manifest_yaml = yaml.safe_dump(job_manifest, sort_keys=False)

    if args.output is not None:
        args.output.write_text(manifest_yaml, encoding="utf-8")

    if args.dry_run:
        sys.stdout.write(manifest_yaml)
        return

    _kubectl_apply(args.kubeconfig, manifest_yaml)

    print(f"Submitted job {job_name} in namespace {namespace}")
    print(f"Workspace PVC: {args.workspace_pvc}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c trainer -f"
    )
    print(f"Results: /workspace/repo/experiments/reranker/")

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
    embedding_provider: str,
    embedding_model: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    neural_epochs: int,
    edge_pred_checkpoint: str | None = None,
    edge_pred_top_m: int = 200,
    has_cache_pvc: bool = False,
    has_chromadb_pvc: bool = False,
) -> str:
    if edge_pred_checkpoint:
        augment_block = dedent(f"""\
            echo "==========================================="
            echo "=== Stage 2.5: Augment SSGs (edge pred) ==="
            echo "==========================================="
            python -m pip install torch_geometric
            python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu124.html 2>/dev/null || true
            python scripts/augment_ssgs.py \\
                --checkpoint {edge_pred_checkpoint} \\
                --manifest configs/phases/phase0_tasks.json \\
                --ssg-dir /workspace/cache/ssg \\
                --chromadb-dir "$CHROMADB_DIR" \\
                --output-dir /workspace/cache/ssg_augmented \\
                --feature-cache /workspace/cache/edge_pred/features \\
                --candidate-cache /workspace/cache/edge_pred/candidates \\
                --top-m {edge_pred_top_m} \\
                --device cuda
            SSG_DIR="/workspace/cache/ssg_augmented"
            echo "Augmented SSG repos: $(ls $SSG_DIR 2>/dev/null | wc -l)"
        """)
    else:
        augment_block = dedent("""\
            SSG_DIR="/workspace/cache/ssg"
        """)

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
            # Ensure a writable ChromaDB Jina copy (PersistentClient needs write access).
            # Remove any stale symlink from previous runs pointing to a read-only PVC.
            if [ -L "/workspace/cache/chromadb_jina" ]; then
                echo "Removing stale chromadb_jina symlink -> $(readlink /workspace/cache/chromadb_jina)"
                rm -f "/workspace/cache/chromadb_jina"
            fi
            JINA_LINKED=0
            if [ -d "/workspace/cache/chromadb_jina" ] && [ "$(ls -A /workspace/cache/chromadb_jina 2>/dev/null)" ]; then
                echo "Writable ChromaDB Jina already present at /workspace/cache/chromadb_jina"
                JINA_LINKED=1
            else
                for chroma_base in /chromadb-pvc/cache /cache/cache; do
                    for name in chromadb_jinaai_jina-code-embeddings-0.5b chromadb_jina_256; do
                        if [ -d "$chroma_base/$name" ] && [ "$(ls -A "$chroma_base/$name" 2>/dev/null)" ]; then
                            rm -rf "/workspace/cache/chromadb_jina"
                            echo "Copying ChromaDB Jina from $chroma_base/$name ..."
                            cp -r "$chroma_base/$name" "/workspace/cache/chromadb_jina"
                            chmod -R u+w "/workspace/cache/chromadb_jina"
                            echo "Copied ChromaDB Jina ($(du -sh /workspace/cache/chromadb_jina | cut -f1))"
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
        """)

    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*

        git config --global --add safe.directory '*'

        mkdir -p /workspace/cache /workspace/cache/edge_pred/features \\
                 /workspace/cache/edge_pred/candidates

        {cache_setup}

        # Create remaining dirs only if not already symlinked
        for d in ssg chromadb repos; do
            [ -e "/workspace/cache/$d" ] || [ -L "/workspace/cache/$d" ] || mkdir -p "/workspace/cache/$d"
        done

        # Clone repo
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        python -m pip install --upgrade pip
        if command -v nvidia-smi &>/dev/null; then
            python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        else
            python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        fi
        python -m pip install -e ".[dev,retrieval,data,experiments]"
        python -m pip install xgboost rank-bm25
        if [ "{embedding_provider}" != "voyageai" ]; then
            python -m pip install "sentence-transformers>=5.0.0" "transformers>=4.53.0"
            python -m pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not available"
        fi

        echo "==========================================="
        echo "=== Stage 1: Regenerate 500-task manifest ==="
        echo "==========================================="
        python scripts/build_phase0_manifest.py --max-tasks 0
        echo "Manifest tasks: $(python3 -c 'import json; print(len(json.load(open(\"configs/phases/phase0_tasks.json\"))))')"

        # Resolve ChromaDB dir: prefer linked Jina cache, fall back to default
        CHROMADB_DIR="/workspace/cache/chromadb_jina"
        if [ ! -d "$CHROMADB_DIR" ] || [ -z "$(ls -A "$CHROMADB_DIR" 2>/dev/null)" ]; then
            CHROMADB_DIR="/workspace/cache/chromadb"
        fi
        echo "Using ChromaDB at: $CHROMADB_DIR"

        echo "==========================================="
        echo "=== Stage 2: Build missing SSGs + index ==="
        echo "==========================================="
        SSG_COUNT=$(ls /workspace/cache/ssg 2>/dev/null | wc -l)
        CHROMA_OK=0
        for d in /workspace/cache/chromadb_jina /workspace/cache/chromadb; do
            if [ -d "$d" ] && [ "$(ls -A "$d" 2>/dev/null)" ]; then
                CHROMA_OK=1; break
            fi
        done
        echo "Guard check: SSG_dirs=$SSG_COUNT chroma=$CHROMA_OK"
        if [ "$SSG_COUNT" -ge 5 ] && [ "$CHROMA_OK" -eq 1 ]; then
            echo "Phase 1 caches found — skipping setup_phase1.py"
        else
            echo "Phase 1 caches incomplete — running setup_phase1.py"
            # Remove broken symlinks that would crash pathlib.mkdir
            [ -L "/workspace/cache/repos" ] && [ ! -d "/workspace/cache/repos" ] && rm -f "/workspace/cache/repos"
            python scripts/setup_phase1.py \\
                --provider {embedding_provider} \\
                --cache-dir /workspace/cache/repos \\
                --ssg-dir /workspace/cache/ssg \\
                --persist-dir "$CHROMADB_DIR"
        fi

        {augment_block}

        echo "==========================================="
        echo "=== Stage 3: Reranker Training           ==="
        echo "==========================================="
        python scripts/train_reranker.py \\
            --manifest configs/phases/phase0_tasks.json \\
            --split-file configs/phases/reranker_split.json \\
            --ssg-dir "$SSG_DIR" \\
            --chromadb-dir "$CHROMADB_DIR" \\
            --output-dir experiments/reranker \\
            --embedding-provider {embedding_provider} \\
            --embedding-model {embedding_model} \\
            --n-estimators {n_estimators} \\
            --max-depth {max_depth} \\
            --learning-rate {learning_rate} \\
            --neural-epochs {neural_epochs}

        echo "==========================================="
        echo "=== Training Complete                    ==="
        echo "==========================================="
        echo "Results:"
        cat experiments/reranker/summary.json
        ls -la experiments/reranker/
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
