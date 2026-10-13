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
    parser = argparse.ArgumentParser(description="Submit VGIB training to Kubernetes")
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
    parser.add_argument("--embedding-model", default="jinaai/jina-code-embeddings-0.5b")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--node-pool", default="h100")
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="64Gi")
    parser.add_argument("--memory-limit", default="128Gi")
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--reranker-dir", default="experiments/reranker",
                        help="Path to reranker output dir (inside repo checkout)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.01, help="Token ratio loss weight")
    parser.add_argument("--gamma", type=float, default=0.001, help="MI loss weight")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-graph-edges", action="store_true",
                        help="Ablation: message passing sees no edges, node features unchanged")
    parser.add_argument("--pooling", choices=("cross_attention", "mean"), default="cross_attention",
                        help="Ablation: mean pooling instead of cross-attention")
    parser.add_argument("--no-lexical-features", action="store_true",
                        help="Ablation: drop the issue-conditioned lexical node features")
    parser.add_argument("--prior-weight", type=float, default=0.0,
                        help="Structural prior KL weight (0 = off)")
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-gamma-s", type=float, default=1.0)
    parser.add_argument("--prior-base-rate", type=float, default=0.5)
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
    parser.add_argument("--reranker-pvc", default=None,
                        help="Reranker workspace PVC (mounted at /reranker, read-only). "
                             "Reuses augmented SSGs, ChromaDB copy, and reranker outputs. "
                             "When set, --edge-pred-checkpoint is ignored (augmentation already done).")
    parser.add_argument("--run-rl", action="store_true",
                        help="After VGIB training, run GRPO RL fine-tuning on the VGIB checkpoint")
    parser.add_argument("--rl-epochs", type=int, default=50)
    parser.add_argument("--rl-lr", type=float, default=1e-5)
    parser.add_argument("--rl-group-size", type=int, default=8)
    parser.add_argument("--rl-patience", type=int, default=10)
    parser.add_argument("--rl-reward-beta", type=float, default=0.5)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.reranker_pvc and not (args.cache_pvc and args.phase2_pvc):
        parser.error(
            "--reranker-pvc requires --cache-pvc (plain SSGs) and --phase2-pvc "
            "(cached repos for token counting)"
        )

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"vgib-train-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    if args.node_name:
        node_selector: dict[str, str] = {"kubernetes.io/hostname": args.node_name}
    else:
        node_selector = {}

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "vgib-training",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": args.node_pool,
    }

    runner_script = _build_runner_script(
        job_name=job_name,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        reranker_dir=args.reranker_dir,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        patience=args.patience,
        model_flags=_model_flags(args),
        edge_pred_checkpoint=args.edge_pred_checkpoint if not args.reranker_pvc else None,
        edge_pred_top_m=args.edge_pred_top_m,
        has_cache_pvc=args.cache_pvc is not None,
        has_chromadb_pvc=args.chromadb_pvc is not None,
        has_reranker_pvc=args.reranker_pvc is not None,
        run_rl=args.run_rl,
        rl_epochs=args.rl_epochs,
        rl_lr=args.rl_lr,
        rl_group_size=args.rl_group_size,
        rl_patience=args.rl_patience,
        rl_reward_beta=args.rl_reward_beta,
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
    if args.reranker_pvc:
        volumes.append(
            {"name": "reranker", "persistentVolumeClaim": {"claimName": args.reranker_pvc, "readOnly": True}},
        )
        mounts.append({"name": "reranker", "mountPath": "/reranker", "readOnly": True})

    resources_spec: dict[str, Any] = {
        "requests": {"cpu": args.cpu_request, "memory": args.memory_request},
        "limits": {
            "cpu": args.cpu_limit,
            "memory": args.memory_limit,
            "nvidia.com/gpu": "1",
        },
    }

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
    if args.reranker_pvc:
        print(f"Reranker PVC (read-only): {args.reranker_pvc}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c trainer -f"
    )
    stages = ["VGIB → experiments/vgib/"]
    if args.run_rl:
        stages.append("RL → experiments/rl_grpo/")
    print(f"Results: {' | '.join(stages)}")

    if args.wait:
        subprocess.run(
            [
                "kubectl", "--kubeconfig", str(args.kubeconfig),
                "wait", "--for=condition=complete", f"--timeout={args.timeout}",
                "-n", namespace, f"job/{job_name}",
            ],
            check=True,
        )


def _model_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.no_graph_edges:
        flags.append("--no-graph-edges")
    if args.pooling != "cross_attention":
        flags.append(f"--pooling {args.pooling}")
    if args.no_lexical_features:
        flags.append("--no-lexical-features")
    if args.prior_weight > 0:
        flags += [
            f"--prior-weight {args.prior_weight}",
            f"--prior-alpha {args.prior_alpha}",
            f"--prior-gamma-s {args.prior_gamma_s}",
            f"--prior-base-rate {args.prior_base_rate}",
        ]
    return flags


def _build_runner_script(
    *,
    job_name: str,
    embedding_model: str,
    embedding_dim: int,
    reranker_dir: str,
    epochs: int,
    lr: float,
    beta: float,
    gamma: float,
    patience: int,
    model_flags: list[str] | None = None,
    edge_pred_checkpoint: str | None = None,
    edge_pred_top_m: int = 200,
    has_cache_pvc: bool = False,
    has_chromadb_pvc: bool = False,
    has_reranker_pvc: bool = False,
    run_rl: bool = False,
    rl_epochs: int = 50,
    rl_lr: float = 1e-5,
    rl_group_size: int = 8,
    rl_patience: int = 10,
    rl_reward_beta: float = 0.5,
) -> str:
    model_flag_lines = "".join(f" \\\n            {flag}" for flag in model_flags or [])

    if has_reranker_pvc:
        data_setup = dedent("""\
            echo "==========================================="
            echo "=== Reranker PVC: Reusing Phase 2+3 artifacts ==="
            echo "==========================================="

            # ChromaDB: copy from reranker PVC (PersistentClient needs write access)
            if [ -L "/workspace/cache/chromadb_jina" ]; then
                rm -f "/workspace/cache/chromadb_jina"
            fi
            if [ -d "/workspace/cache/chromadb_jina" ] && [ "$(ls -A /workspace/cache/chromadb_jina 2>/dev/null)" ]; then
                echo "Writable ChromaDB Jina already present at /workspace/cache/chromadb_jina"
            elif [ -d "/reranker/cache/chromadb_jina" ] && [ "$(ls -A /reranker/cache/chromadb_jina 2>/dev/null)" ]; then
                rm -rf "/workspace/cache/chromadb_jina"
                echo "Copying ChromaDB Jina from reranker PVC..."
                cp -r "/reranker/cache/chromadb_jina" "/workspace/cache/chromadb_jina"
                chmod -R u+w "/workspace/cache/chromadb_jina"
                echo "Copied ChromaDB Jina ($(du -sh /workspace/cache/chromadb_jina | cut -f1))"
            else
                echo "WARNING: No ChromaDB found on reranker PVC"
            fi

            CHROMADB_DIR="/workspace/cache/chromadb_jina"
            if [ ! -d "$CHROMADB_DIR" ] || [ -z "$(ls -A "$CHROMADB_DIR" 2>/dev/null)" ]; then
                CHROMADB_DIR="/workspace/cache/chromadb"
            fi
            echo "Using ChromaDB at: $CHROMADB_DIR"

            # Plain SSGs from the shared cache: the graphs the reranker was trained on.
            # Token count sidecars are built next to them.
            SSG_DIR="/cache/cache/ssg"
            if [ ! -d "$SSG_DIR" ]; then
                echo "ERROR: $SSG_DIR not found on the shared cache PVC"
                exit 1
            fi
            echo "SSG repos: $(ls $SSG_DIR | wc -l)"

            # Reranker outputs: find train_inputs.json on the reranker PVC
            RERANKER_DIR=""
            for loc in "/reranker/cache/reranker_outputs" "/reranker/repo/experiments/reranker"; do
                if [ -d "$loc" ] && [ -f "$loc/train_inputs.json" ]; then
                    RERANKER_DIR="$loc"
                    break
                fi
            done
            if [ -z "$RERANKER_DIR" ]; then
                echo "ERROR: No reranker outputs (train_inputs.json) found on reranker PVC!"
                echo "Searched: /reranker/cache/reranker_outputs, /reranker/repo/experiments/reranker"
                exit 1
            fi
            echo "Reranker outputs at: $RERANKER_DIR"
            ls -la "$RERANKER_DIR/"
        """)
    else:
        cache_setup = ""
        if has_cache_pvc:
            cache_setup = dedent("""\
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
                for d in ssg repos; do
                    if [ -L "/workspace/cache/$d" ]; then continue; fi
                    if [ -d "/chromadb-pvc/cache/$d" ] && [ "$(ls -A /chromadb-pvc/cache/$d 2>/dev/null)" ]; then
                        rm -rf "/workspace/cache/$d"
                        ln -s "/chromadb-pvc/cache/$d" "/workspace/cache/$d"
                        echo "Linked (chromadb-pvc) /chromadb-pvc/cache/$d -> /workspace/cache/$d"
                    fi
                done
            """)

        if edge_pred_checkpoint:
            augment_block = dedent(f"""\
                echo "==========================================="
                echo "=== Stage 1.5: Augment SSGs (edge pred) ==="
                echo "==========================================="
                if [ -d "/workspace/cache/ssg_augmented" ] && [ "$(ls /workspace/cache/ssg_augmented 2>/dev/null | head -3)" ]; then
                    echo "Augmented SSGs already exist — reusing"
                    SSG_DIR="/workspace/cache/ssg_augmented"
                else
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
                fi
                echo "Augmented SSGs: $(ls $SSG_DIR 2>/dev/null | wc -l) repos"
            """)
        else:
            augment_block = 'SSG_DIR="/workspace/cache/ssg"'

        data_setup = dedent(f"""\
            {cache_setup}

            for d in ssg chromadb repos; do
                [ -e "/workspace/cache/$d" ] || [ -L "/workspace/cache/$d" ] || mkdir -p "/workspace/cache/$d"
            done

            RERANKER_SRC="$REPO/{reranker_dir}"
            RERANKER_SAFE="/workspace/cache/reranker_outputs"
            if [ -d "$RERANKER_SRC" ] && [ ! -d "$RERANKER_SAFE" ]; then
                echo "Preserving reranker outputs: $RERANKER_SRC -> $RERANKER_SAFE"
                cp -r "$RERANKER_SRC" "$RERANKER_SAFE"
            fi

            CHROMADB_DIR="/workspace/cache/chromadb_jina"
            if [ ! -d "$CHROMADB_DIR" ] || [ -z "$(ls -A "$CHROMADB_DIR" 2>/dev/null)" ]; then
                CHROMADB_DIR="/workspace/cache/chromadb"
            fi
            echo "Using ChromaDB at: $CHROMADB_DIR"

            {augment_block}

            RERANKER_DIR="/workspace/cache/reranker_outputs"
            if [ -d "$RERANKER_DIR" ]; then
                echo "Reranker outputs found at $RERANKER_DIR"
                ls -la "$RERANKER_DIR/"
            else
                echo "Reranker outputs not found — will generate candidates via retrieval"
            fi
        """)

    if run_rl:
        rl_block = dedent(f"""\
            echo "==========================================="
            echo "=== Stage 3: GRPO RL Fine-tuning        ==="
            echo "==========================================="
            echo "Warm-starting from VGIB checkpoint: experiments/vgib/best.pt"

            python scripts/train_rl.py \\
                --warm-start experiments/vgib/best.pt \\
                --split-json configs/phases/reranker_split.json \\
                --manifest configs/phases/phase0_tasks.json \\
                --reranker-dir "$RERANKER_DIR" \\
                --ssg-dir "$SSG_DIR" \\
                --chromadb-dir "$CHROMADB_DIR" \\
                --output-dir experiments/rl_grpo \\
                --feature-cache-dir /workspace/cache/vgib_features \\
                --issue-cache-dir /workspace/cache/issue_emb \\
                --embedding-model {embedding_model} \\
                --embedding-dim {embedding_dim} \\
                --epochs {rl_epochs} \\
                --lr {rl_lr} \\
                --group-size {rl_group_size} \\
                --patience {rl_patience} \\
                --reward-beta {rl_reward_beta} \\
                --device cuda

            echo "==========================================="
            echo "=== RL Training Complete                 ==="
            echo "==========================================="
            echo "RL Results:"
            cat experiments/rl_grpo/summary.json 2>/dev/null || true
            ls -la experiments/rl_grpo/
        """)
    else:
        rl_block = ""

    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*

        git config --global --add safe.directory '*'

        mkdir -p /workspace/cache /workspace/cache/edge_pred/features \\
                 /workspace/cache/edge_pred/candidates \\
                 /workspace/cache/vgib_features /workspace/cache/issue_emb

        # Clone into this job's own directory.  Nothing from earlier runs is
        # deleted or moved: their checkouts, experiments and checkpoints stay
        # exactly where they are.
        REPO=/workspace/runs/{job_name}/repo
        mkdir -p /workspace/runs/{job_name}
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        git clone "$auth_url" $REPO
        unset auth_url
        cd $REPO
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="$REPO${{PYTHONPATH:+:${{PYTHONPATH}}}}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # Seed the shared caches from earlier runs -- they are content-addressed,
        # so still valid.  Copy, never move: the originals stay in place.  This
        # is what spares the next run the hours of feature extraction.
        for d in /workspace/repo/experiments/*/feature_cache /workspace/runs/*/repo/experiments/*/feature_cache; do
            [ -d "$d" ] && cp -rn "$d"/. /workspace/cache/vgib_features/ || true
        done
        for d in /workspace/repo/experiments/*/issue_cache /workspace/runs/*/repo/experiments/*/issue_cache; do
            [ -d "$d" ] && cp -rn "$d"/. /workspace/cache/issue_emb/ || true
        done
        echo "Feature cache: $(ls /workspace/cache/vgib_features | wc -l) files"
        echo "Issue cache: $(ls /workspace/cache/issue_emb | wc -l) files"

        python -m pip install --upgrade pip
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        python -m pip install -e ".[dev,retrieval,data,experiments]"
        python -m pip install "sentence-transformers>=5.0.0" "transformers>=4.53.0"
        python -m pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not available"
        python -m pip install torch_geometric
        python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \\
            -f https://data.pyg.org/whl/torch-2.6.0+cu124.html 2>/dev/null || \\
            echo "Some PyG extensions not available — SAGEConv still works"

        echo "==========================================="
        echo "=== Stage 1: Regenerate task manifest   ==="
        echo "==========================================="
        python scripts/build_phase0_manifest.py --max-tasks 0
        echo "Manifest tasks: $(python3 -c 'import json; print(len(json.load(open("configs/phases/phase0_tasks.json"))))')"

        {data_setup}

        echo "==========================================="
        echo "=== Stage 1.9: Token count sidecars     ==="
        echo "==========================================="
        python scripts/build_token_sidecars.py \\
            --manifest configs/phases/phase0_tasks.json \\
            --split-json configs/phases/reranker_split.json \\
            --ssg-dir "$SSG_DIR" \\
            --cache-dir /phase2/cache/repos

        echo "==========================================="
        echo "=== Stage 2: VGIB Compression Training  ==="
        echo "==========================================="
        echo "ChromaDB dir: $CHROMADB_DIR"
        echo "SSG dir: $SSG_DIR"
        echo "Reranker dir: $RERANKER_DIR"
        echo "Mode: compression (Phase C)"

        python scripts/train_vgib.py \\
            --manifest configs/phases/phase0_tasks.json \\
            --split-json configs/phases/reranker_split.json \\
            --reranker-dir "$RERANKER_DIR" \\
            --ssg-dir "$SSG_DIR" \\
            --chromadb-dir "$CHROMADB_DIR" \\
            --output-dir experiments/vgib \\
            --feature-cache-dir /workspace/cache/vgib_features \\
            --issue-cache-dir /workspace/cache/issue_emb \\
            --embedding-model {embedding_model} \\
            --embedding-dim {embedding_dim} \\
            --epochs {epochs} \\
            --lr {lr} \\
            --beta {beta} \\
            --gamma {gamma} \\
            --patience {patience}{model_flag_lines}

        echo "==========================================="
        echo "=== VGIB Training Complete              ==="
        echo "==========================================="
        echo "VGIB Results:"
        cat experiments/vgib/summary.json 2>/dev/null || true
        ls -la experiments/vgib/

        {rl_block}

        echo "==========================================="
        echo "=== All training complete               ==="
        echo "==========================================="
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
