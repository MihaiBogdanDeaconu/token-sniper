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
    parser = argparse.ArgumentParser(description="Submit HALT training to Kubernetes")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--workspace-pvc", required=True,
                        help="Existing PVC with halt_data logprobs")
    parser.add_argument("--cache-pvc", default=None,
                        help="Shared cache PVC (pip, HF)")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--node-pool", default="default")
    parser.add_argument("--cpu-request", default="4")
    parser.add_argument("--cpu-limit", default="8")
    parser.add_argument("--memory-request", default="16Gi")
    parser.add_argument("--memory-limit", default="32Gi")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Request 1x GPU (small model trains fine on CPU)")
    parser.add_argument("--node-name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune on Collu-Bench from pretrained checkpoint")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"halt-train-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])

    if args.node_name:
        node_selector: dict[str, str] = {"kubernetes.io/hostname": args.node_name}
    else:
        node_selector = {}

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "halt-training",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": args.node_pool,
    }

    runner_script = _build_runner_script(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        gpu=args.gpu,
        finetune=args.finetune,
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
    print(f"Results: /workspace/repo/experiments/guardrails/")

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
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    gpu: bool,
    finetune: bool = False,
) -> str:
    torch_install = (
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cu124"
        if gpu else
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

    data_args = ""
    if finetune:
        data_args = (
            '    --collu-bench-dir /workspace/cache/halt_data/collu_bench \\\n'
            '    --checkpoint /workspace/cache/halt_checkpoints/pretrain_best.pt \\\n'
            '    --no-pos-weight \\\n'
        )
        stage_name = "HALT Fine-tune (Collu-Bench)"
    else:
        data_args = '    --data-dir "$DATA_DIR" \\\n'
        stage_name = "HALT Classifier Training"

    pretrain_block = ""
    if finetune:
        pretrain_block = dedent("""\

        # Stage 0: ensure pretrained checkpoint exists
        CKPT_DIR="/workspace/cache/halt_checkpoints"
        mkdir -p "$CKPT_DIR"
        if [ ! -f "$CKPT_DIR/pretrain_best.pt" ]; then
            echo "No pretrained checkpoint found — running HaluEval pretrain first"
            python scripts/train_halt.py \\
                --data-dir "$DATA_DIR" \\
                --output-dir /tmp/halt_pretrain \\
                --epochs 30 --batch-size 64 --lr 0.0001 --patience 7
            cp /tmp/halt_pretrain/best.pt "$CKPT_DIR/pretrain_best.pt"
            echo "Pretrain checkpoint saved to $CKPT_DIR/pretrain_best.pt"
        else
            echo "Found pretrained checkpoint at $CKPT_DIR/pretrain_best.pt"
        fi
        """)

    return dedent(f"""\
        set -euo pipefail

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*

        git config --global --add safe.directory '*'

        # Clone repo
        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        python -m pip install --upgrade pip
        {torch_install}
        python -m pip install scikit-learn datasets
        python -m pip install -e ".[dev]"

        DATA_DIR="/workspace/cache/halt_data/logprobs_gen"
{pretrain_block}
        echo "==========================================="
        echo "=== {stage_name}  ==="
        echo "==========================================="

        echo "HaluEval dir: $DATA_DIR"
        ls -lh "$DATA_DIR/" 2>/dev/null || echo "HaluEval data not found"

        COLLU_DIR="/workspace/cache/halt_data/collu_bench"
        echo "Collu-Bench dir: $COLLU_DIR"
        ls "$COLLU_DIR/" 2>/dev/null || echo "Collu-Bench data not found"

        python scripts/train_halt.py \\
        {data_args}    --output-dir experiments/guardrails \\
            --epochs {epochs} \\
            --batch-size {batch_size} \\
            --lr {lr} \\
            --patience {patience}

        # Cache the fine-tuned checkpoint to PVC for inference jobs
        if [ -f experiments/guardrails/best.pt ]; then
            CKPT_DIR="/workspace/cache/halt_checkpoints"
            mkdir -p "$CKPT_DIR"
            cp experiments/guardrails/best.pt "$CKPT_DIR/finetune_best.pt"
            echo "Fine-tune checkpoint cached to $CKPT_DIR/finetune_best.pt"
        fi

        echo "==========================================="
        echo "=== Training Complete                   ==="
        echo "==========================================="
        echo "Results:"
        cat experiments/guardrails/summary.json
        ls -la experiments/guardrails/
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
