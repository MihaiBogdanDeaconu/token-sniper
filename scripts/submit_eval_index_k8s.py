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
    parser = argparse.ArgumentParser(description="Index eval datasets on the eval PVC")
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--eval-pvc", default="token-sniper-eval-cache")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--node-pool", default="a100")
    parser.add_argument("--cpu-request", default="8")
    parser.add_argument("--cpu-limit", default="16")
    parser.add_argument("--memory-request", default="32Gi")
    parser.add_argument("--memory-limit", default="64Gi")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", default="24h")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    namespace = args.namespace or _namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"eval-index-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])
    _require_pushed(git_ref)

    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "eval-index",
        "runai/node-pool": args.node_pool,
    }

    runner_script = dedent("""\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ca-certificates git
        rm -rf /var/lib/apt/lists/*
        git config --global --add safe.directory '*'

        auth_url="$(python3 -c 'import os; url=os.environ["TOKEN_SNIPER_REPO_URL"]; tok=os.environ["GITHUB_TOKEN"]; host=url.split("://",1)[-1]; print("https://x-access-token:"+tok+"@"+host)')"
        rm -rf /workspace/repo
        git clone "$auth_url" /workspace/repo
        unset auth_url
        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${PYTHONPATH:+:${PYTHONPATH}}"

        python -m pip install --upgrade pip
        if command -v nvidia-smi >/dev/null 2>&1; then
            python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
        else
            python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        fi
        python -m pip install -e ".[retrieval,data,experiments]"
        python -m pip install "sentence-transformers>=5.0.0"

        python3 - <<'PY'
        import json
        from pathlib import Path

        pin = json.loads(Path("configs/phases/eval.json").read_text())
        specs = [
            ("lite_274", "swebench_lite"),
            ("verified_500", "swebench_verified"),
            ("locbench_560", "locbench"),
        ]
        out_dir = Path("/eval/manifests")
        out_dir.mkdir(parents=True, exist_ok=True)
        union = []
        for key, raw_key in specs:
            selected = set(pin[key]["instance_ids"])
            rows = []
            with (Path("/eval/raw") / raw_key / "instances.jsonl").open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["instance_id"] not in selected:
                        continue
                    files = []
                    for ln in str(row.get("patch") or "").splitlines():
                        if ln.startswith("+++ b/"):
                            path = ln[6:].strip()
                            if path and path != "/dev/null":
                                files.append(path)
                    files = sorted(set(files))
                    n = len(files)
                    bucket = "single_file" if n <= 1 else ("multi_file_2_3" if n <= 3 else "multi_file_4_plus")
                    rows.append({
                        "instance_id": row["instance_id"],
                        "repo": row["repo"],
                        "base_commit": row["base_commit"],
                        "difficulty": row.get("difficulty") or None,
                        "gold_patch_files": files,
                        "gold_patch_file_count": n,
                        "bucket": bucket,
                    })
            missing = selected - {r["instance_id"] for r in rows}
            if missing:
                raise SystemExit(f"{key}: {len(missing)} IDs missing from raw jsonl")
            (out_dir / f"{key}_tasks.json").write_text(json.dumps(rows, indent=2) + "\\n")
            union.extend(rows)
            print(key, len(rows))
        (out_dir / "all_tasks.json").write_text(json.dumps(union, indent=2) + "\\n")
        print("union", len(union))
        PY

        python scripts/setup_phase1.py \\
            --manifest /eval/manifests/all_tasks.json \\
            --cache-dir /eval/repos \\
            --ssg-dir /eval/ssg \\
            --persist-dir /eval/chromadb \\
            --provider sentence-transformers \\
            --workers 1

        python3 - <<'PY'
        import sys
        from pathlib import Path
        sys.path.insert(0, "/workspace/repo/scripts")
        from train_reranker import _save_inputs, build_reranker_inputs
        from token_sniper.evaluation.phase0 import load_phase0_manifest
        from token_sniper.retrieval.config import EmbeddingConfig
        from token_sniper.retrieval.embeddings import create_embedder
        from token_sniper.retrieval.retriever import BM25Retriever, DenseRetriever
        from token_sniper.retrieval.vectorstore import EntityVectorStore

        embedder = create_embedder(EmbeddingConfig(
            provider="sentence-transformers",
            name="jinaai/jina-code-embeddings-0.5b",
        ))
        store = EntityVectorStore(persist_directory="/eval/chromadb", collection_prefix="token_sniper")
        dense = DenseRetriever(embedder=embedder, vector_store=store)
        bm25 = BM25Retriever(vector_store=store)
        dest_dir = Path("/eval/reranker_inputs")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for key, dataset, split in (
            ("lite_274", "princeton-nlp/SWE-bench_Lite", "test"),
            ("verified_500", "princeton-nlp/SWE-bench_Verified", "test"),
            ("locbench_560", "czlll/Loc-Bench_V1", "test"),
        ):
            dest = dest_dir / f"{key}.json"
            if dest.exists():
                print("exists", dest)
                continue
            tasks = load_phase0_manifest(f"/eval/manifests/{key}_tasks.json").tasks
            inputs = build_reranker_inputs(
                tasks=tasks,
                ssg_dir=Path("/eval/ssg"),
                embedder=embedder,
                vector_store=store,
                retriever=dense,
                bm25_retriever=bm25,
                dataset_name=dataset,
                split=split,
                label=key,
            )
            _save_inputs(inputs, dest)
        PY
    """)

    resources: list[dict[str, Any]] = [
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
                        "schedulerName": "runai-scheduler",
                        "volumes": [
                            {"name": "eval", "persistentVolumeClaim": {"claimName": args.eval_pvc}},
                        ],
                        "containers": [{
                            "name": "indexer",
                            "image": args.runner_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", runner_script],
                            "env": [
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "TOKEN_SNIPER_REPO_URL", "value": repo_url},
                                {"name": "TOKEN_SNIPER_GIT_REF", "value": git_ref},
                                {"name": "HF_HOME", "value": "/eval/hf"},
                                {"name": "TRANSFORMERS_CACHE", "value": "/eval/hf/transformers"},
                                {
                                    "name": "GITHUB_TOKEN",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": args.github_secret_name,
                                            "key": args.github_secret_key,
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": args.cpu_request,
                                    "memory": args.memory_request,
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": args.cpu_limit,
                                    "memory": args.memory_limit,
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "volumeMounts": [
                                {"name": "eval", "mountPath": "/eval"},
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

    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not github_token:
        try:
            github_token = _git_output(["gh", "auth", "token"])
        except (FileNotFoundError, subprocess.CalledProcessError):
            github_token = ""
    if github_token:
        secret_yaml = yaml.safe_dump({
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": args.github_secret_name, "namespace": namespace},
            "type": "Opaque",
            "stringData": {args.github_secret_key: github_token},
        }, sort_keys=False)
        _kubectl_apply(args.kubeconfig, secret_yaml)
    _kubectl_apply(args.kubeconfig, manifests_yaml)
    print(f"Submitted job {job_name} in namespace {namespace}")
    print(
        f"Follow logs: kubectl --kubeconfig {args.kubeconfig} "
        f"logs -n {namespace} job/{job_name} -c indexer -f"
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


def _git_output(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _require_pushed(git_ref: str) -> None:
    dirty = _git_output(["git", "status", "--porcelain"])
    if dirty:
        raise SystemExit("Working tree is dirty. Commit and push before submitting.")
    remote_branches = _git_output(["git", "branch", "-r", "--contains", git_ref])
    if not remote_branches:
        raise SystemExit(f"{git_ref} is not on any remote branch. Push before submitting.")


def _kubectl_apply(kubeconfig: Path, manifest_yaml: str) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest_yaml,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
