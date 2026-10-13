from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from token_sniper.agent.config import load_base_config, load_yaml_config
from token_sniper.evaluation.k8s import (
    Phase0KubernetesJobConfig,
    dump_phase0_job_resources,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", type=Path, default=Path("kubelet.yaml"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--phase-config",
        type=Path,
        default=Path("configs/phases/phase0_oracle.yaml"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--pvc-name", default=None)
    parser.add_argument("--node-pool", default="default")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--git-ref", default=None)
    parser.add_argument("--runner-image", default="python:3.12-slim")
    parser.add_argument(
        "--skip-runtime-install",
        action="store_true",
        help="Assume the runner image already contains OS and Python dependencies.",
    )
    parser.add_argument("--storage-class", default="local-path")
    parser.add_argument("--storage-size", default="200Gi")
    parser.add_argument("--cache-pvc-name", default=None)
    parser.add_argument(
        "--create-cache-pvc",
        action="store_true",
        help="Create a reusable cache PVC for Hugging Face, repo, and Docker caches.",
    )
    parser.add_argument("--cache-storage-class", default="local-path")
    parser.add_argument("--cache-storage-size", default="400Gi")
    parser.add_argument("--github-secret-name", default="token-sniper-github")
    parser.add_argument("--github-secret-key", default="token")
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--qwen-base-url", default="http://qwen-vllm:8077/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["blind_agent", "oracle_retrieval"],
        default=["blind_agent", "oracle_retrieval"],
        help="Subset of Phase 0 conditions to execute in this job.",
    )
    parser.add_argument("--agent-workers", type=int, default=None)
    parser.add_argument("--condition-workers", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--runner-cpu-request", default="32")
    parser.add_argument("--runner-cpu-limit", default="64")
    parser.add_argument("--runner-memory-request", default="128Gi")
    parser.add_argument("--runner-memory-limit", default="192Gi")
    parser.add_argument("--dind-cpu-request", default="8")
    parser.add_argument("--dind-cpu-limit", default="16")
    parser.add_argument("--dind-memory-request", default="16Gi")
    parser.add_argument("--dind-memory-limit", default="32Gi")
    parser.add_argument("--node-name", default=None)
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the job to complete after submission.",
    )
    parser.add_argument(
        "--timeout",
        default="12h",
        help="Timeout passed to kubectl wait when --wait is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered PVC and Job YAML without creating resources.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the rendered PVC and Job YAML.",
    )
    args = parser.parse_args()

    base_config = load_yaml_config(args.base_config)
    agent_config = load_base_config(args.base_config)
    namespace = args.namespace or load_namespace_from_kubeconfig(args.kubeconfig)
    job_name = args.job_name or f"phase0-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    pvc_name = args.pvc_name or f"{job_name}-workspace"
    repo_url = args.repo_url or _git_output(["git", "remote", "get-url", "origin"])
    git_ref = args.git_ref or _git_output(["git", "rev-parse", "HEAD"])
    model_name = args.model or agent_config.models.agent.name
    agent_workers = _resolve_positive_override(
        override=args.agent_workers,
        fallback=base_config.get("evaluation", {}).get("agent_workers", 4),
        field_name="agent_workers",
    )
    condition_workers = _resolve_positive_override(
        override=args.condition_workers,
        fallback=base_config.get("evaluation", {}).get("condition_workers", 1),
        field_name="condition_workers",
    )
    harness_workers = _resolve_positive_override(
        override=args.max_workers,
        fallback=base_config.get("evaluation", {}).get("harness", {}).get("max_workers", 4),
        field_name="max_workers",
    )
    harness_batch_size = _resolve_positive_override(
        override=args.eval_batch_size,
        fallback=base_config.get("evaluation", {})
        .get("harness", {})
        .get("submission_batch_size", 8),
        field_name="eval_batch_size",
    )
    node_selector = (
        {"kubernetes.io/hostname": args.node_name}
        if args.node_name
        else {"node-role.kubernetes.io/runai-cpu-worker": "true"}
    )

    config = Phase0KubernetesJobConfig(
        namespace=namespace,
        job_name=job_name,
        pvc_name=pvc_name,
        node_pool=args.node_pool,
        repo_url=repo_url,
        git_ref=git_ref,
        github_secret_name=args.github_secret_name,
        github_secret_key=args.github_secret_key,
        runner_image=args.runner_image,
        install_repo_runtime=not args.skip_runtime_install,
        storage_class_name=args.storage_class,
        storage_size=args.storage_size,
        cache_pvc_name=args.cache_pvc_name,
        create_cache_pvc=args.create_cache_pvc,
        cache_storage_class_name=args.cache_storage_class,
        cache_storage_size=args.cache_storage_size,
        model_name=model_name,
        qwen_base_url=args.qwen_base_url,
        base_config_path=args.base_config.as_posix(),
        phase_config_path=args.phase_config.as_posix(),
        manifest_path=args.manifest.as_posix(),
        conditions=args.conditions,
        agent_workers=agent_workers,
        condition_workers=condition_workers,
        harness_workers=harness_workers,
        harness_batch_size=harness_batch_size,
        runner_cpu_request=args.runner_cpu_request,
        runner_cpu_limit=args.runner_cpu_limit,
        runner_memory_request=args.runner_memory_request,
        runner_memory_limit=args.runner_memory_limit,
        dind_cpu_request=args.dind_cpu_request,
        dind_cpu_limit=args.dind_cpu_limit,
        dind_memory_request=args.dind_memory_request,
        dind_memory_limit=args.dind_memory_limit,
        node_selector=node_selector,
    )

    resources_yaml = dump_phase0_job_resources(config)
    if args.output is not None:
        args.output.write_text(resources_yaml, encoding="utf-8")

    if args.dry_run:
        sys.stdout.write(resources_yaml)
        return

    github_token = args.github_token or resolve_github_token()
    secret_yaml = yaml.safe_dump(
        build_github_secret_manifest(
            namespace=namespace,
            secret_name=args.github_secret_name,
            secret_key=args.github_secret_key,
            token=github_token,
        ),
        sort_keys=False,
    )
    _kubectl_apply(args.kubeconfig, secret_yaml)
    _kubectl_apply(args.kubeconfig, resources_yaml)

    print(f"submitted job {job_name} in namespace {namespace}")
    print(f"workspace pvc: {pvc_name}")
    if args.cache_pvc_name:
        print(f"shared cache pvc: {args.cache_pvc_name}")
    print(
        f"follow logs: kubectl --kubeconfig {args.kubeconfig} logs -n {namespace} job/{job_name} -c runner -f"
    )
    cache_path = "/cache/swebench" if args.cache_pvc_name else "/workspace/repo/data/swebench"
    print(f"artifacts path inside pvc: /workspace/repo/experiments/phase0 and {cache_path}")

    if args.wait:
        _kubectl_wait(args.kubeconfig, namespace=namespace, job_name=job_name, timeout=args.timeout)


def load_namespace_from_kubeconfig(kubeconfig_path: Path) -> str:
    raw = yaml.safe_load(kubeconfig_path.read_text(encoding="utf-8")) or {}
    current_context_name = raw.get("current-context")
    for context_entry in raw.get("contexts", []):
        if context_entry.get("name") != current_context_name:
            continue
        context = context_entry.get("context", {})
        namespace = context.get("namespace")
        if isinstance(namespace, str) and namespace:
            return namespace
    return "default"


def build_github_secret_manifest(
    *,
    namespace: str,
    secret_name: str,
    secret_key: str,
    token: str,
) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "Opaque",
        "stringData": {secret_key: token},
    }


def resolve_github_token() -> str:
    for env_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value
    return _run_capture(["gh", "auth", "token"])


def _kubectl_apply(kubeconfig: Path, manifest_yaml: str) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
        input=manifest_yaml,
        text=True,
        check=True,
    )


def _kubectl_wait(kubeconfig: Path, *, namespace: str, job_name: str, timeout: str) -> None:
    subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "wait",
            "--for=condition=complete",
            f"--timeout={timeout}",
            "-n",
            namespace,
            f"job/{job_name}",
        ],
        check=True,
    )


def _git_output(command: list[str]) -> str:
    return _run_capture(command)


def _run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}.")
    return resolved


def _resolve_positive_override(*, override: Any, fallback: Any, field_name: str) -> int:
    if override is not None:
        return _coerce_positive_int(override, field_name=field_name)
    return _coerce_positive_int(fallback, field_name=field_name)


if __name__ == "__main__":
    main()
