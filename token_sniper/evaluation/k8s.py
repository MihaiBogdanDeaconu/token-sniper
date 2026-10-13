from __future__ import annotations

from textwrap import dedent
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field


class Phase0KubernetesJobConfig(BaseModel):
    namespace: str
    job_name: str
    pvc_name: str
    node_pool: str = "default"
    repo_url: str
    git_ref: str
    github_secret_name: str
    github_secret_key: str = "token"
    runner_image: str = "python:3.12-slim"
    install_repo_runtime: bool = True
    storage_class_name: str = "local-path"
    storage_size: str = "200Gi"
    cache_pvc_name: str | None = None
    create_cache_pvc: bool = False
    cache_storage_class_name: str = "local-path"
    cache_storage_size: str = "400Gi"
    model_name: str
    qwen_base_url: str = "http://qwen-vllm:8077/v1"
    base_config_path: str = "configs/base.yaml"
    phase_config_path: str = "configs/phases/phase0_oracle.yaml"
    manifest_path: str = "configs/phases/phase0_tasks.json"
    conditions: list[str] = Field(
        default_factory=lambda: ["blind_agent", "oracle_retrieval"],
        min_length=1,
    )
    agent_workers: int = Field(default=4, gt=0)
    condition_workers: int = Field(default=1, gt=0)
    harness_workers: int = Field(default=4, gt=0)
    harness_batch_size: int = Field(default=8, gt=0)
    runner_cpu_request: str = "32"
    runner_cpu_limit: str = "64"
    runner_memory_request: str = "128Gi"
    runner_memory_limit: str = "192Gi"
    dind_cpu_request: str = "8"
    dind_cpu_limit: str = "16"
    dind_memory_request: str = "16Gi"
    dind_memory_limit: str = "32Gi"
    node_selector: dict[str, str] = Field(
        default_factory=lambda: {"node-role.kubernetes.io/runai-cpu-worker": "true"}
    )

    model_config = {"frozen": True}


def build_phase0_runner_script(config: Phase0KubernetesJobConfig) -> str:
    install_runtime_block = ""
    if config.install_repo_runtime:
        install_runtime_block = dedent(
            """
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends ca-certificates git
            rm -rf /var/lib/apt/lists/*
            """
        ).strip()

    pip_install_block = ""
    if config.install_repo_runtime:
        pip_install_block = dedent(
            """
            python -m pip install --upgrade pip
            python -m pip install -e ".[agent,experiments,data]"
            """
        ).strip()

    cache_setup_block = dedent(
        """
        if [ -d /cache ]; then
          mkdir -p /cache/runner/pip /cache/runner/huggingface /cache/runner/xdg /cache/swebench
        fi
        """
    ).strip()
    docker_cleanup_block = dedent(
        """
        cleanup_docker_daemon() {
          python - <<'PY'
        from pathlib import Path
        import os
        import signal
        import time

        def iter_dockerd_pids() -> list[int]:
            pids: list[int] = []
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                try:
                    cmdline = Path(f"/proc/{entry.name}/cmdline").read_bytes().decode(
                        "utf-8", "ignore"
                    )
                except OSError:
                    continue
                if "dockerd" in cmdline:
                    pids.append(int(entry.name))
            return pids

        for pid in iter_dockerd_pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue

        deadline = time.time() + 30
        while time.time() < deadline:
            if not iter_dockerd_pids():
                break
            time.sleep(1)
        PY
        }
        trap cleanup_docker_daemon EXIT
        """
    ).strip()
    wait_for_docker_block = dedent(
        """
        until python - <<'PY'
        from urllib.request import urlopen

        try:
            with urlopen("http://127.0.0.1:2375/_ping", timeout=2) as response:
                body = response.read().decode("utf-8").strip()
        except Exception:
            raise SystemExit(1)

        raise SystemExit(0 if body == "OK" else 1)
        PY
        do
          echo "waiting for docker sidecar"
          sleep 2
        done
        """
    ).strip()
    clone_repo_block = dedent(
        """
        rm -rf /workspace/repo
        auth_repo_url="$(python - <<'PY'
        import os

        repo_url = os.environ["TOKEN_SNIPER_REPO_URL"]
        token = os.environ["GITHUB_TOKEN"]
        if not repo_url.startswith("https://"):
            raise SystemExit(f"Unsupported repo URL for private clone: {repo_url}")
        print(f"https://x-access-token:{token}@{repo_url[len('https://'):]}")
        PY
        )"
        git clone "$auth_repo_url" /workspace/repo
        unset auth_repo_url

        cd /workspace/repo
        git checkout "$TOKEN_SNIPER_GIT_REF"
        export PYTHONPATH="/workspace/repo${PYTHONPATH:+:${PYTHONPATH}}"
        """
    ).strip()
    run_phase0_block = "\n".join(
        [
            "python scripts/run_phase0.py \\",
            f'  --base-config "{config.base_config_path}" \\',
            f'  --phase-config "{config.phase_config_path}" \\',
            f'  --manifest "{config.manifest_path}" \\',
            "  --conditions \\",
            *[f'  "{condition}" \\' for condition in config.conditions[:-1]],
            f'  "{config.conditions[-1]}" \\',
            f'  --agent-workers "{config.agent_workers}" \\',
            f'  --condition-workers "{config.condition_workers}" \\',
            f'  --eval-batch-size "{config.harness_batch_size}" \\',
            f'  --max-workers "{config.harness_workers}"',
        ]
    )
    blocks = [
        "set -euo pipefail",
        install_runtime_block,
        cache_setup_block,
        docker_cleanup_block,
        wait_for_docker_block,
        clone_repo_block,
        pip_install_block,
        run_phase0_block,
        'echo "phase0 artifacts persisted in /workspace/repo/experiments/phase0"',
    ]
    return "\n\n".join(block for block in blocks if block)


def build_phase0_job_resources(config: Phase0KubernetesJobConfig) -> list[dict[str, Any]]:
    labels = {
        "app.kubernetes.io/name": "token-sniper",
        "app.kubernetes.io/component": "phase0-runner",
        "app.kubernetes.io/part-of": "token-sniper",
        "runai/node-pool": config.node_pool,
    }
    workspace_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": config.pvc_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": config.storage_class_name,
            "resources": {"requests": {"storage": config.storage_size}},
        },
    }
    resources: list[dict[str, Any]] = [workspace_pvc]
    if config.cache_pvc_name and config.create_cache_pvc:
        resources.append(
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": config.cache_pvc_name,
                    "namespace": config.namespace,
                    "labels": labels,
                },
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "storageClassName": config.cache_storage_class_name,
                    "resources": {"requests": {"storage": config.cache_storage_size}},
                },
            }
        )

    runner_env: list[dict[str, Any]] = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "DOCKER_HOST", "value": "tcp://127.0.0.1:2375"},
        {"name": "TOKEN_SNIPER_REPO_URL", "value": config.repo_url},
        {"name": "TOKEN_SNIPER_GIT_REF", "value": config.git_ref},
        {"name": "TOKEN_SNIPER_AGENT_PROVIDER", "value": "local"},
        {"name": "TOKEN_SNIPER_AGENT_BASE_URL", "value": config.qwen_base_url},
        {"name": "TOKEN_SNIPER_AGENT_MODEL", "value": config.model_name},
        {"name": "TOKEN_SNIPER_AGENT_API_KEY_ENV", "value": "LOCAL_OPENAI_API_KEY"},
        {"name": "LOCAL_OPENAI_API_KEY", "value": "local-placeholder"},
        {"name": "TOKEN_SNIPER_AGENT_WORKERS", "value": str(config.agent_workers)},
        {"name": "TOKEN_SNIPER_CONDITION_WORKERS", "value": str(config.condition_workers)},
        {"name": "TOKEN_SNIPER_HARNESS_MAX_WORKERS", "value": str(config.harness_workers)},
        {"name": "TOKEN_SNIPER_HARNESS_BATCH_SIZE", "value": str(config.harness_batch_size)},
        {
            "name": "GITHUB_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": config.github_secret_name,
                    "key": config.github_secret_key,
                }
            },
        },
    ]
    if config.cache_pvc_name:
        runner_env.extend(
            [
                {"name": "PIP_CACHE_DIR", "value": "/cache/runner/pip"},
                {"name": "HF_HOME", "value": "/cache/runner/huggingface"},
                {"name": "HF_HUB_CACHE", "value": "/cache/runner/huggingface/hub"},
                {
                    "name": "HF_DATASETS_CACHE",
                    "value": "/cache/runner/huggingface/datasets",
                },
                {"name": "XDG_CACHE_HOME", "value": "/cache/runner/xdg"},
                {"name": "TOKEN_SNIPER_SWEBENCH_CACHE", "value": "/cache/swebench"},
            ]
        )

    volumes: list[dict[str, Any]] = [
        {
            "name": "workspace",
            "persistentVolumeClaim": {"claimName": config.pvc_name},
        }
    ]
    runner_mounts: list[dict[str, Any]] = [{"name": "workspace", "mountPath": "/workspace"}]
    dind_mounts: list[dict[str, Any]]
    if config.cache_pvc_name:
        volumes.append(
            {
                "name": "shared-cache",
                "persistentVolumeClaim": {"claimName": config.cache_pvc_name},
            }
        )
        runner_mounts.append({"name": "shared-cache", "mountPath": "/cache"})
        dind_mounts = [{"name": "shared-cache", "mountPath": "/var/lib/docker"}]
    else:
        volumes.append({"name": "docker-graph-storage", "emptyDir": {}})
        dind_mounts = [{"name": "docker-graph-storage", "mountPath": "/var/lib/docker"}]

    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": config.job_name,
            "namespace": config.namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "shareProcessNamespace": True,
                    "nodeSelector": config.node_selector,
                    "volumes": volumes,
                    "containers": [
                        {
                            "name": "runner",
                            "image": config.runner_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc", build_phase0_runner_script(config)],
                            "env": runner_env,
                            "resources": {
                                "requests": {
                                    "cpu": config.runner_cpu_request,
                                    "memory": config.runner_memory_request,
                                },
                                "limits": {
                                    "cpu": config.runner_cpu_limit,
                                    "memory": config.runner_memory_limit,
                                },
                            },
                            "volumeMounts": runner_mounts,
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
                                "requests": {
                                    "cpu": config.dind_cpu_request,
                                    "memory": config.dind_memory_request,
                                },
                                "limits": {
                                    "cpu": config.dind_cpu_limit,
                                    "memory": config.dind_memory_limit,
                                },
                            },
                            "volumeMounts": dind_mounts,
                        },
                    ],
                },
            },
        },
    }
    resources.append(job)
    return resources


def dump_phase0_job_resources(config: Phase0KubernetesJobConfig) -> str:
    return cast(
        "str",
        yaml.safe_dump_all(
            build_phase0_job_resources(config),
            sort_keys=False,
        ),
    )
