from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.submit_phase0_k8s import build_github_secret_manifest, load_namespace_from_kubeconfig
from token_sniper.evaluation.k8s import (
    Phase0KubernetesJobConfig,
    build_phase0_job_resources,
    build_phase0_runner_script,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_phase0_job_resources_includes_cache_volume_and_runner_env() -> None:
    config = Phase0KubernetesJobConfig(
        namespace="runai-romania-dev",
        job_name="phase0-test",
        pvc_name="phase0-test-workspace",
        cache_pvc_name="phase0-cache",
        create_cache_pvc=True,
        repo_url="https://github.com/reticle-ai/token-sniper.git",
        git_ref="abc123",
        github_secret_name="token-sniper-github",
        model_name="Qwen/Qwen3.5-35B-A3B-FP8",
        agent_workers=8,
        condition_workers=2,
        harness_workers=12,
        harness_batch_size=6,
    )

    resources = build_phase0_job_resources(config)

    assert len(resources) == 3
    workspace_pvc, cache_pvc, job = resources
    assert workspace_pvc["kind"] == "PersistentVolumeClaim"
    assert workspace_pvc["metadata"]["name"] == "phase0-test-workspace"
    assert cache_pvc["metadata"]["name"] == "phase0-cache"
    assert cache_pvc["spec"]["resources"]["requests"]["storage"] == "400Gi"

    pod_spec = job["spec"]["template"]["spec"]
    runner = pod_spec["containers"][0]
    dind = pod_spec["containers"][1]
    runner_env = {entry["name"]: entry for entry in runner["env"]}

    assert job["metadata"]["labels"]["runai/node-pool"] == "default"
    assert pod_spec["nodeSelector"] == {"node-role.kubernetes.io/runai-cpu-worker": "true"}
    assert pod_spec["shareProcessNamespace"] is True
    assert runner_env["TOKEN_SNIPER_AGENT_WORKERS"]["value"] == "8"
    assert runner_env["TOKEN_SNIPER_CONDITION_WORKERS"]["value"] == "2"
    assert runner_env["TOKEN_SNIPER_HARNESS_MAX_WORKERS"]["value"] == "12"
    assert runner_env["TOKEN_SNIPER_HARNESS_BATCH_SIZE"]["value"] == "6"
    assert runner_env["TOKEN_SNIPER_SWEBENCH_CACHE"]["value"] == "/cache/swebench"
    assert {"name": "shared-cache", "mountPath": "/cache"} in runner["volumeMounts"]
    assert dind["volumeMounts"] == [{"name": "shared-cache", "mountPath": "/var/lib/docker"}]


def test_build_phase0_runner_script_targets_phase0_cli_and_prebuilt_image_mode() -> None:
    config = Phase0KubernetesJobConfig(
        namespace="runai-romania-dev",
        job_name="phase0-test",
        pvc_name="phase0-test-workspace",
        repo_url="https://github.com/reticle-ai/token-sniper.git",
        git_ref="abc123",
        github_secret_name="token-sniper-github",
        model_name="Qwen/Qwen3.5-35B-A3B-FP8",
        install_repo_runtime=False,
        conditions=["oracle_retrieval"],
        agent_workers=16,
        condition_workers=2,
        harness_workers=24,
        harness_batch_size=10,
    )

    script = build_phase0_runner_script(config)

    assert "git clone" in script
    assert "http://127.0.0.1:2375/_ping" in script
    assert 'export PYTHONPATH="/workspace/repo${PYTHONPATH:+:${PYTHONPATH}}"' in script
    assert "trap cleanup_docker_daemon EXIT" in script
    assert "pip install -e" not in script
    assert "\nPY\n" in script
    assert "\n        PY\n" not in script
    assert "--conditions \\" in script
    assert '  "oracle_retrieval" \\' in script
    assert '--condition-workers "2"' in script
    assert '--eval-batch-size "10"' in script


def test_build_phase0_runner_script_runtime_install_mode_keeps_pip_install() -> None:
    config = Phase0KubernetesJobConfig(
        namespace="runai-romania-dev",
        job_name="phase0-test",
        pvc_name="phase0-test-workspace",
        repo_url="https://github.com/reticle-ai/token-sniper.git",
        git_ref="abc123",
        github_secret_name="token-sniper-github",
        model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    )

    script = build_phase0_runner_script(config)

    assert "apt-get install -y --no-install-recommends ca-certificates git" in script
    assert 'pip install -e ".[agent,experiments,data]"' in script


def test_load_namespace_from_kubeconfig_reads_current_context_namespace(tmp_path: Path) -> None:
    kubeconfig_path = tmp_path / "kubeconfig.yaml"
    kubeconfig_path.write_text(
        "\n".join(
            [
                "apiVersion: v1",
                "contexts:",
                "- name: ctx-a",
                "  context:",
                "    namespace: runai-romania-dev",
                "current-context: ctx-a",
            ]
        ),
        encoding="utf-8",
    )

    assert load_namespace_from_kubeconfig(kubeconfig_path) == "runai-romania-dev"


def test_build_github_secret_manifest_uses_string_data() -> None:
    manifest = build_github_secret_manifest(
        namespace="runai-romania-dev",
        secret_name="token-sniper-github",
        secret_key="token",
        token="ghp_example",
    )

    assert manifest["kind"] == "Secret"
    assert manifest["metadata"]["namespace"] == "runai-romania-dev"
    assert manifest["stringData"]["token"] == "ghp_example"
