from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from token_sniper.agent import docker_tools


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


@patch("token_sniper.agent.docker_tools.time.sleep")
@patch("token_sniper.agent.docker_tools.subprocess.run")
def test_docker_pull_succeeds_first_attempt(mock_run: MagicMock, mock_sleep: MagicMock) -> None:
    mock_run.return_value = _completed()

    docker_tools._docker_pull("repo/image:tag", max_attempts=3)

    assert mock_run.call_count == 1
    mock_sleep.assert_not_called()


@patch("token_sniper.agent.docker_tools.time.sleep")
@patch("token_sniper.agent.docker_tools.subprocess.run")
def test_docker_pull_retries_then_succeeds(mock_run: MagicMock, mock_sleep: MagicMock) -> None:
    mock_run.side_effect = [
        _completed(returncode=1, stderr="toomanyrequests"),
        _completed(returncode=1, stderr="toomanyrequests"),
        _completed(),
    ]

    docker_tools._docker_pull(
        "repo/image:tag",
        max_attempts=5,
        base_delay_s=1,
        max_delay_s=10,
    )

    assert mock_run.call_count == 3
    assert mock_sleep.call_count == 2


@patch("token_sniper.agent.docker_tools.time.sleep")
@patch("token_sniper.agent.docker_tools.subprocess.run")
def test_docker_pull_raises_after_max_attempts(mock_run: MagicMock, mock_sleep: MagicMock) -> None:
    mock_run.return_value = _completed(returncode=1, stderr="rate limit")

    with pytest.raises(docker_tools.DockerStartError, match="Failed to pull"):
        docker_tools._docker_pull("repo/image:tag", max_attempts=2, base_delay_s=0.01)

    assert mock_run.call_count == 2
    assert mock_sleep.call_count == 1


@patch("token_sniper.agent.docker_tools._docker_run_task_container")
@patch("token_sniper.agent.docker_tools._docker_pull")
@patch("token_sniper.agent.docker_tools._remove_container")
@patch("token_sniper.agent.docker_tools.subprocess.run")
def test_start_task_container_pulls_when_image_missing(
    mock_run: MagicMock,
    mock_remove: MagicMock,
    mock_pull: MagicMock,
    mock_run_container: MagicMock,
) -> None:
    mock_run.return_value = _completed(returncode=1)

    name = docker_tools.start_task_container(
        "org__repo-1",
        image_tag="starryzhang/sweb.eval.x86_64.org_1776_repo-1:latest",
    )

    assert name == "swebench-org__repo-1"
    mock_pull.assert_called_once()
    mock_run_container.assert_called_once()


@patch("token_sniper.agent.docker_tools._docker_run_task_container")
@patch("token_sniper.agent.docker_tools._remove_container")
@patch("token_sniper.agent.docker_tools.subprocess.run")
def test_start_task_container_skips_pull_when_image_present(
    mock_run: MagicMock,
    mock_remove: MagicMock,
    mock_run_container: MagicMock,
) -> None:
    mock_run.return_value = _completed()

    docker_tools.start_task_container(
        "org__repo-1",
        image_tag="starryzhang/sweb.eval.x86_64.org_1776_repo-1:latest",
    )

    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0][:2] == ["docker", "image"]
