from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from token_sniper.agent.providers import LLMResponse, ToolCall
from token_sniper.agent.runner import AgentRunSettings, run_agent_session


class StubClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        del tools, response_format
        self.calls.append(deepcopy(messages))
        return self._responses.pop(0)


def test_run_agent_session_executes_tools_and_writes_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def old_name():\n    return 1\n", encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    client = StubClient(
        [
            LLMResponse(
                content="Need to inspect the file first.",
                tool_calls=[ToolCall(id="tool-1", name="cat", arguments={"path": "app.py"})],
                prompt_tokens=10,
                completion_tokens=4,
            ),
            LLMResponse(
                content=(
                    "--- a/app.py\n"
                    "+++ b/app.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    "-def old_name():\n"
                    "+def new_name():\n"
                    "     return 1\n"
                ),
                prompt_tokens=12,
                completion_tokens=6,
            ),
        ]
    )

    result = run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Rename old_name to new_name.",
            output_dir=str(output_dir),
        ),
        client,
    )

    assert result.patch_path is not None
    assert Path(result.patch_path).exists()
    assert Path(result.session_log_path).exists()
    assert result.total_prompt_tokens == 22
    assert result.total_completion_tokens == 10
    assert len(result.turns) == 2
    assert result.turns[0].tool_calls[0].result["path"] == "app.py"


def test_oracle_mode_preloads_requested_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(
                content=("--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")
            )
        ]
    )

    run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Update the constant.",
            output_dir=str(tmp_path / "artifacts"),
            mode="oracle",
            oracle_files=["target.py"],
        ),
        client,
    )

    oracle_message = client.calls[0][2]["content"]
    assert "Oracle mode is enabled." in oracle_message
    assert "Path: target.py" in oracle_message
    assert "VALUE = 1" in oracle_message


def test_oracle_mode_notes_missing_files_without_crashing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(
                content=("--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")
            )
        ]
    )

    run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Update the constant.",
            output_dir=str(tmp_path / "artifacts"),
            mode="oracle",
            oracle_files=["target.py", "created_later.py"],
        ),
        client,
    )

    oracle_message = client.calls[0][2]["content"]
    assert "Path: target.py" in oracle_message
    assert "created_later.py" in oracle_message
    assert "do not exist at the base commit" in oracle_message


def test_oracle_mode_accepts_explicit_oracle_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(
                content=("--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")
            )
        ]
    )

    run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Update the constant.",
            output_dir=str(tmp_path / "artifacts"),
            mode="oracle",
            oracle_context=(
                "Oracle mode is enabled.\n\nPath: target.py\n```python\n   1: VALUE = 1\n```"
            ),
        ),
        client,
    )

    oracle_message = client.calls[0][2]["content"]
    assert "Path: target.py" in oracle_message
    assert "VALUE = 1" in oracle_message


def test_invalid_patch_triggers_retry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('x')\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(content="I would change app.py."),
            LLMResponse(
                content=(
                    "```diff\n"
                    "--- a/app.py\n"
                    "+++ b/app.py\n"
                    "@@ -1 +1 @@\n"
                    "-print('x')\n"
                    "+print('y')\n"
                    "```"
                )
            ),
        ]
    )

    result = run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Change the printed value.",
            output_dir=str(tmp_path / "artifacts"),
        ),
        client,
    )

    assert len(result.turns) == 2
    retry_message = client.calls[1][-1]["content"]
    assert "valid unified diff patch" in retry_message


def test_malformed_patch_triggers_parse_validation_and_repair(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('x')\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(content=("--- a/app.py\n+++ b/app.py\n-print('x')\n+print('y')\n")),
            LLMResponse(
                content=("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('x')\n+print('y')\n")
            ),
        ]
    )

    result = run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Change the printed value.",
            output_dir=str(tmp_path / "artifacts"),
        ),
        client,
    )

    assert result.patch_path is not None
    assert result.turns[0].patch_validation is not None
    assert result.turns[0].patch_validation.validation_status == "rejected"
    assert result.turns[0].patch_validation.validation_stage == "parse"
    assert result.turns[0].patch_validation.repair_prompt_sent is True
    assert "touched files are relevant" in client.calls[1][-1]["content"]


def test_submit_patch_rejects_invalid_paths_before_submission(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('x')\n", encoding="utf-8")

    client = StubClient(
        [
            LLMResponse(
                content="Ready to submit.",
                tool_calls=[
                    ToolCall(
                        id="tool-1",
                        name="submit_patch",
                        arguments={
                            "patch_content": (
                                "diff --git a/../outside.py b/../outside.py\n"
                                "--- a/../outside.py\n"
                                "+++ b/../outside.py\n"
                                "@@ -1 +1 @@\n"
                                "-print('x')\n"
                                "+print('y')\n"
                            )
                        },
                    )
                ],
            ),
            LLMResponse(
                content=("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('x')\n+print('y')\n")
            ),
        ]
    )

    result = run_agent_session(
        AgentRunSettings(
            repo_root=str(repo),
            issue_description="Change the printed value.",
            output_dir=str(tmp_path / "artifacts"),
        ),
        client,
    )

    assert result.patch_path is not None
    assert result.turns[0].patch_validation is not None
    assert result.turns[0].patch_validation.validation_status == "rejected"
    assert result.turns[0].patch_validation.validation_stage == "path"
    assert result.turns[0].tool_calls[0].result["status"] == "rejected"
    validation_message = result.turns[0].patch_validation.validation_message
    assert validation_message is not None
    assert "invalid path outside the repository" in validation_message
    repair_prompt = client.calls[1][-1]["content"]
    assert repair_prompt is not None
    assert "diff headers and hunks are valid" in repair_prompt


def test_failed_run_writes_session_log(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('x')\n", encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    client = StubClient(
        [
            LLMResponse(content="I would change app.py."),
            LLMResponse(content="Still thinking."),
        ]
    )

    with pytest.raises(RuntimeError, match="maximum number of turns"):
        run_agent_session(
            AgentRunSettings(
                repo_root=str(repo),
                issue_description="Change the printed value.",
                output_dir=str(output_dir),
                max_turns=2,
            ),
            client,
        )

    session_path = output_dir / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["status"] == "failed"
    assert "maximum number of turns" in session["error_message"]
    assert len(session["turns"]) == 2


def test_oracle_mode_rejects_context_overflow(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("VALUE = 1\n" * 100, encoding="utf-8")

    client = StubClient([])

    with pytest.raises(ValueError, match="Oracle context exceeds"):
        run_agent_session(
            AgentRunSettings(
                repo_root=str(repo),
                issue_description="Update the constant.",
                output_dir=str(tmp_path / "artifacts"),
                mode="oracle",
                oracle_files=["target.py"],
                context_window=50,
                max_output_tokens=10,
            ),
            client,
        )


def test_agent_config_rejects_max_tokens_larger_than_context() -> None:
    from token_sniper.agent.config import BaseConfig

    with pytest.raises(ValueError, match="max_tokens must be smaller"):
        BaseConfig.model_validate(
            {
                "models": {
                    "agent": {
                        "name": "test-model",
                        "provider": "local",
                        "base_url": "http://localhost:8000/v1",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                        "max_tokens": 4096,
                        "context_window": 4096,
                    }
                }
            }
        )
