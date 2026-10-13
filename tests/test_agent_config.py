from __future__ import annotations

from typing import TYPE_CHECKING

from token_sniper.agent.config import load_base_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_load_base_config_reads_base_url_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TOKEN_SNIPER_AGENT_BASE_URL", raising=False)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".env").write_text(
        "TOKEN_SNIPER_AGENT_BASE_URL=http://127.0.0.1:9000/v1\n",
        encoding="utf-8",
    )

    config_path = repo_root / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  agent:",
                "    name: local-qwen",
                "    provider: local",
                "    base_url: ${TOKEN_SNIPER_AGENT_BASE_URL:-http://localhost:8000/v1}",
                "    api_key_env: LOCAL_OPENAI_API_KEY",
            ]
        ),
        encoding="utf-8",
    )

    config = load_base_config(config_path)
    assert config.models.agent.base_url == "http://127.0.0.1:9000/v1"


def test_load_base_config_uses_default_when_dotenv_value_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TOKEN_SNIPER_AGENT_BASE_URL", raising=False)
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  agent:",
                "    name: local-qwen",
                "    provider: local",
                "    base_url: ${TOKEN_SNIPER_AGENT_BASE_URL:-http://localhost:8000/v1}",
                "    api_key_env: LOCAL_OPENAI_API_KEY",
            ]
        ),
        encoding="utf-8",
    )

    config = load_base_config(config_path)
    assert config.models.agent.base_url == "http://localhost:8000/v1"
