from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, model_validator

AgentProvider = Literal["openrouter", "local"]
AgentMode = Literal["blind", "oracle"]
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


class PathsConfig(BaseModel):
    experiments_dir: str = "experiments/"

    model_config = {"frozen": True}


class AgentModelConfig(BaseModel):
    name: str
    provider: AgentProvider
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    presence_penalty: float = Field(default=1.5, ge=-2.0, le=2.0)
    max_tokens: int = Field(default=16_384, gt=0)
    context_window: int = Field(default=65_536, gt=0)
    max_turns: int = Field(default=100, gt=0)
    base_url: str | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def set_provider_defaults(self) -> AgentModelConfig:
        if self.provider == "openrouter":
            if self.base_url is None:
                self.base_url = "https://openrouter.ai/api/v1"
            if self.api_key_env is None:
                self.api_key_env = "OPENROUTER_API_KEY"
        else:
            if self.base_url is None:
                self.base_url = "http://localhost:8000/v1"
            if self.api_key_env is None:
                self.api_key_env = "LOCAL_OPENAI_API_KEY"
        if self.max_tokens >= self.context_window:
            raise ValueError(
                "models.agent.max_tokens must be smaller than models.agent.context_window."
            )
        return self

    model_config = {"frozen": True}


class ModelsConfig(BaseModel):
    agent: AgentModelConfig

    model_config = {"frozen": True}


class BaseConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig

    model_config = {"frozen": True}


def load_yaml_config(config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    _load_dotenv_from_parents(path.parent)
    raw_data = cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return cast("dict[str, Any]", _expand_env_references(raw_data))


def load_base_config(config_path: Path | str) -> BaseConfig:
    return BaseConfig.model_validate(load_yaml_config(config_path))


def _load_dotenv_from_parents(start_dir: Path) -> None:
    for directory in (start_dir, *start_dir.parents):
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            _load_dotenv_file(dotenv_path)
            return


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _expand_env_references(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_references(item) for item in value]
    if isinstance(value, str):
        return _expand_env_string(value)
    return value


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default_value = match.group(2)
        resolved = os.environ.get(env_name)
        if resolved is not None:
            return resolved
        if default_value is not None:
            return default_value
        raise ValueError(f"Environment variable {env_name!r} is required by the config.")

    return _ENV_REFERENCE_PATTERN.sub(replace, value)
