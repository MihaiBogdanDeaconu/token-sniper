from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from token_sniper.agent.config import AgentModelConfig


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    token_logprobs: list[list[float]] | None = None

    model_config = {"frozen": True}


class ChatModelClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        pass


class OpenAIChatModelClient:
    def __init__(
        self,
        config: AgentModelConfig,
        *,
        collect_logprobs: bool = False,
        top_logprobs: int = 20,
    ) -> None:
        self._config = config
        self._collect_logprobs = collect_logprobs
        self._top_logprobs = top_logprobs
        api_key_name = config.api_key_env or ""
        api_key = os.environ.get(api_key_name, "")
        if config.provider == "openrouter" and not api_key:
            raise RuntimeError(
                f"Missing API key for OpenRouter. Set the {api_key_name} environment variable."
            )
        if not api_key:
            api_key = "local-placeholder"

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for the agent scaffold. "
                "Install the project with the 'agent' extra."
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._config.name,
            "messages": cast("Any", messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "top_p": self._config.top_p,
            "presence_penalty": self._config.presence_penalty,
            "timeout": self._config.request_timeout_seconds,
            "extra_body": {"top_k": self._config.top_k},
        }
        if tools:
            kwargs["tools"] = cast("Any", tools)
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._collect_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self._top_logprobs
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        tool_calls: list[ToolCall] = []
        for index, raw_tool_call in enumerate(message.tool_calls or []):
            raw_function = getattr(raw_tool_call, "function", None)
            if raw_function is None:
                continue
            raw_arguments = raw_function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": raw_arguments}
            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.id or f"tool-call-{index}",
                    name=raw_function.name,
                    arguments=arguments,
                )
            )

        return LLMResponse(
            content=_flatten_message_content(message.content),
            tool_calls=tool_calls,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=choice.finish_reason,
            token_logprobs=_extract_top_k_logprobs(choice, self._top_logprobs)
            if self._collect_logprobs else None,
        )


def _extract_top_k_logprobs(choice: Any, top_k: int) -> list[list[float]] | None:
    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return None
    content = getattr(logprobs_obj, "content", None)
    if not content:
        return None

    series: list[list[float]] = []
    for token_obj in content:
        top_list = getattr(token_obj, "top_logprobs", []) or []
        vals = [getattr(entry, "logprob", -100.0) for entry in top_list[:top_k]]
        while len(vals) < top_k:
            vals.append(-100.0)
        series.append(vals)
    return series if series else None


def _flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, sort_keys=True))
        return "\n".join(part for part in parts if part)
    return str(content)
