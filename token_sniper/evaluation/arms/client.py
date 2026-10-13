from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any

from token_sniper.agent.providers import OpenAIChatModelClient

if TYPE_CHECKING:
    from pathlib import Path

    from token_sniper.agent.config import AgentModelConfig

CACHED_TOKENS_HINT = (
    "start the vLLM server with --enable-prompt-tokens-details (and --enable-prefix-caching) "
    "so it reports usage.prompt_tokens_details.cached_tokens"
)


def usage_row(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        raise RuntimeError("the provider returned no usage for a chat completion")
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    if cached is None:
        raise RuntimeError(f"the provider did not report cached prompt tokens: {CACHED_TOKENS_HINT}")
    return {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "cached_tokens": int(cached),
    }


class _Completions:
    def __init__(self, inner: Any, seed: int, usage_log: Path) -> None:
        self._inner = inner
        self._seed = seed
        self._usage_log = usage_log
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        if "seed" in kwargs:
            raise ValueError("the request already carries a seed")
        response = self._inner.create(**kwargs, seed=self._seed)
        row = usage_row(response)
        with self._lock, self._usage_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return response


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _SeededOpenAI:
    def __init__(self, inner: Any, seed: int, usage_log: Path) -> None:
        self.chat = _Chat(_Completions(inner.chat.completions, seed, usage_log))


class SeededUsageClient(OpenAIChatModelClient):
    def __init__(self, config: AgentModelConfig, *, seed: int, usage_log: Path) -> None:
        super().__init__(config)
        self._client = _SeededOpenAI(self._client, seed, usage_log)


def read_usage(usage_log: Path) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0}
    if not usage_log.exists():
        return totals
    for line in usage_log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            totals[key] += row[key]
        totals["calls"] += 1
    return totals
