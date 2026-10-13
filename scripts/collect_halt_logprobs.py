from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import subprocess

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

VLLM_BASE = "http://qwen-vllm:8077/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
TOP_K = 20
MAX_CONCURRENT = 8
MAX_GEN_TOKENS = 128
TEMPERATURE = 0.0


def _build_requests_qa(examples: list[dict]) -> list[tuple[str, int, int]]:
    reqs = []
    for i, ex in enumerate(examples):
        prompt_correct = (
            f"Based on the following knowledge, answer the question.\n"
            f"Knowledge: {ex['knowledge']}\n"
            f"Question: {ex['question']}\n"
            f"Answer:"
        )
        reqs.append((prompt_correct, 0, i))

        prompt_halluc = (
            f"Answer the following question. You may make reasonable guesses.\n"
            f"Question: {ex['question']}\n"
            f"Answer:"
        )
        reqs.append((prompt_halluc, 1, i))
    return reqs


def _build_requests_summarization(examples: list[dict]) -> list[tuple[str, int, int]]:
    reqs = []
    for i, ex in enumerate(examples):
        prompt_correct = (
            f"Summarize the following document accurately.\n"
            f"Document: {ex['document'][:1500]}\n"
            f"Summary:"
        )
        reqs.append((prompt_correct, 0, i))

        prompt_halluc = (
            f"Summarize the following document.\n"
            f"Document: {ex['document'][:200]}\n"
            f"Summary:"
        )
        reqs.append((prompt_halluc, 1, i))
    return reqs


def _build_requests_dialogue(examples: list[dict]) -> list[tuple[str, int, int]]:
    reqs = []
    for i, ex in enumerate(examples):
        prompt_correct = (
            f"Given the knowledge and dialogue, provide an accurate response.\n"
            f"Knowledge: {ex['knowledge']}\n"
            f"Dialogue: {ex['dialogue_history']}\n"
            f"Response:"
        )
        reqs.append((prompt_correct, 0, i))

        prompt_halluc = (
            f"Continue the following dialogue naturally.\n"
            f"Dialogue: {ex['dialogue_history']}\n"
            f"Response:"
        )
        reqs.append((prompt_halluc, 1, i))
    return reqs


def _build_requests_general(examples: list[dict]) -> list[tuple[str, int, int]]:
    reqs = []
    for i, ex in enumerate(examples):
        prompt = f"User: {ex['user_query']}\nAssistant:"
        label = 1 if ex.get("hallucination", "").lower() == "yes" else 0
        reqs.append((prompt, label, i))
    return reqs


SPLIT_BUILDERS = {
    "qa": _build_requests_qa,
    "summarization": _build_requests_summarization,
    "dialogue": _build_requests_dialogue,
    "general": _build_requests_general,
}


@dataclass
class LogprobRequest:
    split: str
    idx: int
    label: int
    prompt: str


def build_requests(data_dir: Path) -> list[LogprobRequest]:
    requests: list[LogprobRequest] = []
    for split_name, builder in SPLIT_BUILDERS.items():
        fpath = data_dir / f"{split_name}.json"
        if not fpath.exists():
            log.warning("Missing %s, skipping", fpath)
            continue
        with open(fpath) as f:
            examples = json.load(f)
        log.info("Building requests for %s (%d examples)", split_name, len(examples))
        for prompt, label, idx in builder(examples):
            requests.append(LogprobRequest(
                split=split_name, idx=idx, label=label, prompt=prompt,
            ))
    return requests


def load_completed(out_dir: Path) -> set[str]:
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for jsonl_file in out_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(f"{rec['split']}_{rec['idx']}_{rec['label']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


async def collect_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    req: LogprobRequest,
    circuit: "_CircuitBreaker",
    *,
    temperature: float = TEMPERATURE,
) -> dict[str, Any] | None:
    if circuit.tripped:
        return None

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": MAX_GEN_TOKENS,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    if temperature > 0:
        payload["top_p"] = 0.95

    url = f"{VLLM_BASE}/chat/completions"

    async with sem:
        if circuit.tripped:
            return None
        for attempt in range(3):
            try:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=45),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt + 1)
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("[%s/%d] HTTP %d: %s", req.split, req.idx, resp.status, body[:200])
                        await asyncio.sleep(2 ** attempt)
                        continue
                    data = await resp.json()

                circuit.record_success()

                if "choices" not in data:
                    log.warning("[%s/%d] No choices: %s", req.split, req.idx, str(data)[:200])
                    return None

                choice = data["choices"][0]
                logprobs_content = choice.get("logprobs", {}).get("content", [])

                if not logprobs_content or len(logprobs_content) < 3:
                    return None

                series = []
                for token_obj in logprobs_content:
                    top_list = token_obj.get("top_logprobs", [])
                    vals = [entry["logprob"] for entry in top_list[:TOP_K]]
                    while len(vals) < TOP_K:
                        vals.append(-100.0)
                    series.append(vals)

                return {
                    "logprobs": series,
                    "label": req.label,
                    "split": req.split,
                    "idx": req.idx,
                    "n_tokens": len(series),
                }

            except Exception as e:
                circuit.record_failure()
                if circuit.tripped:
                    return None
                log.warning(
                    "[%s/%d] attempt %d: %s: %s",
                    req.split, req.idx, attempt, type(e).__name__, e,
                )
                await asyncio.sleep(min(2 ** attempt + 1, 5))

    return None


class _CircuitBreaker:
    def __init__(self, threshold: int = 6) -> None:
        self._consecutive_failures = 0
        self._threshold = threshold
        self.tripped = False

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self.tripped = True

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def reset(self) -> None:
        self._consecutive_failures = 0
        self.tripped = False


async def _restart_vllm_and_wait() -> None:
    log.info("Triggering vLLM rollout restart...")
    subprocess.run(
        ["kubectl", "rollout", "restart", "deployment", "qwen-vllm"],
        capture_output=True, timeout=30,
    )
    for attempt in range(60):
        await asyncio.sleep(15)
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{VLLM_BASE.rsplit('/', 1)[0]}/health",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        async with s.post(
                            f"{VLLM_BASE}/chat/completions",
                            json={
                                "model": MODEL,
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 1,
                            },
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as test_resp:
                            if test_resp.status == 200:
                                log.info("vLLM recovered after %ds", (attempt + 1) * 15)
                                return
        except Exception:
            pass
        if attempt % 4 == 0:
            log.info("Waiting for vLLM... (%ds)", (attempt + 1) * 15)
    log.error("vLLM did not recover after 15 minutes!")


async def collect_all(
    requests: list[LogprobRequest],
    out_dir: Path,
    *,
    concurrency: int = MAX_CONCURRENT,
    temperature: float = TEMPERATURE,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    split_files: dict[str, Any] = {}
    split_counts: dict[str, int] = {}
    split_failures: dict[str, int] = {}

    for split_name in SPLIT_BUILDERS:
        split_files[split_name] = open(out_dir / f"{split_name}.jsonl", "a")
        split_counts[split_name] = 0
        split_failures[split_name] = 0

    total = len(requests)
    done = 0
    t0 = time.time()
    sem = asyncio.Semaphore(concurrency)
    circuit = _CircuitBreaker(threshold=6)

    batch_size = 32
    batch_start = 0
    while batch_start < total:
        batch = requests[batch_start : batch_start + batch_size]
        circuit.reset()

        connector = aiohttp.TCPConnector(
            limit=concurrency + 4,
            limit_per_host=concurrency + 4,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [collect_one(session, sem, r, circuit, temperature=temperature) for r in batch]
            results = await asyncio.gather(*tasks)

        if circuit.tripped:
            ok_results = [(req_item, result) for req_item, result in zip(batch, results) if result is not None]
            for req_item, result in ok_results:
                done += 1
                split_files[req_item.split].write(json.dumps(result) + "\n")
                split_counts[req_item.split] += 1
            for f in split_files.values():
                f.flush()

            failed_reqs = [req_item for req_item, result in zip(batch, results) if result is None]
            log.warning(
                "Circuit breaker tripped: %d ok, %d failed in batch. Restarting vLLM...",
                len(ok_results), len(failed_reqs),
            )
            await _restart_vllm_and_wait()

            if failed_reqs:
                requests[batch_start : batch_start + batch_size] = failed_reqs + [
                    req_item for req_item, result in zip(batch, results) if result is not None
                ][-0:]
                total = len(requests)
            continue

        for req_item, result in zip(batch, results):
            done += 1
            if result is not None:
                split_files[req_item.split].write(json.dumps(result) + "\n")
                split_counts[req_item.split] += 1
            else:
                split_failures[req_item.split] += 1

        for f in split_files.values():
            f.flush()

        batch_start += batch_size
        await asyncio.sleep(0.5)

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = total - batch_start
        eta = remaining / rate if rate > 0 else 0
        log.info(
            "Progress: %d/%d (%.1f%%) | %.1f req/s | ETA %.0fs",
            done, total, 100 * done / total, rate, eta,
        )

    for f in split_files.values():
        f.close()

    log.info("=== Collection complete ===")
    for split_name in SPLIT_BUILDERS:
        log.info(
            "  %s: %d ok, %d failed",
            split_name, split_counts[split_name], split_failures[split_name],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect HALT logprobs from Qwen vLLM")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/cache/halt_data/halueval"))
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/cache/halt_data/logprobs_gen"))
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT)
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature (0.0 = greedy, 0.6 = match agent inference)",
    )
    args = parser.parse_args()

    log.info("Loading HaluEval data from %s", args.data_dir)
    all_requests = build_requests(args.data_dir)
    log.info("Total requests built: %d (temperature=%.2f)", len(all_requests), args.temperature)

    completed = load_completed(args.out_dir)
    if completed:
        log.info("Found %d already-completed, resuming...", len(completed))
        all_requests = [
            r for r in all_requests
            if f"{r.split}_{r.idx}_{r.label}" not in completed
        ]
        log.info("Remaining requests: %d", len(all_requests))

    if not all_requests:
        log.info("Nothing to do!")
        return

    asyncio.run(collect_all(
        all_requests, args.out_dir,
        concurrency=args.concurrency,
        temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()
