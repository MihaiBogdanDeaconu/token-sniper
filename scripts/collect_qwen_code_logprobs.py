from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

VLLM_BASE = "http://qwen-vllm:8077/v1"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
TOP_K = 20
MAX_GEN_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.95


@dataclass
class CodeTask:
    task_id: str
    prompt: str
    test_code: str
    entry_point: str


def load_humaneval() -> list[CodeTask]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    tasks = []
    for row in ds:
        tasks.append(CodeTask(
            task_id=row["task_id"],
            prompt=row["prompt"],
            test_code=row["test"],
            entry_point=row["entry_point"],
        ))
    return tasks


def load_mbpp() -> list[CodeTask]:
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="train")
    tasks = []
    for row in ds:
        prompt = (
            f"\"\"\"\n{row['prompt']}\n"
            f"Example: {row['test_list'][0]}\n\"\"\"\n"
        )
        test_code = "\n".join(row["test_list"])
        tasks.append(CodeTask(
            task_id=f"MBPP/{row['task_id']}",
            prompt=prompt,
            test_code=test_code,
            entry_point="",
        ))
    return tasks


async def generate_sample(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    task: CodeTask,
    sample_idx: int,
) -> dict[str, Any] | None:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. "
                    "Return only code, no markdown fences or explanation.\n\n"
                    f"```python\n{task.prompt}\n```"
                ),
            },
        ],
        "max_tokens": MAX_GEN_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }

    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{VLLM_BASE}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("[%s sample=%d] HTTP %d: %s", task.task_id, sample_idx, resp.status, body[:200])
                        await asyncio.sleep(2 ** attempt)
                        continue
                    data = await resp.json()

                choice = data["choices"][0]
                msg = choice["message"]
                completion = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                lp_content = choice.get("logprobs", {}).get("content", [])

                if not lp_content or len(lp_content) < 3:
                    return None
                if not completion:
                    return {
                        "task_id": task.task_id,
                        "sample_idx": sample_idx,
                        "completion": "",
                        "reasoning": reasoning,
                        "logprobs": [
                            [e["logprob"] for e in (tok.get("top_logprobs") or [])[:TOP_K]]
                            + [-100.0] * max(0, TOP_K - len(tok.get("top_logprobs") or []))
                            for tok in lp_content
                        ],
                        "n_tokens": len(lp_content),
                        "no_final_answer": True,
                    }

                logprobs = []
                for tok_obj in lp_content:
                    top = tok_obj.get("top_logprobs", [])
                    vals = [e["logprob"] for e in top[:TOP_K]]
                    while len(vals) < TOP_K:
                        vals.append(-100.0)
                    logprobs.append(vals)

                return {
                    "task_id": task.task_id,
                    "sample_idx": sample_idx,
                    "completion": completion,
                    "reasoning": reasoning,
                    "logprobs": logprobs,
                    "n_tokens": len(logprobs),
                    "no_final_answer": False,
                }
            except Exception as e:
                log.warning(
                    "[%s sample=%d] attempt %d: %s: %s",
                    task.task_id, sample_idx, attempt, type(e).__name__, e,
                )
                await asyncio.sleep(min(2 ** attempt + 1, 8))
    return None


def _strip_markdown_fences(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _run_in_subprocess(code: str, timeout: float = 5.0) -> tuple[bool, str]:
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout or "non-zero exit")[:500]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def execute_humaneval(
    task: CodeTask, completion: str,
) -> tuple[bool, str]:
    code = _strip_markdown_fences(completion)
    full_program = task.prompt + code + "\n\n" + task.test_code + (
        f"\n\ncheck({task.entry_point})\n"
    )
    return _run_in_subprocess(full_program)


def execute_mbpp(task: CodeTask, completion: str) -> tuple[bool, str]:
    code = _strip_markdown_fences(completion)
    full_program = code + "\n\n" + task.test_code
    return _run_in_subprocess(full_program)


def execute(task: CodeTask, completion: str, dataset: str) -> tuple[bool, str]:
    if dataset == "humaneval":
        return execute_humaneval(task, completion)
    return execute_mbpp(task, completion)


async def collect_all(
    tasks: list[CodeTask],
    samples_per_task: int,
    out_path: Path,
    concurrency: int,
    dataset: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add((rec["task_id"], rec["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    log.info("Resume: %d records already done", len(done_keys))

    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4, limit_per_host=concurrency + 4)

    request_units = [
        (t, i) for t in tasks for i in range(samples_per_task)
        if (t.task_id, i) not in done_keys
    ]
    log.info("Pending generations: %d", len(request_units))

    t0 = time.time()
    pass_count = fail_count = error_count = 0
    n_done = 0

    out_f = open(out_path, "a")

    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = 32
        for batch_start in range(0, len(request_units), batch_size):
            batch = request_units[batch_start: batch_start + batch_size]
            coros = [generate_sample(session, sem, t, i) for t, i in batch]
            results = await asyncio.gather(*coros)

            for (task, sample_idx), result in zip(batch, results):
                if result is None:
                    error_count += 1
                    continue

                if result.get("no_final_answer"):
                    passed = False
                    err = "NO_FINAL_ANSWER (model reasoned but never produced code)"
                else:
                    passed, err = execute(task, result["completion"], dataset)

                if passed:
                    pass_count += 1
                else:
                    fail_count += 1

                out_record = {
                    "task_id": result["task_id"],
                    "sample_idx": result["sample_idx"],
                    "label": 0 if passed else 1,
                    "binary_label": 0 if passed else 1,
                    "logprobs": result["logprobs"],
                    "n_tokens": result["n_tokens"],
                    "passed": passed,
                    "no_final_answer": result.get("no_final_answer", False),
                    "exec_error": err if not passed else "",
                }
                out_f.write(json.dumps(out_record) + "\n")
                out_f.flush()
                n_done += 1

            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1.0)
            remaining = len(request_units) - batch_start - batch_size
            eta = remaining / max(rate, 0.01)
            log.info(
                "Progress: %d/%d (%.1f%%) | pass=%d fail=%d err=%d | %.2f gen/s | ETA %.0fs",
                n_done, len(request_units), 100 * n_done / max(len(request_units), 1),
                pass_count, fail_count, error_count, rate, eta,
            )

    out_f.close()
    log.info("=== Collection complete ===")
    log.info(
        "Total: %d records (%d pass, %d fail, %d errors). Pass rate: %.1f%%",
        n_done, pass_count, fail_count, error_count,
        100 * pass_count / max(pass_count + fail_count, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    log.info("Loading %s tasks", args.dataset)
    if args.dataset == "humaneval":
        tasks = load_humaneval()
    else:
        tasks = load_mbpp()

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    log.info("Loaded %d tasks", len(tasks))

    asyncio.run(collect_all(
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        out_path=args.out,
        concurrency=args.concurrency,
        dataset=args.dataset,
    ))


if __name__ == "__main__":
    main()
