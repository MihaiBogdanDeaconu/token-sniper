from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

from datasets import load_dataset


def _summ(xs: list[int]) -> dict[str, float | int]:
    return {
        "min": min(xs),
        "max": max(xs),
        "mean": round(statistics.mean(xs), 2),
        "median": float(statistics.median(xs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SWE-bench Verified dataset statistics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/datasets/swe_bench_verified_stats.json"),
        help="Path to write JSON summary",
    )
    args = parser.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    repos: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    ftp_counts: list[int] = []
    ptp_counts: list[int] = []
    patch_lines: list[int] = []
    patch_chars: list[int] = []
    ps_chars: list[int] = []
    hint_nonempty = 0

    for row in ds:
        repos[row["repo"]] += 1
        d = row.get("difficulty") or "unknown"
        difficulties[str(d)] += 1
        try:
            ftp_raw = row["FAIL_TO_PASS"]
            ptp_raw = row["PASS_TO_PASS"]
            ftp = json.loads(ftp_raw) if isinstance(ftp_raw, str) else ftp_raw
            ptp = json.loads(ptp_raw) if isinstance(ptp_raw, str) else ptp_raw
        except (json.JSONDecodeError, TypeError):
            ftp, ptp = [], []
        ftp_counts.append(len(ftp))
        ptp_counts.append(len(ptp))
        p = row["patch"] or ""
        patch_chars.append(len(p))
        patch_lines.append(p.count("\n") + (1 if p else 0))
        ps = row["problem_statement"] or ""
        ps_chars.append(len(ps))
        if (row.get("hints_text") or "").strip():
            hint_nonempty += 1

    out = {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "generated_at": date.today().isoformat(),
        "num_instances": len(ds),
        "repos": dict(sorted(repos.items(), key=lambda x: (-x[1], x[0]))),
        "num_repos": len(repos),
        "difficulty": dict(difficulties),
        "fail_to_pass_tests_per_instance": _summ(ftp_counts),
        "pass_to_pass_tests_per_instance": _summ(ptp_counts),
        "gold_patch": {"lines": _summ(patch_lines), "chars": _summ(patch_chars)},
        "problem_statement": {"chars": _summ(ps_chars)},
        "hints_text_nonempty_fraction": round(hint_nonempty / len(ds), 4),
        "schema_columns": ds.column_names,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
