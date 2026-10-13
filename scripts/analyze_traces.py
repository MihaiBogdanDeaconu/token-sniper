from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/workspace/repo/experiments/halt_swebench",
    )
    args = parser.parse_args()
    rd = Path(args.results_dir)

    with open(rd / "results.jsonl") as f:
        results = [json.loads(l) for l in f]

    patched = []
    no_patch = []
    for r in results:
        tid = r["instance_id"]
        pp = rd / tid / "agent_output" / "patch.diff"
        has_patch = pp.is_file() and pp.stat().st_size > 0
        (patched if has_patch else no_patch).append(tid)

    print(f"Total tasks: {len(results)}")
    print(f"With patches: {len(patched)}")
    print(f"Without patches: {len(no_patch)}")

    print("\n" + "=" * 70)
    print("FAILED TASKS (no patch) - last tool calls")
    print("=" * 70)
    for tid in sorted(no_patch):
        sp = rd / tid / "agent_output" / "session.json"
        if not sp.is_file():
            print(f"\n{tid}: NO SESSION")
            continue
        session = json.loads(sp.read_text())
        turns = session.get("turns", [])
        wall = session.get("total_wall_time_seconds", 0)
        patches = session.get("patch_attempt_count", 0)
        valid = session.get("valid_patch_attempt_count", 0)

        tool_sequence = []
        for t in turns:
            for tc in t.get("tool_calls", []):
                name = tc.get("tool_name", "?")
                tool_sequence.append(name)

        last_10 = tool_sequence[-10:]
        unique_last_10 = len(set(last_10))

        print(f"\n{tid}")
        print(f"  turns={len(turns)}, wall={wall:.0f}s, patch_attempts={patches}, valid={valid}")
        print(f"  tool sequence (last 10): {last_10}")
        if unique_last_10 <= 2:
            print(f"  ** LIKELY LOOP (only {unique_last_10} unique tools in last 10)")

        for t in reversed(turns[-3:]):
            text = (t.get("assistant_text", "") or "").strip()
            if text:
                print(f"  last text: {text[:200]}")
                break

    print("\n" + "=" * 70)
    print("PATCHED TASKS - summary")
    print("=" * 70)
    for tid in sorted(patched):
        sp = rd / tid / "agent_output" / "session.json"
        if not sp.is_file():
            print(f"  {tid}: NO SESSION")
            continue
        session = json.loads(sp.read_text())
        turns = session.get("turns", [])
        wall = session.get("total_wall_time_seconds", 0)
        patches = session.get("patch_attempt_count", 0)
        valid = session.get("valid_patch_attempt_count", 0)
        halt = session.get("halt_submit_score")
        halt_str = f", halt={halt:.3f}" if halt is not None else ""

        tool_sequence = []
        for t in turns:
            for tc in t.get("tool_calls", []):
                name = tc.get("tool_name", "?")
                tool_sequence.append(name)

        tool_counts: dict[str, int] = {}
        for name in tool_sequence:
            tool_counts[name] = tool_counts.get(name, 0) + 1

        print(f"  {tid}: {len(turns)} turns, {wall:.0f}s, patches={valid}{halt_str}")
        print(f"    tools: {dict(sorted(tool_counts.items(), key=lambda x: -x[1]))}")

    report_file = Path("token-sniper-agent.grade-run.json")
    if report_file.is_file():
        report = json.loads(report_file.read_text())
        resolved = set(report.get("resolved_ids", []))
        print(f"\n{'=' * 70}")
        print(f"SWEBENCH GRADING: {len(resolved)} resolved")
        print(f"{'=' * 70}")
        for tid in sorted(patched):
            status = "RESOLVED" if tid in resolved else "FAILED"
            print(f"  {status:10s} {tid}")


if __name__ == "__main__":
    main()
