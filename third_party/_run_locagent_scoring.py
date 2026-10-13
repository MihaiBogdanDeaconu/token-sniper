"""Subprocess entry point: calls the vendored, untouched LocAgent
evaluate_results() under its own Python 3.13 venv (third_party/locagent/.venv)
and prints the result as JSON on stdout.

Kept outside third_party/locagent/ so the vendored tree stays an exact,
unmodified copy of upstream (see third_party/LOCAGENT_COMMIT.txt) -- this
file is token-sniper glue code, not vendored code.

Usage:
    third_party/locagent/.venv/bin/python third_party/_run_locagent_scoring.py '<json args>'

json args: {"loc_file": str, "dataset": str, "split": str,
            "k_values_list": [[..],[..],[..]], "selected_list": [str] | null}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "locagent"))

LEVEL2KEY = {"file": "found_files", "module": "found_modules", "function": "found_entities"}


def main() -> None:
    args = json.loads(sys.argv[1])

    from evaluation import eval_metric  # type: ignore[import-not-found]

    df = eval_metric.evaluate_results(
        args["loc_file"],
        LEVEL2KEY,
        dataset=args["dataset"],
        split=args["split"],
        selected_list=args.get("selected_list"),
        k_values_list=args.get("k_values_list"),
    )
    # df has a 3-level column MultiIndex (file/module/function, then metric@k).
    payload = {
        "columns": [list(c) for c in df.columns],
        "values": df.iloc[0].tolist(),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
