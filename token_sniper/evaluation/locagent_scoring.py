from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

DatasetKey = Literal["lite_274", "locbench_560"]

_HF_DATASET_BY_KEY: dict[DatasetKey, str] = {
    "lite_274": "czlll/SWE-bench_Lite",
    "locbench_560": "czlll/Loc-Bench_V1",
}

LITE_K_GROUPS: list[list[int]] = [[1, 3, 5], [5, 10], [5, 10]]
LOCBENCH_K_GROUPS: list[list[int]] = [[5, 10], [10, 15], [10, 15]]

_K_GROUPS_BY_KEY: dict[DatasetKey, list[list[int]]] = {
    "lite_274": LITE_K_GROUPS,
    "locbench_560": LOCBENCH_K_GROUPS,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAGENT_VENV_PYTHON = _REPO_ROOT / "third_party" / "locagent" / ".venv" / "bin" / "python"
_RUNNER_SCRIPT = _REPO_ROOT / "third_party" / "_run_locagent_scoring.py"


class ScoringResult(dict):
    pass


def score_predictions(
    predictions_path: Path,
    *,
    dataset: DatasetKey,
    split: str = "test",
    selected_list: list[str] | None = None,
    k_values_list: list[list[int]] | None = None,
    python: Path = _LOCAGENT_VENV_PYTHON,
) -> ScoringResult:
    if not python.exists():
        raise RuntimeError(
            f"LocAgent scoring interpreter not found at {python}. "
            "Create it: /usr/local/bin/python3.13 -m venv third_party/locagent/.venv "
            "&& third_party/locagent/.venv/bin/pip install pandas torch datasets"
        )

    args = {
        "loc_file": str(predictions_path),
        "dataset": _HF_DATASET_BY_KEY[dataset],
        "split": split,
        "selected_list": selected_list,
        "k_values_list": k_values_list or _K_GROUPS_BY_KEY[dataset],
    }
    proc = subprocess.run(
        [str(python), str(_RUNNER_SCRIPT), json.dumps(args)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"LocAgent scoring subprocess failed:\n{proc.stderr}")

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    result: ScoringResult = ScoringResult()
    for (level, metric_at_k), value in zip(payload["columns"], payload["values"], strict=True):
        result.setdefault(level, {})[metric_at_k] = value
    return result
