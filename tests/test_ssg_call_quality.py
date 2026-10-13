from __future__ import annotations

from pathlib import Path

from token_sniper.graph.calls_evaluation import evaluate_ssg_calls_against_gold, load_call_gold_json

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "minimal_pkg"
GOLD_PATH = FIXTURE / "calls_gold.json"


def test_load_gold_json() -> None:
    gold = load_call_gold_json(GOLD_PATH)
    assert ("pack.main.Child.run", "pack.utils.helper") in gold
    assert len(gold) == 5


def test_call_metrics_against_minimal_fixture() -> None:
    result = evaluate_ssg_calls_against_gold(FIXTURE, GOLD_PATH)
    assert result["true_positives"] == 5
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0
