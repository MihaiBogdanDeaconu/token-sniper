from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_PATH = Path("third_party/locagent/evaluation/loc_output/locagent/claude_3-5/loc_outputs.jsonl")

PUBLISHED = {
    ("file", "Acc@5"): 0.9416,
    ("module", "Acc@10"): 0.8759,
    ("function", "Acc@10"): 0.7737,
}

TOLERANCE = 1e-3


def main() -> int:
    from token_sniper.evaluation.locagent_scoring import score_predictions

    if not PREDICTIONS_PATH.exists():
        logger.error("Missing %s -- did the LocAgent vendoring go through?", PREDICTIONS_PATH)
        return 1

    result = score_predictions(PREDICTIONS_PATH, dataset="lite_274")

    ok = True
    for (level, metric), expected in PUBLISHED.items():
        actual = result.get(level, {}).get(metric)
        if actual is None:
            logger.error("%s.%s missing from scoring output", level, metric)
            ok = False
            continue
        diff = abs(actual - expected)
        status = "OK" if diff <= TOLERANCE else "MISMATCH"
        if diff > TOLERANCE:
            ok = False
        logger.info("%s.%s: expected=%.4f actual=%.4f diff=%.4f [%s]", level, metric, expected, actual, diff, status)

    if not ok:
        logger.error("verify_locagent_baseline: FAILED -- do not trust our own T1/T2 numbers yet.")
        return 1
    logger.info("verify_locagent_baseline: PASS -- harness reproduces LocAgent's published Lite-274 numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
