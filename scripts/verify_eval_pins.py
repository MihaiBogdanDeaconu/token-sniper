from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_PIN_PATH = Path("configs/phases/eval.json")

CHECKS = [
    {
        "pin_key": "lite_274",
        "hf_id": "czlll/SWE-bench_Lite",
        "split": "test",
        "required_fields": ["edit_functions"],
    },
    {
        "pin_key": "locbench_560",
        "hf_id": "czlll/Loc-Bench_V1",
        "split": "test",
        "required_fields": ["edit_functions", "category"],
    },
]


def _verify_one(pin: dict, check: dict) -> bool:
    from datasets import load_dataset

    pin_key = check["pin_key"]
    hf_id = check["hf_id"]
    split = check["split"]

    pinned_ids = set(pin[pin_key]["instance_ids"])
    logger.info("Loading %s split=%s to verify against %s (%d pinned IDs)", hf_id, split, pin_key, len(pinned_ids))

    ds = load_dataset(hf_id, split=split)
    live_ids = set(ds["instance_id"])

    missing_from_live = pinned_ids - live_ids
    extra_in_live = live_ids - pinned_ids

    ok = True
    if missing_from_live:
        ok = False
        logger.error(
            "%s: %d pinned IDs not found in %s: %s",
            pin_key, len(missing_from_live), hf_id,
            sorted(missing_from_live)[:10],
        )
    if extra_in_live:
        logger.warning(
            "%s: %d IDs present in %s but not pinned (expected if %s is a strict subset): %s",
            pin_key, len(extra_in_live), hf_id, pin_key,
            sorted(extra_in_live)[:10],
        )

    missing_fields = [f for f in check["required_fields"] if f not in ds.column_names]
    if missing_fields:
        ok = False
        logger.error("%s: %s is missing required field(s) %s", pin_key, hf_id, missing_fields)

    if ok:
        logger.info("%s: OK — %d/%d IDs match, required fields present", pin_key, len(pinned_ids), len(pinned_ids))
    return ok


def main() -> int:
    pin = json.loads(EVAL_PIN_PATH.read_text())

    all_ok = True
    for check in CHECKS:
        all_ok = _verify_one(pin, check) and all_ok

    if not all_ok:
        logger.error("verify_eval_pins: FAILED — see above. Pin file was NOT modified.")
        return 1
    logger.info("verify_eval_pins: PASS — all pinned instance IDs match their published czlll source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
