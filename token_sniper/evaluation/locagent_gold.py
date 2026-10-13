from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

DatasetKey = Literal["lite_274", "locbench_560"]

_HF_DATASET_BY_KEY: dict[DatasetKey, str] = {
    "lite_274": "czlll/SWE-bench_Lite",
    "locbench_560": "czlll/Loc-Bench_V1",
}


@dataclass(frozen=True)
class LocAgentGoldLabels:
    instance_id: str
    gold_files: list[str]
    gold_modules: list[str]
    gold_entities: list[str]


def extract_gold_labels(instance_id: str, edit_functions: list[str]) -> LocAgentGoldLabels:
    gold_files: list[str] = []
    gold_modules: list[str] = []
    gold_entities: list[str] = []

    for func in edit_functions:
        fn, _, qualname = func.partition(":")

        if fn not in gold_files:
            gold_files.append(fn)

        mname = qualname.split(".")[0]
        mid = f"{fn}:{mname}"
        if mid not in gold_modules:
            gold_modules.append(mid)

        entity_qualname = qualname
        if entity_qualname.endswith(".__init__"):
            entity_qualname = entity_qualname[: -len(".__init__")]
        eid = f"{fn}:{entity_qualname}"
        if eid not in gold_entities:
            gold_entities.append(eid)

    return LocAgentGoldLabels(
        instance_id=instance_id,
        gold_files=gold_files,
        gold_modules=gold_modules,
        gold_entities=gold_entities,
    )


@lru_cache(maxsize=4)
def _load_hf_rows(hf_id: str, split: str) -> tuple[dict[str, Any], ...]:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split)
    return tuple(dict(row) for row in ds)


def dataset_rows(dataset_key: DatasetKey, *, split: str = "test") -> tuple[dict[str, Any], ...]:
    return _load_hf_rows(_HF_DATASET_BY_KEY[dataset_key], split)


def build_gold_index(dataset_key: DatasetKey, *, split: str = "test") -> dict[str, LocAgentGoldLabels]:
    return {
        row["instance_id"]: extract_gold_labels(row["instance_id"], row["edit_functions"])
        for row in dataset_rows(dataset_key, split=split)
    }
