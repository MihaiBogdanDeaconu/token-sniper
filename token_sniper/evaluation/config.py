from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CacheLevel = Literal["none", "base", "env", "instance"]


class PathsConfig(BaseModel):
    experiments_dir: str = "experiments/"

    model_config = {"frozen": True}


class RetrievalConfig(BaseModel):
    top_k_values: list[int] = Field(default_factory=lambda: [10, 20, 50, 100, 200])

    model_config = {"frozen": True}


class HarnessConfig(BaseModel):
    dataset_name: str = "SWE-bench-Live/SWE-bench-Live"
    split: str = "verified"
    max_workers: int = Field(default=1, gt=0)
    cache_level: CacheLevel = "env"
    clean: bool = False
    force_rebuild: bool = False
    open_file_limit: int = Field(default=4096, gt=0)
    timeout_seconds: int = Field(default=1800, gt=0)
    namespace: str | None = None
    rewrite_reports: bool = False
    modal: bool = False
    instance_image_tag: str = "latest"
    env_image_tag: str = "latest"

    model_config = {"frozen": True}


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "pass_at_1",
            "pass_at_3",
            "token_efficiency",
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_map",
        ]
    )
    num_seeds: int = Field(default=3, gt=0)
    significance_test: str = "mcnemar"
    significance_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)

    model_config = {"frozen": True}


class MLflowConfig(BaseModel):
    tracking_uri: str = "experiments/mlflow/"
    experiment_prefix: str = "token-sniper"

    model_config = {"frozen": True}


class EvaluationBaseConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    model_config = {"frozen": True}


def load_evaluation_base_config(config_path: Path | str) -> EvaluationBaseConfig:
    path = Path(config_path).resolve()
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvaluationBaseConfig.model_validate(raw_data)
