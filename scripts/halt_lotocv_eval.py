from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from token_sniper.guardrails.halt_dataset import AgentTraceDataset, halt_collate_fn
from token_sniper.guardrails.halt_model import HALTClassifier, HALTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_logits(
    ds: AgentTraceDataset,
    model: HALTClassifier,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=halt_collate_fn,
    )
    all_logits: list[float] = []
    all_labels: list[float] = []
    record_idx = 0
    task_ids: list[str] = []
    for batch in loader:
        logits = model(
            batch["logprobs"].to(device),
            batch["halt_features"].to(device),
            batch["atp"].to(device),
            batch["rank_encoding"].to(device),
            batch["attention_mask"].to(device),
        )
        labels = batch["labels"]
        for i in range(logits.size(0)):
            all_logits.append(float(logits[i].cpu()))
            all_labels.append(float(labels[i].cpu()))
            task_ids.append(ds.records[record_idx]["task_id"])
            record_idx += 1
    return np.array(all_logits), np.array(all_labels), task_ids


def aggregate_per_task(
    logits: np.ndarray,
    labels: np.ndarray,
    task_ids: list[str],
) -> dict[str, dict]:
    by_task: dict[str, dict] = defaultdict(lambda: {"logits": [], "label": None})
    for lg, lb, tid in zip(logits, labels, task_ids):
        by_task[tid]["logits"].append(lg)
        if by_task[tid]["label"] is None:
            by_task[tid]["label"] = lb

    aggregated = {}
    for tid, d in by_task.items():
        lgs = d["logits"]
        aggregated[tid] = {
            "n_turns": len(lgs),
            "logit_mean": float(np.mean(lgs)),
            "logit_max": float(np.max(lgs)),
            "logit_min": float(np.min(lgs)),
            "label": float(d["label"]),
            "binary": int(d["label"] >= 0.5),
        }
    return aggregated


def run_lotocv(
    per_task: dict[str, dict],
    aggregation: str = "logit_mean",
) -> dict:
    tasks = sorted(per_task.keys())
    X = np.array([per_task[t][aggregation] for t in tasks]).reshape(-1, 1)
    y_soft = np.array([per_task[t]["label"] for t in tasks])
    y_binary = np.array([per_task[t]["binary"] for t in tasks])

    if len(set(y_binary)) < 2:
        return {"error": "single class", "n_tasks": len(tasks)}

    cv_preds = np.zeros(len(tasks))
    fold_results = []
    for i, tid in enumerate(tasks):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_binary, i)

        if len(set(y_train)) < 2:
            cv_preds[i] = 0.5
            fold_results.append({"task": tid, "skipped": True})
            continue

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        prob = float(lr.predict_proba(X[i:i+1])[0, 1])
        cv_preds[i] = prob
        fold_results.append({
            "task": tid,
            "test_logit": float(X[i, 0]),
            "test_label": int(y_binary[i]),
            "test_prob": prob,
            "a": float(lr.coef_[0][0]),
            "b": float(lr.intercept_[0]),
        })

    lotocv_auroc = float(roc_auc_score(y_binary, cv_preds))

    lr_full = LogisticRegression(max_iter=1000)
    lr_full.fit(X, y_binary)
    full_a = float(lr_full.coef_[0][0])
    full_b = float(lr_full.intercept_[0])
    full_train_probs = lr_full.predict_proba(X)[:, 1]
    full_train_auroc = float(roc_auc_score(y_binary, full_train_probs))

    raw_probs = 1.0 / (1.0 + np.exp(-X.flatten()))
    raw_auroc = float(roc_auc_score(y_binary, raw_probs))

    return {
        "aggregation": aggregation,
        "n_tasks": len(tasks),
        "n_resolved": int((y_binary == 0).sum()),
        "n_failed": int((y_binary == 1).sum()),
        "raw_auroc_per_task": raw_auroc,
        "full_train_platt_auroc": full_train_auroc,
        "lotocv_platt_auroc": lotocv_auroc,
        "full_platt_a": full_a,
        "full_platt_b": full_b,
        "fold_results": fold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-task-out CV for HALT calibration")
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading agent traces from %s", args.training_data)
    ds = AgentTraceDataset(args.training_data)
    logger.info("Records: %d", len(ds))

    if len(ds) == 0:
        logger.error("Empty dataset")
        return

    logger.info("Loading checkpoint from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = HALTConfig(**ckpt["config"])
    model = HALTClassifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info("Collecting logits...")
    logits, labels, task_ids = collect_logits(ds, model, device)
    logger.info("Collected %d records from %d unique tasks", len(logits), len(set(task_ids)))

    per_task = aggregate_per_task(logits, labels, task_ids)

    results = {}
    for agg in ["logit_mean", "logit_max", "logit_min"]:
        logger.info("Running LOTOCV with aggregation=%s", agg)
        results[agg] = run_lotocv(per_task, aggregation=agg)
        info = results[agg]
        logger.info(
            "  [%s] raw_per_task=%.4f, full_platt=%.4f, lotocv_platt=%.4f (n=%d, %d resolved / %d failed)",
            agg, info.get("raw_auroc_per_task", 0), info.get("full_train_platt_auroc", 0),
            info.get("lotocv_platt_auroc", 0), info.get("n_tasks", 0),
            info.get("n_resolved", 0), info.get("n_failed", 0),
        )

    output_data = {
        "n_records": len(ds),
        "n_tasks": len(per_task),
        "per_task_logits": per_task,
        "lotocv_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", args.output)

    print("\n" + "=" * 70)
    print("LEAVE-ONE-TASK-OUT CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    for agg, info in results.items():
        if "error" in info:
            print(f"  {agg}: {info['error']}")
            continue
        print(f"\n  Aggregation: {agg}")
        print(f"    Raw HALT (no calibration):         AUROC = {info['raw_auroc_per_task']:.4f}")
        print(f"    Platt scaling (full train fit):    AUROC = {info['full_train_platt_auroc']:.4f}")
        print(f"    Platt scaling (LOTOCV, honest):    AUROC = {info['lotocv_platt_auroc']:.4f}")
        print(f"    n_tasks={info['n_tasks']}, resolved={info['n_resolved']}, failed={info['n_failed']}")


if __name__ == "__main__":
    main()
