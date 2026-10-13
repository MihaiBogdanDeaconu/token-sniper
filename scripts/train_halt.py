from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from token_sniper.guardrails.halt_dataset import (
    AgentTraceDataset,
    ColluBenchDataset,
    HALTDataset,
    halt_collate_fn,
)
from token_sniper.guardrails.halt_model import HALTClassifier, HALTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_SPLITS = ["qa", "summarization", "dialogue", "general"]
VAL_FRACTION = 0.2


def _stratified_split(
    ds: HALTDataset,
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    by_group: dict[tuple, list[int]] = {}
    for i, rec in enumerate(ds.records):
        if "task_id" in rec:
            key = ("agent", rec["task_id"])
        else:
            key = (rec.get("split", "unknown"), rec.get("idx", i))
        by_group.setdefault(key, []).append(i)

    groups_by_split: dict[str, list[tuple]] = {}
    for key in by_group:
        groups_by_split.setdefault(key[0], []).append(key)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for split_name, group_keys in sorted(groups_by_split.items()):
        rng.shuffle(group_keys)
        n_val_groups = max(1, int(len(group_keys) * val_fraction))
        for gk in group_keys[:n_val_groups]:
            val_idx.extend(by_group[gk])
        for gk in group_keys[n_val_groups:]:
            train_idx.extend(by_group[gk])
        logger.info(
            "  Split '%s': %d groups, %d val / %d train",
            split_name, len(group_keys), n_val_groups,
            len(group_keys) - n_val_groups,
        )

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.cpu().numpy()
    y_true = (labels_np >= 0.5).astype(int)

    metrics: dict[str, float] = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["auroc"] = 0.0
    metrics["accuracy"] = float(accuracy_score(y_true, preds))
    metrics["precision"] = float(precision_score(y_true, preds, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, preds, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, preds, zero_division=0))
    metrics["mae"] = float(((probs - labels_np) ** 2).mean() ** 0.5)
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(probs, labels_np)
        metrics["spearman"] = float(rho) if rho == rho else 0.0
    except ImportError:
        metrics["spearman"] = 0.0
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    nan_batches = 0

    for batch in loader:
        logprobs = batch["logprobs"].to(device)
        halt_feats = batch["halt_features"].to(device)
        atp = batch["atp"].to(device)
        rank_enc = batch["rank_encoding"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(logprobs, halt_feats, atp, rank_enc, mask)
        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if nan_batches > 0:
        logger.warning("Skipped %d NaN/Inf batches during training", nan_batches)

    if not all_logits:
        return {"loss": float("nan"), "auroc": 0.0, "accuracy": 0.0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    metrics = compute_metrics(all_logits_t, all_labels_t)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        logprobs = batch["logprobs"].to(device)
        halt_feats = batch["halt_features"].to(device)
        atp = batch["atp"].to(device)
        rank_enc = batch["rank_encoding"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(logprobs, halt_feats, atp, rank_enc, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits_t = torch.cat(all_logits)
    all_labels_t = torch.cat(all_labels)
    metrics = compute_metrics(all_logits_t, all_labels_t)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HALT hallucination classifier")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dir with {split}.jsonl logprob files (HaluEval)")
    parser.add_argument("--collu-bench-dir", type=Path, default=None,
                        help="Dir with Collu-Bench Arrow dataset")
    parser.add_argument("--agent-data", type=Path, default=None,
                        help="JSONL with agent edit-turn logprobs + labels (from analyze_halt_swebench.py)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/guardrails"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="Disable pos_weight in BCE loss (use for balanced fine-tuning)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the classification head (linear probe).")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Just print pre-sigmoid logit distribution on the data and exit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_list = []
    if args.data_dir:
        logger.info("Loading HaluEval data (splits: %s)", ALL_SPLITS)
        halueval_ds = HALTDataset(args.data_dir, splits=ALL_SPLITS, max_seq_len=args.max_seq_len)
        if len(halueval_ds) > 0:
            datasets_list.append(halueval_ds)
    if args.collu_bench_dir:
        logger.info("Loading Collu-Bench data from %s", args.collu_bench_dir)
        collu_ds = ColluBenchDataset(args.collu_bench_dir, max_seq_len=args.max_seq_len)
        if len(collu_ds) > 0:
            datasets_list.append(collu_ds)
    if args.agent_data:
        logger.info("Loading agent trace data from %s", args.agent_data)
        agent_ds = AgentTraceDataset(args.agent_data, max_seq_len=args.max_seq_len)
        if len(agent_ds) > 0:
            datasets_list.append(agent_ds)

    if not datasets_list:
        logger.error("No data loaded. Provide --data-dir and/or --collu-bench-dir")
        return

    if len(datasets_list) == 1:
        full_ds = datasets_list[0]
    else:
        from torch.utils.data import ConcatDataset
        full_ds = ConcatDataset(datasets_list)
        full_ds.records = []
        for ds in datasets_list:
            full_ds.records.extend(ds.records)

    if len(full_ds) == 0:
        logger.error("No records after loading")
        return

    train_idx, val_idx = _stratified_split(full_ds, args.val_fraction, rng)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_labels = [float(full_ds.records[i]["label"]) for i in train_idx]
    val_labels = [float(full_ds.records[i]["label"]) for i in val_idx]
    soft_mode = any(0.0 < v < 1.0 for v in train_labels + val_labels)
    logger.info(
        "Train: %d samples (mean_label=%.3f), Val: %d samples (mean_label=%.3f) [%s]",
        len(train_ds),
        sum(train_labels) / max(len(train_labels), 1),
        len(val_ds),
        sum(val_labels) / max(len(val_labels), 1),
        "soft labels" if soft_mode else "binary",
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=halt_collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=halt_collate_fn, num_workers=2, pin_memory=True,
    ) if len(val_ds) > 0 else None

    cfg = HALTConfig(
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    model = HALTClassifier(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded checkpoint from %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))
        has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
        has_inf = any(torch.isinf(p).any().item() for p in model.parameters())
        logger.info("Checkpoint health: NaN=%s, Inf=%s", has_nan, has_inf)

    if args.diagnostic_only:
        _run_diagnostic(model, train_loader, val_loader, device, args.output_dir)
        return

    if args.freeze_backbone:
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            if name.startswith("head."):
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
                frozen += p.numel()
        logger.info(
            "Linear probe mode: frozen=%s params, trainable=%s params (head only)",
            f"{frozen:,}", f"{trainable:,}",
        )

    logger.info("Model parameters: %s", f"{model.num_parameters:,}")

    if args.no_pos_weight or soft_mode:
        criterion = nn.BCEWithLogitsLoss()
        if soft_mode:
            logger.info("Soft-label mode: BCEWithLogitsLoss (no pos_weight)")
        else:
            logger.info("Using BCEWithLogitsLoss without pos_weight")
    else:
        pos_weight = torch.tensor(
            [(len(train_labels) - sum(train_labels)) / max(sum(train_labels), 1)],
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("pos_weight=%.3f", pos_weight.item())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auroc = 0.0
    best_epoch = -1
    patience_left = args.patience
    history: list[dict] = []

    probe_batch = next(iter(train_loader))
    with torch.no_grad():
        pb_logits = model(
            probe_batch["logprobs"].to(device),
            probe_batch["halt_features"].to(device),
            probe_batch["atp"].to(device),
            probe_batch["rank_encoding"].to(device),
            probe_batch["attention_mask"].to(device),
        )
        pb_loss = criterion(pb_logits, probe_batch["labels"].to(device))
        logger.info(
            "Probe batch: logits range=[%.3f, %.3f], loss=%.4f, "
            "feat NaN=%d, logprob NaN=%d",
            pb_logits.min().item(), pb_logits.max().item(), pb_loss.item(),
            probe_batch["halt_features"].isnan().sum().item(),
            probe_batch["logprobs"].isnan().sum().item(),
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)

        logger.info(
            "Epoch %2d/%d | train loss=%.4f auroc=%.4f | val loss=%.4f auroc=%.4f | lr=%.2e | %.1fs",
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["auroc"],
            val_metrics.get("loss", 0.0), val_metrics.get("auroc", 0.0),
            scheduler.get_last_lr()[0], elapsed,
        )

        improved = False
        if val_metrics:
            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                best_epoch = epoch
                improved = True
                patience_left = args.patience
                _save_checkpoint(model, cfg, epoch, val_metrics, args.output_dir / "best.pt")
                logger.info("  -> New best val AUROC=%.4f, saved checkpoint", best_auroc)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                    break
        else:
            _save_checkpoint(model, cfg, epoch, train_metrics, args.output_dir / "best.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "total_epochs": epoch,
        "model_params": model.num_parameters,
        "split_strategy": "stratified_80_20_per_category",
        "val_fraction": args.val_fraction,
        "all_splits": ALL_SPLITS,
        "config": {
            "model_dim": cfg.model_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "ffn_dim": cfg.ffn_dim,
            "top_k": cfg.top_k,
            "max_seq_len": cfg.max_seq_len,
        },
        "history": history,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Training complete. Summary -> %s", summary_path)
    logger.info("Best val AUROC=%.4f at epoch %d", best_auroc, best_epoch)


def _save_checkpoint(
    model: HALTClassifier,
    cfg: HALTConfig,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "epoch": epoch,
        "metrics": metrics,
    }, path)


@torch.no_grad()
def _run_diagnostic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    output_dir: Path,
) -> None:
    import numpy as np

    model.eval()

    def _collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        all_logits, all_labels = [], []
        for batch in loader:
            logits = model(
                batch["logprobs"].to(device),
                batch["halt_features"].to(device),
                batch["atp"].to(device),
                batch["rank_encoding"].to(device),
                batch["attention_mask"].to(device),
            )
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
        return np.concatenate(all_logits), np.concatenate(all_labels)

    train_logits, train_labels = _collect(train_loader)
    val_logits = val_labels = None
    if val_loader is not None:
        val_logits, val_labels = _collect(val_loader)

    def _stats(name: str, logits: np.ndarray, labels: np.ndarray) -> dict:
        probs = 1.0 / (1.0 + np.exp(-logits))
        out = {
            "n": int(len(logits)),
            "logit_mean": float(logits.mean()),
            "logit_stdev": float(logits.std()),
            "logit_min": float(logits.min()),
            "logit_max": float(logits.max()),
            "prob_mean": float(probs.mean()),
            "prob_min": float(probs.min()),
            "prob_max": float(probs.max()),
        }
        logger.info(
            "[%s] n=%d, logit mean=%.3f stdev=%.3f range=[%.3f, %.3f] "
            "prob mean=%.4f range=[%.4f, %.4f]",
            name, out["n"], out["logit_mean"], out["logit_stdev"],
            out["logit_min"], out["logit_max"],
            out["prob_mean"], out["prob_min"], out["prob_max"],
        )

        try:
            from sklearn.metrics import roc_auc_score
            y_true = (labels >= 0.5).astype(int)
            if len(set(y_true)) > 1:
                out["raw_auroc"] = float(roc_auc_score(y_true, probs))
                logger.info("[%s] Raw AUROC = %.4f", name, out["raw_auroc"])
        except (ValueError, ImportError):
            out["raw_auroc"] = None

        return out

    train_stats = _stats("TRAIN", train_logits, train_labels)
    val_stats = _stats("VAL", val_logits, val_labels) if val_logits is not None else None

    platt_result: dict[str, Any] = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        y_train = (train_labels >= 0.5).astype(int)
        if len(set(y_train)) > 1:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(train_logits.reshape(-1, 1), y_train)
            platt_a = float(lr.coef_[0][0])
            platt_b = float(lr.intercept_[0])
            train_calibrated = lr.predict_proba(train_logits.reshape(-1, 1))[:, 1]
            platt_result["a"] = platt_a
            platt_result["b"] = platt_b
            platt_result["train_auroc"] = float(roc_auc_score(y_train, train_calibrated))
            logger.info(
                "Platt scaling: a=%.4f, b=%.4f -> train AUROC=%.4f",
                platt_a, platt_b, platt_result["train_auroc"],
            )
            if val_logits is not None:
                y_val = (val_labels >= 0.5).astype(int)
                if len(set(y_val)) > 1:
                    val_calibrated = lr.predict_proba(val_logits.reshape(-1, 1))[:, 1]
                    platt_result["val_auroc"] = float(roc_auc_score(y_val, val_calibrated))
                    logger.info("Platt scaling val AUROC=%.4f", platt_result["val_auroc"])
    except Exception as e:
        logger.warning("Platt scaling failed: %s", e)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic.json").write_text(
        json.dumps({
            "train": train_stats,
            "val": val_stats,
            "platt_scaling": platt_result,
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("Diagnostic saved to %s", output_dir / "diagnostic.json")


if __name__ == "__main__":
    main()
