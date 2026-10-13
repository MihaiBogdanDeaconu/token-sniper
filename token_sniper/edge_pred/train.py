from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn

from token_sniper.edge_pred.dataset import TaskGraph
from token_sniper.edge_pred.model import EdgePredictor, EdgePredictorConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    output_dir: Path
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    negative_ratio: int = 5
    pos_weight: float = 5.0
    grad_accum_graphs: int = 4
    patience: int = 5
    hidden_dim: int = 384
    out_dim: int = 256
    val_fraction: float = 0.1
    seed: int = 42
    device: str | None = None
    mlflow_experiment: str = "phase2_edge_predictor"
    max_grad_norm: float = 1.0
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    lr_min_factor: float = 0.01


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.pos_weight + (1 - targets) * 1.0
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * bce).mean()


import torch.nn.functional as F  # noqa: E402


def _split_positives_for_validation(
    tasks: Sequence[TaskGraph],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[TaskGraph], dict[str, torch.Tensor]]:
    val_positives: dict[str, torch.Tensor] = {}
    out: list[TaskGraph] = []
    for tg in tasks:
        n_pos = tg.positive_pairs.shape[0]
        if n_pos == 0:
            val_positives[tg.instance_id] = torch.empty((0, 2), dtype=torch.long)
            out.append(tg)
            continue
        n_val = max(1, int(round(n_pos * val_fraction)))
        idx = list(range(n_pos))
        rng.shuffle(idx)
        val_idx = sorted(idx[:n_val])
        train_idx = sorted(idx[n_val:])
        val_positives[tg.instance_id] = tg.positive_pairs[val_idx].clone()
        tg_copy = TaskGraph(
            instance_id=tg.instance_id,
            repo=tg.repo,
            node_ids=tg.node_ids,
            features=tg.features,
            edge_index=tg.edge_index,
            positive_pairs=tg.positive_pairs[train_idx].clone(),
            candidate_pairs=tg.candidate_pairs,
            trace_edges=tg.trace_edges,
        )
        out.append(tg_copy)
    return out, val_positives


def _forward_task(model: EdgePredictor, tg: TaskGraph, device: torch.device) -> torch.Tensor:
    feats = tg.features
    return model.encode(
        feats.type_onehot.to(device),
        feats.signature_emb.to(device),
        feats.file_idx.to(device),
        feats.degree_log.to(device),
        tg.edge_index.to(device),
    )


def _score_pairs_batched(
    model: EdgePredictor,
    h: torch.Tensor,
    pairs: torch.Tensor,
    batch_size: int = 200_000,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0, device=h.device)
    outs: list[torch.Tensor] = []
    for i in range(0, pairs.shape[0], batch_size):
        outs.append(model.score_pairs(h, pairs[i : i + batch_size].to(h.device)))
    return torch.cat(outs, dim=0)


def _auc_roc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.numel() == 0 or labels.min() == labels.max():
        return 0.5
    return float(roc_auc_score(labels.cpu().numpy(), logits.detach().cpu().numpy()))


def train_edge_predictor(
    train_tasks: Sequence[TaskGraph],
    cfg: TrainConfig,
) -> Path:
    from token_sniper.edge_pred.features import _pick_device
    device = torch.device(cfg.device or _pick_device())
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger.info("Preparing train/val split (val_fraction=%.2f)", cfg.val_fraction)
    train_split, val_positives = _split_positives_for_validation(train_tasks, cfg.val_fraction, rng)

    num_files = max(max(tg.features.file_idx.max().item() + 1 for tg in train_tasks), 1)

    model_cfg = EdgePredictorConfig(
        num_files=int(num_files),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
    )
    model = EdgePredictor(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.lr_min_factor
    )

    if cfg.use_focal_loss:
        loss_fn = FocalLossWithLogits(gamma=cfg.focal_gamma, pos_weight=cfg.pos_weight)
        logger.info(
            "Using Focal Loss (gamma=%.1f, pos_weight=%.1f)",
            cfg.focal_gamma, cfg.pos_weight,
        )
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight], device=device))
        logger.info("Using BCEWithLogitsLoss (pos_weight=%.1f)", cfg.pos_weight)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    best_auc = -1.0
    patience_left = cfg.patience

    mlflow_enabled = False
    try:
        import mlflow
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "hidden_dim": cfg.hidden_dim,
            "out_dim": cfg.out_dim,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "negative_ratio": cfg.negative_ratio,
            "pos_weight": cfg.pos_weight,
            "epochs": cfg.epochs,
            "train_tasks": len(train_tasks),
            "num_files": num_files,
            "seed": cfg.seed,
            "use_focal_loss": cfg.use_focal_loss,
            "focal_gamma": cfg.focal_gamma,
            "max_grad_norm": cfg.max_grad_norm,
        })
        mlflow_enabled = True
    except ImportError:
        logger.warning("MLflow not available, continuing without run tracking")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %d total params, %d trainable, device=%s",
        total_params, trainable_params, device,
    )

    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            shuffled = list(train_split)
            rng.shuffle(shuffled)
            epoch_loss = 0.0
            step = 0
            opt.zero_grad(set_to_none=True)

            for tg in shuffled:
                if tg.positive_pairs.numel() == 0:
                    continue
                h = _forward_task(model, tg, device)

                pos = tg.positive_pairs.to(device)
                n_neg = min(pos.shape[0] * cfg.negative_ratio, tg.candidate_pairs.shape[0])
                negs = tg.sample_negatives(n_neg, rng).to(device)
                if negs.numel() == 0:
                    continue
                pairs = torch.cat([pos, negs], dim=0)
                labels = torch.cat([
                    torch.ones(pos.shape[0], device=device),
                    torch.zeros(negs.shape[0], device=device),
                ])
                logits = model.score_pairs(h, pairs)
                loss = loss_fn(logits, labels) / cfg.grad_accum_graphs
                loss.backward()
                step += 1
                epoch_loss += loss.item() * cfg.grad_accum_graphs

                if step % cfg.grad_accum_graphs == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if step % cfg.grad_accum_graphs != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            model.eval()
            with torch.no_grad():
                logits_all: list[torch.Tensor] = []
                labels_all: list[torch.Tensor] = []
                for tg in train_split:
                    val_pos = val_positives.get(tg.instance_id, torch.empty((0, 2), dtype=torch.long))
                    if val_pos.numel() == 0:
                        continue
                    h = _forward_task(model, tg, device)
                    negs = tg.sample_negatives(val_pos.shape[0] * cfg.negative_ratio, rng)
                    if negs.numel() == 0:
                        continue
                    pairs = torch.cat([val_pos, negs], dim=0).to(device)
                    labels = torch.cat([
                        torch.ones(val_pos.shape[0]),
                        torch.zeros(negs.shape[0]),
                    ]).to(device)
                    logits = model.score_pairs(h, pairs)
                    logits_all.append(logits)
                    labels_all.append(labels)
                if logits_all:
                    v_logits = torch.cat(logits_all)
                    v_labels = torch.cat(labels_all)
                    val_auc = _auc_roc(v_logits, v_labels)
                else:
                    val_auc = 0.5

            elapsed = time.perf_counter() - t0
            logger.info(
                "epoch %03d | loss=%.4f | val_auc=%.4f | lr=%.6f | %.1fs",
                epoch, epoch_loss / max(step, 1), val_auc, current_lr, elapsed,
            )
            if mlflow_enabled:
                import mlflow
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_loss / max(step, 1),
                        "val_auc": val_auc,
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            if val_auc > best_auc:
                best_auc = val_auc
                patience_left = cfg.patience
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "train_cfg": {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in cfg.__dict__.items()
                        },
                        "best_val_auc": best_auc,
                        "epoch": epoch,
                    },
                    best_path,
                )
                logger.info("  -> saved new best checkpoint (%.4f)", best_auc)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping at epoch %d (best val_auc=%.4f)", epoch, best_auc)
                    break

    finally:
        if mlflow_enabled:
            import mlflow
            mlflow.log_metric("best_val_auc", best_auc)
            mlflow.log_artifact(str(best_path))
            mlflow.end_run()

    return best_path
