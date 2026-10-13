from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

import torch

from token_sniper.edge_pred.dataset import TaskGraph
from token_sniper.edge_pred.model import EdgePredictor, EdgePredictorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgePredictionMetrics:
    instance_id: str
    num_positives: int
    num_negatives: int
    auc_roc: float
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]


def load_checkpoint(path, device: torch.device) -> EdgePredictor:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = EdgePredictorConfig(**payload["model_cfg"])
    model = EdgePredictor(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def evaluate_edge_predictor(
    model: EdgePredictor,
    test_tasks: Sequence[TaskGraph],
    *,
    k_values: Sequence[int] = (50, 100, 200, 500),
    negative_ratio: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[EdgePredictionMetrics]:
    from sklearn.metrics import roc_auc_score

    device = device or next(model.parameters()).device
    rng = random.Random(seed)
    out: list[EdgePredictionMetrics] = []

    num_files = model.cfg.num_files

    for tg in test_tasks:
        if tg.positive_pairs.numel() == 0:
            logger.warning("[%s] no positives -> skipped", tg.instance_id)
            continue

        with torch.no_grad():
            feats = tg.features
            file_idx = feats.file_idx.clamp(max=num_files - 1).to(device)
            h = model.encode(
                feats.type_onehot.to(device),
                feats.signature_emb.to(device),
                file_idx,
                feats.degree_log.to(device),
                tg.edge_index.to(device),
            )
            pos = tg.positive_pairs.to(device)
            n_neg = min(pos.shape[0] * negative_ratio, tg.candidate_pairs.shape[0])
            negs = tg.sample_negatives(n_neg, rng).to(device)
            if negs.numel() == 0:
                logger.warning("[%s] no negatives sampled → skipped", tg.instance_id)
                continue

            pairs = torch.cat([pos, negs], dim=0)
            logits = model.score_pairs(h, pairs).cpu()
            labels = torch.cat([
                torch.ones(pos.shape[0]),
                torch.zeros(negs.shape[0]),
            ])

            if labels.min() == labels.max():
                auc = 0.5
            else:
                auc = float(roc_auc_score(labels.numpy(), logits.numpy()))

            all_pairs = tg.candidate_pairs.to(device)
            pos_set = {tuple(p.tolist()) for p in pos.cpu()}
            all_logits = []
            for i in range(0, all_pairs.shape[0], 200_000):
                all_logits.append(model.score_pairs(h, all_pairs[i : i + 200_000]))
            all_logits_t = torch.cat(all_logits).cpu() if all_logits else torch.empty(0)
            all_pairs_cpu = tg.candidate_pairs
            prec_k: dict[int, float] = {}
            rec_k: dict[int, float] = {}
            if all_logits_t.numel() > 0:
                order = torch.argsort(all_logits_t, descending=True)
                for k in k_values:
                    top = order[:k].tolist()
                    hits = sum(1 for idx in top if tuple(all_pairs_cpu[idx].tolist()) in pos_set)
                    prec_k[k] = hits / max(k, 1)
                    rec_k[k] = hits / max(len(pos_set), 1)

        out.append(EdgePredictionMetrics(
            instance_id=tg.instance_id,
            num_positives=pos.shape[0],
            num_negatives=negs.shape[0],
            auc_roc=auc,
            precision_at_k=prec_k,
            recall_at_k=rec_k,
        ))
    return out


def aggregate_metrics(metrics: Sequence[EdgePredictionMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    auc = sum(m.auc_roc for m in metrics) / len(metrics)
    out: dict[str, float] = {"mean_auc_roc": auc, "task_count": float(len(metrics))}
    if metrics[0].precision_at_k:
        for k in metrics[0].precision_at_k:
            out[f"mean_precision_at_{k}"] = sum(m.precision_at_k[k] for m in metrics) / len(metrics)
            out[f"mean_recall_at_{k}"] = sum(m.recall_at_k[k] for m in metrics) / len(metrics)
    return out
