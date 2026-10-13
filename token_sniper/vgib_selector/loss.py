from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as func


class VGIBLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.01,
        gamma: float = 1.0,
        target_retention: float = 0.3,
        alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.target_retention = target_retention
        self.alpha = alpha

    def forward(
        self,
        contrastive_logits: torch.Tensor,
        lambda_softs: list[torch.Tensor],
        node_embeddings_list: list[torch.Tensor],
        retention_probs_list: list[torch.Tensor] | None = None,
        gold_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss = func.cross_entropy(
            contrastive_logits.unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=contrastive_logits.device),
        )

        mi_terms = [
            _mutual_information_upper_bound(lam, h)
            for lam, h in zip(lambda_softs, node_embeddings_list, strict=True)
        ]
        mi_loss = torch.stack(mi_terms).mean()

        p_vs = retention_probs_list if retention_probs_list is not None else lambda_softs
        ret_terms = [(p.mean() - self.target_retention) ** 2 for p in p_vs]
        ret_loss = torch.stack(ret_terms).mean()

        gold_loss = torch.tensor(0.0, device=contrastive_logits.device)
        if self.alpha > 0 and gold_labels is not None:
            gold_loss = _gold_recall_loss(p_vs[0], gold_labels)

        total = (
            cls_loss
            + self.alpha * gold_loss
            + self.beta * mi_loss
            + self.gamma * ret_loss
        )
        return {
            "total": total,
            "cls": cls_loss,
            "gold": gold_loss,
            "mi": mi_loss,
            "ret": ret_loss,
        }


class VGIBNodeLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.01,
        gamma: float = 1.0,
        target_retention: float = 0.3,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.target_retention = target_retention

    def forward(
        self,
        retention_probs: torch.Tensor,
        gold_labels: torch.Tensor,
        lambda_soft: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n_pos = gold_labels.sum().clamp(min=1.0)
        n_neg = (gold_labels.numel() - n_pos).clamp(min=1.0)
        pos_weight = (n_neg / n_pos).detach()

        node_loss = func.binary_cross_entropy(
            retention_probs,
            gold_labels,
            weight=torch.where(gold_labels > 0.5, pos_weight, torch.ones_like(gold_labels)),
        )

        mi_loss = _mutual_information_upper_bound(lambda_soft, node_embeddings)

        ret_loss = (retention_probs.mean() - self.target_retention) ** 2

        total = node_loss + self.beta * mi_loss + self.gamma * ret_loss
        return {"total": total, "node": node_loss, "mi": mi_loss, "ret": ret_loss}


def _gold_recall_loss(
    retention_probs: torch.Tensor,
    gold_labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    gold_mask = gold_labels > 0.5
    if not gold_mask.any():
        return torch.tensor(0.0, device=retention_probs.device, dtype=retention_probs.dtype)
    return -torch.log(retention_probs[gold_mask].clamp(min=eps)).mean()


class RecallConstrainedTokenLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.01,
        gamma: float = 0.001,
        prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.prior_weight = prior_weight

    def forward(
        self,
        retention_probs: torch.Tensor,
        gold_node_mask: torch.Tensor,
        token_counts: torch.Tensor,
        lambda_soft: torch.Tensor,
        node_embeddings: torch.Tensor,
        prior_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        gold_mask = gold_node_mask > 0.5
        if not gold_mask.any():
            raise ValueError("no gold node in this subgraph: the recall term is undefined")
        soft_recall = retention_probs[gold_mask].mean()
        recall_loss = -soft_recall

        total_tokens = token_counts.sum().clamp(min=1.0)
        retained_tokens = (retention_probs * token_counts).sum()
        token_ratio = retained_tokens / total_tokens

        mi_loss = _mutual_information_upper_bound(lambda_soft, node_embeddings)

        total = recall_loss + self.beta * token_ratio + self.gamma * mi_loss

        prior_kl = torch.zeros((), device=retention_probs.device, dtype=retention_probs.dtype)
        if self.prior_weight > 0:
            if prior_probs is None:
                raise ValueError("prior_weight > 0 but no structural prior was given for this task")
            prior_kl = bernoulli_kl(retention_probs, prior_probs).mean()
            total = total + self.prior_weight * prior_kl

        return {
            "total": total,
            "recall": -recall_loss,
            "token_ratio": token_ratio,
            "mi": mi_loss,
            "prior_kl": prior_kl,
            "recall_loss": recall_loss,
        }


def structural_prior(
    ppr: torch.Tensor,
    seed_mask: torch.Tensor,
    *,
    alpha: float,
    gamma_s: float,
    base_rate: float = 0.5,
) -> torch.Tensor:
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    peak = ppr.max() if ppr.numel() else ppr.new_zeros(())
    ppr_rel = ppr / peak if peak > 0 else torch.zeros_like(ppr)
    logit = math.log(base_rate / (1.0 - base_rate))
    return torch.sigmoid(logit + alpha * ppr_rel + gamma_s * seed_mask)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def _mutual_information_upper_bound(
    lambda_soft: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    m = float(h.shape[0])
    if m < 1:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    mu_h = h.mean(dim=0)
    sigma_h = h.std(dim=0).clamp(min=1e-6)

    a_g = ((1.0 - lambda_soft) ** 2).mean()

    deviation = (h - mu_h.unsqueeze(0)) / sigma_h.unsqueeze(0)
    weighted = lambda_soft.unsqueeze(-1) * deviation
    b_g_vec = weighted.mean(dim=0)
    b_g_sq = (b_g_vec**2).mean()

    eps = 1e-8
    loss = -0.5 * torch.log(a_g + eps) + 0.5 * a_g + 0.5 * b_g_sq
    return loss
