from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as func

from token_sniper.vgib_selector.backbone import GraphSAGEConfig, GraphSAGEEncoder

if TYPE_CHECKING:
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask


@dataclass
class TemperatureScheduler:
    start: float = 1.0
    end: float = 0.5
    total_epochs: int = 50

    def get(self, epoch: int) -> float:
        if self.total_epochs <= 1:
            return self.end
        frac = min(epoch / (self.total_epochs - 1), 1.0)
        return self.start + frac * (self.end - self.start)


def _concrete_relaxation(
    log_alpha: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    u = torch.rand_like(log_alpha).clamp(1e-6, 1.0 - 1e-6)
    logistic_noise = torch.log(u) - torch.log(1.0 - u)
    return torch.sigmoid((log_alpha + logistic_noise) / temperature)


class VGIBSelector(nn.Module):
    def __init__(self, cfg: GraphSAGEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphSAGEEncoder(cfg)

        self.retention_head = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.out_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.out_dim // 2, 1),
        )

        self.issue_proj = nn.Linear(cfg.task_emb_dim, cfg.out_dim)
        self._attn_scale = 1.0 / math.sqrt(cfg.out_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(cfg.out_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        type_onehot: torch.Tensor,
        signature_emb: torch.Tensor,
        file_idx: torch.Tensor,
        degree_log: torch.Tensor,
        clustering: torch.Tensor,
        task_emb: torch.Tensor,
        edge_index: torch.Tensor,
        temperature: float = 1.0,
        lexical: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(
            type_onehot,
            signature_emb,
            file_idx,
            degree_log,
            clustering,
            task_emb,
            edge_index,
            lexical,
        )

        retention_logits = self.retention_head(h).squeeze(-1)
        p_v = torch.sigmoid(retention_logits)

        issue_query = self.issue_proj(task_emb[0:1])

        if self.training:
            lambda_soft = _concrete_relaxation(retention_logits, temperature)

            mu_h = h.mean(dim=0, keepdim=True)
            sigma_h = h.std(dim=0, keepdim=True).clamp(min=1e-6)
            epsilon = torch.randn_like(h) * sigma_h + mu_h

            lambda_expanded = lambda_soft.unsqueeze(-1)
            z = lambda_expanded * h + (1.0 - lambda_expanded) * epsilon

            pooled = self._pool(issue_query, z)
        else:
            lambda_soft = p_v
            weighted_h = p_v.unsqueeze(-1) * h
            pooled = self._pool(issue_query, weighted_h)

        prediction_logit = self.prediction_head(pooled).squeeze()

        return {
            "prediction_logit": prediction_logit,
            "retention_probs": p_v,
            "lambda_soft": lambda_soft,
            "node_embeddings": h,
        }

    def forward_task(
        self,
        task: RerankedVGIBTask,
        device: torch.device,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        feats = task.features
        n = len(feats.node_ids)
        lexical = task.lexical_features.to(device) if task.lexical_features is not None else None
        result: dict[str, torch.Tensor] = self(
            feats.type_onehot.to(device),
            feats.signature_emb.to(device),
            feats.file_idx.to(device),
            feats.degree_log.to(device),
            feats.clustering.to(device),
            task.task_embedding.unsqueeze(0).expand(n, -1).to(device),
            task.edge_index.to(device),
            temperature=temperature,
            lexical=lexical,
        )
        return result

    def _pool(self, query: torch.Tensor, node_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "mean":
            return node_values.mean(dim=0, keepdim=True)
        return self._cross_attention_pool(query, node_values)

    def _cross_attention_pool(
        self,
        query: torch.Tensor,
        node_values: torch.Tensor,
    ) -> torch.Tensor:
        attn_scores = (query @ node_values.T) * self._attn_scale
        attn_weights = func.softmax(attn_scores, dim=-1)
        return attn_weights @ node_values
