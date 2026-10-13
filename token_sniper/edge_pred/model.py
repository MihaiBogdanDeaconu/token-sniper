from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for Phase 2. "
        "Install it alongside torch via the instructions in PHASE2_RUN_GUIDE.md."
    ) from exc

from token_sniper.edge_pred.features import EMBEDDING_DIM, NODE_TYPES


@dataclass(frozen=True)
class EdgePredictorConfig:
    num_files: int
    file_emb_dim: int = 64
    hidden_dim: int = 384
    out_dim: int = 256
    num_sage_layers: int = 3
    mlp_hidden: int = 384
    dropout: float = 0.2

    @property
    def in_dim(self) -> int:
        return len(NODE_TYPES) + EMBEDDING_DIM + self.file_emb_dim + 1


class EdgePredictor(nn.Module):
    def __init__(self, cfg: EdgePredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.file_emb = nn.Embedding(cfg.num_files, cfg.file_emb_dim)
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)

        layers = []
        res_projs = []
        in_d = cfg.hidden_dim
        for i in range(cfg.num_sage_layers):
            out_d = cfg.out_dim if i == cfg.num_sage_layers - 1 else cfg.hidden_dim
            layers.append(SAGEConv(in_d, out_d, aggr="mean"))
            if in_d != out_d:
                res_projs.append(nn.Linear(in_d, out_d, bias=False))
            else:
                res_projs.append(nn.Identity())
            in_d = out_d
        self.sage_layers = nn.ModuleList(layers)
        self.res_projs = nn.ModuleList(res_projs)

        self.dropout = nn.Dropout(cfg.dropout)

        jk_dim = cfg.hidden_dim * (cfg.num_sage_layers - 1) + cfg.out_dim
        self.jk_proj = nn.Linear(jk_dim, cfg.out_dim)
        self.post_sage_norm = nn.LayerNorm(cfg.out_dim)

        scorer_in = 4 * cfg.out_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_in),
            nn.Linear(scorer_in, cfg.mlp_hidden),
            nn.GELU(),
            nn.BatchNorm1d(cfg.mlp_hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden // 2),
            nn.GELU(),
            nn.BatchNorm1d(cfg.mlp_hidden // 2),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden // 2, 1),
        )

    def encode(
        self,
        type_onehot: torch.Tensor,
        signature_emb: torch.Tensor,
        file_idx: torch.Tensor,
        degree_log: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                type_onehot,
                signature_emb,
                self.file_emb(file_idx),
                degree_log.unsqueeze(-1),
            ],
            dim=-1,
        )
        x = F.gelu(self.input_proj(x))
        x = self.dropout(x)

        layer_outputs: list[torch.Tensor] = []
        for i, (layer, res_proj) in enumerate(zip(self.sage_layers, self.res_projs)):
            x_prev = x
            x = layer(x, edge_index)
            x = x + res_proj(x_prev)
            if i < len(self.sage_layers) - 1:
                x = F.gelu(x)
                x = self.dropout(x)
            layer_outputs.append(x)

        jk = torch.cat(layer_outputs, dim=-1)
        x = self.jk_proj(jk)
        return self.post_sage_norm(x)

    def score_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        hu = h[pairs[:, 0]]
        hv = h[pairs[:, 1]]
        z = torch.cat([hu, hv, (hu - hv).abs(), hu * hv], dim=-1)
        return self.scorer(z).squeeze(-1)
