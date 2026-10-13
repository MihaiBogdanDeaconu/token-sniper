from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HALTConfig:
    top_k: int = 20
    top_k_proj_dim: int = 16
    halt_feature_dim: int = 5
    atp_dim: int = 1
    rank_enc_dim: int = 1
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1


class HALTClassifier(nn.Module):
    def __init__(self, cfg: HALTConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = HALTConfig()
        self.cfg = cfg

        input_dim = (
            cfg.top_k_proj_dim
            + cfg.atp_dim
            + cfg.rank_enc_dim
            + cfg.halt_feature_dim
        )

        self.logprob_proj = nn.Sequential(
            nn.Linear(cfg.top_k, cfg.top_k_proj_dim),
            nn.GELU(),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len + 1, cfg.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.model_dim),
            nn.Linear(cfg.model_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        logprobs: Tensor,
        halt_features: Tensor,
        atp: Tensor,
        rank_encoding: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        B, T, _ = logprobs.shape

        lp_proj = self.logprob_proj(logprobs)
        x = torch.cat([lp_proj, atp, rank_encoding, halt_features], dim=-1)
        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        positions = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)
        src_key_padding_mask = ~full_mask

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        cls_out = x[:, 0]
        logits = self.head(cls_out).squeeze(-1)
        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
