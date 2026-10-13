from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as func

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for Phase 3 VGIB. Install via: pip install -e '.[ml]'"
    ) from exc

EMBEDDING_DIM = 256
NODE_TYPES = ("file", "class", "function", "method", "module_preamble")
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

NODE_FEATURE_VERSION = 1
"""Type one-hot, source embedding, file index, log1p(degree), clustering (features.py)."""

LEXICAL_FEATURE_VERSION = 1
"""Issue-conditioned per-node features (features.py:lexical_node_features)."""

LEXICAL_FEATURE_NAMES = (
    "name_in_issue",
    "qualname_part_frac",
    "qualname_in_issue",
    "file_in_issue",
    "file_in_traceback",
    "line_in_traceback",
    "bm25_log",
    "bm25_rel",
    "issue_cosine",
)

POOLING_MODES = ("cross_attention", "mean")


@dataclass(frozen=True)
class GraphSAGEConfig:
    num_files: int
    source_emb_dim: int = EMBEDDING_DIM
    file_emb_dim: int = 64
    hidden_dim: int = 384
    out_dim: int = 256
    num_sage_layers: int = 3
    dropout: float = 0.2
    task_emb_dim: int = EMBEDDING_DIM
    node_feature_version: int = NODE_FEATURE_VERSION
    """Checkpoints from before versioning lack the field and get 1, which is what they used."""
    lexical_dim: int = 0
    """0 = no lexical features (every checkpoint before them), else ``len(LEXICAL_FEATURE_NAMES)``."""
    lexical_version: int = 0
    use_edges: bool = True
    """False is the "- graph edges" ablation: message passing sees no edges, node features are unchanged."""
    pooling: str = "cross_attention"

    def __post_init__(self) -> None:
        if self.node_feature_version != NODE_FEATURE_VERSION:
            raise ValueError(
                f"checkpoint node features are v{self.node_feature_version}, this code builds "
                f"v{NODE_FEATURE_VERSION}: retrain, or check out the commit that trained it"
            )
        lexical = (self.lexical_dim, self.lexical_version)
        if lexical != (0, 0) and lexical != (len(LEXICAL_FEATURE_NAMES), LEXICAL_FEATURE_VERSION):
            raise ValueError(
                f"checkpoint lexical features are v{self.lexical_version} ({self.lexical_dim}-d), this "
                f"code computes v{LEXICAL_FEATURE_VERSION} ({len(LEXICAL_FEATURE_NAMES)}-d): retrain, "
                "or check out the commit that trained it"
            )
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {self.pooling!r}")

    @property
    def node_feature_dim(self) -> int:
        return len(NODE_TYPES) + self.source_emb_dim + self.file_emb_dim + 2 + self.lexical_dim

    @property
    def in_dim(self) -> int:
        return self.node_feature_dim + self.task_emb_dim


class GraphSAGEEncoder(nn.Module):
    def __init__(self, cfg: GraphSAGEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.file_emb = nn.Embedding(cfg.num_files, cfg.file_emb_dim)
        self.input_proj = nn.Linear(cfg.in_dim, cfg.hidden_dim)

        layers: list[SAGEConv] = []
        res_projs: list[nn.Module] = []
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

    def forward(
        self,
        type_onehot: torch.Tensor,
        signature_emb: torch.Tensor,
        file_idx: torch.Tensor,
        degree_log: torch.Tensor,
        clustering: torch.Tensor,
        task_emb: torch.Tensor,
        edge_index: torch.Tensor,
        lexical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        file_idx_clamped = file_idx.clamp(max=self.cfg.num_files - 1)
        parts = [
            type_onehot,
            signature_emb,
            self.file_emb(file_idx_clamped),
            degree_log.unsqueeze(-1),
            clustering.unsqueeze(-1),
        ]
        if self.cfg.lexical_dim:
            if lexical is None:
                raise ValueError(
                    "this model was trained with lexical node features: pass the task's "
                    "lexical_features (VGIBSelector.forward_task does)"
                )
            if lexical.shape != (type_onehot.shape[0], self.cfg.lexical_dim):
                raise ValueError(
                    f"lexical features have shape {tuple(lexical.shape)}, "
                    f"expected ({type_onehot.shape[0]}, {self.cfg.lexical_dim})"
                )
            parts.append(lexical)
        parts.append(task_emb)
        x = torch.cat(parts, dim=-1)
        if not self.cfg.use_edges:
            edge_index = edge_index.new_empty((2, 0))
        x = func.gelu(self.input_proj(x))
        x = self.dropout(x)

        layer_outputs: list[torch.Tensor] = []
        for i, (layer, res_proj) in enumerate(zip(self.sage_layers, self.res_projs, strict=True)):
            x_prev = x
            x = layer(x, edge_index)
            x = x + res_proj(x_prev)
            if i < len(self.sage_layers) - 1:
                x = func.gelu(x)
                x = self.dropout(x)
            layer_outputs.append(x)

        jk = torch.cat(layer_outputs, dim=-1)
        x = self.jk_proj(jk)
        out: torch.Tensor = self.post_sage_norm(x)
        return out
