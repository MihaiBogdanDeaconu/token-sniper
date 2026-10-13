from __future__ import annotations

import copy
import logging
from typing import Any

import networkx as nx
import torch

from token_sniper.edge_pred.dataset import TaskGraph
from token_sniper.edge_pred.model import EdgePredictor
from token_sniper.graph.schema import EdgeAttrs

logger = logging.getLogger(__name__)


def predict_edge_scores(
    model: EdgePredictor,
    tg: TaskGraph,
    *,
    device: torch.device | None = None,
    batch_size: int = 200_000,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    feats = tg.features
    num_files = model.cfg.num_files
    with torch.no_grad():
        h = model.encode(
            feats.type_onehot.to(device),
            feats.signature_emb.to(device),
            feats.file_idx.clamp(max=num_files - 1).to(device),
            feats.degree_log.to(device),
            tg.edge_index.to(device),
        )
        outs: list[torch.Tensor] = []
        for i in range(0, tg.candidate_pairs.shape[0], batch_size):
            chunk = tg.candidate_pairs[i : i + batch_size].to(device)
            outs.append(model.score_pairs(h, chunk).cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0)


def augment_graph_with_top_m(
    graph: nx.MultiDiGraph[Any],
    tg: TaskGraph,
    scores: torch.Tensor,
    m: int,
) -> nx.MultiDiGraph[Any]:
    augmented = copy.deepcopy(graph)
    if scores.numel() == 0 or m <= 0:
        return augmented
    top = torch.topk(scores, k=min(m, scores.numel())).indices.tolist()
    added = 0
    for idx in top:
        src_idx, dst_idx = tg.candidate_pairs[idx].tolist()
        src = tg.node_ids[src_idx]
        dst = tg.node_ids[dst_idx]
        if src not in augmented or dst not in augmented:
            continue
        if augmented.has_edge(src, dst):
            data = augmented.get_edge_data(src, dst) or {}
            if any(d.get("edge_type") == "calls" for d in data.values()):
                continue
        attrs = EdgeAttrs(
            edge_type="calls",
            confidence="low",
            resolved=True,
        ).model_dump()
        attrs["learned"] = True
        attrs["predicted_score"] = float(scores[idx].item())
        augmented.add_edge(src, dst, **attrs)
        added += 1
    logger.info("Augmented graph with %d / %d requested predicted edges", added, m)
    return augmented
