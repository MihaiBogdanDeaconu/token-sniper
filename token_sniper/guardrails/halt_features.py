from __future__ import annotations

import math

import torch
from torch import Tensor


def normalized_entropy(logprobs: Tensor) -> Tensor:
    probs = logprobs.exp().clamp(min=1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    raw_entropy = -(probs * probs.log()).sum(dim=-1)
    max_entropy = math.log(logprobs.size(-1))
    return raw_entropy / max_entropy if max_entropy > 0 else raw_entropy


def rank_proxy(logprobs: Tensor) -> Tensor:
    probs = logprobs.exp().clamp(min=1e-12)
    top1 = probs[:, 0]
    top2 = probs[:, 1].clamp(min=1e-12)
    return (top1 / (top1 + top2)).clamp(0.0, 1.0)


def average_logprob(logprobs: Tensor) -> Tensor:
    return logprobs.mean(dim=-1)


def decision_entropy_change(logprobs: Tensor) -> Tensor:
    ent = normalized_entropy(logprobs)
    delta = torch.zeros_like(ent)
    delta[1:] = ent[1:] - ent[:-1]
    return delta


def top1_top2_gap(logprobs: Tensor) -> Tensor:
    return logprobs[:, 0] - logprobs[:, 1]


def extract_halt_features(logprobs: Tensor) -> Tensor:
    logprobs = logprobs.clamp(min=-30.0, max=0.0)
    feats = torch.stack([
        normalized_entropy(logprobs),
        rank_proxy(logprobs),
        average_logprob(logprobs),
        decision_entropy_change(logprobs),
        top1_top2_gap(logprobs),
    ], dim=-1)
    return torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
