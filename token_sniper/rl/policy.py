from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask
    from token_sniper.vgib_selector.selector import VGIBSelector

_LOG_EPS = 1e-7


@dataclass
class PolicyOutput:
    actions: torch.Tensor
    """[N] binary retention mask (0 = drop, 1 = keep)."""

    log_prob: torch.Tensor
    """Scalar: sum of per-node log π(a_i | s) under the current policy."""

    retention_probs: torch.Tensor
    """[N] raw sigmoid p_v from the retention head (for KL computation).
    Always computed at sampling_temperature=1.0 so KL is well-defined."""

    node_embeddings: torch.Tensor
    """[N, D] pre-noise node embeddings from the backbone."""

    entropy: torch.Tensor
    """Scalar: sum of per-node Bernoulli entropy H(p_v)."""


class VGIBPolicy(nn.Module):
    def __init__(
        self,
        selector: VGIBSelector,
        temperature: float = 1.0,
        sampling_temperature: float = 1.0,
        warm_start_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.selector = selector
        self.temperature = temperature
        self.sampling_temperature = sampling_temperature
        self.warm_start_beta = warm_start_beta
        """β of the supervised selector this policy started from (None if unknown)."""

    def forward(
        self,
        task: RerankedVGIBTask,
        device: torch.device,
    ) -> PolicyOutput:
        out = self.selector.forward_task(task, device, temperature=self.temperature)

        p_v = out["retention_probs"]

        if self.training:
            if self.sampling_temperature != 1.0:
                logits = torch.log(
                    p_v.clamp(min=_LOG_EPS) / (1.0 - p_v).clamp(min=_LOG_EPS)
                )
                p_v_sample = torch.sigmoid(logits / self.sampling_temperature)
            else:
                p_v_sample = p_v
            actions = torch.bernoulli(p_v_sample)
        else:
            p_v_sample = p_v
            actions = (p_v >= 0.5).float()

        p_sample_clamped = p_v_sample.clamp(min=_LOG_EPS, max=1.0 - _LOG_EPS)
        log_probs_per_node = (
            actions * torch.log(p_sample_clamped)
            + (1.0 - actions) * torch.log(1.0 - p_sample_clamped)
        )
        log_prob = log_probs_per_node.sum()

        p_clamped = p_v.clamp(min=_LOG_EPS, max=1.0 - _LOG_EPS)
        entropy_per_node = -(
            p_clamped * torch.log(p_clamped)
            + (1.0 - p_clamped) * torch.log(1.0 - p_clamped)
        )
        entropy = entropy_per_node.mean()

        return PolicyOutput(
            actions=actions,
            log_prob=log_prob,
            retention_probs=p_v,
            node_embeddings=out["node_embeddings"],
            entropy=entropy,
        )

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def set_sampling_temperature(self, sampling_temperature: float) -> None:
        self.sampling_temperature = sampling_temperature

    @property
    def device(self) -> torch.device:
        return next(self.selector.parameters()).device
