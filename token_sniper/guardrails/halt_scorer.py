from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from token_sniper.guardrails.halt_dataset import _rank_encoding
from token_sniper.guardrails.halt_features import extract_halt_features
from token_sniper.guardrails.halt_model import HALTClassifier, HALTConfig

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 256
TOP_K = 20


class HALTScorer:
    def __init__(self, model: HALTClassifier, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> HALTScorer:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = HALTConfig(**ckpt["config"])
        model = HALTClassifier(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])

        logger.info(
            "HALT scorer loaded from %s (epoch %d, %d params)",
            path, ckpt.get("epoch", -1), model.num_parameters,
        )
        return cls(model, device)

    @torch.no_grad()
    def score_logprobs(
        self,
        logprobs: list[list[float]] | Tensor,
        *,
        max_seq_len: int = MAX_SEQ_LEN,
        top_k: int = TOP_K,
    ) -> float:
        lp = self._prepare_logprobs(logprobs, max_seq_len, top_k)
        feats = extract_halt_features(lp)
        atp = lp[:, 0:1]
        rank_enc = _rank_encoding(lp)

        lp_b = lp.unsqueeze(0).to(self.device)
        feats_b = feats.unsqueeze(0).to(self.device)
        atp_b = atp.unsqueeze(0).to(self.device)
        rank_b = rank_enc.unsqueeze(0).to(self.device)
        mask = torch.ones(1, lp_b.size(1), dtype=torch.bool, device=self.device)

        logit = self.model(lp_b, feats_b, atp_b, rank_b, mask)
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def score_batch(
        self,
        batch_logprobs: list[list[list[float]]],
        *,
        max_seq_len: int = MAX_SEQ_LEN,
        top_k: int = TOP_K,
    ) -> list[float]:
        tensors = [
            self._prepare_logprobs(lp, max_seq_len, top_k)
            for lp in batch_logprobs
        ]
        lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)

        feat_list = [extract_halt_features(t) for t in tensors]
        atp_list = [t[:, 0:1] for t in tensors]
        rank_list = [_rank_encoding(t) for t in tensors]

        from torch.nn.utils.rnn import pad_sequence

        lp_pad = pad_sequence(tensors, batch_first=True, padding_value=-100.0)
        feat_pad = pad_sequence(feat_list, batch_first=True, padding_value=0.0)
        atp_pad = pad_sequence(atp_list, batch_first=True, padding_value=-100.0)
        rank_pad = pad_sequence(rank_list, batch_first=True, padding_value=0.0)

        max_len = lp_pad.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

        lp_pad = lp_pad.to(self.device)
        feat_pad = feat_pad.to(self.device)
        atp_pad = atp_pad.to(self.device)
        rank_pad = rank_pad.to(self.device)
        mask = mask.to(self.device)

        logits = self.model(lp_pad, feat_pad, atp_pad, rank_pad, mask)
        probs = torch.sigmoid(logits).cpu().tolist()
        return probs

    @staticmethod
    def _prepare_logprobs(
        logprobs: list[list[float]] | Tensor,
        max_seq_len: int,
        top_k: int,
    ) -> Tensor:
        if isinstance(logprobs, Tensor):
            lp = logprobs.float()
        else:
            lp = torch.tensor(logprobs, dtype=torch.float32)

        lp = lp[:max_seq_len]
        lp = torch.nan_to_num(lp, nan=-30.0, posinf=0.0, neginf=-30.0)
        lp = lp.clamp(min=-30.0, max=0.0)

        if lp.size(-1) < top_k:
            pad = torch.full((lp.size(0), top_k - lp.size(-1)), -100.0)
            lp = torch.cat([lp, pad], dim=-1)
        elif lp.size(-1) > top_k:
            lp = lp[:, :top_k]

        lp, _ = lp.sort(dim=-1, descending=True)
        return lp
