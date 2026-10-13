from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from token_sniper.guardrails.halt_features import extract_halt_features

logger = logging.getLogger(__name__)

TOP_K = 20
MAX_SEQ_LEN = 256


class HALTDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        *,
        splits: list[str] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        top_k: int = TOP_K,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.records: list[dict[str, Any]] = []

        data_dir = Path(data_dir)
        all_splits = splits or ["qa", "summarization", "dialogue", "general"]

        for split_name in all_splits:
            fpath = data_dir / f"{split_name}.jsonl"
            if not fpath.exists():
                logger.warning("Missing %s, skipping", fpath)
                continue
            with open(fpath) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if len(rec.get("logprobs", [])) >= 3:
                            self.records.append(rec)
                    except (json.JSONDecodeError, KeyError):
                        continue

        logger.info(
            "Loaded %d records from %s (splits: %s)",
            len(self.records), data_dir, all_splits,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rec = self.records[idx]
        logprobs_raw = rec["logprobs"]

        seq_len = min(len(logprobs_raw), self.max_seq_len)
        logprobs_raw = logprobs_raw[:seq_len]

        lp = torch.tensor(logprobs_raw, dtype=torch.float32)
        if lp.size(-1) < self.top_k:
            pad = torch.full(
                (lp.size(0), self.top_k - lp.size(-1)), -100.0,
            )
            lp = torch.cat([lp, pad], dim=-1)
        elif lp.size(-1) > self.top_k:
            lp = lp[:, : self.top_k]

        lp_sorted, _ = lp.sort(dim=-1, descending=True)

        halt_feats = extract_halt_features(lp_sorted)

        atp = lp_sorted[:, 0:1]
        rank_enc = _rank_encoding(lp_sorted)

        return {
            "logprobs": lp_sorted,
            "halt_features": halt_feats,
            "atp": atp,
            "rank_encoding": rank_enc,
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


def _rank_encoding(logprobs: Tensor) -> Tensor:
    k = logprobs.size(-1)
    rank_ratio = torch.zeros(logprobs.size(0), 1)
    probs = logprobs.exp().clamp(min=1e-12)
    top1_mass = probs[:, 0] / probs.sum(dim=-1)
    rank_ratio[:, 0] = torch.sigmoid(4.0 * (top1_mass - 0.5))
    return rank_ratio


MIN_PREFIX_LEN = 5


class ColluBenchDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        *,
        max_seq_len: int = MAX_SEQ_LEN,
        top_k: int = TOP_K,
        min_prefix_len: int = MIN_PREFIX_LEN,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.records: list[dict[str, Any]] = []

        data_dir = Path(data_dir)

        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(data_dir))
        except Exception:
            logger.error("Failed to load Collu-Bench from %s", data_dir)
            return

        skipped = 0
        for row in ds:
            halluc_idx = row["hallucination_token_index"]
            steps = row["token_logprobs"]
            if not steps or halluc_idx is None:
                skipped += 1
                continue

            logprobs = _extract_topk_logprobs(steps, top_k)
            total_len = len(logprobs)
            group_key = f"{row.get('dataset', 'unk')}_{row['task_id']}"

            if halluc_idx >= min_prefix_len and halluc_idx <= total_len:
                prefix = logprobs[:halluc_idx][:max_seq_len]
                if len(prefix) >= 3:
                    self.records.append({
                        "logprobs": prefix,
                        "label": 0,
                        "split": "collu_bench",
                        "idx": group_key,
                        "source": row.get("model", ""),
                    })

            if halluc_idx < total_len:
                suffix = logprobs[halluc_idx:][:max_seq_len]
                if len(suffix) >= 3:
                    self.records.append({
                        "logprobs": suffix,
                        "label": 1,
                        "split": "collu_bench",
                        "idx": group_key,
                        "source": row.get("model", ""),
                    })

        logger.info(
            "ColluBench: %d records from %d examples (%d skipped), "
            "pos=%.1f%%",
            len(self.records), len(ds), skipped,
            100 * sum(r["label"] for r in self.records)
            / max(len(self.records), 1),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rec = self.records[idx]
        logprobs_raw = rec["logprobs"]

        seq_len = min(len(logprobs_raw), self.max_seq_len)
        logprobs_raw = logprobs_raw[:seq_len]

        lp = torch.tensor(logprobs_raw, dtype=torch.float32)
        lp = torch.nan_to_num(lp, nan=-30.0, posinf=0.0, neginf=-30.0)
        lp = lp.clamp(min=-30.0, max=0.0)

        if lp.size(-1) < self.top_k:
            pad = torch.full(
                (lp.size(0), self.top_k - lp.size(-1)), -100.0,
            )
            lp = torch.cat([lp, pad], dim=-1)
        elif lp.size(-1) > self.top_k:
            lp = lp[:, :self.top_k]

        lp_sorted, _ = lp.sort(dim=-1, descending=True)
        halt_feats = extract_halt_features(lp_sorted)
        atp = lp_sorted[:, 0:1]
        rank_enc = _rank_encoding(lp_sorted)

        return {
            "logprobs": lp_sorted,
            "halt_features": halt_feats,
            "atp": atp,
            "rank_encoding": rank_enc,
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


class AgentTraceDataset(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        *,
        max_seq_len: int = MAX_SEQ_LEN,
        top_k: int = TOP_K,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.records: list[dict[str, Any]] = []

        data_path = Path(data_path)
        if not data_path.exists():
            logger.error("Agent trace data not found: %s", data_path)
            return

        with open(data_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("logprobs") and len(rec["logprobs"]) >= 3:
                        self.records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue

        labels = [float(r["label"]) for r in self.records]
        if labels:
            mean_lbl = sum(labels) / len(labels)
            n_hard_pos = sum(1 for v in labels if v >= 0.5)
            is_soft = any(0.0 < v < 1.0 for v in labels)
            logger.info(
                "AgentTrace: %d records, mean_label=%.3f, %s>=0.5=%.1f%% (%s labels)",
                len(self.records), mean_lbl,
                "soft" if is_soft else "hard",
                100 * n_hard_pos / len(labels),
                "soft" if is_soft else "binary",
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rec = self.records[idx]
        logprobs_raw = rec["logprobs"]

        seq_len = min(len(logprobs_raw), self.max_seq_len)
        logprobs_raw = logprobs_raw[:seq_len]

        lp = torch.tensor(logprobs_raw, dtype=torch.float32)
        lp = torch.nan_to_num(lp, nan=-30.0, posinf=0.0, neginf=-30.0)
        lp = lp.clamp(min=-30.0, max=0.0)

        if lp.size(-1) < self.top_k:
            pad = torch.full(
                (lp.size(0), self.top_k - lp.size(-1)), -100.0,
            )
            lp = torch.cat([lp, pad], dim=-1)
        elif lp.size(-1) > self.top_k:
            lp = lp[:, :self.top_k]

        lp_sorted, _ = lp.sort(dim=-1, descending=True)
        halt_feats = extract_halt_features(lp_sorted)
        atp = lp_sorted[:, 0:1]
        rank_enc = _rank_encoding(lp_sorted)

        return {
            "logprobs": lp_sorted,
            "halt_features": halt_feats,
            "atp": atp,
            "rank_encoding": rank_enc,
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "length": torch.tensor(seq_len, dtype=torch.long),
        }


def _extract_topk_logprobs(
    steps: list[list[dict]], top_k: int = TOP_K,
) -> list[list[float]]:
    result: list[list[float]] = []
    for step in steps:
        vals: list[float] = []
        for entry in step[:top_k]:
            v = entry.get("logprob")
            if v is None or not isinstance(v, (int, float)):
                vals.append(-30.0)
            else:
                vals.append(float(v))
        while len(vals) < top_k:
            vals.append(-100.0)
        result.append(vals)
    return result


def halt_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    lengths = torch.stack([b["length"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    logprobs = pad_sequence(
        [b["logprobs"] for b in batch], batch_first=True, padding_value=-100.0,
    )
    halt_features = pad_sequence(
        [b["halt_features"] for b in batch], batch_first=True, padding_value=0.0,
    )
    atp = pad_sequence(
        [b["atp"] for b in batch], batch_first=True, padding_value=-100.0,
    )
    rank_enc = pad_sequence(
        [b["rank_encoding"] for b in batch], batch_first=True, padding_value=0.0,
    )

    max_len = logprobs.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "logprobs": logprobs,
        "halt_features": halt_features,
        "atp": atp,
        "rank_encoding": rank_enc,
        "attention_mask": mask,
        "labels": labels,
        "lengths": lengths,
    }
