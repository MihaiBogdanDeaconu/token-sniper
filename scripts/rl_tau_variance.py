from __future__ import annotations

import argparse
import csv
import logging
import statistics
from pathlib import Path

import torch

from scripts.train_rl import add_data_args, load_rl_data, subsample
from token_sniper.rl.rewards import ProxyReward
from token_sniper.rl.rollout import RolloutBuffer, group_reward_std
from token_sniper.rl.trainer import disable_dropout, load_warm_start_policy

logger = logging.getLogger(__name__)

ZERO_STD = 1e-9


def tau_variance_rows(
    policy,
    tasks: list,
    *,
    taus: list[float],
    group_size: int,
    reward_fn: ProxyReward,
    device: torch.device,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    disable_dropout(policy)
    per_task: list[dict] = []
    summary: list[dict] = []
    for tau in taus:
        torch.manual_seed(seed)
        buffer = RolloutBuffer(group_size=group_size)
        rows = []
        for task in tasks:
            group = buffer.collect_group(policy, task, reward_fn, device, rollout_temperature=tau)
            n = len(group)
            rows.append({
                "tau_s": tau,
                "instance_id": task.instance_id,
                "n_nodes": len(task.features.node_ids),
                "group_reward_std": group_reward_std(group),
                "distinct_masks": len({tuple(t.actions.to(torch.uint8).tolist()) for t in group}),
                "mean_reward": sum(t.reward for t in group) / n,
                "mean_recall": sum(t.reward_signal.recall for t in group) / n,
                "mean_token_ratio": sum(t.reward_signal.token_ratio for t in group) / n,
            })
        stds = [r["group_reward_std"] for r in rows]
        summary.append({
            "tau_s": tau,
            "n_tasks": len(rows),
            "group_size": group_size,
            "mean_group_std": statistics.fmean(stds),
            "median_group_std": statistics.median(stds),
            "frac_zero_std_groups": sum(s < ZERO_STD for s in stds) / len(stds),
            "mean_distinct_masks": statistics.fmean(r["distinct_masks"] for r in rows),
            "mean_reward": statistics.fmean(r["mean_reward"] for r in rows),
            "mean_recall": statistics.fmean(r["mean_recall"] for r in rows),
            "mean_token_ratio": statistics.fmean(r["mean_token_ratio"] for r in rows),
        })
        logger.info(
            "tau_s=%.2f: within-group std mean=%.4f median=%.4f, %.0f%% zero-std groups, "
            "%.1f distinct masks / %d",
            tau, summary[-1]["mean_group_std"], summary[-1]["median_group_std"],
            100 * summary[-1]["frac_zero_std_groups"], summary[-1]["mean_distinct_masks"], group_size,
        )
        per_task.extend(rows)
    return per_task, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(
        description="Within-group reward std vs rollout temperature",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Policy to sample from -- the supervised warm-start for the figure.")
    parser.add_argument("--taus", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--task-split", choices=("train", "val"), default="train")
    parser.add_argument("--num-tasks", type=int, default=None, help="Seeded subsample; all when omitted.")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--reward-beta", type=float, default=0.5)
    parser.add_argument("--reward-trace-weight", type=float, default=0.1)
    parser.add_argument("--reward-recall-bonus", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data = load_rl_data(args)
    pool = data.val_tasks if args.task_split == "val" else data.train_tasks
    tasks = subsample(pool, args.num_tasks, args.seed)
    policy = load_warm_start_policy(args.checkpoint, device=args.device)
    reward_fn = ProxyReward(
        beta=args.reward_beta, trace_weight=args.reward_trace_weight,
        recall_bonus=args.reward_recall_bonus,
    )
    per_task, summary = tau_variance_rows(
        policy, tasks, taus=args.taus, group_size=args.group_size,
        reward_fn=reward_fn, device=policy.device, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tau_variance_per_task.csv", per_task)
    write_csv(args.output_dir / "tau_variance.csv", summary)
    logger.info("Wrote %s", args.output_dir / "tau_variance.csv")


if __name__ == "__main__":
    main()
