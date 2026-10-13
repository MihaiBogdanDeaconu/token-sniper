from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from token_sniper.rl.policy import VGIBPolicy
from token_sniper.rl.rewards import ProxyReward
from token_sniper.rl.rollout import RolloutBuffer, Trajectory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from token_sniper.rl.rewards import LLMJudgeReward, RewardFunction
    from token_sniper.vgib_selector.dataset import RerankedVGIBTask

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    output_dir: Path
    """Directory for checkpoints and logs."""

    group_size: int = 8
    """Rollouts per task per epoch.  G=8 balances variance vs. compute."""

    clip_eps: float = 0.2
    """PPO clipping epsilon for the surrogate objective."""

    kl_coeff: float = 0.1
    """Initial KL penalty coefficient.  Adaptively adjusted.
    Calibrated for per-node-averaged KL (O(1) scale)."""

    kl_target: float = 0.01
    """Target per-node KL divergence.  With .mean() over N≈3860 nodes,
    a per-node target of 0.01 nats allows meaningful policy drift while
    preventing catastrophic divergence from the reference."""

    entropy_coeff: float = 0.1
    """Entropy bonus coefficient for exploration.  Calibrated for
    per-node-averaged entropy (≈0.025 nats/node for near-deterministic
    Phase C checkpoint).  With 0.1 * 0.025 = 0.0025, the entropy bonus
    is comparable in scale to the surrogate (≈0.05) without overwhelming it."""

    lr: float = 1e-5
    """Learning rate — much lower than supervised pretrain (1e-4)."""

    weight_decay: float = 1e-5
    lr_min_factor: float = 0.01
    max_grad_norm: float = 1.0

    epochs: int = 50
    patience: int = 10
    val_fraction: float = 0.15

    grad_accum_steps: int = 32
    """Accumulate gradients over this many tasks before optimizer.step().
    Simulates a larger effective batch size, reduces optimizer overhead,
    and improves GPU utilisation on the H200 cluster.  The per-task loss
    is divided by grad_accum_steps before backward to maintain gradient scale."""

    reward_beta: float = 0.5
    """Token compression penalty weight for ProxyReward.  Increased from
    0.3 → 0.5 to produce a stronger within-group reward gradient."""

    reward_trace_weight: float = 0.1
    """Trace connectivity bonus weight for ProxyReward."""

    reward_recall_bonus: float = 0.5
    """Joint quality bonus scale: recall_bonus * recall * (1-token_ratio).
    Non-constant even at 100% recall because (1-token_ratio) varies per rollout."""

    tau_start: float = 1.0
    tau_end: float = 0.5

    rollout_temperature: float = 2.0
    """Temperature applied during rollout sampling (higher → more diverse masks).
    Decoupled from the policy inference temperature: rollout sampling uses this
    value, while the GRPO loss computation uses the annealing tau schedule.
    Phase C checkpoints have near-binary p_v; rollout_temperature=2.0 flattens
    them enough to generate meaningfully diverse G=8 rollouts per task."""

    seed: int = 42
    device: str | None = None
    mlflow_experiment: str = "phase_rl_grpo"

    reward_source: str = "proxy"
    """'proxy' (zero cost) or 'verified' (agent + SWE-bench tests per rollout;
    the trainer then needs an ``LLMJudgeReward``).  The same reward scores
    the rollouts and the validation masks, so checkpoint selection follows it."""


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _grouped_repo_split(
    tasks: list,
    val_fraction: float,
    rng: random.Random,
) -> tuple[list, list]:
    from collections import defaultdict

    by_repo: dict[str, list] = defaultdict(list)
    for t in tasks:
        by_repo[t.repo].append(t)

    repos = sorted(by_repo.keys())
    rng.shuffle(repos)

    target_val = int(len(tasks) * val_fraction)
    val_tasks: list = []
    train_tasks: list = []

    for repo in repos:
        if len(val_tasks) < target_val:
            val_tasks.extend(by_repo[repo])
        else:
            train_tasks.extend(by_repo[repo])

    if not val_tasks or not train_tasks:
        logger.warning(
            "Grouped split degenerate; falling back to proportional repo split",
        )
        val_tasks, train_tasks = [], []
        for repo in repos:
            repo_tasks = list(by_repo[repo])
            rng.shuffle(repo_tasks)
            n_val = max(1, int(len(repo_tasks) * val_fraction))
            val_tasks.extend(repo_tasks[:n_val])
            train_tasks.extend(repo_tasks[n_val:])

    rng.shuffle(train_tasks)
    rng.shuffle(val_tasks)

    val_repos = len({t.repo for t in val_tasks})
    train_repos = len({t.repo for t in train_tasks})
    overlap = {t.repo for t in val_tasks} & {t.repo for t in train_tasks}
    logger.info(
        "Grouped repo split: %d train tasks (%d repos), %d val tasks (%d repos), "
        "%d repo overlap",
        len(train_tasks), train_repos, len(val_tasks), val_repos, len(overlap),
    )
    return train_tasks, val_tasks


class GRPOTrainer:
    def __init__(
        self,
        policy: VGIBPolicy,
        cfg: GRPOConfig,
        verified_reward: LLMJudgeReward | None = None,
    ) -> None:
        if cfg.reward_source not in ("proxy", "verified"):
            raise ValueError(f"reward_source must be 'proxy' or 'verified', got {cfg.reward_source!r}")
        if (cfg.reward_source == "verified") != (verified_reward is not None):
            raise ValueError(
                "pass verified_reward exactly when reward_source='verified' "
                f"(reward_source={cfg.reward_source!r}, verified_reward={verified_reward!r})"
            )
        self.policy = policy
        self.cfg = cfg

        self.device = torch.device(cfg.device or _pick_device())
        self.policy.to(self.device)

        disable_dropout(self.policy)

        self.ref_policy = VGIBPolicy(
            copy.deepcopy(policy.selector),
            temperature=policy.temperature,
        )
        self.ref_policy.to(self.device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        self.proxy_reward = ProxyReward(
            beta=cfg.reward_beta,
            trace_weight=cfg.reward_trace_weight,
            recall_bonus=cfg.reward_recall_bonus,
        )
        self.reward_fn: RewardFunction = verified_reward or self.proxy_reward

        self.buffer = RolloutBuffer(group_size=cfg.group_size)

        self.optimiser = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser,
            T_max=cfg.epochs,
            eta_min=cfg.lr * cfg.lr_min_factor,
        )

        from token_sniper.vgib_selector.selector import TemperatureScheduler
        self.temp_sched = TemperatureScheduler(
            start=cfg.tau_start,
            end=cfg.tau_end,
            total_epochs=cfg.epochs,
        )

        self._kl_coeff = cfg.kl_coeff

    def train(
        self,
        train_tasks: Sequence[RerankedVGIBTask],
        *,
        val_tasks: Sequence[RerankedVGIBTask] | None = None,
    ) -> Path:
        rng = random.Random(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        if val_tasks is not None:
            trn_tasks = list(train_tasks)
            val_tasks = list(val_tasks)
            logger.info(
                "GRPO | Using pre-defined split: %d train, %d val tasks",
                len(trn_tasks), len(val_tasks),
            )
        else:
            task_list = list(train_tasks)
            trn_tasks, val_tasks = _grouped_repo_split(
                task_list, self.cfg.val_fraction, rng,
            )
        logger.info("GRPO | Train: %d tasks, Val: %d tasks", len(trn_tasks), len(val_tasks))

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.cfg.output_dir / "best_rl.pt"
        best_val_reward = float("-inf")
        patience_left = self.cfg.patience

        mlflow_enabled = False
        try:
            import mlflow

            mlflow.set_experiment(self.cfg.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params({
                "mode": "grpo",
                "group_size": self.cfg.group_size,
                "clip_eps": self.cfg.clip_eps,
                "kl_coeff": self.cfg.kl_coeff,
                "kl_target": self.cfg.kl_target,
                "entropy_coeff": self.cfg.entropy_coeff,
                "lr": self.cfg.lr,
                "epochs": self.cfg.epochs,
                "reward_beta": self.cfg.reward_beta,
                "reward_trace_weight": self.cfg.reward_trace_weight,
                "reward_recall_bonus": self.cfg.reward_recall_bonus,
                "tau_start": self.cfg.tau_start,
                "tau_end": self.cfg.tau_end,
                "seed": self.cfg.seed,
                "train_tasks": len(trn_tasks),
                "val_tasks": len(val_tasks),
                "reward_source": self.cfg.reward_source,
            })
            mlflow_enabled = True
        except ImportError:
            logger.warning("MLflow not available, continuing without run tracking")

        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info(
            "GRPO: %d params, device=%s, G=%d, clip=%.2f, kl=%.4f",
            total_params, self.device, self.cfg.group_size,
            self.cfg.clip_eps, self._kl_coeff,
        )

        try:
            for epoch in range(1, self.cfg.epochs + 1):
                t0 = time.perf_counter()
                temperature = self.temp_sched.get(epoch - 1)
                self.policy.set_temperature(temperature)

                self.buffer.clear()
                rng.shuffle(trn_tasks)

                self.buffer.collect_groups(
                    self.policy,
                    [t for t in trn_tasks if len(t.features.node_ids) > 0],
                    self.reward_fn,
                    self.device,
                    rollout_temperature=self.cfg.rollout_temperature,
                )

                rollout_stats = self.buffer.stats()

                self.policy.train()
                epoch_surrogate = 0.0
                epoch_kl = 0.0
                epoch_entropy = 0.0
                epoch_total = 0.0
                n_updates = 0
                n_accumulated = 0

                self.optimiser.zero_grad(set_to_none=True)

                for task_idx, task in enumerate(trn_tasks):
                    group = self.buffer.get_group(task.instance_id)
                    if not group:
                        continue

                    loss = self._compute_grpo_loss(task, group)

                    if loss is not None:
                        scaled = loss["total"] / self.cfg.grad_accum_steps
                        scaled.backward()
                        n_accumulated += 1

                        epoch_surrogate += loss["surrogate"].item()
                        epoch_kl += loss["kl"].item()
                        epoch_entropy += loss["entropy"].item()
                        epoch_total += loss["total"].item()
                        n_updates += 1

                    is_last_task = (task_idx == len(trn_tasks) - 1)
                    if n_accumulated > 0 and (
                        n_accumulated % self.cfg.grad_accum_steps == 0
                        or is_last_task
                    ):
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(),
                            max_norm=self.cfg.max_grad_norm,
                        )
                        self.optimiser.step()
                        self.optimiser.zero_grad(set_to_none=True)

                self.scheduler.step()

                val_reward, val_recall, val_token = self._evaluate(val_tasks)

                if n_updates > 0:
                    mean_kl = epoch_kl / n_updates
                    self._adapt_kl_coeff(mean_kl)

                elapsed = time.perf_counter() - t0
                current_lr = float(self.scheduler.get_last_lr()[0])
                ns = max(n_updates, 1)

                logger.info(
                    "epoch %03d | loss=%.4f (surr=%.4f kl=%.4f ent=%.4f) | "
                    "r=%.3f±%.3f (group σ=%.3f) recall=%.3f tok=%.3f | "
                    "val_r=%.3f val_rec=%.3f val_tok=%.3f | "
                    "kl_c=%.4f tau=%.3f lr=%.6f | steps=%d | %.1fs",
                    epoch,
                    epoch_total / ns, epoch_surrogate / ns,
                    epoch_kl / ns, epoch_entropy / ns,
                    rollout_stats["mean_reward"],
                    rollout_stats["std_reward"],
                    rollout_stats["mean_group_std"],
                    rollout_stats["mean_recall"],
                    rollout_stats["mean_token_ratio"],
                    val_reward, val_recall, val_token,
                    self._kl_coeff, temperature, current_lr,
                    n_updates // self.cfg.grad_accum_steps + (
                        1 if n_updates % self.cfg.grad_accum_steps else 0
                    ),
                    elapsed,
                )

                if rollout_stats["mean_group_std"] < 0.005:
                    logger.warning(
                        "epoch %03d | within-group reward std=%.4f < 0.005: reward "
                        "saturated — check rollout_temperature and reward weights",
                        epoch, rollout_stats["mean_group_std"],
                    )

                if mlflow_enabled:
                    import mlflow

                    mlflow.log_metrics({
                        "train_loss": epoch_total / ns,
                        "train_surrogate": epoch_surrogate / ns,
                        "train_kl": epoch_kl / ns,
                        "train_entropy": epoch_entropy / ns,
                        "train_reward": rollout_stats["mean_reward"],
                        "train_group_reward_std": rollout_stats["mean_group_std"],
                        "train_recall": rollout_stats["mean_recall"],
                        "train_token_ratio": rollout_stats["mean_token_ratio"],
                        "val_reward": val_reward,
                        "val_recall": val_recall,
                        "val_token_ratio": val_token,
                        "kl_coeff": self._kl_coeff,
                        "temperature": temperature,
                        "learning_rate": current_lr,
                    }, step=epoch)

                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    patience_left = self.cfg.patience
                    self._save_checkpoint(best_path, epoch, val_reward)
                    logger.info(
                        "  -> saved best RL checkpoint (val_reward=%.4f)",
                        best_val_reward,
                    )
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        logger.info(
                            "Early stopping at epoch %d (best val_reward=%.4f)",
                            epoch, best_val_reward,
                        )
                        break

        finally:
            if mlflow_enabled:
                import mlflow

                mlflow.log_metric("best_val_reward", best_val_reward)
                if best_path.exists():
                    mlflow.log_artifact(str(best_path))
                mlflow.end_run()

        return best_path

    def _compute_grpo_loss(
        self,
        task: RerankedVGIBTask,
        group: list[Trajectory],
    ) -> dict[str, torch.Tensor] | None:
        if len(task.features.node_ids) == 0:
            return None

        current_output = self.policy(task, self.device)
        p_v = current_output.retention_probs

        tau_s = self.cfg.rollout_temperature
        if tau_s != 1.0:
            logits = torch.log(
                p_v.clamp(min=1e-7) / (1.0 - p_v).clamp(min=1e-7)
            )
            p_v_sample = torch.sigmoid(logits / tau_s)
        else:
            p_v_sample = p_v
        p_sample_clamped = p_v_sample.clamp(min=1e-7, max=1.0 - 1e-7)

        with torch.no_grad():
            ref_output = self.ref_policy(task, self.device)

        kl = _bernoulli_kl(p_v, ref_output.retention_probs)

        surrogate_terms: list[torch.Tensor] = []

        for trajectory in group:
            advantage = trajectory.advantage

            actions_device = trajectory.actions.to(self.device)
            current_log_prob = (
                actions_device * torch.log(p_sample_clamped)
                + (1.0 - actions_device) * torch.log(1.0 - p_sample_clamped)
            ).sum()

            old_log_prob = trajectory.log_prob
            log_ratio = current_log_prob - old_log_prob
            ratio = torch.exp(log_ratio)

            adv_tensor = torch.tensor(
                advantage, device=self.device, dtype=torch.float32,
            )
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps,
            )
            surrogate = torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)
            surrogate_terms.append(surrogate)

        if not surrogate_terms:
            return None

        surrogate_loss = -torch.stack(surrogate_terms).mean()
        entropy_loss = -current_output.entropy

        total = (
            surrogate_loss
            + self._kl_coeff * kl
            + self.cfg.entropy_coeff * entropy_loss
        )

        return {
            "total": total,
            "surrogate": surrogate_loss,
            "kl": kl,
            "entropy": entropy_loss,
        }

    def _evaluate(
        self,
        tasks: Sequence[RerankedVGIBTask],
    ) -> tuple[float, float, float]:
        self.policy.eval()
        items = []
        with torch.no_grad():
            for task in tasks:
                if len(task.features.node_ids) == 0:
                    continue
                items.append((task, self.policy(task, self.device).actions.detach()))

        if not items:
            return 0.0, 0.0, 0.0
        signals = self.reward_fn.compute_many(items)
        n = len(signals)
        return (
            sum(s.task_reward for s in signals) / n,
            sum(s.recall for s in signals) / n,
            sum(s.token_ratio for s in signals) / n,
        )

    def _adapt_kl_coeff(self, mean_kl: float) -> None:
        if mean_kl > 1.5 * self.cfg.kl_target:
            self._kl_coeff *= 1.1
        elif mean_kl < 0.5 * self.cfg.kl_target:
            self._kl_coeff /= 1.1
        self._kl_coeff = max(0.01, min(self._kl_coeff, 1.0))

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_reward: float,
    ) -> None:
        torch.save(
            {
                "model_state": self.policy.selector.state_dict(),
                "model_cfg": self.policy.selector.cfg.__dict__,
                "grpo_cfg": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in self.cfg.__dict__.items()
                },
                "mode": "grpo",
                "epoch": epoch,
                "best_val_reward": val_reward,
                "kl_coeff": self._kl_coeff,
                "beta": self.policy.warm_start_beta,
            },
            path,
        )


def disable_dropout(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def _bernoulli_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    p_c = p.clamp(min=eps, max=1.0 - eps)
    q_c = q.clamp(min=eps, max=1.0 - eps)

    kl = (
        p_c * (torch.log(p_c) - torch.log(q_c))
        + (1.0 - p_c) * (torch.log(1.0 - p_c) - torch.log(1.0 - q_c))
    )
    return kl.mean()


def load_warm_start_policy(
    checkpoint_path: Path | str,
    *,
    device: str | None = None,
) -> VGIBPolicy:
    from token_sniper.vgib_selector.backbone import GraphSAGEConfig
    from token_sniper.vgib_selector.selector import VGIBSelector

    dev = device or _pick_device()
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    model_cfg_dict = ckpt["model_cfg"]
    model_cfg = GraphSAGEConfig(**model_cfg_dict)
    selector = VGIBSelector(model_cfg)
    selector.load_state_dict(ckpt["model_state"])
    selector.to(dev)

    beta = ckpt["beta"] if "beta" in ckpt else ckpt.get("train_cfg", {}).get("beta")
    policy = VGIBPolicy(selector, temperature=1.0, warm_start_beta=beta)
    logger.info(
        "Loaded warm-start from %s (epoch=%d, val_loss=%.4f)",
        checkpoint_path,
        ckpt.get("epoch", -1),
        ckpt.get("best_val_loss", float("inf")),
    )
    return policy
