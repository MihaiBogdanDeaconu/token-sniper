from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

from token_sniper.vgib_selector.backbone import (
    LEXICAL_FEATURE_NAMES,
    LEXICAL_FEATURE_VERSION,
    NODE_FEATURE_VERSION,
    GraphSAGEConfig,
)
from token_sniper.vgib_selector.dataset import (
    RerankedVGIBTask,
    VGIBTask,
    build_contrastive_batches,
    gold_labels_tensor,
)
from token_sniper.vgib_selector.features import BUNDLE_VERSION
from token_sniper.vgib_selector.loss import (
    RecallConstrainedTokenLoss,
    VGIBLoss,
    VGIBNodeLoss,
    structural_prior,
)
from token_sniper.vgib_selector.selector import TemperatureScheduler, VGIBSelector

logger = logging.getLogger(__name__)


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
            "Grouped split degenerate (%d train repos, %d val repos); "
            "falling back to proportional repo split",
            len([r for r in repos if by_repo[r][0] in train_tasks]),
            len([r for r in repos if by_repo[r][0] in val_tasks]),
        )
        val_tasks, train_tasks = [], []
        for repo in repos:
            repo_tasks = by_repo[repo]
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


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class VGIBTrainConfig:
    output_dir: Path
    beta: float = 0.01
    gamma: float = 1.0
    alpha: float = 0.0
    target_retention: float = 0.1
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 384
    out_dim: int = 256
    negatives_per_task: int = 10
    patience: int = 8
    val_fraction: float = 0.15
    tau_start: float = 1.0
    tau_end: float = 0.5
    max_val_token_ratio: float = 0.5
    """Compression a checkpoint must reach before it can be selected."""
    seed: int = 42
    device: str | None = None
    max_grad_norm: float = 1.0
    lr_min_factor: float = 0.01
    mlflow_experiment: str = "phase3_vgib"
    lexical_features: bool = True
    use_edges: bool = True
    pooling: str = "cross_attention"
    prior_weight: float = 0.0
    prior_alpha: float = 1.0
    prior_gamma_s: float = 1.0
    prior_base_rate: float = 0.5


def _forward_batch_item(
    model: VGIBSelector,
    task: VGIBTask,
    task_emb: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> dict[str, torch.Tensor]:
    feats = task.features
    n = len(feats.node_ids)
    task_emb_broadcast = task_emb.unsqueeze(0).expand(n, -1).to(device)

    result: dict[str, torch.Tensor] = model(
        feats.type_onehot.to(device),
        feats.signature_emb.to(device),
        feats.file_idx.to(device),
        feats.degree_log.to(device),
        feats.clustering.to(device),
        task_emb_broadcast,
        task.edge_index.to(device),
        temperature=temperature,
    )
    return result


def train_vgib_selector(
    train_tasks: Sequence[VGIBTask],
    cfg: VGIBTrainConfig,
) -> Path:
    device = torch.device(cfg.device or _pick_device())
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    task_list = list(train_tasks)
    trn_tasks, val_tasks = _grouped_repo_split(task_list, cfg.val_fraction, rng)
    logger.info("Train: %d tasks, Val: %d tasks", len(trn_tasks), len(val_tasks))

    num_files = max(
        (int(t.features.file_idx.max().item()) + 1 for t in train_tasks),
        default=1,
    )

    model_cfg = GraphSAGEConfig(num_files=num_files, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim)
    model = VGIBSelector(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.epochs,
        eta_min=cfg.lr * cfg.lr_min_factor,
    )
    loss_fn = VGIBLoss(
        beta=cfg.beta, gamma=cfg.gamma,
        target_retention=cfg.target_retention, alpha=cfg.alpha,
    )
    temp_sched = TemperatureScheduler(start=cfg.tau_start, end=cfg.tau_end, total_epochs=cfg.epochs)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    best_val_loss = float("inf")
    patience_left = cfg.patience

    gold_cache: dict[str, torch.Tensor] = {}
    if cfg.alpha > 0:
        for t in task_list:
            gold_cache[t.instance_id] = gold_labels_tensor(t.features.node_ids, t.gold_node_ids)
        total_gold = sum(int(g.sum().item()) for g in gold_cache.values())
        total_nodes = sum(len(g) for g in gold_cache.values())
        logger.info(
            "Gold recall (alpha=%.3f): %d gold nodes / %d total (%.2f%%)",
            cfg.alpha, total_gold, total_nodes, 100.0 * total_gold / max(total_nodes, 1),
        )

    mlflow_enabled = False
    try:
        import mlflow

        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(
            {
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "alpha": cfg.alpha,
                "target_retention": cfg.target_retention,
                "hidden_dim": cfg.hidden_dim,
                "out_dim": cfg.out_dim,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "negatives_per_task": cfg.negatives_per_task,
                "tau_start": cfg.tau_start,
                "tau_end": cfg.tau_end,
                "seed": cfg.seed,
                "train_tasks": len(trn_tasks),
                "val_tasks": len(val_tasks),
                "loss_type": "infonce+gold_recall" if cfg.alpha > 0 else "infonce",
                "pooling": "cross_attention",
            }
        )
        mlflow_enabled = True
    except ImportError:
        logger.warning("MLflow not available, continuing without run tracking")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, device=%s, beta=%.4f", total_params, device, cfg.beta)

    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            temperature = temp_sched.get(epoch - 1)

            model.train()
            batches = build_contrastive_batches(
                trn_tasks,
                negatives_per_task=cfg.negatives_per_task,
                rng=rng,
            )
            rng.shuffle(batches)

            epoch_cls = 0.0
            epoch_gold = 0.0
            epoch_mi = 0.0
            epoch_ret = 0.0
            epoch_total = 0.0
            epoch_acc = 0.0
            n_steps = 0

            for batch in batches:
                opt.zero_grad(set_to_none=True)
                task = batch.task

                all_logits: list[torch.Tensor] = []
                all_lambda_softs: list[torch.Tensor] = []
                all_node_embs: list[torch.Tensor] = []
                all_retention_probs: list[torch.Tensor] = []

                all_embs = [batch.positive_task_emb] + batch.negative_task_embs
                for emb in all_embs:
                    out = _forward_batch_item(model, task, emb, device, temperature)
                    all_logits.append(out["prediction_logit"])
                    all_lambda_softs.append(out["lambda_soft"])
                    all_node_embs.append(out["node_embeddings"])
                    all_retention_probs.append(out["retention_probs"])

                contrastive_logits = torch.stack(all_logits)

                gold = None
                if cfg.alpha > 0 and task.instance_id in gold_cache:
                    gold = gold_cache[task.instance_id].to(device)

                losses = loss_fn(
                    contrastive_logits,
                    all_lambda_softs,
                    all_node_embs,
                    retention_probs_list=all_retention_probs,
                    gold_labels=gold,
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                opt.step()

                epoch_cls += losses["cls"].item()
                epoch_gold += losses["gold"].item()
                epoch_mi += losses["mi"].item()
                epoch_ret += losses["ret"].item()
                epoch_total += losses["total"].item()
                epoch_acc += float(contrastive_logits.argmax().item() == 0)
                n_steps += 1

            scheduler.step()

            model.eval()
            val_batches = build_contrastive_batches(
                val_tasks,
                negatives_per_task=cfg.negatives_per_task,
                rng=rng,
            )
            val_loss = 0.0
            val_retention = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch in val_batches:
                    all_embs = [batch.positive_task_emb] + batch.negative_task_embs
                    v_logits: list[torch.Tensor] = []
                    v_lambda: list[torch.Tensor] = []
                    v_h: list[torch.Tensor] = []
                    v_pv: list[torch.Tensor] = []

                    for emb in all_embs:
                        out = _forward_batch_item(
                            model, batch.task, emb, device, temperature
                        )
                        v_logits.append(out["prediction_logit"])
                        v_lambda.append(out["lambda_soft"])
                        v_h.append(out["node_embeddings"])
                        v_pv.append(out["retention_probs"])

                    c_logits = torch.stack(v_logits)
                    v_gold = None
                    if cfg.alpha > 0 and batch.task.instance_id in gold_cache:
                        v_gold = gold_cache[batch.task.instance_id].to(device)
                    vl = loss_fn(
                        c_logits, v_lambda, v_h,
                        retention_probs_list=v_pv, gold_labels=v_gold,
                    )
                    val_loss += vl["total"].item()
                    val_retention += v_pv[0].mean().item()
                    val_acc += float(c_logits.argmax().item() == 0)

            n_val_batches = max(len(val_batches), 1)
            val_loss /= n_val_batches
            val_retention /= n_val_batches
            val_acc /= n_val_batches

            elapsed = time.perf_counter() - t0
            current_lr = float(scheduler.get_last_lr()[0])
            ns = max(n_steps, 1)
            train_acc = epoch_acc / ns
            logger.info(
                "epoch %03d | loss=%.4f (cls=%.4f gold=%.4f mi=%.4f ret=%.4f) | "
                "acc=%.2f | val=%.4f vacc=%.2f | p_v=%.2f | tau=%.3f | lr=%.6f | %.1fs",
                epoch,
                epoch_total / ns,
                epoch_cls / ns,
                epoch_gold / ns,
                epoch_mi / ns,
                epoch_ret / ns,
                train_acc,
                val_loss,
                val_acc,
                val_retention,
                temperature,
                current_lr,
                elapsed,
            )

            if mlflow_enabled:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_loss": epoch_total / ns,
                        "train_cls_loss": epoch_cls / ns,
                        "train_gold_loss": epoch_gold / ns,
                        "train_mi_loss": epoch_mi / ns,
                        "train_ret_loss": epoch_ret / ns,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_mean_retention": val_retention,
                        "temperature": temperature,
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_left = cfg.patience
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "train_cfg": {
                            k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()
                        },
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                        "beta": cfg.beta,
                        "gamma": cfg.gamma,
                        "alpha": cfg.alpha,
                        "target_retention": cfg.target_retention,
                    },
                    best_path,
                )
                logger.info("  -> saved best checkpoint (val_loss=%.4f)", best_val_loss)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss
                    )
                    break

    finally:
        if mlflow_enabled:
            import mlflow

            mlflow.log_metric("best_val_loss", best_val_loss)
            if best_path.exists():
                mlflow.log_artifact(str(best_path))
            mlflow.end_run()

    return best_path


def _node_metrics(
    p_v: torch.Tensor,
    gold: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred = (p_v >= threshold).float()
    tp = (pred * gold).sum().item()
    fp = (pred * (1.0 - gold)).sum().item()
    fn = ((1.0 - pred) * gold).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def train_vgib_node_supervised(
    train_tasks: Sequence[VGIBTask],
    cfg: VGIBTrainConfig,
) -> Path:
    device = torch.device(cfg.device or _pick_device())
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    task_list = list(train_tasks)
    trn_tasks, val_tasks = _grouped_repo_split(task_list, cfg.val_fraction, rng)
    logger.info("Node-supervised | Train: %d tasks, Val: %d tasks", len(trn_tasks), len(val_tasks))

    num_files = max(
        (int(t.features.file_idx.max().item()) + 1 for t in train_tasks),
        default=1,
    )

    model_cfg = GraphSAGEConfig(num_files=num_files, hidden_dim=cfg.hidden_dim, out_dim=cfg.out_dim)
    model = VGIBSelector(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.lr_min_factor,
    )
    loss_fn = VGIBNodeLoss(beta=cfg.beta, gamma=cfg.gamma, target_retention=cfg.target_retention)
    temp_sched = TemperatureScheduler(start=cfg.tau_start, end=cfg.tau_end, total_epochs=cfg.epochs)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    best_val_loss = float("inf")
    patience_left = cfg.patience

    gold_cache: dict[str, torch.Tensor] = {}
    for t in task_list:
        gold_cache[t.instance_id] = gold_labels_tensor(t.features.node_ids, t.gold_node_ids)

    total_gold = sum(int(g.sum().item()) for g in gold_cache.values())
    total_nodes = sum(len(g) for g in gold_cache.values())
    logger.info(
        "Gold label stats: %d gold nodes / %d total (%.2f%%)",
        total_gold, total_nodes, 100.0 * total_gold / max(total_nodes, 1),
    )

    mlflow_enabled = False
    try:
        import mlflow

        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(
            {
                "mode": "node",
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "target_retention": cfg.target_retention,
                "hidden_dim": cfg.hidden_dim,
                "out_dim": cfg.out_dim,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "tau_start": cfg.tau_start,
                "tau_end": cfg.tau_end,
                "seed": cfg.seed,
                "train_tasks": len(trn_tasks),
                "val_tasks": len(val_tasks),
                "loss_type": "node_bce",
                "total_gold_nodes": total_gold,
                "total_nodes": total_nodes,
            }
        )
        mlflow_enabled = True
    except ImportError:
        logger.warning("MLflow not available, continuing without run tracking")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, device=%s, beta=%.4f (node mode)", total_params, device, cfg.beta)

    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            temperature = temp_sched.get(epoch - 1)

            model.train()
            order = list(range(len(trn_tasks)))
            rng.shuffle(order)

            epoch_node = 0.0
            epoch_mi = 0.0
            epoch_ret = 0.0
            epoch_total = 0.0
            agg_prec = 0.0
            agg_rec = 0.0
            agg_f1 = 0.0
            n_steps = 0

            for idx in order:
                task = trn_tasks[idx]
                opt.zero_grad(set_to_none=True)

                out = _forward_batch_item(model, task, task.task_embedding, device, temperature)
                gold = gold_cache[task.instance_id].to(device)

                losses = loss_fn(
                    out["retention_probs"],
                    gold,
                    out["lambda_soft"],
                    out["node_embeddings"],
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                opt.step()

                epoch_node += losses["node"].item()
                epoch_mi += losses["mi"].item()
                epoch_ret += losses["ret"].item()
                epoch_total += losses["total"].item()

                with torch.no_grad():
                    m = _node_metrics(out["retention_probs"], gold)
                agg_prec += m["precision"]
                agg_rec += m["recall"]
                agg_f1 += m["f1"]
                n_steps += 1

            scheduler.step()

            model.eval()
            val_loss = 0.0
            val_prec = 0.0
            val_rec = 0.0
            val_f1 = 0.0
            val_retention = 0.0
            with torch.no_grad():
                for task in val_tasks:
                    out = _forward_batch_item(model, task, task.task_embedding, device, temperature)
                    gold = gold_cache[task.instance_id].to(device)
                    vl = loss_fn(
                        out["retention_probs"], gold, out["lambda_soft"], out["node_embeddings"],
                    )
                    val_loss += vl["total"].item()
                    val_retention += out["retention_probs"].mean().item()
                    vm = _node_metrics(out["retention_probs"], gold)
                    val_prec += vm["precision"]
                    val_rec += vm["recall"]
                    val_f1 += vm["f1"]

            nv = max(len(val_tasks), 1)
            val_loss /= nv
            val_prec /= nv
            val_rec /= nv
            val_f1 /= nv
            val_retention /= nv

            elapsed = time.perf_counter() - t0
            current_lr = float(scheduler.get_last_lr()[0])
            ns = max(n_steps, 1)
            logger.info(
                "epoch %03d | loss=%.4f (node=%.4f mi=%.4f ret=%.4f) | "
                "P=%.3f R=%.3f F1=%.3f | val=%.4f vF1=%.3f | p_v=%.2f | "
                "tau=%.3f | lr=%.6f | %.1fs",
                epoch,
                epoch_total / ns, epoch_node / ns, epoch_mi / ns, epoch_ret / ns,
                agg_prec / ns, agg_rec / ns, agg_f1 / ns,
                val_loss, val_f1, val_retention,
                temperature, current_lr, elapsed,
            )

            if mlflow_enabled:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_loss": epoch_total / ns,
                        "train_node_loss": epoch_node / ns,
                        "train_mi_loss": epoch_mi / ns,
                        "train_ret_loss": epoch_ret / ns,
                        "train_precision": agg_prec / ns,
                        "train_recall": agg_rec / ns,
                        "train_f1": agg_f1 / ns,
                        "val_loss": val_loss,
                        "val_precision": val_prec,
                        "val_recall": val_rec,
                        "val_f1": val_f1,
                        "val_mean_retention": val_retention,
                        "temperature": temperature,
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_left = cfg.patience
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "train_cfg": {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in cfg.__dict__.items()
                        },
                        "mode": "node",
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                        "beta": cfg.beta,
                        "gamma": cfg.gamma,
                        "target_retention": cfg.target_retention,
                    },
                    best_path,
                )
                logger.info("  -> saved best checkpoint (val_loss=%.4f, vF1=%.3f)", best_val_loss, val_f1)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss,
                    )
                    break

    finally:
        if mlflow_enabled:
            import mlflow

            mlflow.log_metric("best_val_loss", best_val_loss)
            if best_path.exists():
                mlflow.log_artifact(str(best_path))
            mlflow.end_run()

    return best_path


def compression_model_config(cfg: VGIBTrainConfig, tasks: Sequence[RerankedVGIBTask]) -> GraphSAGEConfig:
    source_dims = {int(t.features.signature_emb.shape[1]) for t in tasks}
    task_dims = {int(t.task_embedding.shape[-1]) for t in tasks}
    if len(source_dims) != 1 or len(task_dims) != 1:
        raise ValueError(f"tasks disagree on embedding widths: source {source_dims}, issue {task_dims}")
    return GraphSAGEConfig(
        num_files=max((int(t.features.file_idx.max().item()) + 1 for t in tasks if t.features.node_ids), default=1),
        source_emb_dim=source_dims.pop(),
        task_emb_dim=task_dims.pop(),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        lexical_dim=len(LEXICAL_FEATURE_NAMES) if cfg.lexical_features else 0,
        lexical_version=LEXICAL_FEATURE_VERSION if cfg.lexical_features else 0,
        use_edges=cfg.use_edges,
        pooling=cfg.pooling,
    )


def task_prior(task: RerankedVGIBTask, cfg: VGIBTrainConfig) -> torch.Tensor | None:
    if cfg.prior_weight <= 0:
        return None
    if task.prior_ppr is None or task.prior_seed_mask is None:
        raise ValueError(f"{task.instance_id}: prior_weight > 0 but the task was loaded without prior inputs")
    return structural_prior(
        task.prior_ppr, task.prior_seed_mask,
        alpha=cfg.prior_alpha, gamma_s=cfg.prior_gamma_s, base_rate=cfg.prior_base_rate,
    )


def gold_within_file_topk(
    retention_probs: torch.Tensor,
    node_to_file: torch.Tensor,
    gold_node_mask: torch.Tensor,
    k: int,
) -> bool:
    gold_index = (gold_node_mask > 0.5).nonzero(as_tuple=True)[0]
    for i in gold_index.tolist():
        same_file = node_to_file == node_to_file[i]
        at_least_as_good = (retention_probs >= retention_probs[i]) & same_file
        if int(at_least_as_good.sum().item()) - 1 >= k:
            return False
    return True


def gold_rank_percentile(
    retention_probs: torch.Tensor,
    node_to_file: torch.Tensor,
    gold_node_mask: torch.Tensor,
) -> float:
    scores: list[float] = []
    for i in (gold_node_mask > 0.5).nonzero(as_tuple=True)[0].tolist():
        same_file = node_to_file == node_to_file[i]
        n = int(same_file.sum().item())
        if n < 2:
            scores.append(0.5)
            continue
        better = int(((retention_probs > retention_probs[i]) & same_file).sum().item())
        tied = int(((retention_probs == retention_probs[i]) & same_file).sum().item()) - 1
        rank = better + tied / 2.0
        scores.append(1.0 - rank / (n - 1))
    return sum(scores) / len(scores) if scores else 0.5


def train_vgib_compression(
    train_tasks: Sequence[RerankedVGIBTask],
    cfg: VGIBTrainConfig,
    *,
    val_tasks: Sequence[RerankedVGIBTask],
) -> Path:
    device = torch.device(cfg.device or _pick_device())
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    trn_tasks = list(train_tasks)
    val_tasks = list(val_tasks)
    task_list = trn_tasks + val_tasks
    logger.info("Compression | Train: %d tasks, Val: %d tasks", len(trn_tasks), len(val_tasks))

    model_cfg = compression_model_config(cfg, task_list)
    model = VGIBSelector(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * cfg.lr_min_factor,
    )
    loss_fn = RecallConstrainedTokenLoss(beta=cfg.beta, gamma=cfg.gamma, prior_weight=cfg.prior_weight)
    priors = {t.instance_id: task_prior(t, cfg) for t in task_list}
    feature_versions = {
        "bundle": BUNDLE_VERSION,
        "node": NODE_FEATURE_VERSION,
        "lexical": model_cfg.lexical_version,
    }
    temp_sched = TemperatureScheduler(start=cfg.tau_start, end=cfg.tau_end, total_epochs=cfg.epochs)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "best.pt"
    best_val_pct = -1.0
    best_val_func10 = -1.0
    best_val_loss = float("inf")
    patience_left = cfg.patience
    first_epoch_within_budget: int | None = None

    total_nodes = sum(len(t.features.node_ids) for t in task_list)
    gold_nodes_total = sum(int(t.gold_node_mask.sum().item()) for t in task_list)
    logger.info(
        "Dataset: %d total nodes, %d gold nodes across %d tasks",
        total_nodes, gold_nodes_total, len(task_list),
    )

    mlflow_enabled = False
    try:
        import mlflow

        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(
            {
                "mode": "compression",
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "hidden_dim": cfg.hidden_dim,
                "out_dim": cfg.out_dim,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "tau_start": cfg.tau_start,
                "tau_end": cfg.tau_end,
                "seed": cfg.seed,
                "train_tasks": len(trn_tasks),
                "val_tasks": len(val_tasks),
                "loss_type": "recall_constrained_token",
                "lexical_features": cfg.lexical_features,
                "use_edges": cfg.use_edges,
                "pooling": cfg.pooling,
                "prior_weight": cfg.prior_weight,
                "prior_alpha": cfg.prior_alpha,
                "prior_gamma_s": cfg.prior_gamma_s,
                "prior_base_rate": cfg.prior_base_rate,
                "feature_versions": str(feature_versions),
            }
        )
        mlflow_enabled = True
    except ImportError:
        logger.warning("MLflow not available, continuing without run tracking")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %d params, device=%s, beta=%.4f (compression mode)",
        total_params, device, cfg.beta,
    )

    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            temperature = temp_sched.get(epoch - 1)

            model.train()
            order = list(range(len(trn_tasks)))
            rng.shuffle(order)

            epoch_recall = 0.0
            epoch_token = 0.0
            epoch_mi = 0.0
            epoch_prior = 0.0
            epoch_total = 0.0
            n_steps = 0

            for idx in order:
                task = trn_tasks[idx]
                if len(task.features.node_ids) == 0:
                    continue

                opt.zero_grad(set_to_none=True)
                out = model.forward_task(task, device, temperature)
                prior = priors[task.instance_id]

                losses = loss_fn(
                    out["retention_probs"],
                    task.gold_node_mask.to(device),
                    task.token_counts.to(device),
                    out["lambda_soft"],
                    out["node_embeddings"],
                    prior_probs=prior.to(device) if prior is not None else None,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                opt.step()

                epoch_recall += losses["recall"].item()
                epoch_token += losses["token_ratio"].item()
                epoch_mi += losses["mi"].item()
                epoch_prior += losses["prior_kl"].item()
                epoch_total += losses["total"].item()
                n_steps += 1

            scheduler.step()

            model.eval()
            val_recall = 0.0
            val_token = 0.0
            val_total = 0.0
            val_retention = 0.0
            val_func10 = 0.0
            val_pct = 0.0
            with torch.no_grad():
                for task in val_tasks:
                    if len(task.features.node_ids) == 0:
                        continue
                    out = model.forward_task(task, device, temperature)
                    prior = priors[task.instance_id]
                    vl = loss_fn(
                        out["retention_probs"],
                        task.gold_node_mask.to(device),
                        task.token_counts.to(device),
                        out["lambda_soft"],
                        out["node_embeddings"],
                        prior_probs=prior.to(device) if prior is not None else None,
                    )
                    val_total += vl["total"].item()
                    val_recall += vl["recall"].item()
                    val_token += vl["token_ratio"].item()
                    val_retention += out["retention_probs"].mean().item()
                    p_v_cpu = out["retention_probs"].cpu()
                    val_func10 += float(gold_within_file_topk(
                        p_v_cpu, task.node_to_file, task.gold_node_mask, k=10,
                    ))
                    val_pct += gold_rank_percentile(
                        p_v_cpu, task.node_to_file, task.gold_node_mask,
                    )

            nv = max(len(val_tasks), 1)
            val_total /= nv
            val_recall /= nv
            val_token /= nv
            val_retention /= nv
            val_func10 /= nv
            val_pct /= nv

            elapsed = time.perf_counter() - t0
            current_lr = float(scheduler.get_last_lr()[0])
            ns = max(n_steps, 1)
            logger.info(
                "epoch %03d | loss=%.4f (recall=%.3f token=%.3f mi=%.4f prior=%.4f) | "
                "val=%.4f vRecall=%.3f vToken=%.3f vFunc@10=%.3f vPct=%.3f | p_v=%.2f | "
                "tau=%.3f | lr=%.6f | %.1fs",
                epoch,
                epoch_total / ns, epoch_recall / ns, epoch_token / ns, epoch_mi / ns, epoch_prior / ns,
                val_total, val_recall, val_token, val_func10, val_pct, val_retention,
                temperature, current_lr, elapsed,
            )

            if mlflow_enabled:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_loss": epoch_total / ns,
                        "train_recall": epoch_recall / ns,
                        "train_token_ratio": epoch_token / ns,
                        "train_mi_loss": epoch_mi / ns,
                        "train_prior_kl": epoch_prior / ns,
                        "val_loss": val_total,
                        "val_recall": val_recall,
                        "val_token_ratio": val_token,
                        "val_func_at_10": val_func10,
                        "val_gold_percentile": val_pct,
                        "val_mean_retention": val_retention,
                        "temperature": temperature,
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            within_budget = val_token <= cfg.max_val_token_ratio
            if within_budget and first_epoch_within_budget is None:
                first_epoch_within_budget = epoch
            if within_budget and val_pct > best_val_pct:
                best_val_pct = val_pct
                best_val_func10 = val_func10
                best_val_loss = val_total
                patience_left = cfg.patience
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "train_cfg": {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in cfg.__dict__.items()
                        },
                        "mode": "compression",
                        "feature_versions": feature_versions,
                        "first_epoch_within_budget": first_epoch_within_budget,
                        "best_val_gold_percentile": best_val_pct,
                        "best_val_func_at_10": best_val_func10,
                        "best_val_loss": best_val_loss,
                        "val_token_ratio": val_token,
                        "epoch": epoch,
                        "beta": cfg.beta,
                        "gamma": cfg.gamma,
                    },
                    best_path,
                )
                logger.info(
                    "  -> saved best (vPct=%.3f, vFunc@10=%.3f, val_loss=%.4f, recall=%.3f, token=%.3f)",
                    best_val_pct, best_val_func10, best_val_loss, val_recall, val_token,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d (best vPct=%.3f)", epoch, best_val_pct,
                    )
                    break

    finally:
        if mlflow_enabled:
            import mlflow

            mlflow.log_metric("best_val_gold_percentile", best_val_pct)
            mlflow.log_metric("best_val_func_at_10", best_val_func10)
            if first_epoch_within_budget is not None:
                mlflow.log_metric("first_epoch_within_budget", first_epoch_within_budget)
            mlflow.log_metric("best_val_loss", best_val_loss)
            if best_path.exists():
                mlflow.log_artifact(str(best_path))
            mlflow.end_run()

    return best_path
