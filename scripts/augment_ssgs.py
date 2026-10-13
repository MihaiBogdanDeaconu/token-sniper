from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment SSGs with Phase 2 edge predictor predictions",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("data/edge_pred/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("configs/phases/phase0_tasks.json"),
    )
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--chromadb-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ssg_augmented"))
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--candidate-cache", type=Path, default=None)
    parser.add_argument("--top-m", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from token_sniper.edge_pred.augment import augment_graph_with_top_m, predict_edge_scores
    from token_sniper.edge_pred.dataset import load_task_graph_for_scoring
    from token_sniper.edge_pred.evaluate import load_checkpoint
    from token_sniper.edge_pred.features import _pick_device
    from token_sniper.graph.io import read_ssg_json, write_ssg_json
    from token_sniper.vgib_selector.features import ChromaSourceEncoder

    device = torch.device(args.device or _pick_device())
    logger.info("Loading edge predictor from %s (device=%s)", args.checkpoint, device)
    model = load_checkpoint(str(args.checkpoint), device)

    encoder = ChromaSourceEncoder(chromadb_dir=str(args.chromadb_dir))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    logger.info("Manifest: %d tasks", len(manifest))

    seen: dict[tuple[str, str], Path] = {}
    for t in manifest:
        repo = t["repo"]
        commit_short = t["base_commit"][:12]
        key = (repo, commit_short)
        if key in seen:
            continue
        repo_slug = repo.replace("/", "__")
        ssg_path = args.ssg_dir / repo_slug / commit_short / "ssg.json"
        if ssg_path.exists():
            seen[key] = ssg_path

    logger.info("Found %d unique SSGs to augment", len(seen))

    if args.feature_cache:
        args.feature_cache.mkdir(parents=True, exist_ok=True)
    if args.candidate_cache:
        args.candidate_cache.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    augmented = 0
    skipped = 0
    prev_repo: str | None = None

    for i, ((repo, commit_short), ssg_path) in enumerate(seen.items()):
        repo_slug = repo.replace("/", "__")
        out_path = args.output_dir / repo_slug / commit_short / "ssg.json"

        if out_path.exists():
            skipped += 1
            if (i + 1) % 50 == 0:
                logger.info(
                    "Progress: %d/%d (augmented=%d, skipped=%d)",
                    i + 1, len(seen), augmented, skipped,
                )
            continue

        try:
            tg = load_task_graph_for_scoring(
                instance_id=f"{repo_slug}@{commit_short}",
                repo=repo,
                ssg_path=ssg_path,
                encoder=encoder,
                feature_cache_dir=args.feature_cache,
                candidate_cache_dir=args.candidate_cache,
            )

            if tg.candidate_pairs.numel() == 0:
                logger.warning("[%s@%s] 0 candidates — copying original SSG", repo, commit_short)
                graph = read_ssg_json(ssg_path)
                write_ssg_json(graph, out_path)
                augmented += 1
                continue

            scores = predict_edge_scores(model, tg, device=device)
            graph = read_ssg_json(ssg_path)
            aug = augment_graph_with_top_m(graph, tg, scores, m=args.top_m)
            write_ssg_json(aug, out_path)
            augmented += 1

        except Exception:
            logger.exception("[%s@%s] failed — skipping", repo, commit_short)
            skipped += 1

        if prev_repo is not None and repo != prev_repo:
            encoder.evict_collection(prev_repo)
        prev_repo = repo

        if (i + 1) % 25 == 0:
            gc.collect()
            logger.info(
                "Progress: %d/%d (augmented=%d, skipped=%d)",
                i + 1, len(seen), augmented, skipped,
            )

    logger.info(
        "Done. Augmented %d SSGs, skipped %d. Output: %s",
        augmented, skipped, args.output_dir,
    )


if __name__ == "__main__":
    main()
