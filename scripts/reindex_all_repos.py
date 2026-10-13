from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index repos with improved pipeline")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--repo-cache-dir", type=Path, default=Path("/workspace/cache/repos"))
    parser.add_argument("--chromadb-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="jinaai/jina-code-embeddings-0.5b")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    args = parser.parse_args()

    from token_sniper.evaluation.phase0 import load_phase0_manifest
    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.retriever import build_entity_index
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    manifest = load_phase0_manifest(args.manifest)
    logger.info("Manifest: %d tasks", len(manifest.tasks))

    embed_cfg = EmbeddingConfig(
        provider=args.embedding_provider,
        name=args.embedding_model,
        dimension=args.embedding_dim,
    )
    embedder = create_embedder(embed_cfg)

    vector_store = EntityVectorStore(
        persist_directory=str(args.chromadb_dir),
        collection_prefix="token_sniper",
    )

    repos_seen: set[str] = set()
    total_entities = 0
    t0 = time.monotonic()

    for i, task in enumerate(manifest.tasks):
        commit_short = task.base_commit[:12]
        collection_repo = f"{task.repo}@{commit_short}"
        if collection_repo in repos_seen:
            continue
        repos_seen.add(collection_repo)

        repo_slug = task.repo.replace("/", "__")
        repo_dir = args.repo_cache_dir / repo_slug
        if not repo_dir.exists():
            logger.warning("[%d] Repo not found: %s", i, repo_dir)
            continue

        import subprocess
        try:
            subprocess.run(
                ["git", "checkout", task.base_commit],
                cwd=repo_dir, capture_output=True, timeout=30,
            )
        except Exception:
            logger.warning("[%d] git checkout failed for %s", i, collection_repo)

        logger.info("[%d/%d] Indexing %s...", len(repos_seen), len(manifest.tasks), collection_repo)

        try:
            stats = build_entity_index(
                repo_root=repo_dir,
                repo_name=collection_repo,
                embedder=embedder,
                vector_store=vector_store,
            )
            n = stats.get("entity_count", 0)
            total_entities += n
            logger.info(
                "  -> %d entities (%.1fs embed, %.1fs index)",
                n, stats.get("embed_time_s", 0), stats.get("index_time_s", 0),
            )
        except Exception:
            logger.warning("  -> FAILED", exc_info=True)

    elapsed = time.monotonic() - t0
    logger.info(
        "=== Re-indexing complete: %d repos, %d entities, %.1fs ===",
        len(repos_seen), total_entities, elapsed,
    )


if __name__ == "__main__":
    main()
