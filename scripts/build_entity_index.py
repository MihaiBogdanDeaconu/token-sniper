from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and persist the code entity embedding index for a repository.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the repository checkout.",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository identifier (e.g. 'django/django').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["voyageai", "nomic-local"],
        default="voyageai",
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/chromadb/",
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection-prefix",
        type=str,
        default="token_sniper",
        help="Prefix for ChromaDB collection names.",
    )
    parser.add_argument(
        "--token-threshold",
        type=int,
        default=512,
        help="Token threshold for switching to signature summaries.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete the existing collection before indexing.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path to write indexing stats JSON.",
    )
    args = parser.parse_args()

    if not args.repo.is_dir():
        logger.error("Repository path does not exist or is not a directory: %s", args.repo)
        sys.exit(1)

    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.retriever import build_entity_index
    from token_sniper.retrieval.vectorstore import EntityVectorStore

    embedding_config = EmbeddingConfig(
        provider=args.provider,
        name="voyage-code-3" if args.provider == "voyageai" else "nomic-embed-code-v1",
    )
    embedder = create_embedder(embedding_config)
    logger.info("Using embedding provider: %s (dim=%d)", args.provider, embedder.dimension)

    vector_store = EntityVectorStore(
        persist_directory=args.persist_dir,
        collection_prefix=args.collection_prefix,
    )

    if args.reindex:
        logger.info("Deleting existing collection for %s", args.repo_name)
        vector_store.delete_collection(args.repo_name)

    stats = build_entity_index(
        repo_root=args.repo,
        repo_name=args.repo_name,
        embedder=embedder,
        vector_store=vector_store,
        token_threshold=args.token_threshold,
    )

    collection_stats = vector_store.get_collection_stats(args.repo_name)
    stats["collection_stats"] = collection_stats

    logger.info("Indexing complete: %s", json.dumps(stats, indent=2))

    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Stats written to %s", args.stats_output)


if __name__ == "__main__":
    main()
