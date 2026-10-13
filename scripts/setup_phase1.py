from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SSG_TIMING = "ssg_timing.json"
INDEX_TIMING = "index_timing.json"


def _write_timing(path: Path, timing: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Phase 1: clone repos, build SSGs, index entities")
    parser.add_argument("--manifest", type=Path, default=Path("configs/phases/phase0_tasks.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/swebench/repos"))
    parser.add_argument("--ssg-dir", type=Path, default=Path("data/ssg"))
    parser.add_argument("--persist-dir", type=str, default="data/chromadb/")
    parser.add_argument("--provider", choices=["voyageai", "nomic-local", "sentence-transformers"], default="voyageai")
    parser.add_argument("--collection-prefix", default="token_sniper")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without doing it")
    parser.add_argument("--skip-clone", action="store_true", help="Skip repo cloning (assume already cached)")
    parser.add_argument("--skip-ssg", action="store_true", help="Skip SSG building (assume already built)")
    parser.add_argument("--skip-index", action="store_true", help="Skip entity indexing (assume already indexed)")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="SSG process-pool size. 0 = min(cpu/2, 16)",
    )
    parser.add_argument("--repos", nargs="+", help="Optional list of repos to process (e.g. django/django astropy/astropy)")
    args = parser.parse_args()

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    logger.info("Loaded %d tasks from %s", len(manifest_data), args.manifest)

    repo_commits: dict[str, dict[str, list[str]]] = {}
    for task in manifest_data:
        repo = task["repo"]
        if args.repos and repo not in args.repos:
            continue
        commit = task["base_commit"]
        repo_commits.setdefault(repo, {}).setdefault(commit, []).append(task["instance_id"])

    total_snapshots = sum(len(commits) for commits in repo_commits.values())
    logger.info(
        "Found %d unique repos, %d unique (repo, commit) pairs",
        len(repo_commits),
        total_snapshots,
    )

    if args.dry_run:
        for repo, commits in sorted(repo_commits.items()):
            logger.info("  %s: %d commits", repo, len(commits))
            for commit, task_ids in sorted(commits.items()):
                logger.info("    %s (%d tasks: %s)", commit[:12], len(task_ids), ", ".join(task_ids[:3]))
        logger.info("Dry run complete. Use without --dry-run to execute.")
        return

    if not args.skip_clone:
        _clone_repos(repo_commits, cache_dir=args.cache_dir)

    if not args.skip_ssg:
        _build_ssgs(repo_commits, cache_dir=args.cache_dir, ssg_dir=args.ssg_dir, workers=args.workers)

    if not args.skip_index:
        _index_entities(
            repo_commits,
            cache_dir=args.cache_dir,
            ssg_dir=args.ssg_dir,
            persist_dir=args.persist_dir,
            provider=args.provider,
            collection_prefix=args.collection_prefix,
        )

    logger.info("Phase 1 setup complete!")


def _clone_repos(
    repo_commits: dict[str, dict[str, list[str]]],
    *,
    cache_dir: Path,
) -> None:
    from token_sniper.evaluation.phase0 import ensure_repo_cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo in sorted(repo_commits):
        for commit in sorted(repo_commits[repo]):
            logger.info("Ensuring repo cached: %s @ %s", repo, commit[:12])
            try:
                ensure_repo_cached(repo=repo, base_commit=commit, cache_dir=cache_dir)
            except RuntimeError as exc:
                logger.error("Failed to cache %s @ %s: %s — skipping", repo, commit[:12], exc)


def _build_one_ssg(args: tuple[str, str, Path, Path]) -> tuple[str, str, bool, str]:
    repo, commit, cache_dir, ssg_dir = args
    import shutil

    from token_sniper.evaluation.phase0 import materialize_repo_snapshot
    from token_sniper.graph.io import write_ssg_json
    from token_sniper.graph.ssg import build_ssg

    commit_short = commit[:12]
    repo_slug = repo.replace("/", "__")
    ssg_path = ssg_dir / repo_slug / commit_short / "ssg.json"
    skip_marker = ssg_dir / repo_slug / commit_short / ".skip"

    if ssg_path.exists():
        return repo, commit_short, True, "cached"
    if skip_marker.exists():
        return repo, commit_short, False, f"skipped ({skip_marker.read_text().strip()})"

    work_dir = ssg_dir / repo_slug / commit_short / "work"
    t_step = time.perf_counter()
    try:
        repo_root = materialize_repo_snapshot(
            repo=repo, base_commit=commit,
            cache_dir=cache_dir, work_dir=work_dir,
        )
        t0 = time.perf_counter()
        graph = build_ssg(repo_root)
        elapsed = time.perf_counter() - t0
        write_ssg_json(graph, ssg_path)
        _write_timing(ssg_path.parent / SSG_TIMING, {
            "repo": repo, "commit": commit, "seconds": time.perf_counter() - t_step, "build_ssg_seconds": elapsed,
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        })
        msg = f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges in {elapsed:.1f}s"
        return repo, commit_short, True, msg
    except Exception as exc:
        skip_marker.parent.mkdir(parents=True, exist_ok=True)
        skip_marker.write_text(str(exc))
        return repo, commit_short, False, str(exc)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _build_ssgs(
    repo_commits: dict[str, dict[str, list[str]]],
    *,
    cache_dir: Path,
    ssg_dir: Path,
    workers: int = 0,
) -> None:
    import multiprocessing as mp

    ssg_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, Path, Path]] = []
    for repo in sorted(repo_commits):
        for commit in sorted(repo_commits[repo]):
            tasks.append((repo, commit, cache_dir, ssg_dir))

    if workers <= 0:
        workers = min(mp.cpu_count() // 2, 16)
    logger.info("Building SSGs: %d tasks, %d parallel workers", len(tasks), workers)

    built = skipped = failed = 0
    with mp.Pool(workers) as pool:
        for repo, commit_short, ok, msg in pool.imap_unordered(_build_one_ssg, tasks):
            if ok and msg == "cached":
                logger.info("SSG already exists: %s @ %s", repo, commit_short)
                skipped += 1
            elif ok:
                logger.info("SSG built: %s @ %s — %s", repo, commit_short, msg)
                built += 1
            else:
                logger.error("Failed to build SSG for %s @ %s: %s", repo, commit_short, msg)
                failed += 1

    logger.info("SSG summary: %d built, %d cached, %d failed", built, skipped, failed)


def _index_entities(
    repo_commits: dict[str, dict[str, list[str]]],
    *,
    cache_dir: Path,
    ssg_dir: Path,
    persist_dir: str,
    provider: str,
    collection_prefix: str,
) -> None:
    import hashlib
    import shutil

    import numpy as np

    from token_sniper.evaluation.phase0 import materialize_repo_snapshot
    from token_sniper.retrieval.config import EmbeddingConfig
    from token_sniper.retrieval.embeddings import create_embedder
    from token_sniper.retrieval.retriever import prepare_entities_for_indexing
    from token_sniper.retrieval.vectorstore import EntityVectorStore
    from token_sniper.parse import extract_python_entities

    model_names = {
        "voyageai": "voyage-code-3",
        "nomic-local": "nomic-embed-code-v1",
        "sentence-transformers": "jinaai/jina-code-embeddings-0.5b",
    }
    embedding_config = EmbeddingConfig(
        provider=provider,
        name=model_names.get(provider, "jinaai/jina-code-embeddings-0.5b"),
    )

    if provider == "voyageai":
        api_key = embedding_config.resolve_api_key()
        if api_key:
            masked = api_key[:8] + "..." + api_key[-4:]
            logger.info("Embedding provider: VOYAGE CODE 3 (API key: %s)", masked)
        else:
            logger.error(
                "VOYAGE_API_KEY not set! Use --provider sentence-transformers "
                "for free local GPU embeddings"
            )
            raise SystemExit(1)
    else:
        logger.info("Embedding provider: %s (%s)", provider, embedding_config.name)

    embedder = create_embedder(embedding_config)
    vector_store = EntityVectorStore(
        persist_directory=persist_dir,
        collection_prefix=collection_prefix,
    )

    embed_cache: dict[str, list[float]] = {}
    total_embedded = 0
    total_cache_hits = 0

    for repo in sorted(repo_commits):
        for commit in sorted(repo_commits[repo]):
            commit_short = commit[:12]
            collection_repo = f"{repo}@{commit_short}"

            stats = vector_store.get_collection_stats(collection_repo)
            if stats["entity_count"] > 0:
                logger.info(
                    "Already indexed: %s @ %s (%d entities)",
                    repo, commit_short, stats["entity_count"],
                )
                continue

            t_step = time.perf_counter()
            work_dir = (
                Path(persist_dir).resolve().parent
                / "index_work"
                / repo.replace("/", "__")
                / commit_short
            )
            try:
                repo_root = materialize_repo_snapshot(
                    repo=repo, base_commit=commit,
                    cache_dir=cache_dir, work_dir=work_dir,
                )

                root = Path(repo_root).resolve()
                py_files = sorted(p for p in root.glob("**/*.py") if p.is_file())
                logger.info("Indexing: %s @ %s (%d .py files)", repo, commit_short, len(py_files))

                from token_sniper.parse import CodeEntity
                all_entities: list[CodeEntity] = []
                for abs_path in py_files:
                    rel = abs_path.relative_to(root)
                    try:
                        raw = extract_python_entities(abs_path, project_root=root)
                        for e in raw:
                            all_entities.append(e.model_copy(update={"file_path": rel.as_posix()}))
                    except (OSError, UnicodeDecodeError, ValueError):
                        continue

                if not all_entities:
                    logger.warning("No entities found in %s @ %s", repo, commit_short)
                    continue

                entities_data, texts = prepare_entities_for_indexing(
                    all_entities, repo=collection_repo,
                )

                text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
                new_indices = [i for i, h in enumerate(text_hashes) if h not in embed_cache]
                new_texts = [texts[i] for i in new_indices]
                cache_hits = len(texts) - len(new_texts)

                if new_texts:
                    t0 = time.perf_counter()
                    new_embeddings = embedder.embed_documents(new_texts)
                    elapsed = time.perf_counter() - t0
                    for idx, emb in zip(new_indices, new_embeddings):
                        embed_cache[text_hashes[idx]] = emb
                    total_embedded += len(new_texts)
                    logger.info(
                        "Embedded %d new texts in %.1fs (cache hits: %d/%d = %.0f%%)",
                        len(new_texts), elapsed, cache_hits, len(texts),
                        100 * cache_hits / max(len(texts), 1),
                    )
                else:
                    logger.info(
                        "100%% cache hit: all %d texts already embedded", len(texts),
                    )

                total_cache_hits += cache_hits

                all_embeddings = [embed_cache[h] for h in text_hashes]

                t1 = time.perf_counter()
                count = vector_store.index_entities(
                    entities_data=entities_data,
                    repo=collection_repo,
                    embeddings=all_embeddings,
                )
                logger.info(
                    "Indexed %d entities for %s @ %s in %.1fs",
                    count, repo, commit_short, time.perf_counter() - t1,
                )
                _write_timing(ssg_dir / repo.replace("/", "__") / commit_short / INDEX_TIMING, {
                    "repo": repo, "commit": commit, "seconds": time.perf_counter() - t_step,
                    "provider": provider, "embedding_model": embedding_config.name, "entities": count,
                    "embedded_texts": len(new_texts), "embedding_cache_hits": cache_hits,
                })
            except Exception as exc:
                logger.error("Failed to index %s @ %s: %s", repo, commit_short, exc)
                continue
            finally:
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "Indexing complete: %d unique texts embedded, %d cache hits saved",
        total_embedded, total_cache_hits,
    )


if __name__ == "__main__":
    main()
