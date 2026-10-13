from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from token_sniper.evaluation.phase0 import load_phase0_manifest
from token_sniper.graph.io import read_ssg_json
from token_sniper.vgib_selector.token_counts import (
    count_ssg_tokens,
    load_tokenizer,
    read_token_sidecar,
    sidecar_path,
    write_token_sidecar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-SSG token count sidecars")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--ssg-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Existing bare repo cache: <cache-dir>/owner__name")
    args = parser.parse_args()

    tasks = {t.instance_id: t for t in load_phase0_manifest(args.manifest).tasks}
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    instance_ids = split["train"] + split["val"]
    missing = [i for i in instance_ids if i not in tasks]
    if missing:
        raise ValueError(f"{len(missing)} split instances are not in {args.manifest}: {missing[:10]}")

    pairs = sorted({(tasks[i].repo, tasks[i].base_commit) for i in instance_ids})
    logger.info("%d instances -> %d (repo, commit) SSGs", len(instance_ids), len(pairs))

    tokenizer = load_tokenizer()
    for n, (repo, commit) in enumerate(pairs, 1):
        ssg_path = args.ssg_dir / repo.replace("/", "__") / commit[:12] / "ssg.json"
        if sidecar_path(ssg_path).exists():
            read_token_sidecar(ssg_path)
            logger.info("[%d/%d] %s@%s: sidecar valid", n, len(pairs), repo, commit[:12])
            continue

        t0 = time.monotonic()
        graph = read_ssg_json(ssg_path)
        git_dir = args.cache_dir / repo.replace("/", "__")
        if not git_dir.is_dir():
            raise FileNotFoundError(f"no cached repo for {repo} at {git_dir}")
        counts = count_ssg_tokens(graph, git_dir, commit, tokenizer)
        write_token_sidecar(ssg_path, counts)
        logger.info(
            "[%d/%d] %s@%s: %d nodes, %d tokens in %.1fs",
            n, len(pairs), repo, commit[:12], len(counts), sum(counts.values()), time.monotonic() - t0,
        )


if __name__ == "__main__":
    main()
