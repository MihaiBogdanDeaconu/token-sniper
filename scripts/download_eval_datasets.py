from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS: list[dict[str, Any]] = [
    {
        "key": "swebench_lite",
        "hf_id": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "expected_rows": 300,
    },
    {
        "key": "swebench_verified",
        "hf_id": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "expected_rows": 500,
    },
    {
        "key": "locbench",
        "hf_id": "czlll/Loc-Bench_V1",
        "split": "test",
        "expected_rows": 560,
    },
]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _export_jsonl(rows: list[dict[str, Any]], dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, ensure_ascii=False))
            handle.write("\n")
    return _sha256_file(dest)


def _hf_revision(hf_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(hf_id)
        return str(getattr(info, "sha", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resolve HF revision for %s: %s", hf_id, exc)
        return ""


def _load_split(hf_id: str, split: str) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    revision = _hf_revision(hf_id)
    kwargs: dict[str, Any] = {"path": hf_id, "split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(**kwargs)
    return [dict(row) for row in ds], revision


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw eval datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args()

    raw_root = args.output_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    errors: list[str] = []

    for spec in DATASETS:
        dest_dir = raw_root / spec["key"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = dest_dir / "instances.jsonl"
        logger.info("Downloading %s split=%s", spec["hf_id"], spec["split"])
        try:
            rows, revision = _load_split(spec["hf_id"], spec["split"])
        except Exception as exc:
            msg = f"Failed to download {spec['hf_id']}: {exc}"
            logger.error(msg)
            errors.append(msg)
            continue

        n_rows = len(rows)
        if n_rows != spec["expected_rows"]:
            msg = f"{spec['hf_id']} expected {spec['expected_rows']} rows, got {n_rows}"
            logger.error(msg)
            errors.append(msg)

        digest = _export_jsonl(rows, jsonl_path)
        entry = {
            "key": spec["key"],
            "hf_id": spec["hf_id"],
            "split": spec["split"],
            "row_count": n_rows,
            "expected_rows": spec["expected_rows"],
            "revision": revision,
            "jsonl_path": str(jsonl_path.relative_to(args.output_dir)),
            "sha256": digest,
        }
        _write_json(dest_dir / "meta.json", entry)
        entries.append(entry)
        logger.info("Wrote %s (%d rows, sha256=%s)", jsonl_path, n_rows, digest)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "datasets": entries,
        "ok": not errors,
        "errors": errors,
    }
    _write_json(raw_root / "download_manifest.json", manifest)
    lines = [
        "# Evaluation dataset download pin",
        "",
        f"Created: `{manifest['created_at']}`",
        "",
        "| Dataset | Split | Rows | SHA-256 |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['hf_id']}` | {entry['split']} | {entry['row_count']} "
            f"| `{entry['sha256']}` |"
        )
    lines.append("")
    (raw_root / "DOWNLOAD.md").write_text("\n".join(lines), encoding="utf-8")

    if errors:
        logger.error("Download finished with errors:\n%s", "\n".join(errors))
        raise SystemExit(1)
    logger.info("Download complete. Manifest: %s", raw_root / "download_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
