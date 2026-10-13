from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts._check_swebench_overlap import main, repo_from_instance_id

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST = [
    {"instance_id": "flexget__flexget-1", "repo": "Flexget/Flexget"},
    {"instance_id": "owner__train-2", "repo": "owner/train"},
    {"instance_id": "owner__held-3", "repo": "owner/held"},
]


def _pin(ids: list[str]) -> dict:
    return {
        "dataset": "fake/bench",
        "instance_ids": ids,
        "repos": sorted({repo_from_instance_id(i) for i in ids}),
    }


def _write(tmp_path: Path, eval_ids: list[str], split: dict | None = None) -> list[str]:
    split = split or {
        "train": ["flexget__flexget-1"],
        "val": ["owner__train-2"],
        "train_repos": ["Flexget/Flexget"],
        "val_repos": ["owner/train"],
    }
    pin = {
        "lite_274": _pin(eval_ids),
        "verified_500": _pin(["other__repo-1"]),
        "locbench_560": _pin(["other__repo-2"]),
    }
    for name, payload in (("eval.json", pin), ("manifest.json", MANIFEST), ("split.json", split)):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return [
        "--eval-pin", str(tmp_path / "eval.json"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--split", str(tmp_path / "split.json"),
        "--report", str(tmp_path / "out" / "report.json"),
    ]


def _report(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))


def test_repo_from_instance_id() -> None:
    assert repo_from_instance_id("Agenta-AI__agenta-1639") == "Agenta-AI/agenta"
    assert repo_from_instance_id("scikit-learn__scikit-learn-10297") == "scikit-learn/scikit-learn"
    with pytest.raises(ValueError, match="does not match"):
        repo_from_instance_id("not-an-id")


def test_clean_passes(tmp_path: Path) -> None:
    assert main(_write(tmp_path, ["clean__repo-1"])) == 0
    report = _report(tmp_path)
    assert report["verdict"]["status"] == "pass"
    assert report["training"] == {"n_instances": 2, "n_repos": 2}
    assert report["eval_sets"]["lite_274"]["repo_overlap"] == []


def test_instance_id_overlap_fails_case_insensitively(tmp_path: Path, capsys) -> None:
    assert main(_write(tmp_path, ["Flexget__Flexget-1"])) == 1
    report = _report(tmp_path)
    overlap = report["eval_sets"]["lite_274"]["instance_id_overlap"]
    assert overlap == [{
        "instance_id": "Flexget__Flexget-1",
        "train_instance_id": "flexget__flexget-1",
        "sources": [f"{tmp_path / 'split.json'}:train"],
    }]
    assert report["verdict"]["status"] == "fail"
    assert "[id]   lite_274: Flexget__Flexget-1" in capsys.readouterr().err


def test_id_overlap_fails_even_with_allow_flag(tmp_path: Path) -> None:
    args = _write(tmp_path, ["owner__train-2"]) + ["--allow-repo-overlap", "same repo, later period"]
    assert main(args) == 1
    assert _report(tmp_path)["verdict"]["status"] == "fail"


def test_repo_overlap_fails_without_allow(tmp_path: Path, capsys) -> None:
    assert main(_write(tmp_path, ["owner__train-99", "owner__train-100", "clean__repo-1"])) == 1
    s = _report(tmp_path)["eval_sets"]["lite_274"]
    assert s["instance_id_overlap"] == []
    assert s["repo_overlap"] == [{
        "repo": "owner/train", "train_names": ["owner/train"],
        "eval_instances": 2, "train_instances": 1,
    }]
    assert s["repo_overlap_eval_instances"] == 2
    assert "[repo] lite_274: owner/train (eval=2, train=1)" in capsys.readouterr().err


def test_repo_overlap_allowed_is_recorded(tmp_path: Path) -> None:
    args = _write(tmp_path, ["owner__train-99"]) + ["--allow-repo-overlap", "disjoint time window"]
    assert main(args) == 0
    verdict = _report(tmp_path)["verdict"]
    assert verdict["status"] == "pass_with_allowed_repo_overlap"
    assert verdict["repo_overlap_allowed"] is True
    assert verdict["repo_overlap_allow_reason"] == "disjoint time window"


def test_empty_allow_reason_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        main(_write(tmp_path, ["clean__repo-1"]) + ["--allow-repo-overlap", " "])


def test_holdout_overlap_is_informational(tmp_path: Path) -> None:
    assert main(_write(tmp_path, ["owner__held-3"])) == 0
    report = _report(tmp_path)
    held = report["eval_sets"]["lite_274"]["holdout_manifest_overlap"]
    assert held["instance_ids"] == ["owner__held-3"]
    assert held["repos"][0]["holdout_instances"] == 1
    assert report["verdict"]["status"] == "pass"


def test_split_id_missing_from_manifest_raises(tmp_path: Path) -> None:
    split = {"train": ["ghost__repo-1"], "val": []}
    with pytest.raises(ValueError, match="not in the task manifest"):
        main(_write(tmp_path, ["clean__repo-1"], split))


def test_declared_split_repos_mismatch_raises(tmp_path: Path) -> None:
    split = {"train": ["owner__train-2"], "val": [], "train_repos": ["owner/wrong"]}
    with pytest.raises(ValueError, match="disagree with the manifest"):
        main(_write(tmp_path, ["clean__repo-1"], split))


def test_eval_declared_repos_mismatch_raises(tmp_path: Path) -> None:
    args = _write(tmp_path, ["clean__repo-1"])
    pin = json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))
    pin["lite_274"]["repos"] = ["clean/other"]
    (tmp_path / "eval.json").write_text(json.dumps(pin), encoding="utf-8")
    with pytest.raises(ValueError, match="declared repos disagree"):
        main(args)
