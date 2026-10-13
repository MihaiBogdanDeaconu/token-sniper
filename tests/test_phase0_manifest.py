import json
from collections import Counter
from typing import Any
from unittest.mock import patch

from scripts.build_phase0_manifest import (
    assign_bucket,
    build_phase0_manifest_tasks,
    extract_patched_files,
)


def test_extract_patched_files() -> None:
    patch_text = """
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old
+new
"""
    files = extract_patched_files(patch_text)
    assert files == ["file1.py", "file2.py"]


def test_assign_bucket() -> None:
    assert assign_bucket(0) == "single_file"
    assert assign_bucket(1) == "single_file"
    assert assign_bucket(2) == "multi_file_2_3"
    assert assign_bucket(3) == "multi_file_2_3"
    assert assign_bucket(4) == "multi_file_4_plus"
    assert assign_bucket(10) == "multi_file_4_plus"


def test_build_phase0_manifest_enforces_minimum_repo_coverage() -> None:
    mock_data: dict[str, dict[str, Any]] = {}
    repos = [f"repo_{index}" for index in range(10)]

    for repo_index, repo_name in enumerate(repos):
        for task_index in range(5):
            instance_id = f"{repo_index:02d}_sf_{task_index:02d}"
            mock_data[instance_id] = {
                "instance_id": instance_id,
                "repo": repo_name,
                "base_commit": "123",
                "patch": f"+++ b/{repo_name}_single_{task_index}.py\n@@",
            }
        for task_index in range(5):
            instance_id = f"{repo_index:02d}_mf23_{task_index:02d}"
            mock_data[instance_id] = {
                "instance_id": instance_id,
                "repo": repo_name,
                "base_commit": "123",
                "patch": (
                    f"+++ b/{repo_name}_two_{task_index}_a.py\n@@\n"
                    f"+++ b/{repo_name}_two_{task_index}_b.py\n@@"
                ),
            }
        for task_index in range(2):
            instance_id = f"{repo_index:02d}_mf4_{task_index:02d}"
            mock_data[instance_id] = {
                "instance_id": instance_id,
                "repo": repo_name,
                "base_commit": "123",
                "patch": (
                    f"+++ b/{repo_name}_four_{task_index}_a.py\n@@\n"
                    f"+++ b/{repo_name}_four_{task_index}_b.py\n@@\n"
                    f"+++ b/{repo_name}_four_{task_index}_c.py\n@@\n"
                    f"+++ b/{repo_name}_four_{task_index}_d.py\n@@"
                ),
            }

    manifest_tasks = build_phase0_manifest_tasks(mock_data)

    assert len(manifest_tasks) == 50
    buckets = [task.bucket for task in manifest_tasks]
    assert buckets.count("single_file") == 20
    assert buckets.count("multi_file_2_3") == 20
    assert buckets.count("multi_file_4_plus") == 10
    repo_counts = Counter(task.repo for task in manifest_tasks)
    assert all(repo_counts[repo_name] >= 3 for repo_name in repos)


@patch("scripts.build_phase0_manifest._load_dataset_rows")
def test_manifest_builder_main_writes_manifest(mock_load: Any, tmp_path: Any) -> None:
    mock_load.return_value = {
        "task-1": {
            "instance_id": "task-1",
            "repo": "repo_a",
            "base_commit": "123",
            "patch": "+++ b/file1.py\n@@",
        },
        "task-2": {
            "instance_id": "task-2",
            "repo": "repo_b",
            "base_commit": "456",
            "patch": "+++ b/file2.py\n@@\n+++ b/file3.py\n@@",
        },
    }

    import scripts.build_phase0_manifest

    output_file = tmp_path / "manifest.json"
    with patch("sys.argv", ["build_phase0_manifest.py", "--output", str(output_file)]):
        scripts.build_phase0_manifest.main()

    written = json.loads(output_file.read_text())
    assert len(written) == 2
    assert written[0]["instance_id"] == "task-1"
