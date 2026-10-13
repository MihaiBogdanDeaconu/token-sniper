from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from token_sniper.agent.config import load_base_config
from token_sniper.agent.providers import OpenAIChatModelClient
from token_sniper.agent.runner import AgentRunSettings, run_agent_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Token Sniper shell agent for one task.")
    parser.add_argument("--repo", type=Path, required=True, help="Repository root to inspect.")
    parser.add_argument("--issue", type=str, help="Issue description or task prompt text.")
    parser.add_argument(
        "--issue-file",
        type=Path,
        help="Path to a text file containing the issue description.",
    )
    parser.add_argument(
        "--mode",
        choices=("blind", "oracle"),
        default="blind",
        help="Blind mode starts with no preloaded files; oracle mode injects known files.",
    )
    parser.add_argument(
        "--oracle-file",
        action="append",
        default=[],
        help="Repo-relative file path to preload in oracle mode. Can be provided multiple times.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="YAML config file containing model settings.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Optional directory for patch and session artifacts."
    )
    parser.add_argument(
        "--max-turns", type=int, help="Optional override for the agent max turn budget."
    )
    args = parser.parse_args()

    issue_description = _load_issue_description(issue=args.issue, issue_file=args.issue_file)
    config = load_base_config(args.config)
    output_dir = args.output_dir or _default_output_dir(
        experiments_dir=config.paths.experiments_dir,
        mode=args.mode,
    )
    run_settings = AgentRunSettings(
        repo_root=str(args.repo.resolve()),
        issue_description=issue_description,
        output_dir=str(output_dir.resolve()),
        mode=args.mode,
        oracle_files=list(args.oracle_file),
        max_turns=args.max_turns or config.models.agent.max_turns,
        context_window=config.models.agent.context_window,
        max_output_tokens=config.models.agent.max_tokens,
    )
    client = OpenAIChatModelClient(config.models.agent)
    result = run_agent_session(run_settings, client)
    print(json.dumps(result.model_dump(), indent=2))


def _load_issue_description(*, issue: str | None, issue_file: Path | None) -> str:
    if issue:
        return issue
    if issue_file is not None:
        return issue_file.read_text(encoding="utf-8")
    raise ValueError("Provide either --issue or --issue-file.")


def _default_output_dir(*, experiments_dir: str, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(experiments_dir) / "phase0" / f"{mode}_agent_{timestamp}"


if __name__ == "__main__":
    main()
