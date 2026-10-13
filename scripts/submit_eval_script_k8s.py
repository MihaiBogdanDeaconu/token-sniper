from __future__ import annotations

import argparse
import shlex

from scripts.submit_eval_retrieval_k8s import add_job_args, submit_eval_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one repo script on the eval PVC")
    add_job_args(parser)
    parser.add_argument("script", nargs=argparse.REMAINDER, help="-- <script.py> [args...]")
    args = parser.parse_args()
    command = args.script[1:] if args.script[:1] == ["--"] else args.script
    if not command:
        parser.error("give the script and its arguments after --")
    submit_eval_job(args, component="eval-script", body=lambda _job_name: f"python {shlex.join(command)}\n")


if __name__ == "__main__":
    main()
