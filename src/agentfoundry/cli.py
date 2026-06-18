"""
agentfoundry/cli.py - AgentFoundry CLI 入口

提供 agentfoundry run <task.yaml> 命令，启动最小 RunOrchestrator。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentfoundry.runtime.orchestrator import RunOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfoundry", description="AgentFoundry runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task.yaml file")
    run_parser.add_argument("task_yaml", type=Path, help="path to task.yaml")
    run_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(".runs"),
        help="directory for episode packages (default: .runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析 CLI 参数，运行 orchestrator，并输出机器可读的最小结果。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = RunOrchestrator(runs_root=args.runs_root).run(args.task_yaml)
        print(f"status={result.status.value}")
        print(f"episode_path={result.episode_path}")
        return 0 if result.status.value == "completed" else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
