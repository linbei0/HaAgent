"""
agentfoundry/cli.py - AgentFoundry CLI 入口

提供 agentfoundry run <task.yaml> 和 agentfoundry inspect <episode_path> 命令。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

    inspect_parser = subparsers.add_parser("inspect", help="inspect an episode package")
    inspect_parser.add_argument("episode_path", type=Path, help="path to an episode directory")
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

    if args.command == "inspect":
        try:
            print(render_episode_summary(args.episode_path))
        except EpisodeInspectError as error:
            print(f"error: {error}")
            return 1
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


class EpisodeInspectError(RuntimeError):
    """Raised when an episode package cannot be inspected safely."""


def render_episode_summary(episode_path: Path) -> str:
    """读取 episode package，并生成面向人的审计摘要。"""
    required_files = [
        "context-manifest.json",
        "transcript.jsonl",
        "tool-calls.jsonl",
        "verification/commands.jsonl",
        "failure-attribution.md",
    ]
    for relative_path in required_files:
        path = episode_path / relative_path
        if not path.exists():
            raise EpisodeInspectError(f"missing required episode file: {relative_path}")

    context_manifest = _read_json(episode_path / "context-manifest.json")
    transcript = _read_jsonl(episode_path / "transcript.jsonl")
    tool_calls = _read_jsonl(episode_path / "tool-calls.jsonl")
    verification = _read_jsonl(episode_path / "verification" / "commands.jsonl")
    failure_attribution = (episode_path / "failure-attribution.md").read_text(encoding="utf-8").strip()

    state_flow = [
        record["status"]
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    final_status = state_flow[-1] if state_flow else "unknown"
    model_calls = [
        record
        for record in transcript
        if record.get("event") == "model_call"
    ]

    lines = [
        "Run Summary",
        f"- episode_path: {episode_path}",
        f"- status: {final_status}",
        f"- provider: {context_manifest.get('summary', {}).get('provider', 'unknown')}",
        f"- context_count: {context_manifest.get('context_count', 0)}",
        "",
        "State Flow",
        f"- {' -> '.join(state_flow) if state_flow else 'none'}",
        "",
        "Contexts",
    ]
    lines.extend(_format_contexts(context_manifest.get("contexts", [])))
    lines.extend(["", "Model Calls"])
    lines.extend(_format_model_calls(model_calls))
    lines.extend(["", "Tool Calls"])
    lines.extend(_format_tool_calls(tool_calls))
    lines.extend(["", "Verification"])
    lines.extend(_format_verification(verification))
    lines.extend(["", "Failure Attribution", failure_attribution])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _format_contexts(contexts: list[dict[str, Any]]) -> list[str]:
    if not contexts:
        return ["- none"]
    return [
        (
            f"- {context['context_id']}: "
            f"{context['model_input_path']} | {context['manifest_path']}"
        )
        for context in contexts
    ]


def _format_model_calls(model_calls: list[dict[str, Any]]) -> list[str]:
    if not model_calls:
        return ["- none"]
    return [
        (
            f"- provider={call.get('provider', 'unknown')} "
            f"context_id={call.get('context_id', 'unknown')}"
        )
        for call in model_calls
    ]


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    if not tool_calls:
        return ["- none"]
    return [
        f"- {call.get('tool_name', 'unknown')}: {call.get('status', 'unknown')}"
        for call in tool_calls
    ]


def _format_verification(commands: list[dict[str, Any]]) -> list[str]:
    if not commands:
        return ["- none"]
    lines = []
    for command in commands:
        lines.append(
            (
                f"- {command.get('command', '')}: {command.get('status', 'unknown')} "
                f"(exit_code={command.get('exit_code')})"
            ),
        )
        if command.get("timeout"):
            lines.append("  timeout: true")
        if command.get("stdout_excerpt"):
            lines.append(f"  stdout: {command['stdout_excerpt']}")
        if command.get("stderr_excerpt"):
            lines.append(f"  stderr: {command['stderr_excerpt']}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
