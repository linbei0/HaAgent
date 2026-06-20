"""
agentfoundry/cli.py - AgentFoundry CLI 入口

提供 agentfoundry run <task.yaml> 和 agentfoundry inspect <episode_path> 命令。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentfoundry.models.gateway import OpenAIResponsesGateway
from agentfoundry.runtime.episode_validator import (
    EpisodeValidationError,
    load_validated_episode_package,
    read_episode_metadata,
    read_failure_record,
)
from agentfoundry.runtime.eval_export import export_eval_case
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
    run_parser.add_argument(
        "--provider",
        choices=["fake", "openai"],
        default="fake",
        help="model provider to use (default: fake)",
    )
    run_parser.add_argument(
        "--model",
        help="OpenAI model name; only used when --provider openai",
    )
    run_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible Responses API base URL; only used when --provider openai",
    )

    inspect_parser = subparsers.add_parser("inspect", help="inspect an episode package")
    inspect_parser.add_argument("episode_path", type=Path, help="path to an episode directory")

    export_eval_parser = subparsers.add_parser("export-eval", help="export an eval case JSON")
    export_eval_parser.add_argument(
        "episode_paths",
        nargs="+",
        type=Path,
        help="path to one or more episode directories",
    )
    export_eval_parser.add_argument(
        "--output",
        type=Path,
        help="write eval case JSON to this file instead of stdout",
    )
    export_eval_parser.add_argument(
        "--output-dir",
        type=Path,
        help="write one eval case JSON file per episode into this existing directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析 CLI 参数，运行 orchestrator，并输出机器可读的最小结果。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.provider == "openai":
            gateway_kwargs = {}
            if args.model is not None:
                gateway_kwargs["model"] = args.model
            if args.base_url is not None:
                gateway_kwargs["base_url"] = args.base_url
            model_gateway = OpenAIResponsesGateway(**gateway_kwargs)
            result = RunOrchestrator(
                runs_root=args.runs_root,
                model_gateway=model_gateway,
            ).run(args.task_yaml)
        else:
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

    if args.command == "export-eval":
        return _handle_export_eval(args.episode_paths, args.output, args.output_dir)

    parser.error(f"unknown command: {args.command}")
    return 2


class EpisodeInspectError(RuntimeError):
    """Raised when an episode package cannot be inspected safely."""


def _handle_export_eval(
    episode_paths: list[Path],
    output_path: Path | None,
    output_dir: Path | None,
) -> int:
    """处理 eval case 单文件和批量导出命令。"""
    if len(episode_paths) == 1:
        return _export_single_eval_case(episode_paths[0], output_path, output_dir)
    if output_path is not None:
        print("error: --output can only be used with a single episode path")
        return 1
    if output_dir is None:
        print("error: multiple episode paths require --output-dir")
        return 1
    if not output_dir.exists():
        print(f"error: output directory does not exist: {output_dir}")
        return 1
    if not output_dir.is_dir():
        print(f"error: output directory is not a directory: {output_dir}")
        return 1
    records = []
    for episode_path in episode_paths:
        target_path = output_dir / f"{episode_path.name}.json"
        try:
            _write_eval_case_file(episode_path, target_path)
        except EpisodeValidationError as error:
            message = str(error)
            records.append(
                {
                    "episode_path": str(episode_path),
                    "status": "error",
                    "output_file": None,
                    "error": message,
                },
            )
            print(f"error={episode_path}: {message}")
            continue
        records.append(
            {
                "episode_path": str(episode_path),
                "status": "success",
                "output_file": str(target_path),
                "error": None,
            },
        )
        print(f"exported_eval_case={target_path}")
    _write_eval_dataset_manifest(output_dir, records)
    failure_count = sum(1 for record in records if record["status"] == "error")
    return 1 if failure_count else 0


def _export_single_eval_case(
    episode_path: Path,
    output_path: Path | None,
    output_dir: Path | None,
) -> int:
    if output_dir is not None:
        print("error: --output-dir requires multiple episode paths")
        return 1
    try:
        eval_case = export_eval_case(episode_path)
    except EpisodeValidationError as error:
        print(f"error: {error}")
        return 1
    output = json.dumps(eval_case, ensure_ascii=False, indent=2)
    if output_path is not None:
        if not output_path.parent.exists():
            print(f"error: output parent directory does not exist: {output_path.parent}")
            return 1
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"exported_eval_case={output_path}")
        return 0
    print(output)
    return 0


def _write_eval_case_file(episode_path: Path, output_path: Path) -> None:
    eval_case = export_eval_case(episode_path)
    output = json.dumps(eval_case, ensure_ascii=False, indent=2)
    output_path.write_text(output + "\n", encoding="utf-8")


def _write_eval_dataset_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
    success_count = sum(1 for record in records if record["status"] == "success")
    failure_count = sum(1 for record in records if record["status"] == "error")
    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_episode_summary(episode_path: Path) -> str:
    """读取 episode package，并生成面向人的审计摘要。"""
    try:
        episode_metadata, warnings = read_episode_metadata(episode_path)
        if episode_metadata is not None:
            package_view = load_validated_episode_package(episode_path)
            episode_metadata = package_view.episode_metadata
            context_manifest = package_view.context_manifest
            plan = package_view.plan
            transcript = package_view.transcript
            tool_calls = package_view.tool_calls
            verification = package_view.verification_commands
            failure_record = package_view.failure_record
            sandbox = _read_optional_json(episode_path / "sandbox.json")
        else:
            _ensure_legacy_inspect_files(episode_path)
            context_manifest = _read_json(episode_path / "context-manifest.json")
            plan = None
            transcript = _read_jsonl(episode_path / "transcript.jsonl")
            tool_calls = _read_jsonl(episode_path / "tool-calls.jsonl")
            verification = _read_jsonl(episode_path / "verification" / "commands.jsonl")
            failure_record = read_failure_record(episode_path)
            sandbox = _read_optional_json(episode_path / "sandbox.json")
    except EpisodeValidationError as error:
        raise EpisodeInspectError(str(error)) from error

    failure_attribution = (episode_path / "failure-attribution.md").read_text(encoding="utf-8").strip()

    state_flow = [
        record["status"]
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    final_status = state_flow[-1] if state_flow else "unknown"
    if episode_metadata:
        final_status = episode_metadata.get("status", final_status)
    model_calls = [
        record
        for record in transcript
        if record.get("event") == "model_call"
    ]

    lines = [
        *warnings,
        "Run Summary",
        f"- episode_path: {episode_path}",
        f"- episode_version: {episode_metadata.get('episode_version', 'legacy') if episode_metadata else 'legacy'}",
        f"- status: {final_status}",
        f"- provider: {_summary_provider(episode_metadata, context_manifest)}",
        f"- context_count: {context_manifest.get('context_count', 0)}",
        "",
        "State Flow",
        f"- {' -> '.join(state_flow) if state_flow else 'none'}",
        "",
        "Contexts",
    ]
    lines.extend(_format_contexts(context_manifest.get("contexts", [])))
    lines.extend(["", "Plan"])
    lines.extend(_format_plan(plan))
    lines.extend(["", "Sandbox"])
    lines.extend(_format_sandbox(sandbox))
    lines.extend(["", "Next Actions"])
    lines.extend(_format_next_actions(episode_path, context_manifest.get("contexts", [])))
    lines.extend(["", "Model Calls"])
    lines.extend(_format_model_calls(model_calls))
    lines.extend(["", "Tool Calls"])
    lines.extend(_format_tool_calls(tool_calls))
    lines.extend(["", "Approval Summary"])
    lines.extend(_format_approval_summary(tool_calls))
    lines.extend(["", "Tool Argument Errors"])
    lines.extend(_format_tool_argument_errors(tool_calls))
    lines.extend(["", "Verification"])
    lines.extend(_format_verification(verification))
    lines.extend(["", "Structured Failure"])
    lines.extend(_format_failure_record(failure_record))
    lines.extend(["", "Failure Attribution", failure_attribution])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _ensure_legacy_inspect_files(episode_path: Path) -> None:
    """legacy episode 没有 v1 根 schema，只校验 inspect 展示必须读取的文件。"""
    required_files = [
        "context-manifest.json",
        "transcript.jsonl",
        "tool-calls.jsonl",
        "verification/commands.jsonl",
        "failure-attribution.md",
    ]
    for relative_path in required_files:
        if not (episode_path / relative_path).exists():
            raise EpisodeInspectError(f"missing required episode file: {relative_path}")


def _summary_provider(
    episode_metadata: dict[str, Any] | None,
    context_manifest: dict[str, Any],
) -> str:
    if episode_metadata and episode_metadata.get("provider"):
        return str(episode_metadata["provider"])
    return str(context_manifest.get("summary", {}).get("provider", "unknown"))


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


def _format_plan(plan: dict[str, Any] | None) -> list[str]:
    if plan is None:
        return ["- legacy episode without plan.json"]
    planned_steps = plan.get("planned_steps", [])
    if not planned_steps:
        return ["- none"]
    return [f"- {step}" for step in planned_steps]


def _format_sandbox(sandbox: dict[str, Any] | None) -> list[str]:
    if sandbox is None:
        return ["- legacy/missing"]
    resource_limits = sandbox.get("resource_limits", {})
    if not isinstance(resource_limits, dict):
        resource_limits = {}
    return [
        f"- filesystem_boundary: {sandbox.get('filesystem_boundary', 'unknown')}",
        f"- network_policy: {sandbox.get('network_policy', 'unknown')}",
        f"- process_policy: {sandbox.get('process_policy', 'unknown')}",
        f"- credential_policy: {sandbox.get('credential_policy', 'unknown')}",
        (
            "- command_timeout_seconds: "
            f"{resource_limits.get('command_timeout_seconds', 'unknown')}"
        ),
    ]


def _format_next_actions(episode_path: Path, contexts: list[dict[str, Any]]) -> list[str]:
    if not contexts:
        return ["- none"]
    lines = []
    for context in contexts:
        context_id = str(context.get("context_id", "unknown"))
        manifest_path = context.get("manifest_path")
        if not isinstance(manifest_path, str):
            lines.append(f"- {context_id}: legacy/missing")
            continue
        next_action = _read_context_next_action(episode_path / manifest_path)
        if next_action is None:
            lines.append(f"- {context_id}: legacy/missing")
            continue
        tool_name = next_action.get("based_on_tool_name")
        based_on_tool_name = str(tool_name) if tool_name is not None else "none"
        lines.append(
            (
                f"- {context_id}: status={next_action.get('status', 'unknown')} "
                f"based_on_tool_name={based_on_tool_name} "
                f"reason={next_action.get('reason', '')}"
            ),
        )
    return lines


def _read_context_next_action(path: Path) -> dict[str, Any] | None:
    try:
        context_manifest = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    next_action = context_manifest.get("next_action")
    if not isinstance(next_action, dict):
        return None
    return next_action


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


def _format_approval_summary(tool_calls: list[dict[str, Any]]) -> list[str]:
    if not tool_calls:
        return ["- none"]
    lines = []
    for call in tool_calls:
        tool_name = call.get("tool_name", "unknown")
        policy = call.get("policy")
        approval = policy.get("approval") if isinstance(policy, dict) else None
        if not isinstance(policy, dict) or not isinstance(approval, dict):
            lines.append(f"- {tool_name}: legacy/missing")
            continue
        required = "true" if approval.get("required") is True else "false"
        lines.append(
            (
                f"- {tool_name}: action={policy.get('action', 'missing')} "
                f"approval.required={required} "
                f"approval.status={approval.get('status', 'missing')} "
                f"approval.reason={approval.get('reason', 'legacy/missing')}"
            ),
        )
    return lines


def _format_tool_argument_errors(tool_calls: list[dict[str, Any]]) -> list[str]:
    errors = []
    for call in tool_calls:
        error = call.get("error")
        if isinstance(error, dict) and error.get("type") == "tool_argument_invalid":
            errors.append(
                f"- {call.get('tool_name', 'unknown')}: {error.get('message', '')}",
            )
    if not errors:
        return ["- none"]
    return errors


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


def _format_failure_record(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return ["- legacy episode without failure.json"]
    if record.get("status") == "success":
        return ["- status: success"]
    failure = record.get("failure") or {}
    return [
        f"- status: {record.get('status', 'unknown')}",
        f"- category: {failure.get('category', 'unknown')}",
        f"- stage: {failure.get('stage', 'unknown')}",
        f"- evidence: {failure.get('evidence', '')}",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
