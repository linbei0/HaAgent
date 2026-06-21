"""
haagent/cli.py - HaAgent CLI 入口

提供 run、smoke、inspect 和 export-eval 命令的参数解析与输出展示。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haagent.models.gateway import OpenAIChatCompletionsGateway, OpenAIResponsesGateway
from haagent.models.provider_profile import ProviderProfile, ProviderProfileError, load_provider_profile
from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.eval_export import export_eval_case
from haagent.runtime.orchestrator import RunOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haagent", description="HaAgent runtime CLI")
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
        choices=["fake", "openai", "openai-chat"],
        default="fake",
        help="model provider to use (default: fake)",
    )
    run_parser.add_argument(
        "--profile",
        help="provider profile name from .haagent/providers.json",
    )
    run_parser.add_argument(
        "--model",
        help="OpenAI model name; only used when --provider openai",
    )
    run_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible Responses API base URL; only used when --provider openai",
    )
    run_parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=3,
        help="maximum model/tool turns before failing the run (default: 3)",
    )

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="run the minimal HaAgent smoke suite",
    )
    smoke_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(".runs"),
        help="directory for episode packages (default: .runs)",
    )
    smoke_parser.add_argument(
        "--profile",
        help="real provider profile name from .haagent/providers.json",
    )
    smoke_parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=12,
        help="maximum model/tool turns per smoke task (default: 12)",
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
        try:
            model_gateway = _build_run_model_gateway(args)
        except ProviderProfileError as error:
            print(f"error: {error}")
            return 1
        if model_gateway is not None:
            result = RunOrchestrator(
                runs_root=args.runs_root,
                model_gateway=model_gateway,
                max_turns=args.max_turns,
            ).run(args.task_yaml)
        else:
            result = RunOrchestrator(
                runs_root=args.runs_root,
                max_turns=args.max_turns,
            ).run(args.task_yaml)
        _print_run_summary(result)
        return 0 if result.status.value == "completed" else 1

    if args.command == "smoke":
        return _handle_smoke(args)

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


@dataclass(frozen=True)
class SmokeDefinition:
    name: str
    task_path: Path
    requires_profile: bool


@dataclass(frozen=True)
class SmokeResult:
    name: str
    status: str
    episode_path: Path | None
    failed_stage: str | None = None
    failure_category: str | None = None
    reason: str | None = None


SMOKE_DEFINITIONS = [
    SmokeDefinition(
        name="hello",
        task_path=PROJECT_ROOT / "examples/tasks/hello.yaml",
        requires_profile=False,
    ),
    SmokeDefinition(
        name="real_file_read",
        task_path=PROJECT_ROOT / "examples/tasks/openai_chat_file_read_smoke.yaml",
        requires_profile=True,
    ),
    SmokeDefinition(
        name="real_edit_verify",
        task_path=PROJECT_ROOT / "examples/tasks/openai_chat_edit_smoke.yaml",
        requires_profile=True,
    ),
]


def _handle_smoke(args: argparse.Namespace) -> int:
    smoke_definitions = [
        definition
        for definition in SMOKE_DEFINITIONS
        if not definition.requires_profile or args.profile is not None
    ]
    exit_code = 0
    for definition in smoke_definitions:
        result = _run_smoke_definition(definition, args)
        _print_smoke_result(result)
        if result.status != "completed":
            exit_code = 1
    return exit_code


def _run_smoke_definition(definition: SmokeDefinition, args: argparse.Namespace) -> SmokeResult:
    model_gateway = None
    if definition.requires_profile:
        try:
            model_gateway = _gateway_from_profile(load_provider_profile(args.profile))
        except ProviderProfileError as error:
            return SmokeResult(
                name=definition.name,
                status="failed",
                episode_path=None,
                failed_stage="configuration",
                failure_category="Provider Profile Error",
                reason=str(error),
            )
    result = RunOrchestrator(
        runs_root=args.runs_root,
        model_gateway=model_gateway,
        max_turns=args.max_turns,
    ).run(definition.task_path)
    if result.status.value == "completed":
        return SmokeResult(definition.name, result.status.value, result.episode_path)
    stage, category, reason = _run_failure_summary(result.episode_path)
    return SmokeResult(
        name=definition.name,
        status=result.status.value,
        episode_path=result.episode_path,
        failed_stage=stage,
        failure_category=category,
        reason=reason,
    )


def _run_failure_summary(episode_path: Path) -> tuple[str, str, str]:
    try:
        package_view = load_inspect_episode_package(episode_path)
    except EpisodeValidationError as error:
        return "summary", "Episode Summary Error", str(error)
    failure = package_view.failure_record.get("failure")
    if not isinstance(failure, dict):
        return "unknown", "unknown", ""
    return (
        str(failure.get("stage", "unknown")),
        str(failure.get("category", "unknown")),
        str(failure.get("evidence", "")),
    )


def _print_smoke_result(result: SmokeResult) -> None:
    print(f"smoke={result.name}")
    print(f"status={result.status}")
    episode_path = "none" if result.episode_path is None else str(result.episode_path)
    print(f"episode_path={episode_path}")
    if result.status != "completed":
        print(f"failed_stage={_summary_value(result.failed_stage or 'unknown')}")
        print(f"failure_category={_summary_value(result.failure_category or 'unknown')}")
        print(f"reason={_summary_value(result.reason or '')}")


def _build_run_model_gateway(args: argparse.Namespace):
    if args.profile is not None:
        if args.provider != "fake" or args.model is not None or args.base_url is not None:
            raise ProviderProfileError(
                "--profile cannot be combined with --provider, --model, or --base-url",
            )
        return _gateway_from_profile(load_provider_profile(args.profile))

    if args.provider in {"openai", "openai-chat"}:
        gateway_kwargs = {}
        if args.model is not None:
            gateway_kwargs["model"] = args.model
        if args.base_url is not None:
            gateway_kwargs["base_url"] = args.base_url
        gateway_class = (
            OpenAIResponsesGateway
            if args.provider == "openai"
            else OpenAIChatCompletionsGateway
        )
        return gateway_class(**gateway_kwargs)
    return None


def _gateway_from_profile(profile: ProviderProfile):
    gateway_kwargs = {
        "api_key": profile.api_key,
        "model": profile.model,
        "base_url": profile.base_url,
    }
    if profile.provider == "openai":
        return OpenAIResponsesGateway(**gateway_kwargs)
    if profile.provider == "openai-chat":
        return OpenAIChatCompletionsGateway(**gateway_kwargs)
    raise ProviderProfileError(f"unsupported provider in profile: {profile.provider}")


def _print_run_summary(result) -> None:
    """输出短 run 摘要；完整复盘仍交给 inspect。"""
    print(f"status={result.status.value}")
    print(f"episode_path={result.episode_path}")
    try:
        package_view = load_inspect_episode_package(result.episode_path)
    except EpisodeValidationError as error:
        print(f"summary_error={_summary_value(str(error))}")
        return

    print(f"provider={_summary_provider(package_view.episode_metadata)}")
    if result.status.value == "completed":
        print(f"final_response={_summary_value(_run_final_response(package_view.transcript))}")
        return

    failure = package_view.failure_record.get("failure")
    if not isinstance(failure, dict):
        failure = {}
    print(f"failed_stage={_summary_value(str(failure.get('stage', 'unknown')))}")
    print(f"failure_category={_summary_value(str(failure.get('category', 'unknown')))}")
    print(f"reason={_summary_value(str(failure.get('evidence', '')))}")


def _run_final_response(transcript: list[dict[str, Any]]) -> str:
    response = _last_model_response(transcript)
    if response is None:
        return "none"
    return str(response.get("content", ""))


def _summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    return _excerpt(normalized, limit)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-turns must be a positive integer")
    return parsed


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
        package_view = load_inspect_episode_package(episode_path)
    except EpisodeValidationError as error:
        raise EpisodeInspectError(str(error)) from error
    episode_metadata = package_view.episode_metadata
    context_manifest = package_view.context_manifest
    plan = package_view.plan
    transcript = package_view.transcript
    tool_calls = package_view.tool_calls
    verification = package_view.verification_commands
    verification_reached = package_view.verification_reached
    failure_record = package_view.failure_record
    sandbox = package_view.sandbox

    failure_attribution = (episode_path / "failure-attribution.md").read_text(encoding="utf-8").strip()

    state_flow = [
        record["status"]
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    final_status = state_flow[-1] if state_flow else "unknown"
    final_status = episode_metadata.get("status", final_status)
    model_calls = [
        record
        for record in transcript
        if record.get("event") == "model_call"
    ]

    lines = [
        "Run Summary",
        f"- episode_path: {episode_path}",
        f"- episode_version: {episode_metadata.get('episode_version', 'unknown')}",
        f"- status: {final_status}",
        f"- provider: {_summary_provider(episode_metadata)}",
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
    lines.extend(["", "Final Response"])
    lines.extend(_format_final_response(transcript))
    lines.extend(["", "Tool Calls"])
    lines.extend(_format_tool_calls(tool_calls))
    lines.extend(["", "Approval Summary"])
    lines.extend(_format_approval_summary(tool_calls))
    lines.extend(["", "Tool Argument Errors"])
    lines.extend(_format_tool_argument_errors(tool_calls))
    lines.extend(["", "Verification"])
    lines.extend(_format_verification(verification, verification_reached))
    lines.extend(["", "Structured Failure"])
    lines.extend(_format_failure_record(failure_record))
    lines.extend(["", "Failure Attribution", failure_attribution])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_provider(episode_metadata: dict[str, Any]) -> str:
    return str(episode_metadata.get("provider", "unknown"))


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


def _format_plan(plan: dict[str, Any]) -> list[str]:
    planned_steps = plan.get("planned_steps", [])
    if not planned_steps:
        return ["- none"]
    return [f"- {step}" for step in planned_steps]


def _format_sandbox(sandbox: dict[str, Any]) -> list[str]:
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
        next_action = _read_context_next_action(episode_path / str(context["manifest_path"]))
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


def _read_context_next_action(path: Path) -> dict[str, Any]:
    context_manifest = _read_json(path)
    next_action = context_manifest.get("next_action")
    if not isinstance(next_action, dict):
        raise EpisodeInspectError(f"{path.name} next_action must be an object")
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


def _format_final_response(transcript: list[dict[str, Any]]) -> list[str]:
    response = _last_model_response(transcript)
    if response is None:
        return ["- none"]
    tool_calls = response.get("tool_calls", [])
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    content = str(response.get("content", ""))
    return [
        (
            f"- provider={response.get('provider', 'unknown')} "
            f"turn={response.get('turn', 'unknown')} "
            f"tool_call_count={tool_call_count}"
        ),
        f"- content: {_excerpt(content)}",
    ]


def _last_model_response(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return record
    return None


def _excerpt(content: str, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "... [truncated]"


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
        if policy is None and _policy_not_evaluated(call):
            error = call.get("error") if isinstance(call.get("error"), dict) else {}
            lines.append(f"- {tool_name}: policy=not_evaluated reason={error.get('message', '')}")
            continue
        approval = policy["approval"]
        required = "true" if approval.get("required") is True else "false"
        lines.append(
            (
                f"- {tool_name}: action={policy['action']} "
                f"approval.required={required} "
                f"approval.status={approval['status']} "
                f"approval.reason={approval['reason']}"
            ),
        )
    return lines


def _policy_not_evaluated(call: dict[str, Any]) -> bool:
    error = call.get("error")
    return (
        call.get("status") == "error"
        and isinstance(error, dict)
        and error.get("type") in {"tool_not_allowed", "unknown_tool"}
    )


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


def _format_verification(
    commands: list[dict[str, Any]],
    verification_reached: bool = True,
) -> list[str]:
    if not verification_reached:
        return ["- not reached"]
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


def _format_failure_record(record: dict[str, Any]) -> list[str]:
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
