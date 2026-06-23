"""
haagent/cli.py - HaAgent CLI 入口

提供 run、smoke、inspect 和 export-eval 命令的参数解析与输出展示。
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from haagent.cli_inspect import EpisodeInspectError, render_episode_summary
from haagent.models.gateway import OpenAIChatCompletionsGateway, OpenAIResponsesGateway
from haagent.models.provider_profile import ProviderProfile, ProviderProfileError, load_provider_profile
from haagent.runtime.chat_session import (
    AgentSession,
    CHAT_MAX_TURNS,
    ChatEvent,
    ChatSessionError,
    ChatTurnResult,
)
from haagent.runtime.dogfood import render_dogfood_report, run_dogfood_tasks, skipped_dogfood_report
from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.eval_export import export_eval_case
from haagent.runtime.human_interaction import (
    HumanInteractionRequest,
    HumanInteractionResponse,
)
from haagent.runtime.orchestrator import RunOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHORING_ALLOWED_TOOLS = ["file_list", "file_read", "file_search", "apply_patch", "shell"]
AUTHORING_APPROVED_TOOLS = ["apply_patch", "shell"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haagent", description="HaAgent runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a task.yaml file")
    run_parser.add_argument("task_yaml", nargs="?", type=Path, help="path to task.yaml")
    run_parser.add_argument(
        "--goal",
        help="task goal used when task_yaml is omitted",
    )
    run_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root used when task_yaml is omitted",
    )
    run_parser.add_argument(
        "--verify",
        help="verification command used when task_yaml is omitted",
    )
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

    chat_parser = subparsers.add_parser("chat", help="run a natural language request")
    chat_parser.add_argument(
        "request",
        nargs="?",
        help="natural language request to run in the workspace; omit to enter REPL",
    )
    chat_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for the chat request (default: current directory)",
    )
    chat_parser.add_argument(
        "--resume",
        help="resume a chat session by session id or session package path",
    )
    chat_parser.add_argument(
        "--provider",
        choices=["fake", "openai", "openai-chat"],
        default="fake",
        help="model provider to use (default: fake)",
    )
    chat_parser.add_argument(
        "--profile",
        help="provider profile name from .haagent/providers.json",
    )
    chat_parser.add_argument(
        "--model",
        help="OpenAI model name; only used when --provider openai",
    )
    chat_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible Responses API base URL; only used when --provider openai",
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

    dogfood_parser = subparsers.add_parser(
        "dogfood",
        help="run manual real-model dogfood tasks outside default CI",
    )
    dogfood_parser.add_argument(
        "--runs-root",
        type=Path,
        help="directory for dogfood episode packages; defaults to a temporary directory",
    )
    dogfood_parser.add_argument(
        "--profile",
        help="real provider profile name from .haagent/providers.json",
    )
    dogfood_parser.add_argument(
        "--provider",
        choices=["openai", "openai-chat"],
        help="real provider to use when --profile is omitted",
    )
    dogfood_parser.add_argument(
        "--model",
        help="model name for --provider dogfood runs",
    )
    dogfood_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL for --provider dogfood runs",
    )
    dogfood_parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=16,
        help="maximum model/tool turns per dogfood task (default: 16)",
    )
    dogfood_parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="deny high-risk tool approvals instead of auto-granting them",
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
        generated_task_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            model_gateway = _build_run_model_gateway(args)
            task_path, generated_task_dir = _run_task_path(args)
        except ProviderProfileError as error:
            print(f"error: {error}")
            return 1
        except ValueError as error:
            print(f"error: {error}")
            return 2
        try:
            if model_gateway is not None:
                result = RunOrchestrator(
                    runs_root=args.runs_root,
                    model_gateway=model_gateway,
                    max_turns=args.max_turns,
                ).run(task_path)
            else:
                result = RunOrchestrator(
                    runs_root=args.runs_root,
                    max_turns=args.max_turns,
                ).run(task_path)
            _print_run_summary(result)
            return 0 if result.status.value == "completed" else 1
        finally:
            if generated_task_dir is not None:
                generated_task_dir.cleanup()

    if args.command == "chat":
        try:
            model_gateway = _build_run_model_gateway(args)
        except ProviderProfileError as error:
            print(f"error: {error}")
            return 1
        try:
            if args.resume is None:
                session = AgentSession(
                    workspace_root=args.workspace_root if args.workspace_root is not None else Path.cwd(),
                    runs_root=Path(".runs"),
                    model_gateway=model_gateway,
                    max_turns=CHAT_MAX_TURNS,
                )
            else:
                session = AgentSession.resume(
                    args.resume,
                    runs_root=Path(".runs"),
                    model_gateway=model_gateway,
                    max_turns=CHAT_MAX_TURNS,
                )
        except ChatSessionError as error:
            print(f"error: {error}")
            return 1
        if args.request is None:
            return _run_chat_repl(session)
        result = session.run_prompt_events(
            str(args.request),
            event_sink=_print_chat_event,
            include_session_events=True,
            interaction_handler=_read_chat_interaction,
        )
        _print_chat_turn_result(result)
        return 0 if result.status == "completed" else 1

    if args.command == "smoke":
        return _handle_smoke(args)

    if args.command == "dogfood":
        return _handle_dogfood(args)

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


def _run_task_path(
    args: argparse.Namespace,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.task_yaml is not None:
        return args.task_yaml, None

    for field_name, option_name in [
        ("goal", "--goal"),
        ("workspace_root", "--workspace-root"),
        ("verify", "--verify"),
    ]:
        if getattr(args, field_name) is None:
            raise ValueError(f"{option_name} is required when task_yaml is omitted")

    generated_task_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="haagent-task-",
    )
    task_path = Path(generated_task_dir.name) / "task.yaml"
    _write_authoring_task_yaml(
        task_path,
        goal=str(args.goal),
        workspace_root=args.workspace_root,
        verification_command=str(args.verify),
    )
    return task_path, generated_task_dir


def _write_authoring_task_yaml(
    path: Path,
    *,
    goal: str,
    workspace_root: Path,
    verification_command: str,
) -> None:
    task = {
        "goal": goal,
        "workspace_root": str(workspace_root.resolve()),
        "constraints": [],
        "allowed_tools": list(AUTHORING_ALLOWED_TOOLS),
        "acceptance_criteria": ["Complete the requested goal and pass verification."],
        "verification_commands": [verification_command],
        "policy": {
            "approval_allowed_tools": list(AUTHORING_APPROVED_TOOLS),
            "approved_tools": list(AUTHORING_APPROVED_TOOLS),
        },
    }
    path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")


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


def _handle_dogfood(args: argparse.Namespace) -> int:
    try:
        model_gateway = _build_dogfood_model_gateway(args)
    except ProviderProfileError as error:
        print(render_dogfood_report(skipped_dogfood_report(str(error))))
        return 0
    if model_gateway is None:
        print(render_dogfood_report(skipped_dogfood_report("provide --profile or --provider to run real dogfood")))
        return 0
    report = run_dogfood_tasks(
        model_gateway,
        runs_root=args.runs_root,
        max_turns=args.max_turns,
        auto_approve=not args.no_auto_approve,
    )
    print(render_dogfood_report(report))
    return 0 if report.status == "completed" else 1


def _build_dogfood_model_gateway(args: argparse.Namespace):
    if args.profile is not None:
        if args.provider is not None or args.model is not None or args.base_url is not None:
            raise ProviderProfileError(
                "--profile cannot be combined with --provider, --model, or --base-url",
            )
        return _gateway_from_profile(load_provider_profile(args.profile))
    if args.provider is None:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        raise ProviderProfileError("OPENAI_API_KEY is not set; dogfood skipped")
    return _build_run_model_gateway(args)


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


def _run_chat_repl(session: AgentSession) -> int:
    _print_chat_event(session.session_started_event())
    _print_session_status(session)
    while True:
        try:
            raw_prompt = input("haagent> ")
        except EOFError:
            _print_chat_event(session.session_finished_event())
            print("bye")
            return 0
        prompt = raw_prompt.strip()
        if not prompt:
            continue
        if prompt in {":quit", ":exit"}:
            _print_chat_event(session.session_finished_event())
            print("bye")
            return 0
        if prompt == ":status":
            _print_session_status(session)
            continue
        if prompt == ":new":
            session.new()
            print("session reset")
            continue

        result = session.run_prompt_events(
            prompt,
            event_sink=_print_chat_event,
            interaction_handler=_read_chat_interaction,
        )
        _print_chat_turn_result(result)


def _print_session_status(session: AgentSession) -> None:
    status = session.status()
    print(f"session_id={status['session_id']}")
    print(f"session_path={status['session_path']}")
    print(f"workspace_root={status['workspace_root']}")
    print(f"provider={status['provider']}")
    print(f"turn_count={status['turn_count']}")


def _print_chat_turn_result(result: ChatTurnResult) -> None:
    for line in result.output_lines():
        print(line)


def _print_chat_event(event: ChatEvent) -> None:
    pieces = [f"event={event.event_type}"]
    if event.event_type in {
        "tool_started",
        "tool_finished",
        "tool_failed",
        "approval_requested",
        "approval_granted",
        "approval_denied",
    }:
        tool_name = event.payload.get("tool_name")
        if tool_name is not None:
            pieces.append(f"tool={tool_name}")
    if event.event_type == "tool_started":
        pieces.append(f"args={_format_event_mapping(event.payload.get('args_summary'))}")
    elif event.event_type == "tool_finished":
        status = event.payload.get("status")
        if status is not None:
            pieces.append(f"status={status}")
        pieces.append(f"result={_format_event_mapping(event.payload.get('result_summary'))}")
    elif event.event_type == "tool_failed":
        error_type = event.payload.get("error_type")
        if error_type is not None:
            pieces.append(f"error={error_type}")
        message = event.payload.get("message")
        if message:
            pieces.append(f"message={_shell_token(str(message))}")
    elif event.event_type == "approval_requested":
        question = event.payload.get("question")
        if question:
            pieces.append(f"question={_shell_token(str(question))}")
        pieces.append(f"args={_format_event_mapping(event.payload.get('args_summary'))}")
    elif event.event_type in {"approval_granted", "approval_denied"}:
        approved = event.payload.get("approved")
        if approved is not None:
            pieces.append(f"approved={str(approved).lower()}")
    elif event.event_type == "user_input_requested":
        question = event.payload.get("question")
        if question:
            pieces.append(f"question={_shell_token(str(question))}")
    elif event.event_type == "user_input_received":
        answer_chars = event.payload.get("answer_chars")
        if answer_chars is not None:
            pieces.append(f"answer_chars={answer_chars}")
    elif event.event_type == "assistant_message":
        content = event.payload.get("content")
        if content:
            pieces.append(f"message={_shell_token(str(content))}")
    elif event.event_type == "failure":
        for key in ["failed_stage", "failure_category", "reason"]:
            value = event.payload.get(key)
            if value:
                pieces.append(f"{key}={_shell_token(str(value))}")
    elif event.event_type in {"turn_started", "turn_finished", "session_started", "session_finished"}:
        status = event.payload.get("status")
        if status is not None:
            pieces.append(f"status={status}")
    print(" ".join(pieces))


def _read_chat_interaction(request: HumanInteractionRequest) -> HumanInteractionResponse:
    try:
        if request.interaction_type == "approval":
            raw_answer = input("approve [y/N]> ")
            approved = raw_answer.strip().lower() in {"y", "yes"}
            return HumanInteractionResponse(approved=approved, answer=raw_answer)
        answer = input("answer> ")
        return HumanInteractionResponse(approved=True, answer=answer)
    except EOFError:
        return HumanInteractionResponse(approved=False, answer="")


def _format_event_mapping(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts = []
    for key in sorted(value):
        item = value[key]
        if item is None or item == "":
            continue
        parts.append(f"{key}={_shell_token(str(item))}")
    return ",".join(parts) if parts else "none"


def _shell_token(value: str) -> str:
    compact = _summary_value(" ".join(value.split()), 160)
    if not compact:
        return "none"
    if any(character.isspace() or character in {",", "="} for character in compact):
        return json.dumps(compact, ensure_ascii=False)
    return compact


def _run_final_response(transcript: list[dict[str, Any]]) -> str:
    response = _last_model_response(transcript)
    if response is None:
        return "none"
    return str(response.get("content", ""))


def _summary_provider(episode_metadata: dict[str, Any]) -> str:
    return str(episode_metadata.get("provider", "unknown"))


def _last_model_response(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return record
    return None


def _excerpt(content: str, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "... [truncated]"


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


if __name__ == "__main__":
    raise SystemExit(main())
