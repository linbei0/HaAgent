"""
haagent/cli_commands.py - CLI 子命令处理器

承载各子命令的编排逻辑，保持工具、模型和 episode 调用仍经过 runtime seam。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from haagent.cli_inspect import EpisodeInspectError, render_episode_summary
from haagent.cli_render import (
    print_check_summary,
    print_chat_event,
    print_chat_turn_result,
    print_eval_summary,
    print_run_summary,
    print_smoke_result,
    read_chat_interaction,
    run_chat_repl,
)
from haagent.cli_runtime import CliRuntime, SmokeDefinition
from haagent.models.provider_profile import (
    ProviderProfileError,
    ProviderProfileRecord,
    load_provider_profile,
    save_active_profile,
    save_provider_profile,
)
from haagent.runtime.chat_session import (
    CHAT_MAX_TURNS,
    ChatSessionError,
    find_latest_session,
    list_sessions,
)
from haagent.runtime.checks import run_quality_checks
from haagent.runtime.dogfood import render_dogfood_report, run_dogfood_tasks, skipped_dogfood_report
from haagent.runtime.episode_validator import EpisodeValidationError, load_inspect_episode_package
from haagent.runtime.eval_export import export_eval_case
from haagent.runtime.eval_runner import EvalRunnerError, run_eval_path


AUTHORING_ALLOWED_TOOLS = ["file_list", "file_read", "file_search", "apply_patch", "shell"]
AUTHORING_APPROVED_TOOLS = ["apply_patch", "shell"]


@dataclass(frozen=True)
class SmokeResult:
    name: str
    status: str
    episode_path: Path | None
    failed_stage: str | None = None
    failure_category: str | None = None
    reason: str | None = None


def handle_run(args, runtime: CliRuntime) -> int:
    generated_task_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        model_gateway = runtime.build_run_model_gateway(args)
        task_path, generated_task_dir = run_task_path(args)
    except ProviderProfileError as error:
        print(f"error: {error}")
        return 1
    except ValueError as error:
        print(f"error: {error}")
        return 2
    try:
        if model_gateway is not None:
            result = runtime.orchestrator_cls(
                runs_root=args.runs_root,
                model_gateway=model_gateway,
                max_turns=args.max_turns,
            ).run(task_path)
        else:
            result = runtime.orchestrator_cls(
                runs_root=args.runs_root,
                max_turns=args.max_turns,
            ).run(task_path)
        print_run_summary(result)
        return 0 if result.status.value == "completed" else 1
    finally:
        if generated_task_dir is not None:
            generated_task_dir.cleanup()


def handle_chat(args, runtime: CliRuntime) -> int:
    try:
        model_gateway = runtime.build_run_model_gateway(args)
    except ProviderProfileError as error:
        print(f"error: {error}")
        return 1
    try:
        runs_root = getattr(args, "runs_root", Path(".runs"))
        workspace_root = args.workspace_root if args.workspace_root is not None else Path.cwd()
        if args.resume is not None and getattr(args, "continue_session", False):
            print("error: --resume cannot be combined with --continue")
            return 1
        if getattr(args, "continue_session", False):
            latest = find_latest_session(runs_root, workspace_root)
            if latest is None:
                print("error: 当前目录没有可恢复会话")
                return 1
            session = runtime.session_cls.resume(
                latest.session_path,
                runs_root=runs_root,
                model_gateway=model_gateway,
                max_turns=CHAT_MAX_TURNS,
            )
        elif args.resume is None:
            session = runtime.session_cls(
                workspace_root=workspace_root,
                runs_root=runs_root,
                model_gateway=model_gateway,
                max_turns=CHAT_MAX_TURNS,
            )
        else:
            session = runtime.session_cls.resume(
                args.resume,
                runs_root=runs_root,
                model_gateway=model_gateway,
                max_turns=CHAT_MAX_TURNS,
            )
    except ChatSessionError as error:
        print(f"error: {error}")
        return 1
    if args.request is None:
        return run_chat_repl(session)
    result = session.run_prompt_events(
        str(args.request),
        event_sink=print_chat_event,
        include_session_events=True,
        interaction_handler=read_chat_interaction,
    )
    print_chat_turn_result(result)
    return 0 if result.status == "completed" else 1


def handle_setup(args) -> int:
    try:
        record = ProviderProfileRecord(
            name=_prompt_value("profile name", default="local"),
            provider=_prompt_provider(),
            base_url=_prompt_value("base_url"),
            model=_prompt_value("model"),
            api_key_env=_prompt_value("api_key_env", default="OPENAI_API_KEY"),
        )
        providers_path = save_provider_profile(record)
        settings_path = save_active_profile(record.name)
    except (EOFError, ProviderProfileError) as error:
        print(f"error: {error}")
        return 1
    print(f"providers={providers_path}")
    print(f"settings={settings_path}")
    print(f"active_profile={record.name}")
    print(f"api_key_env={record.api_key_env}")
    print(f"请确认已在环境变量 {record.api_key_env} 中设置真实 API key。")
    return 0


def handle_sessions(args) -> int:
    workspace_root = args.workspace_root if args.workspace_root is not None else Path.cwd()
    try:
        summaries = list_sessions(args.runs_root, workspace_root)
    except ChatSessionError as error:
        print(f"error: {error}")
        return 1
    if not summaries:
        print("no sessions")
        return 0
    for summary in summaries:
        print(f"session_id={summary.session_id}")
        print(f"created_at={summary.created_at}")
        print(f"updated_at={summary.updated_at}")
        print(f"workspace={summary.workspace_root}")
        print(f"turn_count={summary.turn_count}")
        print(f"first_request={summary.first_request}")
        print(f"session_path={summary.session_path}")
    return 0


def handle_smoke(args, runtime: CliRuntime) -> int:
    selected = [
        definition
        for definition in runtime.smoke_definitions()
        if not definition.requires_profile or args.profile is not None
    ]
    exit_code = 0
    for definition in selected:
        result = run_smoke_definition(
            definition,
            args,
            runtime=runtime,
        )
        print_smoke_result(result)
        if result.status != "completed":
            exit_code = 1
    return exit_code


def handle_dogfood(args, runtime: CliRuntime) -> int:
    try:
        model_gateway = runtime.build_dogfood_model_gateway(args)
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


def handle_inspect(args) -> int:
    try:
        print(render_episode_summary(args.episode_path))
    except EpisodeInspectError as error:
        print(f"error: {error}")
        return 1
    return 0


def handle_eval(args, runtime: CliRuntime) -> int:
    try:
        model_gateway = runtime.build_run_model_gateway(args)
        report = run_eval_path(
            args.eval_path,
            runs_root=args.runs_root,
            model_gateway=model_gateway,
        )
    except (ProviderProfileError, EvalRunnerError) as error:
        print(f"error: {error}")
        return 1

    if args.output is not None:
        if not args.output.parent.exists():
            print(f"error: output parent directory does not exist: {args.output.parent}")
            return 1
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"eval_report={args.output}")
    print_eval_summary(report)
    return 0 if report["failed_count"] == 0 and report["error_count"] == 0 else 1


def handle_check(args, runtime: CliRuntime) -> int:
    try:
        model_gateway = runtime.build_run_model_gateway(args)
        report = run_quality_checks(
            eval_path=args.eval_path,
            runs_root=args.runs_root,
            model_gateway=model_gateway,
            run_pytest=bool(args.pytest),
            cwd=Path.cwd(),
        )
    except (ProviderProfileError, EvalRunnerError) as error:
        print(f"error: {error}")
        return 1

    if args.output is not None:
        if not args.output.parent.exists():
            print(f"error: output parent directory does not exist: {args.output.parent}")
            return 1
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"check_report={args.output}")
    print_check_summary(report)
    return 0 if report["status"] == "passed" else 1


def handle_export_eval(args) -> int:
    """处理 eval case 单文件和批量导出命令。"""
    episode_paths = args.episode_paths
    output_path = args.output
    output_dir = args.output_dir
    if len(episode_paths) == 1:
        return export_single_eval_case(episode_paths[0], output_path, output_dir)
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
            write_eval_case_file(episode_path, target_path)
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
    write_eval_dataset_manifest(output_dir, records)
    failure_count = sum(1 for record in records if record["status"] == "error")
    return 1 if failure_count else 0


def run_task_path(
    args,
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
    write_authoring_task_yaml(
        task_path,
        goal=str(args.goal),
        workspace_root=args.workspace_root,
        verification_command=str(args.verify),
    )
    return task_path, generated_task_dir


def write_authoring_task_yaml(
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


def run_smoke_definition(
    definition: SmokeDefinition,
    args,
    *,
    runtime: CliRuntime,
) -> SmokeResult:
    model_gateway = None
    if definition.requires_profile:
        try:
            model_gateway = runtime.gateway_from_profile(load_provider_profile(args.profile))
        except ProviderProfileError as error:
            return SmokeResult(
                name=definition.name,
                status="failed",
                episode_path=None,
                failed_stage="configuration",
                failure_category="Provider Profile Error",
                reason=str(error),
            )
    result = runtime.orchestrator_cls(
        runs_root=args.runs_root,
        model_gateway=model_gateway,
        max_turns=args.max_turns,
    ).run(definition.task_path)
    if result.status.value == "completed":
        return SmokeResult(definition.name, result.status.value, result.episode_path)
    stage, category, reason = run_failure_summary(result.episode_path)
    return SmokeResult(
        name=definition.name,
        status=result.status.value,
        episode_path=result.episode_path,
        failed_stage=stage,
        failure_category=category,
        reason=reason,
    )


def run_failure_summary(episode_path: Path) -> tuple[str, str, str]:
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


def export_single_eval_case(
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


def _prompt_value(label: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    if not value and default is not None:
        value = default
    if not value:
        raise ProviderProfileError(f"{label} is required")
    return value


def _prompt_provider() -> str:
    provider = _prompt_value("provider (openai/openai-chat)", default="openai-chat")
    if provider not in {"openai", "openai-chat"}:
        raise ProviderProfileError(f"unsupported provider: {provider}")
    return provider


def write_eval_case_file(episode_path: Path, output_path: Path) -> None:
    eval_case = export_eval_case(episode_path)
    output = json.dumps(eval_case, ensure_ascii=False, indent=2)
    output_path.write_text(output + "\n", encoding="utf-8")


def write_eval_dataset_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
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
