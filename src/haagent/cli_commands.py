"""
haagent/cli_commands.py - CLI 子命令处理器

承载各子命令的编排逻辑，保持工具、模型和 episode 调用仍经过 runtime seam。
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from haagent.cli_inspect import EpisodeInspectError, render_episode_summary
from haagent.cli_render import (
    print_check_summary,
    print_eval_summary,
    print_run_summary,
    print_smoke_result,
    render_sandbox_doctor,
    render_sandbox_status,
)
from haagent.app.assistant_service import AssistantService
from haagent.channels.process_lock import GatewayInstanceLock
from haagent.cli_runtime import CliRuntime, SmokeDefinition
from haagent.models.config.connections import ProviderProfileError, user_config_dir
from haagent.models.config.credentials import KEYRING_SERVICE_NAME, KeyringCredentialStore
from haagent.runtime.evaluation.checks import run_quality_checks
from haagent.runtime.evaluation.dogfood import render_dogfood_report, run_dogfood_tasks, skipped_dogfood_report
from haagent.runtime.episodes.validator import EpisodeValidationError, load_inspect_episode_package
from haagent.runtime.evaluation.export import export_eval_case
from haagent.runtime.evaluation.runner import EvalRunnerError, run_eval_path
from haagent.runtime.settings import load_runtime_settings
from haagent.runtime.sandbox.status import (
    disable_sandbox,
    enable_docker_sandbox,
    sandbox_doctor_report,
    sandbox_user_status,
)
from haagent.tui.application.app import run_tui


AUTHORING_ALLOWED_TOOLS = ["file_list", "grep", "file_read", "apply_patch", "shell"]
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


def handle_tui_entry(args) -> int:
    if args.resume is not None and args.continue_session:
        print("error: --resume cannot be combined with --continue")
        return 1
    workspace_root = args.workspace_root if args.workspace_root is not None else Path.cwd()
    service = AssistantService(
        workspace_root=workspace_root,
        runs_root=args.runs_root,
        max_turns=load_runtime_settings().interactive_max_turns,
        enable_web=args.enable_web,
        initial_resume=args.resume,
        initial_continue=args.continue_session,
    )
    return run_tui(service)


def handle_tui_migration(_args) -> int:
    print("此交互入口已迁移到 TUI；请运行 haagent 打开 TUI 后完成该操作。")
    return 1


def handle_schedule_worker(args) -> int:
    """高级内部入口：前台/一次性计划 worker；非普通用户主路径。"""
    from haagent.scheduling.worker import run_schedule_worker

    once = bool(getattr(args, "once", False))
    db = getattr(args, "db", None)
    return run_schedule_worker(
        db_path=Path(db) if db is not None else None,
        once=once,
        install_signals=not once,
    )


def handle_gateway(args) -> int:
    """高级渠道网关：status 只读配置；run 前台挂载 Adapter；pair 重发配对码。"""
    action = args.gateway_action
    if action == "status":
        return _gateway_status()
    if action == "run":
        workspace = args.workspace_root if args.workspace_root is not None else Path.cwd()
        return _gateway_run(Path(workspace).resolve())
    if action == "pair":
        return _gateway_pair(args.instance_id)
    print("error: unknown gateway action")
    return 2


def _gateway_model_preflight(workspace_root: Path) -> tuple[bool, str]:
    """
    启动前检查模型 profile/凭据是否可用。

    返回 (ok, message)；失败时 message 供 CLI 短输出，不含 secret。
    """
    try:
        service = AssistantService(workspace_root=workspace_root)
        status = service.workspace.status()
    except Exception as error:
        return False, f"model preflight failed: {type(error).__name__}"
    if status.profile_error:
        return False, f"model profile error: {status.profile_error}"
    if not status.api_key_available:
        return False, "model credential unavailable; configure via TUI first"
    if not status.model and not status.profile_name:
        return False, "no active model profile; configure via TUI first"
    return True, "ok"


def _gateway_status() -> int:
    from haagent.channels.settings import load_channel_settings
    from haagent.channels.state import ChannelStateStore

    config_path = user_config_dir() / "channels.json"
    state_path = user_config_dir() / "channels.sqlite3"
    settings = load_channel_settings(config_path)
    print(f"instances={len(settings.instances)}")
    if not settings.instances:
        print("no channel instances configured; use TUI /channels")
        return 0
    store = KeyringCredentialStore()
    state = ChannelStateStore(state_path)
    try:
        for item in settings.instances:
            try:
                token = store.get_password(KEYRING_SERVICE_NAME, item.credential_username)
                cred = "ok" if token else "missing"
            except Exception:
                cred = "error"
            enabled = "on" if item.enabled else "off"
            # 脱敏摘要：pairing 状态与 cursor 有无，不含明文码/cursor 值。
            summary = state.instance_status_summary(item.id)
            pairing = summary.get("pairing") or "none"
            pairing_detail = state.get_pairing_status(item.id)
            expires = pairing_detail.get("expires_at") or ""
            expires_label = f" expires={expires}" if pairing == "pending" and expires else ""
            print(
                f"{item.id} platform={item.platform} enabled={enabled} "
                f"credential={cred} owner={summary.get('owner', '(unpaired)')} "
                f"pairing={pairing}{expires_label} cursor={summary.get('cursor', 'empty')} "
                f"workspace={item.workspace_root}"
            )
    finally:
        state.close()
    return 0


def _gateway_pair(instance_id: str | None) -> int:
    """重新签发一次性配对码（仅打印一次，不落明文）。"""
    config_dir = user_config_dir()
    target = (instance_id or "").strip()
    if not target:
        from haagent.channels.settings import load_channel_settings

        settings = load_channel_settings(config_dir / "channels.json")
        enabled = [i for i in settings.instances if i.enabled]
        if len(enabled) == 1:
            target = enabled[0].id
        elif not enabled:
            print("error: no channel instances; configure via TUI /channels")
            return 1
        else:
            print("error: multiple instances; pass --instance-id")
            return 2
    channels = AssistantService(workspace_root=Path.cwd()).channels
    try:
        code = channels.issue_pairing_code(target)
    except Exception as error:
        print(f"error: {error}")
        return 1
    print(f"instance={target}")
    print(f"pairing_code={code}")
    print("send in chat: /pair " + code)
    print("(code shown once; expires in 10 minutes)")
    return 0


def _gateway_run(workspace_root: Path) -> int:
    from haagent.channels.settings import load_channel_settings

    config_dir = user_config_dir()
    config_path = config_dir / "channels.json"
    state_path = config_dir / "channels.sqlite3"
    settings = load_channel_settings(config_path)
    enabled = [item for item in settings.instances if item.enabled]
    if not enabled:
        print("error: no enabled channel instances; configure via TUI /channels")
        return 1

    lock = GatewayInstanceLock(config_dir / "gateway.lock")
    if not lock.acquire():
        print("error: channel gateway is already running")
        return 1

    try:
        return _gateway_run_with_lock(
            workspace_root=workspace_root,
            config_path=config_path,
            state_path=state_path,
        )
    finally:
        # 文件锁必须覆盖 preflight、Adapter 构建和完整异步生命周期。
        lock.release()


def _gateway_run_with_lock(
    *, workspace_root: Path, config_path: Path, state_path: Path
) -> int:
    import asyncio

    from haagent.channels.runtime import ChannelGatewayRuntime

    # 有启用渠道后再做模型 preflight，避免空配置时误导成模型错误。
    ok, preflight_msg = _gateway_model_preflight(workspace_root)
    if not ok:
        print(f"error: {preflight_msg}")
        return 1

    def service_factory(root: Path) -> AssistantService:
        return AssistantService(workspace_root=root)

    # 组合根负责装载 settings/state/adapters，CLI 只做前台生命周期。
    runtime = ChannelGatewayRuntime(
        config_path=config_path,
        state_path=state_path,
        default_workspace_root=workspace_root,
        service_factory=service_factory,
        credential_store=KeyringCredentialStore(),
    )
    try:
        adapters = runtime.build_adapters()
    except Exception as error:
        print(f"error: {error}")
        return 1
    if not adapters:
        print("error: no adapters built; check platform support and credentials")
        return 1

    async def _main() -> int:
        assert runtime.manager is not None
        try:
            for adapter in adapters:
                await runtime.manager.attach_adapter(adapter)
                print(f"started instance={adapter.instance_id} platform={adapter.platform}")
        except BaseException:
            # 部分启动失败必须停止已 attach 的 Adapter 并关闭 SQLite，再保留原异常。
            await runtime.stop()
            raise
        print(f"gateway running instances={len(adapters)}; Ctrl+C to stop")
        # 周期性同步 adapter 状态；auth_expired 时打印提示。
        return await _run_gateway_until_cancelled(runtime)

    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        try:
            stop_errors = asyncio.run(runtime.stop())
            for item in stop_errors or []:
                print(f"warning: gateway stop: {item}")
        except Exception as error:
            print(f"warning: gateway stop failed: {error}")
        print("gateway stopped")
        return 0


async def _run_gateway_until_cancelled(runtime: Any, stop_event: Any = None) -> int:
    import asyncio
    import signal

    loop = asyncio.get_running_loop()
    stop = stop_event or asyncio.Event()
    warned_auth: set[str] = set()
    warned_health: set[tuple[str, str, str]] = set()

    def _request_stop() -> None:
        stop.set()

    async def _watch_adapter_health() -> None:
        # 轮询 adapter 状态，auth_expired 时明确提示重新登录。
        while not stop.is_set():
            manager = runtime.manager
            if manager is not None:
                for item in manager.status():
                    instance_id = str(item["instance_id"])
                    state = str(item.get("state") or "")
                    if state == "auth_expired" and instance_id not in warned_auth:
                        warned_auth.add(instance_id)
                        print(
                            f"error: instance={instance_id} auth_expired; "
                            "re-login via TUI /channels then restart gateway"
                        )
                    elif state in {"reconnecting", "failed"}:
                        from haagent.runtime.execution.command import redact_secret_like_text

                        raw_error = str(item.get("last_error") or "").strip()
                        # reconnecting + 无 last_error 是启动/会话建立中的正常态，不告警。
                        if state == "reconnecting" and not raw_error:
                            continue
                        error_text, _ = redact_secret_like_text(raw_error or "unknown error")
                        error_text = error_text[:200]
                        warning_key = (instance_id, state, error_text)
                        if warning_key not in warned_health:
                            warned_health.add(warning_key)
                            print(f"warning: instance={instance_id} state={state} error={error_text}")
                    elif state == "connected":
                        # 恢复后允许相同错误再次出现时重新提示。
                        warned_health.difference_update(key for key in warned_health if key[0] == instance_id)
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

    watch_task = asyncio.create_task(_watch_adapter_health())
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError, ValueError):
                # Windows 上 signal handler 受限；依赖 KeyboardInterrupt。
                pass
        await stop.wait()
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
        stop_errors = await runtime.stop()
        for item in stop_errors or []:
            # 关闭失败可见，避免静默吞掉 adapter 异常。
            print(f"warning: gateway stop: {item}")
        print("gateway stopped")
    return 0


def handle_sandbox(args) -> int:
    action = getattr(args, "sandbox_action", "status")
    if action == "status":
        print(render_sandbox_status(sandbox_user_status()))
        return 0
    if action == "doctor":
        print(render_sandbox_doctor(sandbox_doctor_report(check_disabled=True)))
        return 0
    if action == "enable":
        backend = getattr(args, "backend", "")
        if backend != "docker":
            print("error: only docker sandbox can be enabled")
            return 2
        status = enable_docker_sandbox(
            fail_if_unavailable=bool(getattr(args, "fail_if_unavailable", True)),
        )
        print(render_sandbox_status(status))
        print("note=existing sessions keep their current backend; start a new session for this setting.")
        return 0
    if action == "disable":
        print(render_sandbox_status(disable_sandbox()))
        return 0
    print("error: unknown sandbox action")
    return 2


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
        _safe_print(render_episode_summary(args.episode_path))
    except EpisodeInspectError as error:
        print(f"error: {error}")
        return 1
    return 0


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text)


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
            model_gateway = runtime.build_run_model_gateway(args)
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
