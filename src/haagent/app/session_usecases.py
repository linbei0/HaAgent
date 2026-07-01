"""
haagent/app/session_usecases.py - 会话类应用用例

集中封装 AssistantService 的 session 创建、恢复、运行和权限相关委托逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from haagent.runtime.chat_session import ChatSessionError, find_latest_session, list_sessions

if TYPE_CHECKING:
    from haagent.app.assistant_service import (
        AssistantCancelResult,
        AssistantService,
        AssistantServiceError,
        AssistantSessionStatus,
        AssistantSessionSummary,
        AssistantSessionTurn,
        EventSink,
    )
    from haagent.runtime.human_interaction import HumanInteractionHandler
    from haagent.runtime.path_policy import PathAccess, PermissionMode


def create_session(service: "AssistantService") -> "AssistantSessionStatus":
    try:
        profile = service._load_session_profile()
        service._session = service.session_cls(
            workspace_root=service.workspace_root,
            runs_root=service.runs_root,
            model_gateway=service.gateway_factory(profile),
            model_profile_name=profile.name,
            model_name=profile.model,
            model_base_url=profile.base_url,
            max_turns=service.max_turns,
            enable_web=service.enable_web,
        )
    except Exception as error:
        _raise_service_error(service, error)
    return service._session_status(service._session)


def resume_session(service: "AssistantService", session: str | Path) -> "AssistantSessionStatus":
    try:
        profile = service._load_resume_profile(session)
        service._session = service.session_cls.resume(
            session,
            runs_root=service.runs_root,
            model_gateway=service.gateway_factory(profile),
            model_profile_name=profile.name,
            model_name=profile.model,
            model_base_url=profile.base_url,
            max_turns=service.max_turns,
            enable_web=service.enable_web,
        )
    except Exception as error:
        _raise_service_error(service, error)
    return service._session_status(service._session)


def continue_latest_session(service: "AssistantService") -> "AssistantSessionStatus":
    try:
        latest = find_latest_session(service.runs_root, service.workspace_root)
    except ChatSessionError as error:
        _raise_service_error(service, error)
    if latest is None:
        raise service.error_cls("当前 workspace 没有可恢复会话")
    return resume_session(service, latest.session_path)


def list_sessions_for_workspace(service: "AssistantService") -> list["AssistantSessionSummary"]:
    try:
        return [
            service._session_summary(summary)
            for summary in list_sessions(service.runs_root, service.workspace_root)
        ]
    except ChatSessionError as error:
        _raise_service_error(service, error)


def current_session_history(service: "AssistantService") -> list["AssistantSessionTurn"]:
    if service._session is None:
        return []
    try:
        return [service._session_turn(turn) for turn in service._session.turn_summaries()]
    except ChatSessionError as error:
        _raise_service_error(service, error)


def run_prompt_events(
    service: "AssistantService",
    prompt: str,
    *,
    event_sink: "EventSink | None"=None,
    include_session_events: bool=True,
    interaction_handler: "HumanInteractionHandler | None"=None,
):
    session = _ensure_session(service)
    return session.run_prompt_events(
        prompt,
        event_sink=event_sink,
        include_session_events=include_session_events,
        interaction_handler=interaction_handler,
    )


def cancel_current_run(service: "AssistantService") -> "AssistantCancelResult":
    if service._session is None:
        return service.cancel_result_cls(status="idle", reason="no_active_session")
    service._session.cancel_current_run()
    return service.cancel_result_cls(status="cancelled", reason="user_cancelled")


def set_permission_mode(service: "AssistantService", mode: "PermissionMode") -> "AssistantSessionStatus":
    if mode not in {"request_approval", "auto_approve", "full_access"}:
        raise service.error_cls("permission mode must be request_approval, auto_approve, or full_access")
    session = _ensure_session(service)
    session.set_permission_mode(mode)
    return service._session_status(session)


def set_next_turn_target_paths(service: "AssistantService", paths: list[str | Path]) -> "AssistantSessionStatus":
    session = _ensure_session(service)
    session.set_next_turn_target_paths([Path(path) for path in paths])
    return service._session_status(session)


def add_external_root(
    service: "AssistantService",
    path: str | Path,
    access: "PathAccess",
) -> "AssistantSessionStatus":
    if access not in {"read", "full"}:
        raise service.error_cls("external root access must be read or full")
    session = _ensure_session(service)
    root = Path(path).resolve()
    if not root.exists():
        raise service.error_cls(f"外部目录不存在：{root}")
    if not root.is_dir():
        raise service.error_cls(f"外部路径必须是目录：{root}")
    session.add_external_root(root, access)
    return service._session_status(session)


def remove_external_root(service: "AssistantService", path: str | Path) -> "AssistantSessionStatus":
    session = _ensure_session(service)
    session.remove_external_root(Path(path))
    return service._session_status(session)


def set_external_root_access(
    service: "AssistantService",
    path: str | Path,
    access: "PathAccess",
) -> "AssistantSessionStatus":
    if access not in {"read", "full"}:
        raise service.error_cls("external root access must be read or full")
    session = _ensure_session(service)
    session.set_external_root_access(Path(path), access)
    return service._session_status(session)


def clear_external_roots(service: "AssistantService") -> "AssistantSessionStatus":
    session = _ensure_session(service)
    session.clear_external_roots()
    return service._session_status(session)


def switch_project_root(service: "AssistantService", path: str | Path) -> "AssistantSessionStatus":
    root = Path(path).resolve()
    if not root.exists():
        raise service.error_cls(f"项目目录不存在：{root}")
    if not root.is_dir():
        raise service.error_cls(f"项目路径必须是目录：{root}")
    session = _ensure_session(service)
    service.workspace_root = root
    session.switch_project_root(root)
    return service._session_status(session)


def _ensure_session(service: "AssistantService"):
    if service._session is None:
        create_session(service)
    assert service._session is not None
    return service._session


def _raise_service_error(service: "AssistantService", error: Exception) -> None:
    raise service.error_cls(str(error)) from error
