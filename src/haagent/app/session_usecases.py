"""
haagent/app/session_usecases.py - 会话与权限应用 Module

集中管理 session 生命周期、事件流、附件复用和 path policy 变更。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantCancelResult,
    AssistantServiceError,
    AssistantSessionCompactResult,
    AssistantSessionStatus,
    AssistantSessionSummary,
    AssistantSessionTurn,
    EventSink,
)
from haagent.models.model_ref import ModelRef
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.execution.path_policy import PathAccess, PermissionMode
from haagent.runtime.execution.retry import RetryController
from haagent.runtime.settings import load_runtime_settings
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.attachments import ImageAttachment
from haagent.runtime.session.package import (
    ChatSessionError,
    SessionSummary,
    SessionTurnSummary,
    find_latest_session,
    list_sessions,
)


class AssistantPermissions:
    def __init__(self, context: AssistantContext, sessions: AssistantSessions) -> None:
        self._context = context
        self._sessions = sessions

    def set_mode(self, mode: PermissionMode) -> AssistantSessionStatus:
        if mode not in {"request_approval", "auto_approve", "full_access"}:
            raise AssistantServiceError("permission mode must be request_approval, auto_approve, or full_access")
        session = self._sessions._ensure_session()
        session.set_permission_mode(mode)
        return session_status(session)

    def set_next_turn_targets(self, paths: list[str | Path]) -> AssistantSessionStatus:
        session = self._sessions._ensure_session()
        session.set_next_turn_target_paths([Path(path) for path in paths])
        return session_status(session)

    def add_external_root(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        if access not in {"read", "full"}:
            raise AssistantServiceError("external root access must be read or full")
        root = Path(path).resolve()
        if not root.exists():
            raise AssistantServiceError(f"外部目录不存在：{root}")
        if not root.is_dir():
            raise AssistantServiceError(f"外部路径必须是目录：{root}")
        session = self._sessions._ensure_session()
        session.add_external_root(root, access)
        return session_status(session)

    def remove_external_root(self, path: str | Path) -> AssistantSessionStatus:
        session = self._sessions._ensure_session()
        session.remove_external_root(Path(path))
        return session_status(session)

    def set_external_root_access(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        if access not in {"read", "full"}:
            raise AssistantServiceError("external root access must be read or full")
        session = self._sessions._ensure_session()
        session.set_external_root_access(Path(path), access)
        return session_status(session)

    def clear_external_roots(self) -> AssistantSessionStatus:
        session = self._sessions._ensure_session()
        session.clear_external_roots()
        return session_status(session)

    def switch_project_root(self, path: str | Path) -> AssistantSessionStatus:
        root = Path(path).resolve()
        if not root.exists():
            raise AssistantServiceError(f"项目目录不存在：{root}")
        if not root.is_dir():
            raise AssistantServiceError(f"项目路径必须是目录：{root}")
        session = self._sessions._ensure_session()
        self._context.workspace_root = root
        session.switch_project_root(root)
        return session_status(session)


class AssistantSessions:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context
        self.permissions = AssistantPermissions(context, self)

    @property
    def initial_resume(self) -> str | Path | None:
        return self._context.initial_resume

    @property
    def initial_continue(self) -> bool:
        return self._context.initial_continue

    def create(self) -> AssistantSessionStatus:
        try:
            existing = self._context.session
            # 已有 session 时复用 MCP/gateway，避免 /new 每次重建 runtime。
            if existing is not None:
                if self._context.pending_model_selection is not None:
                    ref = self._load_session_ref()
                    existing.switch_model_gateway(ref, self._gateway_for_ref(ref))
                    self._context.pending_model_selection = None
                existing.new()
            else:
                ref = self._load_session_ref()
                self._context.session = self._context.session_factory(
                    workspace_root=self._context.workspace_root,
                    runs_root=self._context.runs_root,
                    model_gateway=self._gateway_for_ref(ref),
                    model_ref=ref,
                    max_turns=self._context.max_turns,
                    enable_web=self._context.enable_web,
                    skill_catalog=self._context.skill_catalog,
                    instruction_cache=self._context.instruction_cache,
                    tool_schema_cache=self._context.tool_schema_cache,
                )
        except Exception as error:
            raise AssistantServiceError(str(error)) from error
        assert self._context.session is not None
        self._context.status_generation += 1
        return session_status(self._context.session)

    def resume(self, session: str | Path) -> AssistantSessionStatus:
        try:
            existing = self._context.session
            ref = self._load_resume_ref(session)
            # 已有 live session 时就地 reload package，复用 MCP（避免 5–10s 进程级重建）。
            if existing is not None:
                gateway = self._gateway_for_resume(existing, ref)
                existing.reload(
                    session,
                    runs_root=self._context.runs_root,
                    model_gateway=gateway,
                    model_ref=ref,
                    max_turns=self._context.max_turns,
                    enable_web=self._context.enable_web,
                )
            else:
                self._context.session = self._context.session_factory.resume(
                    session,
                    runs_root=self._context.runs_root,
                    model_gateway=self._gateway_for_ref(ref),
                    model_ref=ref,
                    max_turns=self._context.max_turns,
                    enable_web=self._context.enable_web,
                    skill_catalog=self._context.skill_catalog,
                    instruction_cache=self._context.instruction_cache,
                    tool_schema_cache=self._context.tool_schema_cache,
                )
        except Exception as error:
            raise AssistantServiceError(str(error)) from error
        assert self._context.session is not None
        self._context.status_generation += 1
        return session_status(self._context.session)

    def continue_latest(self) -> AssistantSessionStatus:
        try:
            latest = find_latest_session(self._context.runs_root, self._context.workspace_root)
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error
        if latest is None:
            raise AssistantServiceError("当前 workspace 没有可恢复会话")
        return self.resume(latest.session_path)

    def list(self) -> list[AssistantSessionSummary]:
        try:
            return [
                session_summary(summary)
                for summary in list_sessions(self._context.runs_root, self._context.workspace_root)
            ]
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error

    def history(self) -> list[AssistantSessionTurn]:
        if self._context.session is None:
            return []
        try:
            return [session_turn(turn) for turn in self._context.session.turn_summaries()]
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error

    def compact(self) -> AssistantSessionCompactResult:
        session = self._ensure_session()
        try:
            result = session.compact_current_session()
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error
        return AssistantSessionCompactResult(
            applied=result.applied,
            reason=result.reason,
            original_turn_count=result.original_turn_count,
            compacted_turn_count=result.compacted_turn_count,
            preserved_recent_count=result.preserved_recent_count,
            saved_chars=result.saved_chars,
        )

    def run_prompt_events(
        self,
        prompt: str,
        *,
        event_sink: EventSink | None = None,
        include_session_events: bool = True,
        interaction_handler: HumanInteractionHandler | None = None,
        attachments: list[ImageAttachment] | None = None,
    ):
        return self._ensure_session().run_prompt_events(
            prompt,
            event_sink=event_sink,
            include_session_events=include_session_events,
            interaction_handler=interaction_handler,
            attachments=attachments,
        )

    def paste_clipboard_image(self, *, existing: list[ImageAttachment] | None = None) -> ImageAttachment:
        try:
            return self._ensure_session().paste_clipboard_image(existing=existing)
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error

    def cancel_current_run(self) -> AssistantCancelResult:
        if self._context.session is None:
            return AssistantCancelResult(status="idle", reason="no_active_session")
        if not self._context.session.cancel_current_run():
            return AssistantCancelResult(status="idle", reason="no_active_run")
        return AssistantCancelResult(status="cancelled", reason="user_cancelled")

    def _ensure_session(self) -> AgentSession:
        if self._context.session is None:
            self.create()
        assert self._context.session is not None
        return self._context.session

    def _gateway_for_ref(self, ref: ModelRef):
        """每个 session 独占 retry controller；所有模型解析统一经过 ModelRuntime。"""
        assert self._context.model_runtime is not None
        controller = RetryController(load_runtime_settings().model_retry)
        return self._context.model_runtime.create_route_gateway(ref, retry_controller=controller)

    def _gateway_for_resume(
        self,
        existing: AgentSession,
        ref: ModelRef,
    ):
        """模型/variant 未变则复用 gateway；变更时才重建。"""
        if existing.model_ref == ref and existing.model_gateway is not None:
            return existing.model_gateway
        return self._gateway_for_ref(ref)

    def _load_session_ref(self) -> ModelRef:
        assert self._context.model_runtime is not None
        ref = self._context.pending_model_selection or self._context.model_runtime.load_active()
        self._context.model_runtime.resolve(ref)
        return ref

    def _load_resume_ref(self, session: str | Path) -> ModelRef:
        assert self._context.model_runtime is not None
        ref = _session_model_selection(session, self._context.runs_root)
        self._context.model_runtime.resolve(ref)
        return ref


def session_status(session: AgentSession) -> AssistantSessionStatus:
    from haagent.app.workspace_usecases import sandbox_status

    metadata_method = getattr(session.model_gateway, "metadata", None)
    metadata = metadata_method() if callable(metadata_method) else None
    ref = session.model_ref

    return AssistantSessionStatus(
        session_id=session.session_id,
        workspace_root=session.workspace_root,
        runs_root=session.runs_root,
        session_path=session.session_path.resolve(),
        turn_count=session.turn_count,
        max_turns=getattr(session, "max_turns", None),
        provider=session.provider_name,
        model_profile_name=(f"{ref.connection_id}:{ref.model}" if ref else None),
        model_connection_id=ref.connection_id if ref else None,
        model=ref.model if ref else None,
        model_variant=ref.variant if ref else None,
        base_url=metadata.base_url if metadata else None,
        web_enabled=getattr(session, "enable_web", False),
        external_roots=_external_root_summaries(session),
        permission_mode=_session_permission_mode(session),
        sandbox_status=sandbox_status(),
    )


def session_summary(summary: SessionSummary) -> AssistantSessionSummary:
    return AssistantSessionSummary(
        session_id=summary.session_id,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        workspace_root=summary.workspace_root,
        turn_count=summary.turn_count,
        first_request=summary.first_request,
        session_path=summary.session_path,
    )


def session_turn(turn: SessionTurnSummary) -> AssistantSessionTurn:
    return AssistantSessionTurn(
        turn_index=turn.turn_index,
        request=turn.request,
        summary=turn.summary,
        status=turn.status,
        episode_path=turn.episode_path,
        verification_status=turn.verification_status,
        assistant_display_text=turn.assistant_display_text,
    )


def _external_root_summaries(session: AgentSession) -> list[dict[str, str]]:
    policy = getattr(session, "path_policy", None)
    if policy is None:
        return []
    return [
        {"path": str(root.path.resolve()), "access": root.access, "source": root.source}
        for root in policy.external_roots
    ]


def _session_permission_mode(session: AgentSession) -> PermissionMode:
    policy = getattr(session, "path_policy", None)
    mode = getattr(policy, "permission_mode", "request_approval")
    if mode in {"request_approval", "auto_approve", "full_access"}:
        return mode
    return "request_approval"


def _session_model_selection(session: str | Path, runs_root: Path) -> ModelRef:
    raw = Path(session)
    session_path = raw.resolve() if raw.is_absolute() or raw.exists() or raw.name != str(session) else (runs_root / "sessions" / str(session)).resolve()
    metadata_path = session_path / "session.json"
    if not metadata_path.exists():
        raise ChatSessionError(f"session metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ChatSessionError(f"session metadata is invalid JSON: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ChatSessionError(f"session metadata must be an object: {metadata_path}")
    value = metadata.get("model_ref")
    if not isinstance(value, dict):
        raise ChatSessionError(f"session metadata must contain model_ref: {metadata_path}")
    try:
        return ModelRef.from_dict(value)
    except ValueError as error:
        raise ChatSessionError(str(error)) from error
