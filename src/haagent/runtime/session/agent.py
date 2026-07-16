"""
src/haagent/runtime/session/agent.py - 自然语言 Agent 会话

管理 chat 会话状态，并把每条用户请求转成可审计的临时 task contract。
会话 package IO、turn 收尾、路径策略变更与生命周期装配已拆到同级模块。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from haagent.models.types import ModelGateway
from haagent.models.model_ref import ModelRef
from haagent.models.config.connections import user_config_dir
from haagent.memory.extraction import MemoryExtractionRequest, MemoryExtractor
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.session.attachments import (
    AttachmentError,
    ImageAttachment,
    read_clipboard_image_bytes,
    save_clipboard_image,
)
from haagent.runtime.session.lifecycle import (
    SessionRuntimeState,
    apply_state,
    build_create_state,
    build_new_package_state,
    build_resume_state,
)
from haagent.runtime.session.package import (
    ChatSessionError,
    SessionTurnSummary,
    append_turn_record,
    manual_compaction_summary_text,
    merge_image_attachment_history,
    session_turn_summary,
    write_manual_compaction_state,
    write_session_metadata,
)
from haagent.runtime.session.path_mutators import (
    with_external_root_access,
    with_external_root_added,
    with_external_root_removed,
    with_external_roots_cleared,
    with_permission_mode,
    with_project_root,
)
from haagent.runtime.session.turn import ChatTurnRequest, ChatTurnRunner, summary_value as _summary_value
from haagent.runtime.session.turn_completion import (
    ChatTurnResult,
    build_turn_result,
    count_historical_tool_compression_events,
    memory_update_requested,
    task_step_started_event,
    task_turn_closed_events,
    turn_summary,
    with_in_band_verification,
)
from haagent.runtime.session.ui_events import (
    RuntimeUiEventSink,
    emit_runtime_ui_event,
    emit_ui_event,
    failure_notice_event,
    memory_candidates_created_event,
    memory_extraction_warning_event,
    session_finished_event as build_session_finished_event,
    session_started_event as build_session_started_event,
    turn_finished_event as build_turn_finished_event,
    turn_started_event as build_turn_started_event,
)
from haagent.context.compression.full import FullCompactEligibility, maybe_full_compact_messages
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.execution.path_policy import (
    PathAccess,
    PermissionMode,
    serialize_path_policy,
)
from haagent.context.compression.session_memory import (
    DEFAULT_PRESERVED_RECENT_TURNS,
    SESSION_MEMORY_CHAR_LIMIT,
    compact_session_memory,
)
from haagent.runtime.session.working_state import (
    update_working_state,
    working_state_from_dict,
    write_working_state,
)
from haagent.runtime.session.task_ledger import (
    begin_task_ledger_turn,
    update_task_ledger,
    write_task_ledger,
)
from haagent.runtime.settings import DEFAULT_INTERACTIVE_MAX_TURNS

CHAT_MAX_TURNS = DEFAULT_INTERACTIVE_MAX_TURNS

# 对外 re-export 已移至 package / turn_completion；调用方应直接从目标模块 import。
__all__ = [
    "AgentSession",
    "CHAT_MAX_TURNS",
    "ChatSessionError",
    "ChatTurnResult",
    "SessionCompactResult",
    "SessionTurnSummary",
]


@dataclass(frozen=True)
class SessionCompactResult:
    applied: bool
    reason: str
    original_turn_count: int
    compacted_turn_count: int
    preserved_recent_count: int
    saved_chars: int


class AgentSession:
    def __init__(
        self,
        *,
        workspace_root: Path,
        runs_root: Path,
        model_gateway: ModelGateway | None = None,
        model_ref: ModelRef | None = None,
        max_turns: int | None = CHAT_MAX_TURNS,
        session_id: str | None = None,
        memory_extraction_enabled: bool = True,
        enable_web: bool = False,
        allowed_tools_override: list[str] | None = None,
        approval_allowed_tools_override: list[str] | None = None,
        approved_tools_override: list[str] | None = None,
        mcp_runtime: Any | None = None,
        worker_context: dict[str, object] | None = None,
        worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None,
        skill_catalog: Any | None = None,
        instruction_cache: Any | None = None,
        tool_schema_cache: Any | None = None,
    ) -> None:
        state = build_create_state(
            workspace_root=workspace_root,
            runs_root=runs_root,
            model_gateway=model_gateway,
            model_ref=model_ref,
            max_turns=max_turns,
            session_id=session_id,
            memory_extraction_enabled=memory_extraction_enabled,
            enable_web=enable_web,
            allowed_tools_override=allowed_tools_override,
            approval_allowed_tools_override=approval_allowed_tools_override,
            approved_tools_override=approved_tools_override,
            mcp_runtime=mcp_runtime,
            worker_context=worker_context,
            worker_permission_requester=worker_permission_requester,
        )
        apply_state(self, state)
        # cache services 不属于 session package，由 AssistantService 注入并跨 turn 复用。
        self._skill_catalog = skill_catalog
        self._instruction_cache = instruction_cache
        self._tool_schema_cache = tool_schema_cache
        self._write_session_metadata()
        self._write_working_state()
        self._write_task_ledger()

    @classmethod
    def resume(
        cls,
        session: str | Path,
        *,
        runs_root: Path | None = None,
        model_gateway: ModelGateway | None = None,
        model_ref: ModelRef | None = None,
        max_turns: int | None = CHAT_MAX_TURNS,
        enable_web: bool = False,
        mcp_runtime: Any | None = None,
        tool_registry: Any | None = None,
        mcp_settings: Any | None = None,
        mcp_tool_names: list[str] | None = None,
        owns_mcp_runtime: bool | None = None,
        skill_catalog: Any | None = None,
        instruction_cache: Any | None = None,
        tool_schema_cache: Any | None = None,
    ) -> "AgentSession":
        state = build_resume_state(
            session,
            runs_root=runs_root,
            model_gateway=model_gateway,
            model_ref=model_ref,
            max_turns=max_turns,
            enable_web=enable_web,
            mcp_runtime=mcp_runtime,
            tool_registry=tool_registry,
            mcp_settings=mcp_settings,
            mcp_tool_names=mcp_tool_names,
            owns_mcp_runtime=owns_mcp_runtime,
        )
        instance = cls.__new__(cls)
        apply_state(instance, state)
        instance._skill_catalog = skill_catalog
        instance._instruction_cache = instruction_cache
        instance._tool_schema_cache = tool_schema_cache
        return instance

    def reload(
        self,
        session: str | Path,
        *,
        runs_root: Path | None = None,
        model_gateway: ModelGateway | None = None,
        model_ref: ModelRef | None = None,
        max_turns: int | None = None,
        enable_web: bool | None = None,
    ) -> None:
        """把磁盘 session package 装入当前实例，复用 MCP/tool registry（可选换 gateway）。"""
        if self._current_cancellation_token is not None:
            raise ChatSessionError("current task is running")
        previous_gateway = self.model_gateway
        # 未显式传入的字段保持当前 live session 值，避免误清 max_turns/web。
        next_gateway = self.model_gateway if model_gateway is None else model_gateway
        state = build_resume_state(
            session,
            runs_root=self.runs_root if runs_root is None else runs_root,
            model_gateway=next_gateway,
            model_ref=self.model_ref if model_ref is None else model_ref,
            max_turns=self.max_turns if max_turns is None else max_turns,
            enable_web=self.enable_web if enable_web is None else enable_web,
            mcp_runtime=self._mcp_runtime,
            tool_registry=self._tool_registry,
            mcp_settings=self._mcp_settings,
            mcp_tool_names=list(self._mcp_tool_names),
            owns_mcp_runtime=self._owns_mcp_runtime,
        )
        apply_state(self, state)
        # profile 变更时关闭旧 route；复用同一 gateway 时不关闭。
        if previous_gateway is not None and previous_gateway is not next_gateway:
            from haagent.models.http_transport import close_model_gateway

            close_model_gateway(previous_gateway)

    def set_max_turns(self, max_turns: int | None) -> None:
        self.max_turns = max_turns

    @property
    def provider_name(self) -> str:
        if self.model_gateway is None:
            return "fake"
        return self.model_gateway.provider_name

    def run_prompt(
        self,
        prompt: str,
        interaction_handler: HumanInteractionHandler | None = None,
        attachments: list[ImageAttachment] | None = None,
    ) -> ChatTurnResult:
        return self.run_prompt_events(prompt, interaction_handler=interaction_handler, attachments=attachments)

    def run_prompt_events(
        self,
        prompt: str,
        event_sink: RuntimeUiEventSink = None,
        include_session_events: bool = False,
        interaction_handler: HumanInteractionHandler | None = None,
        attachments: list[ImageAttachment] | None = None,
    ) -> ChatTurnResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be non-empty")

        turn_index = self.turn_count + 1
        if include_session_events:
            emit_ui_event(
                event_sink,
                build_session_started_event(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    details=self.status(),
                ),
            )
        emit_ui_event(
            event_sink,
            build_turn_started_event(
                session_id=self.session_id,
                turn_index=turn_index,
                details={"prompt": _summary_value(clean_prompt, 160)},
            ),
        )
        from haagent.runtime.events.bus import RuntimeBusEvent, coerce_bus_event

        runtime_events: list[RuntimeBusEvent] = []
        self._current_cancellation_token = CancellationToken()

        def on_runtime_event(event: RuntimeBusEvent | dict[str, object]) -> None:
            bus_event = coerce_bus_event(event)
            runtime_events.append(bus_event)
            emit_runtime_ui_event(event_sink, bus_event, session_id=self.session_id, turn_index=turn_index)

        target_paths = list(self._next_turn_target_paths)
        self._next_turn_target_paths = []
        session_memory = self._session_memory()
        new_attachments = list(attachments or [])
        prompt_attachments = new_attachments if new_attachments else list(self._last_user_image_attachments)
        self._task_ledger = begin_task_ledger_turn(
            self._task_ledger,
            prompt=clean_prompt,
            turn_index=turn_index,
        )
        self._write_task_ledger()
        started_event = task_step_started_event(self._task_ledger)
        if started_event is not None:
            on_runtime_event(started_event)
        try:
            # prompt 接收点建立单一 trace，覆盖 submit_to_run_start。
            from haagent.runtime.performance import PerformanceTrace

            performance_trace = PerformanceTrace.start()
            result = ChatTurnRunner().run(
                ChatTurnRequest(
                    prompt=clean_prompt,
                    workspace_root=self.workspace_root,
                    runs_root=self.runs_root,
                    model_gateway=self.model_gateway,
                    max_turns=self.max_turns,
                    session_summary=session_memory.summary_text,
                    session_compaction=session_memory.diagnostics,
                    historical_tool_compression_count=self._historical_tool_compression_count,
                    working_state=self._working_state.to_dict() if not self._working_state.is_empty() else None,
                    task_ledger=self._task_ledger.to_dict(),
                    path_policy=self.path_policy,
                    enable_web=self.enable_web,
                    target_paths=target_paths,
                    event_sink=on_runtime_event,
                    interaction_handler=interaction_handler,
                    cancellation_token=self._current_cancellation_token,
                    orchestrator_factory=RunOrchestrator,
                    leader_session_id=self.session_id,
                    tool_registry=self._tool_registry,
                    mcp_runtime=self._mcp_runtime,
                    mcp_tool_names=self._mcp_tool_names,
                    allowed_tools_override=self._allowed_tools_override,
                    approval_allowed_tools_override=self._approval_allowed_tools_override,
                    approved_tools_override=self._approved_tools_override,
                    worker_context=self._worker_context,
                    worker_permission_requester=self._worker_permission_requester,
                    attachments=prompt_attachments,
                    image_attachment_history=self._image_attachment_history,
                    session_interaction_state=self._session_interaction_state,
                    performance_trace=performance_trace,
                    skill_catalog=self._skill_catalog,
                    instruction_cache=self._instruction_cache,
                    tool_schema_cache=self._tool_schema_cache,
                    working_state_sink=self._persist_progress_working_state,
                ),
            )
        except Exception:
            self._current_cancellation_token = None
            raise

        turn_result = self._build_turn_result(clean_prompt, result)
        turn_result = with_in_band_verification(turn_result, runtime_events)
        self.turn_count += 1
        # always 可能在本 turn 的 resolver 中被置位；落盘以便 resume 恢复
        self._write_session_metadata()
        if new_attachments:
            self._last_user_image_attachments = list(new_attachments)
            self._image_attachment_history = merge_image_attachment_history(
                self._image_attachment_history,
                new_attachments,
            )
        self._working_state = update_working_state(
            self._working_state,
            prompt=clean_prompt,
            result=turn_result,
            runtime_events=runtime_events,
        )
        self._write_working_state()
        self._task_ledger = update_task_ledger(
            self._task_ledger,
            prompt=clean_prompt,
            turn_index=turn_index,
            result_status=turn_result.status,
            episode_path=turn_result.episode_path,
            runtime_events=runtime_events,
        )
        self._write_task_ledger()
        for progress_event in task_turn_closed_events(self._task_ledger, turn_result):
            on_runtime_event(progress_event)
        self._historical_tool_compression_count += count_historical_tool_compression_events(runtime_events)
        summary = turn_summary(clean_prompt, turn_result)
        self._summaries.append(summary)
        self._record_turn(clean_prompt, turn_result, summary)
        extraction_result = None
        if self.memory_extraction_enabled and memory_update_requested(runtime_events):
            extraction_result = self._run_memory_extraction(clean_prompt, turn_result, runtime_events)
        if extraction_result is not None and extraction_result.created_count:
            turn_result = replace(
                turn_result,
                memory_candidates_created=extraction_result.created_count,
                memory_extraction_status=extraction_result.status,
                memory_extraction_reason=extraction_result.reason,
            )
            emit_ui_event(
                event_sink,
                memory_candidates_created_event(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    count=extraction_result.created_count,
                    message=f"发现 {extraction_result.created_count} 条可记忆候选，已放入候选队列，等待你确认。",
                ),
            )
        elif extraction_result is not None and extraction_result.status == "error":
            turn_result = replace(
                turn_result,
                memory_extraction_status=extraction_result.status,
                memory_extraction_reason=extraction_result.reason,
            )
            emit_ui_event(
                event_sink,
                memory_extraction_warning_event(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    status=extraction_result.status,
                    reason=extraction_result.reason,
                    message=f"Memory extraction failed: {extraction_result.reason}",
                ),
            )
        if turn_result.status != "completed":
            emit_ui_event(
                event_sink,
                failure_notice_event(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    status=turn_result.status,
                    failed_stage=turn_result.failed_stage,
                    failure_category=turn_result.failure_category,
                    reason=turn_result.reason,
                    episode_path=str(turn_result.episode_path),
                ),
            )
        emit_ui_event(
            event_sink,
            build_turn_finished_event(
                session_id=self.session_id,
                turn_index=turn_index,
                details={
                    "status": turn_result.status,
                    "episode_path": str(turn_result.episode_path),
                    "runtime_event_count": len(runtime_events),
                },
            ),
        )
        if include_session_events:
            emit_ui_event(
                event_sink,
                build_session_finished_event(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    details={"status": turn_result.status},
                ),
            )
        self._current_cancellation_token = None
        return turn_result

    def cancel_current_run(self) -> bool:
        if self._current_cancellation_token is not None:
            self._current_cancellation_token.cancel()
            return True
        return False

    def paste_clipboard_image(self, existing: list[ImageAttachment] | None = None) -> ImageAttachment:
        if self._current_cancellation_token is not None:
            raise ChatSessionError("current task is running")
        try:
            return save_clipboard_image(
                read_clipboard_image_bytes(),
                session_path=self.session_path,
                existing=list(existing or []),
            )
        except AttachmentError as error:
            # 会话层对外统一为 ChatSessionError，避免 UI 区分两种错误类型。
            raise ChatSessionError(str(error)) from error

    def switch_model_gateway(
        self,
        model_ref: ModelRef,
        gateway: ModelGateway,
    ) -> None:
        if self._current_cancellation_token is not None:
            raise ChatSessionError("current task is running")
        # 仅在成功安装新 gateway 后关闭旧 route，安装失败时保留旧连接可用。
        previous = self.model_gateway
        previous_selection = self.model_ref
        self.model_gateway = gateway
        self.model_ref = model_ref
        try:
            self._write_session_metadata()
        except Exception as error:
            self.model_gateway = previous
            self.model_ref = previous_selection
            from haagent.models.http_transport import close_model_gateway

            try:
                close_model_gateway(gateway)
            except Exception as close_error:
                error.add_note(f"failed to close rejected model gateway: {close_error}")
            raise
        if previous is not None and previous is not gateway:
            from haagent.models.http_transport import close_model_gateway

            close_model_gateway(previous)

    def add_external_root(self, path: Path, access: PathAccess) -> None:
        self.path_policy = with_external_root_added(self.path_policy, self.workspace_root, path, access)
        self._write_session_metadata()

    def remove_external_root(self, path: Path) -> None:
        self.path_policy = with_external_root_removed(self.path_policy, self.workspace_root, path)
        self._write_session_metadata()

    def set_external_root_access(self, path: Path, access: PathAccess) -> None:
        self.path_policy = with_external_root_access(self.path_policy, self.workspace_root, path, access)
        self._write_session_metadata()

    def clear_external_roots(self) -> None:
        self.path_policy = with_external_roots_cleared(self.path_policy, self.workspace_root)
        self._write_session_metadata()

    def switch_project_root(self, path: Path) -> None:
        self.workspace_root, self.path_policy = with_project_root(self.path_policy, path)
        self._write_session_metadata()

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.path_policy = with_permission_mode(self.path_policy, self.workspace_root, mode)
        self._write_session_metadata()

    def set_tool_overrides(
        self,
        *,
        allowed_tools: list[str],
        approval_allowed_tools: list[str],
        approved_tools: list[str],
    ) -> None:
        """应用前端无关的工具快照；调度恢复不得继承更宽的历史权限。"""
        self._allowed_tools_override = list(allowed_tools)
        self._approval_allowed_tools_override = list(approval_allowed_tools)
        self._approved_tools_override = list(approved_tools)

    def set_next_turn_target_paths(self, paths: list[Path]) -> None:
        self._next_turn_target_paths = [str(path.resolve()) for path in paths]

    def _run_memory_extraction(
        self,
        prompt: str,
        result: ChatTurnResult,
        runtime_events: list[object],
    ):
        from haagent.runtime.events.bus import bus_event_to_dict, coerce_bus_event

        # 记忆提取仍消费 dict 形态；总线事件在边界序列化，不改变提取 schema。
        dict_events = [bus_event_to_dict(coerce_bus_event(event)) for event in runtime_events]
        return MemoryExtractor().extract(
            MemoryExtractionRequest(
                session_id=self.session_id,
                session_path=self.session_path,
                workspace_root=self.workspace_root,
                turn_index=result.turn_index,
                user_prompt=prompt,
                final_response=result.final_response,
                status=result.status,
                verification_status=result.verification_status,
                episode_path=result.episode_path,
                working_state=self._working_state.to_dict(),
                runtime_events=dict_events,
                model_gateway=self.model_gateway,
            ),
        )

    def status(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_path": str(self.session_path.resolve()),
            "workspace_root": str(self.workspace_root),
            "path_policy": serialize_path_policy(self.path_policy),
            "provider": self.provider_name,
            "turn_count": self.turn_count,
            "working_state": self._working_state.status_summary(),
            "task_ledger": self._task_ledger.status_summary(),
        }

    def mcp_status(self) -> dict[str, object]:
        statuses = self._mcp_runtime.list_statuses()
        return {
            "configured_count": len(statuses),
            "connected_count": sum(1 for item in statuses if item.state == "connected"),
            "failed_count": sum(1 for item in statuses if item.state == "failed"),
            "servers": [
                {
                    "name": item.name,
                    "state": item.state,
                    "detail": item.detail,
                    "tool_count": len(item.tools),
                    "resource_count": len(item.resources),
                }
                for item in statuses
            ],
        }

    def new(self) -> None:
        state = build_new_package_state(self._snapshot_state())
        apply_state(self, state)
        self._write_session_metadata()
        self._write_working_state()
        self._write_task_ledger()

    def _snapshot_state(self) -> SessionRuntimeState:
        return SessionRuntimeState(
            workspace_root=self.workspace_root,
            path_policy=self.path_policy,
            runs_root=self.runs_root,
            model_gateway=self.model_gateway,
            model_ref=self.model_ref,
            max_turns=self.max_turns,
            memory_extraction_enabled=self.memory_extraction_enabled,
            enable_web=self.enable_web,
            allowed_tools_override=self._allowed_tools_override,
            approval_allowed_tools_override=self._approval_allowed_tools_override,
            approved_tools_override=self._approved_tools_override,
            worker_context=self._worker_context,
            worker_permission_requester=self._worker_permission_requester,
            session_id=self.session_id,
            turn_count=self.turn_count,
            summaries=list(self._summaries),
            turn_records=list(self._turn_records),
            manual_compaction_summary=self._manual_compaction_summary,
            manual_compaction_turn_count=self._manual_compaction_turn_count,
            next_turn_target_paths=list(self._next_turn_target_paths),
            last_user_image_attachments=list(self._last_user_image_attachments),
            image_attachment_history=list(self._image_attachment_history),
            historical_tool_compression_count=self._historical_tool_compression_count,
            working_state=self._working_state,
            task_ledger=self._task_ledger,
            current_cancellation_token=self._current_cancellation_token,
            mcp_settings=self._mcp_settings,
            mcp_runtime=self._mcp_runtime,
            owns_mcp_runtime=self._owns_mcp_runtime,
            mcp_tool_names=list(self._mcp_tool_names),
            tool_registry=self._tool_registry,
            session_path=self.session_path,
            created_at=self._created_at,
            session_interaction_state=self._session_interaction_state,
        )

    def _write_task_ledger(self) -> None:
        write_task_ledger(self.session_path / "task-ledger.json", self._task_ledger)

    def close(self) -> None:
        # 先取消 active run，再幂等关闭 gateway/MCP，避免关闭后仍有请求占用连接。
        self.cancel_current_run()
        try:
            store = TeamStore(user_config_dir() / "teams")
            for team in store.list_teams_for_leader(self.session_id):
                store.mark_inactive(team.team_id)
        finally:
            from haagent.models.http_transport import close_model_gateway

            if self.model_gateway is not None:
                close_model_gateway(self.model_gateway)
            if self._owns_mcp_runtime:
                self._mcp_runtime.close()

    def turn_summaries(self) -> list[SessionTurnSummary]:
        """返回 lifecycle 已装载并随 turn 同步更新的会话摘要。"""
        return [session_turn_summary(turn) for turn in self._turn_records]

    def compact_current_session(self) -> SessionCompactResult:
        if self.model_gateway is None:
            raise ChatSessionError("当前会话没有可用模型，无法执行智能压缩")
        if len(self._summaries) <= DEFAULT_PRESERVED_RECENT_TURNS:
            return SessionCompactResult(
                applied=False,
                reason="insufficient_session_history",
                original_turn_count=len(self._summaries),
                compacted_turn_count=0,
                preserved_recent_count=len(self._summaries),
                saved_chars=0,
            )
        messages = [{"role": "user", "content": summary} for summary in self._summaries]
        original_chars = len("\n".join(self._summaries))
        compact_result = maybe_full_compact_messages(
            messages=messages,
            eligibility=FullCompactEligibility(
                eligible=True,
                reason="manual_session_compact",
                trigger_kind="manual_session",
                required_preserve_recent=DEFAULT_PRESERVED_RECENT_TURNS,
            ),
            gateway=self.model_gateway,
            preserve_recent=DEFAULT_PRESERVED_RECENT_TURNS,
        )
        if not compact_result.applied:
            return SessionCompactResult(
                applied=False,
                reason=compact_result.reason,
                original_turn_count=len(self._summaries),
                compacted_turn_count=0,
                preserved_recent_count=compact_result.preserved_recent_count,
                saved_chars=0,
            )
        summary_text_value = manual_compaction_summary_text(compact_result.messages)
        if summary_text_value is None:
            return SessionCompactResult(
                applied=False,
                reason="summary_message_missing",
                original_turn_count=len(self._summaries),
                compacted_turn_count=0,
                preserved_recent_count=compact_result.preserved_recent_count,
                saved_chars=0,
            )
        self._manual_compaction_summary = summary_text_value
        self._manual_compaction_turn_count = max(0, len(self._summaries) - compact_result.preserved_recent_count)
        self._write_manual_compaction_state()
        self._write_session_metadata()
        final_chars = len("\n".join(self._effective_session_summaries()))
        return SessionCompactResult(
            applied=True,
            reason=compact_result.reason,
            original_turn_count=len(self._summaries),
            compacted_turn_count=self._manual_compaction_turn_count,
            preserved_recent_count=compact_result.preserved_recent_count,
            saved_chars=max(0, original_chars - final_chars),
        )

    def _session_memory(self):
        summaries = self._effective_session_summaries()
        keep_recent = DEFAULT_PRESERVED_RECENT_TURNS
        if self._manual_compaction_summary is not None:
            keep_recent += 1
        return compact_session_memory(
            summaries,
            keep_recent=keep_recent,
            memory_char_limit=SESSION_MEMORY_CHAR_LIMIT,
        )

    def _effective_session_summaries(self) -> list[str]:
        if self._manual_compaction_summary is None:
            return list(self._summaries)
        compacted_count = min(max(self._manual_compaction_turn_count, 0), len(self._summaries))
        return [self._manual_compaction_summary, *self._summaries[compacted_count:]]

    def _build_turn_result(self, prompt: str, result) -> ChatTurnResult:
        del prompt
        return build_turn_result(
            session_id=self.session_id,
            turn_index=self.turn_count + 1,
            provider_name=self.provider_name,
            result=result,
        )

    def _record_turn(self, prompt: str, result: ChatTurnResult, summary: str) -> None:
        from haagent.runtime.session.package import assistant_display_text
        from haagent.runtime.session.turn import summary_value

        append_turn_record(
            self.session_path,
            turn_index=result.turn_index,
            request=prompt,
            summary=summary,
            status=result.status,
            episode_path=result.episode_path,
            verification_status=result.verification_status,
            final_response=result.final_response,
        )
        # 与 append_turn_record 写入字段保持一致，供 history 免二次读盘。
        record = {
            "turn_index": result.turn_index,
            "request": summary_value(prompt, 300),
            "summary": summary,
            "status": result.status,
            "episode_path": str(result.episode_path),
            "verification_status": result.verification_status,
            "assistant_display_text": assistant_display_text(result.final_response),
        }
        self._turn_records.append(record)
        self._write_session_metadata()

    def _write_working_state(self) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        write_working_state(self.session_path / "working_state.json", self._working_state)

    def _persist_progress_working_state(self, value: dict[str, object]) -> None:
        """ProgressGuard 进入等待前同步写 session working state，保证中断后可恢复。"""

        self._working_state = working_state_from_dict(value)
        self._write_working_state()

    def _write_session_metadata(self) -> None:
        first_request = "none"
        if self._turn_records:
            request = self._turn_records[0].get("request")
            if isinstance(request, str) and request:
                first_request = request
        write_session_metadata(
            self.session_path,
            session_id=self.session_id,
            workspace_root=self.workspace_root,
            path_policy=self.path_policy,
            provider=self.provider_name,
            model_ref=self.model_ref,
            enable_web=self.enable_web,
            last_user_image_attachments=self._last_user_image_attachments,
            image_attachment_history=self._image_attachment_history,
            created_at=self._created_at,
            turn_count=self.turn_count,
            edit_diff_session_always=self._session_interaction_state.edit_diff_session_always,
            first_request=first_request,
        )

    def _write_manual_compaction_state(self) -> None:
        write_manual_compaction_state(
            self.session_path,
            summary=self._manual_compaction_summary,
            compacted_turn_count=self._manual_compaction_turn_count,
        )
