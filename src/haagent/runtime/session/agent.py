"""
src/haagent/runtime/session/agent.py - 自然语言 Agent 会话

管理 chat 会话状态，并把每条用户请求转成可审计的临时 task contract。
会话 package IO、turn 收尾、路径策略变更与生命周期装配已拆到同级模块。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from haagent.models.types import ModelGateway
from haagent.models.model_ref import ModelRef
from haagent.models.config.connections import user_config_dir
from haagent.memory.extraction import MemoryExtractionRequest, MemoryExtractor
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.steering import SteeringChannel
from haagent.runtime.session.attachments import (
    AttachmentError,
    ImageAttachment,
    read_clipboard_image_bytes,
    save_clipboard_image,
)
from haagent.runtime.session.lifecycle import (
    SessionResources,
    SessionRuntimeState,
    SessionSnapshot,
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
from haagent.runtime.events import PlanningStateEvent, TodoStateEvent
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
    TaskLedgerError,
    cancel_active_todos,
    initialize_todos_from_plan,
    replace_todos,
    update_task_ledger_runtime,
    write_task_ledger,
    write_task_ledger_markdown,
)
from haagent.runtime.session.planning_state import (
    PlanningStateError,
    approve_plan,
    cancel_plan,
    enter_plan_mode as create_planning_state,
    mark_plan_execution_started,
    request_plan_revision,
    submit_plan_revision,
    write_planning_state,
)
from haagent.context.projection import ModelContextFacts
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


# AgentSession 实例自有字段；其余会话字段委托到 Snapshot / Resources。
_SESSION_OWN_ATTRS = frozenset(
    {
        "_snapshot",
        "_resources",
        "_skill_catalog",
        "_instruction_cache",
        "_tool_schema_cache",
        "_model_context_runtime",
        "_closed",
    }
)

# 对外/对内属性名 → SessionSnapshot 字段名
_SNAPSHOT_ATTRS: dict[str, str] = {
    "workspace_root": "workspace_root",
    "path_policy": "path_policy",
    "runs_root": "runs_root",
    "model_ref": "model_ref",
    "enable_web": "enable_web",
    "session_id": "session_id",
    "turn_count": "turn_count",
    "session_path": "session_path",
    "_summaries": "summaries",
    "_turn_records": "turn_records",
    "_manual_compaction_summary": "manual_compaction_summary",
    "_manual_compaction_turn_count": "manual_compaction_turn_count",
    "_last_user_image_attachments": "last_user_image_attachments",
    "_image_attachment_history": "image_attachment_history",
    "_working_state": "working_state",
    "_task_ledger": "task_ledger",
    "_planning_state": "planning_state",
    "_created_at": "created_at",
    "_session_interaction_state": "session_interaction_state",
}

# 对外/对内属性名 → SessionResources 字段名
_RESOURCES_ATTRS: dict[str, str] = {
    "model_gateway": "model_gateway",
    "max_turns": "max_turns",
    "memory_extraction_enabled": "memory_extraction_enabled",
    "_allowed_tools_override": "allowed_tools_override",
    "_approval_allowed_tools_override": "approval_allowed_tools_override",
    "_approved_tools_override": "approved_tools_override",
    "_worker_context": "worker_context",
    "_worker_permission_requester": "worker_permission_requester",
    "_next_turn_target_paths": "next_turn_target_paths",
    "_historical_tool_compression_count": "historical_tool_compression_count",
    "_current_cancellation_token": "current_cancellation_token",
    "_current_steering_channel": "current_steering_channel",
    "_mcp_settings": "mcp_settings",
    "_mcp_runtime": "mcp_runtime",
    "_owns_mcp_runtime": "owns_mcp_runtime",
    "_mcp_tool_names": "mcp_tool_names",
    "_tool_registry": "tool_registry",
}


class AgentSession:
    def __getattr__(self, name: str) -> Any:
        # 失败边界：snapshot/resources 未装配时不假装字段存在。
        snap_field = _SNAPSHOT_ATTRS.get(name)
        if snap_field is not None:
            snapshot = self.__dict__.get("_snapshot")
            if snapshot is None:
                raise AttributeError(name)
            return getattr(snapshot, snap_field)
        res_field = _RESOURCES_ATTRS.get(name)
        if res_field is not None:
            resources = self.__dict__.get("_resources")
            if resources is None:
                raise AttributeError(name)
            return getattr(resources, res_field)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SESSION_OWN_ATTRS:
            object.__setattr__(self, name, value)
            return
        snap_field = _SNAPSHOT_ATTRS.get(name)
        if snap_field is not None:
            snapshot = self.__dict__.get("_snapshot")
            if snapshot is None:
                raise AttributeError(name)
            setattr(snapshot, snap_field, value)
            return
        res_field = _RESOURCES_ATTRS.get(name)
        if res_field is not None:
            resources = self.__dict__.get("_resources")
            if resources is None:
                raise AttributeError(name)
            setattr(resources, res_field, value)
            return
        object.__setattr__(self, name, value)

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
        self._closed = False
        self._write_session_metadata()
        self._write_working_state()
        self._write_task_ledger()
        self._write_planning_state()

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
        instance._closed = False
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
        self._interrupt_background_workers(reason="session reloaded")
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
        # 不用 gateway.provider_name 直接访问：CPython 会把 descriptor 内 AttributeError
        # 清掉后继续走 __getattr__，误报 AgentSession 缺字段。
        gateway = self.model_gateway
        if gateway is None:
            return "fake"
        name = getattr(gateway, "provider_name", None)
        return "fake" if name is None else str(name)

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
        self._current_steering_channel = SteeringChannel()

        def on_runtime_event(event: RuntimeBusEvent | dict[str, object]) -> None:
            bus_event = coerce_bus_event(event)
            runtime_events.append(bus_event)
            emit_runtime_ui_event(event_sink, bus_event, session_id=self.session_id, turn_index=turn_index)

        def todo_state_sink(
            items: list[dict[str, object]],
            explanation: str,
            tool_turn: int | None,
        ) -> dict[str, object]:
            result = self._apply_todo_update(
                items,
                explanation=explanation,
                turn_index=tool_turn or turn_index,
                fallback_goal=clean_prompt,
            )
            active = self._task_ledger.active_todo()
            emit_ui_event(
                event_sink,
                TodoStateEvent(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    total_count=len(self._task_ledger.todos),
                    completed_count=self._task_ledger.status_counts()["completed"],
                    active_item_id=active.id if active is not None else None,
                    active_item_content=active.content if active is not None else "",
                    all_terminal=self._task_ledger.all_terminal(),
                ),
            )
            return result

        def planning_state_handler(
            action: str,
            payload: dict[str, object],
            planning_turn: int,
        ) -> dict[str, object]:
            result = self._handle_planning_state_action(action, payload, planning_turn or turn_index)
            proposal = self._planning_state.proposal
            emit_ui_event(
                event_sink,
                PlanningStateEvent(
                    session_id=self.session_id,
                    turn_index=turn_index,
                    state=self._planning_state.status,
                    plan_id=self._planning_state.plan_id,
                    revision=self._planning_state.revision,
                    step_count=len(proposal.steps) if proposal is not None else 0,
                ),
            )
            return result

        target_paths = list(self._next_turn_target_paths)
        self._next_turn_target_paths = []
        session_memory = self._session_memory()
        new_attachments = list(attachments or [])
        prompt_attachments = new_attachments if new_attachments else list(self._last_user_image_attachments)
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
                    model_ref=self.model_ref,
                    max_turns=self.max_turns,
                    session_summary=session_memory.summary_text,
                    session_compaction=session_memory.diagnostics,
                    historical_tool_compression_count=self._historical_tool_compression_count,
                    working_state=self._working_state.to_dict() if not self._working_state.is_empty() else None,
                    task_ledger=self._task_ledger.to_dict(),
                    planning_state=self._planning_state.to_dict(),
                    path_policy=self.path_policy,
                    enable_web=self.enable_web,
                    target_paths=target_paths,
                    include_memory_tool=self.memory_extraction_enabled,
                    session_path=self.session_path,
                    event_sink=on_runtime_event,
                    interaction_handler=interaction_handler,
                    cancellation_token=self._current_cancellation_token,
                    steering_channel=self._current_steering_channel,
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
                    todo_state_sink=todo_state_sink,
                    planning_state_handler=planning_state_handler,
                    plan_execution_has_active_todos=lambda: (
                        self._planning_state.status == "execution_started"
                        and self._task_ledger.has_active_todos()
                    ),
                    model_context_runtime=self._model_context_runtime,
                ),
            )
        except Exception:
            self._current_cancellation_token = None
            self._current_steering_channel = None
            raise

        # run 结束仍未消费的引导必须交回调用方（TUI 重新排队），不能静默丢弃。
        unconsumed_steering = (
            tuple(self._current_steering_channel.drain())
            if self._current_steering_channel is not None
            else ()
        )
        turn_result = self._build_turn_result(clean_prompt, result)
        turn_result = with_in_band_verification(turn_result, runtime_events)
        if unconsumed_steering:
            turn_result = replace(turn_result, unconsumed_steering=unconsumed_steering)
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
        self._task_ledger = update_task_ledger_runtime(
            self._task_ledger,
            turn_index=turn_index,
            episode_path=turn_result.episode_path,
            runtime_events=runtime_events,
        )
        self._write_task_ledger(episode_path=turn_result.episode_path)
        self._historical_tool_compression_count += count_historical_tool_compression_events(runtime_events)
        summary = turn_summary(
            clean_prompt,
            turn_result,
            steering_texts=_steering_texts_from_events(runtime_events),
        )
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
        self._current_steering_channel = None
        if self._planning_state.status == "approved_pending_execution":
            return self._run_approved_plan_execution(
                event_sink=event_sink,
                interaction_handler=interaction_handler,
            )
        return turn_result

    def steer_current_run(self, text: str) -> bool:
        """向正在运行的任务投递引导消息；无运行中任务时返回 False。"""
        channel = self._current_steering_channel
        if channel is None or self._current_cancellation_token is None:
            return False
        normalized = text.strip()
        if not normalized:
            return False
        channel.post(normalized)
        return True

    def cancel_current_run(self) -> bool:
        changed = False
        if self._planning_state.status in {"planning", "awaiting_confirmation", "approved_pending_execution"}:
            self._planning_state = cancel_plan(self._planning_state, updated_turn=self.turn_count)
            self._write_planning_state()
            changed = True
        if self._task_ledger.has_active_todos():
            self._task_ledger = cancel_active_todos(self._task_ledger, turn_index=self.turn_count)
            self._write_task_ledger()
            changed = True
        if self._current_cancellation_token is not None:
            self._current_cancellation_token.cancel()
            return True
        return changed

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
            "planning_state": {
                "status": self._planning_state.status,
                "plan_id": self._planning_state.plan_id,
                "revision": self._planning_state.revision,
                "is_plan_mode": self._planning_state.is_plan_mode,
            },
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
        self._interrupt_background_workers(reason="new session created")
        state = build_new_package_state(self._snapshot_state())
        apply_state(self, state)
        self._write_session_metadata()
        self._write_working_state()
        self._write_task_ledger()
        self._write_planning_state()

    def _snapshot_state(self) -> SessionRuntimeState:
        # new() 会再构造独立 package；clone 避免共享可变 list。
        return SessionRuntimeState(
            snapshot=self._snapshot.clone(),
            resources=self._resources.clone(),
            model_context_runtime=self._model_context_runtime,
        )

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    @property
    def resources(self) -> SessionResources:
        return self._resources

    def _write_task_ledger(self, episode_path: Path | None = None) -> None:
        write_task_ledger(self.session_path / "task-ledger.json", self._task_ledger)
        # 同步写入 markdown 到 episode 目录，供模型 file_read 回顾完整 Todo 状态。
        if episode_path is not None:
            write_task_ledger_markdown(Path(episode_path) / "task-ledger.md", self._task_ledger)

    def _write_planning_state(self) -> None:
        write_planning_state(self.session_path / "planning-state.json", self._planning_state)

    def enter_plan_mode(self) -> object:
        """显式进入 Plan Mode；运行 turn 或活动 Todo 存在时拒绝。"""

        if self._current_cancellation_token is not None:
            raise ChatSessionError("当前任务仍在运行，不能进入 Plan Mode")
        if self._task_ledger.has_active_todos():
            raise ChatSessionError("当前 Todo 尚未结束，请先完成或取消当前任务")
        if self._planning_state.is_plan_mode:
            raise ChatSessionError("当前已经处于 Plan Mode")
        self._planning_state = create_planning_state(updated_turn=self.turn_count)
        self._write_planning_state()
        return self._planning_state

    def approve_plan_revision(self, plan_id: str, revision: int) -> object:
        self._handle_planning_state_action(
            "approve",
            {"plan_id": plan_id, "revision": revision},
            self.turn_count,
        )
        return self._planning_state

    def submit_plan_feedback(self, plan_id: str, revision: int, feedback: str) -> object:
        if not feedback.strip():
            raise ChatSessionError("Plan 修改意见不能为空")
        self._handle_planning_state_action(
            "feedback",
            {"plan_id": plan_id, "revision": revision, "feedback": feedback},
            self.turn_count,
        )
        return self._planning_state

    def execute_pending_plan_events(
        self,
        *,
        event_sink: RuntimeUiEventSink = None,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        if self._planning_state.status != "approved_pending_execution":
            raise ChatSessionError("当前没有等待执行的已批准 Plan")
        return self._run_approved_plan_execution(
            event_sink=event_sink,
            interaction_handler=interaction_handler,
        )

    def _apply_todo_update(
        self,
        items: list[dict[str, object]],
        *,
        explanation: str,
        turn_index: int,
        fallback_goal: str,
    ) -> dict[str, object]:
        del explanation
        try:
            self._task_ledger = replace_todos(
                self._task_ledger,
                items=items,
                turn_index=turn_index,
                goal=self._task_ledger.goal or fallback_goal[:240],
            )
        except TaskLedgerError:
            # 原子失败边界：不写磁盘，直接把校验错误返回给模型修正。
            raise
        self._write_task_ledger()
        return {
            "items": [item.to_dict() for item in self._task_ledger.todos],
            "counts": self._task_ledger.status_counts(),
            "all_terminal": self._task_ledger.all_terminal(),
        }

    def _handle_planning_state_action(
        self,
        action: str,
        payload: dict[str, object],
        turn_index: int,
    ) -> dict[str, object]:
        try:
            if action == "submit":
                self._planning_state = submit_plan_revision(
                    self._planning_state,
                    payload,
                    updated_turn=turn_index,
                )
            elif action == "feedback":
                self._planning_state = request_plan_revision(
                    self._planning_state,
                    plan_id=str(payload.get("plan_id", "")),
                    revision=int(payload.get("revision", -1)),
                    updated_turn=turn_index,
                )
            elif action == "approve":
                self._planning_state = approve_plan(
                    self._planning_state,
                    plan_id=str(payload.get("plan_id", "")),
                    revision=int(payload.get("revision", -1)),
                    updated_turn=turn_index,
                )
            else:
                raise PlanningStateError(f"unknown planning action: {action}")
        except (PlanningStateError, ValueError) as error:
            raise ChatSessionError(str(error)) from error
        self._write_planning_state()
        return self._planning_state.to_dict()

    def _run_approved_plan_execution(
        self,
        *,
        event_sink: RuntimeUiEventSink,
        interaction_handler: HumanInteractionHandler | None,
    ) -> ChatTurnResult:
        state = self._planning_state
        proposal = state.proposal
        if proposal is None or state.plan_id is None or state.approved_revision is None or state.execution_id is None:
            raise ChatSessionError("批准的 Plan 状态不完整，无法启动执行")
        initialized = any(
            item.source_plan_id == state.plan_id and item.source_plan_revision == state.approved_revision
            for item in self._task_ledger.todos
        )
        if not initialized:
            verification = proposal.verification.description if proposal.verification.required else None
            self._task_ledger = initialize_todos_from_plan(
                goal=proposal.goal,
                plan_id=state.plan_id,
                revision=state.approved_revision,
                steps=[(step.id, step.content) for step in proposal.steps],
                verification=verification,
                turn_index=self.turn_count,
            )
            self._write_task_ledger()
            active = self._task_ledger.active_todo()
            emit_ui_event(
                event_sink,
                TodoStateEvent(
                    session_id=self.session_id,
                    turn_index=self.turn_count,
                    total_count=len(self._task_ledger.todos),
                    completed_count=0,
                    active_item_id=active.id if active is not None else None,
                    active_item_content=active.content if active is not None else "",
                    all_terminal=False,
                ),
            )
        self._planning_state = mark_plan_execution_started(
            state,
            execution_id=state.execution_id,
            updated_turn=self.turn_count + 1,
        )
        self._write_planning_state()
        emit_ui_event(
            event_sink,
            PlanningStateEvent(
                session_id=self.session_id,
                turn_index=self.turn_count + 1,
                state=self._planning_state.status,
                plan_id=self._planning_state.plan_id,
                revision=self._planning_state.revision,
                step_count=len(proposal.steps),
            ),
        )
        # 自动执行不会向 TUI 追加伪造用户消息；结构化 Plan/Todo 通过 session context 注入。
        return self.run_prompt_events(
            proposal.goal,
            event_sink=event_sink,
            include_session_events=False,
            interaction_handler=interaction_handler,
        )

    def _interrupt_background_workers(self, *, reason: str) -> int:
        from haagent.multi_agent.runtime import MultiAgentRuntime

        return MultiAgentRuntime.interrupt_session(
            team_root=user_config_dir() / "teams",
            leader_session_id=self.session_id,
            reason=reason,
        )

    def close(self) -> None:
        # 先封闭重复入口，再取消 active run 和后台 worker，避免重复关闭 gateway/MCP。
        if self._closed:
            return
        self._closed = True
        self.cancel_current_run()
        try:
            self._interrupt_background_workers(reason="session closed")
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
        # 先持久化 rebuild intent；中断最多导致一次额外 rebuild，不能漏掉代际切换。
        self._model_context_runtime.require_rebuild("session_memory_compacted")
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
            recent_turns=self._recent_full_turns(keep_recent),
        )

    def _model_context_facts(self) -> ModelContextFacts:
        """向 Model Context Runtime 提供原始 session 事实，不参与格式化或版本推进。"""

        memory = self._session_memory()
        return ModelContextFacts(
            session_summary=memory.summary_text,
            working_state=(
                self._working_state.to_dict()
                if not self._working_state.is_empty()
                else None
            ),
            task_ledger=self._task_ledger.to_dict(),
            planning_state=self._planning_state.to_dict(),
        )

    def _recent_full_turns(self, keep_recent: int) -> list[dict[str, object]]:
        """从 turn_records 提取最近若干轮完整问答；resume 后由磁盘 records 回填。"""
        records = self._turn_records[-keep_recent:] if keep_recent > 0 else []
        recent: list[dict[str, object]] = []
        for record in records:
            user = str(record.get("request") or "").strip()
            assistant = str(record.get("assistant_display_text") or "").strip()
            if not user and not assistant:
                continue
            recent.append({"request": user, "assistant_display_text": assistant})
        return recent

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
            permission_rules=self._session_interaction_state.permission_rules,
            first_request=first_request,
            session_snapshot_schema_version=self.snapshot.schema_version,
        )

    def _write_manual_compaction_state(self) -> None:
        write_manual_compaction_state(
            self.session_path,
            summary=self._manual_compaction_summary,
            compacted_turn_count=self._manual_compaction_turn_count,
        )

def _steering_texts_from_events(runtime_events: list[object]) -> list[str]:
    from haagent.runtime.events.bus import SteeringInjectedBusEvent, coerce_bus_event

    texts: list[str] = []
    for raw_event in runtime_events:
        event = coerce_bus_event(raw_event)
        if isinstance(event, SteeringInjectedBusEvent) and event.content.strip():
            texts.append(event.content)
    return texts
