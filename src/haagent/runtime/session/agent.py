"""
src/haagent/runtime/session/agent.py - 自然语言 Agent 会话

管理 chat 会话状态，并把每条用户请求转成可审计的临时 task contract。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.settings import load_mcp_settings
from haagent.mcp.tool_adapter import mcp_tool_alias, mcp_tool_definitions
from haagent.models.gateway import ModelGateway
from haagent.memory.extraction import MemoryExtractionRequest, MemoryExtractor
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.session.turn import ChatTurnRequest, ChatTurnRunner, summary_value as _summary_value
from haagent.runtime.events import RuntimeUiEvent
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
from haagent.runtime.episodes.validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.compaction.full import maybe_full_compact_messages
from haagent.runtime.compaction.contract import FullCompactEligibility
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.execution.path_policy import (
    ExternalRoot,
    PathAccess,
    PathPolicy,
    PermissionMode,
    default_path_policy,
    load_path_policy,
    serialize_path_policy,
)
from haagent.runtime.session.memory_compaction import (
    DEFAULT_PRESERVED_RECENT_TURNS,
    SESSION_MEMORY_CHAR_LIMIT,
    compact_session_memory,
)
from haagent.runtime.session.working_state import (
    WorkingStateError,
    empty_working_state,
    load_working_state,
    update_working_state,
    write_working_state,
)
from haagent.tools.registry import default_tool_runtime_registry
CHAT_MAX_TURNS = 20


class ChatSessionError(RuntimeError):
    """Chat session package 损坏或无法恢复时抛出。"""


@dataclass(frozen=True)
class ChatTurnResult:
    session_id: str
    turn_index: int
    status: str
    episode_path: Path
    provider: str
    final_response: str
    verification_status: str
    failed_stage: str = "none"
    failure_category: str = "none"
    reason: str = "none"
    summary_error: str | None = None
    memory_candidates_created: int = 0
    memory_extraction_status: str = "skipped"
    memory_extraction_reason: str = ""

    def output_lines(self) -> list[str]:
        lines = [
            f"status={self.status}",
            f"episode_path={self.episode_path}",
            f"provider={self.provider}",
            f"final_response={_summary_value(self.final_response)}",
            f"verification={self.verification_status}",
        ]
        if self.summary_error is not None:
            lines.append(f"summary_error={_summary_value(self.summary_error)}")
        if self.memory_candidates_created:
            lines.append(f"memory_candidates={self.memory_candidates_created}")
        if self.status != "completed":
            lines.extend(
                [
                    f"failed_stage={_summary_value(self.failed_stage)}",
                    f"failure_category={_summary_value(self.failure_category)}",
                    f"reason={_summary_value(self.reason)}",
                ],
            )
        return lines


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    workspace_root: Path
    turn_count: int
    first_request: str
    session_path: Path


@dataclass(frozen=True)
class SessionTurnSummary:
    turn_index: int
    request: str
    summary: str
    status: str
    episode_path: Path
    verification_status: str


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
        model_profile_name: str | None = None,
        model_name: str | None = None,
        model_base_url: str | None = None,
        max_turns: int = CHAT_MAX_TURNS,
        session_id: str | None = None,
        memory_extraction_enabled: bool = True,
        enable_web: bool = False,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.path_policy = default_path_policy(self.workspace_root)
        self.runs_root = runs_root
        self.model_gateway = model_gateway
        self.model_profile_name = model_profile_name
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.max_turns = max_turns
        self.memory_extraction_enabled = memory_extraction_enabled
        self.enable_web = enable_web
        self.session_id = session_id or _new_session_id()
        self.turn_count = 0
        self._summaries: list[str] = []
        self._manual_compaction_summary: str | None = None
        self._manual_compaction_turn_count = 0
        self._next_turn_target_paths: list[str] = []
        self._tool_result_microcompact_count = 0
        self._working_state = empty_working_state()
        self._current_cancellation_token: CancellationToken | None = None
        self._mcp_settings = load_mcp_settings()
        self._mcp_runtime = SyncMcpRuntime(self._mcp_settings)
        self._mcp_runtime.start()
        self._mcp_tool_names = [
            mcp_tool_alias(tool.server_name, tool.name)
            for tool in self._mcp_runtime.list_tools()
        ]
        self._tool_registry = default_tool_runtime_registry(
            mcp_tool_definitions(self._mcp_runtime.list_tools()),
        )
        self.session_path = self.runs_root / "sessions" / self.session_id
        self._created_at = datetime.now(UTC).isoformat()
        self._write_session_metadata()
        self._write_working_state()

    @classmethod
    def resume(
        cls,
        session: str | Path,
        *,
        runs_root: Path | None = None,
        model_gateway: ModelGateway | None = None,
        model_profile_name: str | None = None,
        model_name: str | None = None,
        model_base_url: str | None = None,
        max_turns: int = CHAT_MAX_TURNS,
        enable_web: bool = False,
    ) -> "AgentSession":
        session_path = _resolve_session_path(session, runs_root or Path(".runs"))
        metadata = _read_session_metadata(session_path)
        turns = _read_session_turns(session_path)

        instance = cls.__new__(cls)
        instance.workspace_root = Path(str(metadata["workspace_root"])).resolve()
        raw_policy = metadata.get("path_policy")
        instance.path_policy = (
            load_path_policy(raw_policy)
            if isinstance(raw_policy, dict)
            else default_path_policy(instance.workspace_root)
        )
        instance.runs_root = session_path.parent.parent
        instance.model_gateway = model_gateway
        instance.model_profile_name = model_profile_name or _optional_string(metadata.get("model_profile_name"))
        instance.model_name = model_name or _optional_string(metadata.get("model"))
        instance.model_base_url = model_base_url or _optional_string(metadata.get("base_url"))
        instance.max_turns = max_turns
        instance.memory_extraction_enabled = True
        instance.enable_web = enable_web
        instance.session_id = str(metadata["session_id"])
        instance.turn_count = int(metadata["turn_count"])
        instance._summaries = [str(turn["summary"]) for turn in turns]
        compaction_summary, compacted_turn_count = _read_manual_compaction_state(session_path)
        instance._manual_compaction_summary = compaction_summary
        instance._manual_compaction_turn_count = compacted_turn_count
        instance._next_turn_target_paths = []
        instance._tool_result_microcompact_count = 0
        try:
            instance._working_state = load_working_state(session_path / "working_state.json")
        except WorkingStateError as error:
            raise ChatSessionError(str(error)) from error
        instance._mcp_settings = load_mcp_settings()
        instance._mcp_runtime = SyncMcpRuntime(instance._mcp_settings)
        instance._mcp_runtime.start()
        instance._mcp_tool_names = [
            mcp_tool_alias(tool.server_name, tool.name)
            for tool in instance._mcp_runtime.list_tools()
        ]
        instance._tool_registry = default_tool_runtime_registry(
            mcp_tool_definitions(instance._mcp_runtime.list_tools()),
        )
        instance.session_path = session_path
        instance._current_cancellation_token = None
        instance._created_at = str(metadata["created_at"])
        return instance

    @property
    def provider_name(self) -> str:
        if self.model_gateway is None:
            return "fake"
        return self.model_gateway.provider_name

    def run_prompt(
        self,
        prompt: str,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        return self.run_prompt_events(prompt, interaction_handler=interaction_handler)

    def run_prompt_events(
        self,
        prompt: str,
        event_sink: RuntimeUiEventSink = None,
        include_session_events: bool = False,
        interaction_handler: HumanInteractionHandler | None = None,
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
        runtime_events: list[dict[str, object]] = []
        self._current_cancellation_token = CancellationToken()

        def on_runtime_event(event: dict[str, object]) -> None:
            runtime_events.append(event)
            emit_runtime_ui_event(event_sink, event, session_id=self.session_id, turn_index=turn_index)

        target_paths = list(self._next_turn_target_paths)
        self._next_turn_target_paths = []
        session_memory = self._session_memory()
        result = ChatTurnRunner().run(
            ChatTurnRequest(
                prompt=clean_prompt,
                workspace_root=self.workspace_root,
                runs_root=self.runs_root,
                model_gateway=self.model_gateway,
                max_turns=self.max_turns,
                session_summary=session_memory.summary_text,
                session_compaction=session_memory.diagnostics,
                tool_result_microcompact_count=self._tool_result_microcompact_count,
                working_state=self._working_state.to_dict() if not self._working_state.is_empty() else None,
                path_policy=self.path_policy,
                enable_web=self.enable_web,
                target_paths=target_paths,
                event_sink=on_runtime_event,
                interaction_handler=interaction_handler,
                cancellation_token=self._current_cancellation_token,
                orchestrator_factory=RunOrchestrator,
                tool_registry=self._tool_registry,
                mcp_runtime=self._mcp_runtime,
                mcp_tool_names=self._mcp_tool_names,
            ),
        )

        turn_result = self._build_turn_result(clean_prompt, result)
        self.turn_count += 1
        self._working_state = update_working_state(
            self._working_state,
            prompt=clean_prompt,
            result=turn_result,
            runtime_events=runtime_events,
        )
        self._write_working_state()
        self._tool_result_microcompact_count += _count_runtime_events(runtime_events, "tool_result_microcompact")
        turn_summary = _turn_summary(clean_prompt, turn_result)
        self._summaries.append(turn_summary)
        self._record_turn(clean_prompt, turn_result, turn_summary)
        extraction_result = None
        if self.memory_extraction_enabled and _memory_update_requested(runtime_events):
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

    def cancel_current_run(self) -> None:
        if self._current_cancellation_token is not None:
            self._current_cancellation_token.cancel()

    def switch_model_gateway(
        self,
        *,
        profile_name: str,
        provider: str,
        model: str,
        base_url: str,
        gateway: ModelGateway,
    ) -> None:
        if self._current_cancellation_token is not None:
            raise ChatSessionError("current task is running")
        self.model_gateway = gateway
        self.model_profile_name = profile_name
        self.model_name = model
        self.model_base_url = base_url
        self._write_session_metadata()

    def add_external_root(self, path: Path, access: PathAccess) -> None:
        resolved_path = path.resolve()
        roots = [root for root in self.path_policy.external_roots if root.path.resolve() != resolved_path]
        roots.append(
            ExternalRoot(
                path=resolved_path,
                access=access,
                source="user",
                created_at=datetime.now(UTC).isoformat(),
            ),
        )
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            external_roots=roots,
            permission_mode=self.path_policy.permission_mode,
        ).resolved()
        self._write_session_metadata()

    def remove_external_root(self, path: Path) -> None:
        resolved_path = path.resolve()
        roots = [root for root in self.path_policy.external_roots if root.path.resolve() != resolved_path]
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            external_roots=roots,
            permission_mode=self.path_policy.permission_mode,
        ).resolved()
        self._write_session_metadata()

    def set_external_root_access(self, path: Path, access: PathAccess) -> None:
        resolved_path = path.resolve()
        roots: list[ExternalRoot] = []
        found = False
        for root in self.path_policy.external_roots:
            if root.path.resolve() == resolved_path:
                roots.append(ExternalRoot(path=resolved_path, access=access, source=root.source, created_at=root.created_at))
                found = True
            else:
                roots.append(root)
        if not found:
            roots.append(
                ExternalRoot(
                    path=resolved_path,
                    access=access,
                    source="user",
                    created_at=datetime.now(UTC).isoformat(),
                ),
            )
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            external_roots=roots,
            permission_mode=self.path_policy.permission_mode,
        ).resolved()
        self._write_session_metadata()

    def clear_external_roots(self) -> None:
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            permission_mode=self.path_policy.permission_mode,
        ).resolved()
        self._write_session_metadata()

    def switch_project_root(self, path: Path) -> None:
        permission_mode = self.path_policy.permission_mode
        self.workspace_root = path.resolve()
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            permission_mode=permission_mode,
        ).resolved()
        self._write_session_metadata()

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if mode not in {"request_approval", "auto_approve", "full_access"}:
            raise ChatSessionError("permission mode must be request_approval, auto_approve, or full_access")
        self.path_policy = PathPolicy(
            project_root=self.workspace_root,
            external_roots=self.path_policy.external_roots,
            permission_mode=mode,
        ).resolved()
        self._write_session_metadata()

    def set_next_turn_target_paths(self, paths: list[Path]) -> None:
        self._next_turn_target_paths = [str(path.resolve()) for path in paths]

    def _run_memory_extraction(
        self,
        prompt: str,
        result: ChatTurnResult,
        runtime_events: list[dict[str, object]],
    ):
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
                runtime_events=runtime_events,
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
        self.session_id = _new_session_id()
        self.turn_count = 0
        self._summaries = []
        self._manual_compaction_summary = None
        self._manual_compaction_turn_count = 0
        self._working_state = empty_working_state()
        self.path_policy = default_path_policy(self.workspace_root)
        self.session_path = self.runs_root / "sessions" / self.session_id
        self._created_at = datetime.now(UTC).isoformat()
        self._write_session_metadata()
        self._write_working_state()

    def close(self) -> None:
        self._mcp_runtime.close()

    def summary_text(self) -> str | None:
        return self._session_memory().summary_text

    def turn_summaries(self) -> list[SessionTurnSummary]:
        """读取当前 session 已记录轮次，供 UI 展示可恢复上下文。"""
        return [_session_turn_summary(turn) for turn in _read_session_turns(self.session_path)]

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
        summary_text = _manual_compaction_summary_text(compact_result.messages)
        if summary_text is None:
            return SessionCompactResult(
                applied=False,
                reason="summary_message_missing",
                original_turn_count=len(self._summaries),
                compacted_turn_count=0,
                preserved_recent_count=compact_result.preserved_recent_count,
                saved_chars=0,
            )
        self._manual_compaction_summary = summary_text
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

    def session_started_event(self) -> RuntimeUiEvent:
        return build_session_started_event(
            session_id=self.session_id,
            turn_index=self.turn_count,
            details=self.status(),
        )

    def session_finished_event(self) -> RuntimeUiEvent:
        return build_session_finished_event(
            session_id=self.session_id,
            turn_index=self.turn_count,
            details={"turn_count": self.turn_count},
        )

    def _build_turn_result(self, prompt: str, result) -> ChatTurnResult:
        try:
            package_view = load_inspect_episode_package(result.episode_path)
        except EpisodeValidationError as error:
            return ChatTurnResult(
                session_id=self.session_id,
                turn_index=self.turn_count + 1,
                status=result.status.value,
                episode_path=result.episode_path,
                provider=self.provider_name,
                final_response="none",
                verification_status="not_run",
                summary_error=str(error),
            )

        failure = package_view.failure_record.get("failure")
        if not isinstance(failure, dict):
            failure = {}
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=self.turn_count + 1,
            status=result.status.value,
            episode_path=result.episode_path,
            provider=str(package_view.episode_metadata.get("provider", self.provider_name)),
            final_response=_run_final_response(package_view.transcript),
            verification_status=_verification_status(
                package_view.verification_commands,
                package_view.verification_reached,
            ),
            failed_stage=str(failure.get("stage", "none")),
            failure_category=str(failure.get("category", "none")),
            reason=str(failure.get("evidence", "none")),
        )

    def _record_turn(self, prompt: str, result: ChatTurnResult, summary: str) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        record = {
            "turn_index": result.turn_index,
            "request": _summary_value(prompt, 300),
            "summary": summary,
            "status": result.status,
            "episode_path": str(result.episode_path),
            "verification_status": result.verification_status,
        }
        with (self.session_path / "turns.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_session_metadata()

    def _write_working_state(self) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        write_working_state(self.session_path / "working_state.json", self._working_state)

    def _write_session_metadata(self) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.session_path / "session.json"
        created_at = self._created_at
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
                created_at = str(existing["created_at"])
        metadata = {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "path_policy": serialize_path_policy(self.path_policy),
            "provider": self.provider_name,
            "model_profile_name": self.model_profile_name,
            "model": self.model_name,
            "base_url": self.model_base_url,
            "enable_web": self.enable_web,
            "created_at": created_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "turn_count": self.turn_count,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_manual_compaction_state(self) -> None:
        state_path = self.session_path / "session_memory.json"
        if self._manual_compaction_summary is None:
            if state_path.exists():
                state_path.unlink()
            return
        self.session_path.mkdir(parents=True, exist_ok=True)
        state = {
            "summary": self._manual_compaction_summary,
            "compacted_turn_count": self._manual_compaction_turn_count,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _turn_summary(prompt: str, result: ChatTurnResult) -> str:
    return "\n".join(
        [
            f"- user_request: {_summary_value(prompt, 160)}",
            f"  status: {result.status}",
            f"  episode_path: {result.episode_path}",
            f"  assistant_final_response: {_summary_value(result.final_response, 220)}",
            f"  verification: {result.verification_status}",
        ],
    )


def _run_final_response(transcript: list[dict[str, Any]]) -> str:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return str(record.get("content", ""))
    return "none"


def _verification_status(commands: list[dict[str, Any]], verification_reached: bool) -> str:
    if not verification_reached or not commands:
        return "not_run"
    if any(command.get("status") != "success" for command in commands):
        return "failed"
    return "success"


def _memory_update_requested(runtime_events: list[dict[str, object]]) -> bool:
    for event in runtime_events:
        if event.get("event_type") != "tool_finished" or event.get("tool_name") != "start_memory_update":
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if result.get("status") == "success" and result.get("memory_update_requested") is True:
            return True
    return False


def _count_runtime_events(runtime_events: list[dict[str, object]], event_type: str) -> int:
    return sum(1 for event in runtime_events if event.get("event_type") == event_type or event.get("event") == event_type)


def _resolve_session_path(session: str | Path, runs_root: Path) -> Path:
    raw = Path(session)
    if raw.is_absolute() or raw.exists() or raw.name != str(session):
        return raw.resolve()
    return (runs_root / "sessions" / str(session)).resolve()


def list_sessions(runs_root: Path, workspace_root: Path) -> list[SessionSummary]:
    """列出当前 workspace 下的 chat 会话摘要。"""
    sessions_root = runs_root / "sessions"
    if not sessions_root.exists():
        return []
    resolved_workspace = workspace_root.resolve()
    summaries: list[SessionSummary] = []
    for session_path in sessions_root.iterdir():
        if not session_path.is_dir():
            continue
        metadata = _read_session_metadata(session_path)
        if Path(str(metadata["workspace_root"])).resolve() != resolved_workspace:
            continue
        turns = _read_session_turns(session_path)
        first_request = str(turns[0]["request"]) if turns else "none"
        summaries.append(
            SessionSummary(
                session_id=str(metadata["session_id"]),
                created_at=str(metadata["created_at"]),
                updated_at=str(metadata["updated_at"]),
                workspace_root=resolved_workspace,
                turn_count=int(metadata["turn_count"]),
                first_request=first_request,
                session_path=session_path.resolve(),
            ),
        )
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)


def find_latest_session(runs_root: Path, workspace_root: Path) -> SessionSummary | None:
    sessions = list_sessions(runs_root, workspace_root)
    return sessions[0] if sessions else None


def _read_session_metadata(session_path: Path) -> dict[str, object]:
    metadata_path = session_path / "session.json"
    if not metadata_path.exists():
        raise ChatSessionError(f"session package missing required file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ChatSessionError(f"invalid session.json: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ChatSessionError(f"invalid session.json: {metadata_path} must contain an object")
    required_fields = ["session_id", "workspace_root", "provider", "created_at", "updated_at", "turn_count"]
    for field_name in required_fields:
        if field_name not in metadata:
            raise ChatSessionError(f"invalid session.json: missing {field_name}")
    for field_name in ["session_id", "workspace_root", "provider", "created_at", "updated_at"]:
        if not isinstance(metadata[field_name], str):
            raise ChatSessionError(f"invalid session.json: {field_name} must be a string")
    if not isinstance(metadata["turn_count"], int) or isinstance(metadata["turn_count"], bool):
        raise ChatSessionError("invalid session.json: turn_count must be an integer")
    if str(metadata["session_id"]) != session_path.name:
        raise ChatSessionError("invalid session.json: session_id does not match session path")
    return metadata


def _read_session_turns(session_path: Path) -> list[dict[str, object]]:
    turns_path = session_path / "turns.jsonl"
    if not turns_path.exists():
        return []
    turns: list[dict[str, object]] = []
    for index, line in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ChatSessionError(f"invalid turns.jsonl line {index}") from error
        if not isinstance(record, dict):
            raise ChatSessionError(f"invalid turns.jsonl line {index}: must contain an object")
        for field_name in ["turn_index", "request", "summary", "status", "episode_path", "verification_status"]:
            if field_name not in record:
                raise ChatSessionError(f"invalid turns.jsonl line {index}: missing {field_name}")
        if not isinstance(record["turn_index"], int) or isinstance(record["turn_index"], bool):
            raise ChatSessionError(f"invalid turns.jsonl line {index}: turn_index must be an integer")
        for field_name in ["request", "summary", "status", "episode_path", "verification_status"]:
            if not isinstance(record[field_name], str):
                raise ChatSessionError(f"invalid turns.jsonl line {index}: {field_name} must be a string")
        turns.append(record)
    return turns


def _read_manual_compaction_state(session_path: Path) -> tuple[str | None, int]:
    state_path = session_path / "session_memory.json"
    if not state_path.exists():
        return None, 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ChatSessionError("invalid session_memory.json") from error
    if not isinstance(state, dict):
        raise ChatSessionError("invalid session_memory.json: must contain an object")
    summary = state.get("summary")
    compacted_turn_count = state.get("compacted_turn_count")
    if not isinstance(summary, str):
        raise ChatSessionError("invalid session_memory.json: summary must be a string")
    if not isinstance(compacted_turn_count, int) or isinstance(compacted_turn_count, bool):
        raise ChatSessionError("invalid session_memory.json: compacted_turn_count must be an integer")
    return summary, max(0, compacted_turn_count)


def _manual_compaction_summary_text(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith("Full Compact Summary:"):
            return content
    return None


def _session_turn_summary(record: dict[str, object]) -> SessionTurnSummary:
    return SessionTurnSummary(
        turn_index=int(record["turn_index"]),
        request=str(record["request"]),
        summary=str(record["summary"]),
        status=str(record["status"]),
        episode_path=Path(str(record["episode_path"])),
        verification_status=str(record["verification_status"]),
    )


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:8]
