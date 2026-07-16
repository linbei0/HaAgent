"""
src/haagent/runtime/session/lifecycle.py - AgentSession 创建/恢复/重置字段装配

统一 init 与 resume 的字段与 MCP/tool registry 装配，避免双路径漂移。
行为与历史 AgentSession 保持一致（含 resume 不恢复 worker override 等既有约定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.settings import load_mcp_settings
from haagent.mcp.tool_adapter import mcp_tool_alias, mcp_tool_definitions
from haagent.models.types import ModelGateway
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.human_interaction_resolver import SessionInteractionState
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, load_path_policy
from haagent.runtime.session.attachments import ImageAttachment
from haagent.runtime.session.package import (
    ChatSessionError,
    new_session_id,
    optional_string,
    read_image_attachment_history,
    read_manual_compaction_state,
    read_session_image_attachments,
    read_session_metadata,
    read_session_turns,
    resolve_session_path,
)
from haagent.runtime.session.task_ledger import (
    TaskLedger,
    TaskLedgerError,
    empty_task_ledger,
    load_task_ledger,
)
from haagent.runtime.session.working_state import (
    WorkingState,
    WorkingStateError,
    empty_working_state,
    load_working_state,
)
from haagent.tools.registry import default_tool_runtime_registry


MODEL_VARIANT_UNSET = object()


@dataclass
class SessionRuntimeState:
    """AgentSession 的可序列化/可装配运行时字段快照。"""

    workspace_root: Path
    path_policy: PathPolicy
    runs_root: Path
    model_gateway: ModelGateway | None
    model_profile_name: str | None
    model_connection_id: str | None
    model_name: str | None
    model_base_url: str | None
    model_variant: str | None
    max_turns: int | None
    memory_extraction_enabled: bool
    enable_web: bool
    allowed_tools_override: list[str] | None
    approval_allowed_tools_override: list[str] | None
    approved_tools_override: list[str] | None
    worker_context: dict[str, object] | None
    worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None
    session_id: str
    turn_count: int
    summaries: list[str]
    # 完整 turn 记录缓存；供 UI history 复用，避免 resume 后再读 turns.jsonl。
    turn_records: list[dict[str, object]]
    manual_compaction_summary: str | None
    manual_compaction_turn_count: int
    next_turn_target_paths: list[str]
    last_user_image_attachments: list[ImageAttachment]
    image_attachment_history: list[ImageAttachment]
    historical_tool_compression_count: int
    working_state: WorkingState
    task_ledger: TaskLedger
    current_cancellation_token: CancellationToken | None
    mcp_settings: Any
    mcp_runtime: Any
    owns_mcp_runtime: bool
    mcp_tool_names: list[str]
    tool_registry: Any
    session_path: Path
    created_at: str
    session_interaction_state: SessionInteractionState


def bootstrap_mcp(mcp_runtime: Any | None = None) -> tuple[Any, Any, bool, list[str], Any]:
    """启动或接管 MCP，返回 (settings, runtime, owns, tool_names, tool_registry)。"""
    if mcp_runtime is None:
        mcp_settings = load_mcp_settings()
        runtime = SyncMcpRuntime(mcp_settings)
        runtime.start()
        owns = True
    else:
        runtime = mcp_runtime
        mcp_settings = runtime.settings
        owns = False
    mcp_tool_names = [
        mcp_tool_alias(tool.server_name, tool.name)
        for tool in runtime.list_tools()
    ]
    tool_registry = default_tool_runtime_registry(
        mcp_tool_definitions(runtime.list_tools()),
    )
    return mcp_settings, runtime, owns, mcp_tool_names, tool_registry


def build_create_state(
    *,
    workspace_root: Path,
    runs_root: Path,
    model_gateway: ModelGateway | None = None,
    model_profile_name: str | None = None,
    model_connection_id: str | None = None,
    model_name: str | None = None,
    model_base_url: str | None = None,
    model_variant: str | None = None,
    max_turns: int | None,
    session_id: str | None = None,
    memory_extraction_enabled: bool = True,
    enable_web: bool = False,
    allowed_tools_override: list[str] | None = None,
    approval_allowed_tools_override: list[str] | None = None,
    approved_tools_override: list[str] | None = None,
    mcp_runtime: Any | None = None,
    worker_context: dict[str, object] | None = None,
    worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None,
) -> SessionRuntimeState:
    resolved_workspace = workspace_root.resolve()
    sid = session_id or new_session_id()
    mcp_settings, runtime, owns, mcp_tool_names, tool_registry = bootstrap_mcp(mcp_runtime)
    return SessionRuntimeState(
        workspace_root=resolved_workspace,
        path_policy=default_path_policy(resolved_workspace),
        runs_root=runs_root,
        model_gateway=model_gateway,
        model_profile_name=model_profile_name,
        model_connection_id=model_connection_id,
        model_name=model_name,
        model_base_url=model_base_url,
        model_variant=model_variant,
        max_turns=max_turns,
        memory_extraction_enabled=memory_extraction_enabled,
        enable_web=enable_web,
        allowed_tools_override=list(allowed_tools_override) if allowed_tools_override is not None else None,
        approval_allowed_tools_override=(
            list(approval_allowed_tools_override)
            if approval_allowed_tools_override is not None
            else None
        ),
        approved_tools_override=list(approved_tools_override) if approved_tools_override is not None else None,
        worker_context=dict(worker_context) if worker_context is not None else None,
        worker_permission_requester=worker_permission_requester,
        session_id=sid,
        turn_count=0,
        summaries=[],
        turn_records=[],
        manual_compaction_summary=None,
        manual_compaction_turn_count=0,
        next_turn_target_paths=[],
        last_user_image_attachments=[],
        image_attachment_history=[],
        historical_tool_compression_count=0,
        working_state=empty_working_state(),
        task_ledger=empty_task_ledger(),
        current_cancellation_token=None,
        mcp_settings=mcp_settings,
        mcp_runtime=runtime,
        owns_mcp_runtime=owns,
        mcp_tool_names=mcp_tool_names,
        tool_registry=tool_registry,
        session_path=runs_root / "sessions" / sid,
        created_at=datetime.now(UTC).isoformat(),
        session_interaction_state=SessionInteractionState(),
    )


def build_resume_state(
    session: str | Path,
    *,
    runs_root: Path | None = None,
    model_gateway: ModelGateway | None = None,
    model_profile_name: str | None = None,
    model_connection_id: str | None = None,
    model_name: str | None = None,
    model_base_url: str | None = None,
    model_variant: str | None | object = MODEL_VARIANT_UNSET,
    max_turns: int | None,
    enable_web: bool = False,
    mcp_runtime: Any | None = None,
    tool_registry: Any | None = None,
    mcp_settings: Any | None = None,
    mcp_tool_names: list[str] | None = None,
    owns_mcp_runtime: bool | None = None,
) -> SessionRuntimeState:
    session_path = resolve_session_path(session, runs_root or Path(".runs"))
    metadata = read_session_metadata(session_path)
    turns = read_session_turns(session_path)
    workspace_root = Path(str(metadata["workspace_root"])).resolve()
    raw_policy = metadata.get("path_policy")
    path_policy = (
        load_path_policy(raw_policy)
        if isinstance(raw_policy, dict)
        else default_path_policy(workspace_root)
    )
    compaction_summary, compacted_turn_count = read_manual_compaction_state(session_path)
    try:
        working_state = load_working_state(session_path / "working_state.json")
    except WorkingStateError as error:
        raise ChatSessionError(str(error)) from error
    try:
        task_ledger = load_task_ledger(session_path / "task-ledger.json")
    except TaskLedgerError as error:
        raise ChatSessionError(str(error)) from error
    # 有现成 MCP 时复用（会话切换热路径）；否则自建。不恢复 worker/tool override。
    if mcp_runtime is not None:
        settings = mcp_settings if mcp_settings is not None else mcp_runtime.settings
        names = (
            list(mcp_tool_names)
            if mcp_tool_names is not None
            else [
                mcp_tool_alias(tool.server_name, tool.name)
                for tool in mcp_runtime.list_tools()
            ]
        )
        registry = (
            tool_registry
            if tool_registry is not None
            else default_tool_runtime_registry(mcp_tool_definitions(mcp_runtime.list_tools()))
        )
        runtime = mcp_runtime
        owns = True if owns_mcp_runtime is None else bool(owns_mcp_runtime)
    else:
        settings, runtime, owns, names, registry = bootstrap_mcp(None)
    return SessionRuntimeState(
        workspace_root=workspace_root,
        path_policy=path_policy,
        runs_root=session_path.parent.parent,
        model_gateway=model_gateway,
        model_profile_name=model_profile_name or optional_string(metadata.get("model_profile_name")),
        model_connection_id=model_connection_id or optional_string(metadata.get("model_connection_id")),
        model_name=model_name or optional_string(metadata.get("model")),
        model_base_url=model_base_url or optional_string(metadata.get("base_url")),
        model_variant=(
            optional_string(metadata["model_variant"])
            if model_variant is MODEL_VARIANT_UNSET
            else model_variant
        ),
        max_turns=max_turns,
        memory_extraction_enabled=True,
        enable_web=enable_web,
        allowed_tools_override=None,
        approval_allowed_tools_override=None,
        approved_tools_override=None,
        worker_context=None,
        worker_permission_requester=None,
        session_id=str(metadata["session_id"]),
        turn_count=int(metadata["turn_count"]),
        summaries=[str(turn["summary"]) for turn in turns],
        turn_records=list(turns),
        manual_compaction_summary=compaction_summary,
        manual_compaction_turn_count=compacted_turn_count,
        next_turn_target_paths=[],
        last_user_image_attachments=read_session_image_attachments(metadata, session_path),
        image_attachment_history=read_image_attachment_history(metadata, session_path),
        historical_tool_compression_count=0,
        working_state=working_state,
        task_ledger=task_ledger,
        current_cancellation_token=None,
        mcp_settings=settings,
        mcp_runtime=runtime,
        owns_mcp_runtime=owns,
        mcp_tool_names=names,
        tool_registry=registry,
        session_path=session_path,
        created_at=str(metadata["created_at"]),
        # resume 同一 session：恢复 edit_diff 始终允许；缺失字段视为 False
        session_interaction_state=SessionInteractionState(
            edit_diff_session_always=bool(metadata.get("edit_diff_session_always", False)),
        ),
    )


def build_new_package_state(state: SessionRuntimeState) -> SessionRuntimeState:
    """在保留 workspace/model/MCP 的前提下重置为新 session package。"""
    sid = new_session_id()
    return SessionRuntimeState(
        workspace_root=state.workspace_root,
        path_policy=default_path_policy(state.workspace_root),
        runs_root=state.runs_root,
        model_gateway=state.model_gateway,
        model_profile_name=state.model_profile_name,
        model_connection_id=state.model_connection_id,
        model_name=state.model_name,
        model_base_url=state.model_base_url,
        model_variant=state.model_variant,
        max_turns=state.max_turns,
        memory_extraction_enabled=state.memory_extraction_enabled,
        enable_web=state.enable_web,
        allowed_tools_override=state.allowed_tools_override,
        approval_allowed_tools_override=state.approval_allowed_tools_override,
        approved_tools_override=state.approved_tools_override,
        worker_context=state.worker_context,
        worker_permission_requester=state.worker_permission_requester,
        session_id=sid,
        turn_count=0,
        summaries=[],
        turn_records=[],
        manual_compaction_summary=None,
        manual_compaction_turn_count=0,
        next_turn_target_paths=state.next_turn_target_paths,
        last_user_image_attachments=[],
        image_attachment_history=[],
        historical_tool_compression_count=state.historical_tool_compression_count,
        working_state=empty_working_state(),
        task_ledger=empty_task_ledger(),
        current_cancellation_token=state.current_cancellation_token,
        mcp_settings=state.mcp_settings,
        mcp_runtime=state.mcp_runtime,
        owns_mcp_runtime=state.owns_mcp_runtime,
        mcp_tool_names=state.mcp_tool_names,
        tool_registry=state.tool_registry,
        session_path=state.runs_root / "sessions" / sid,
        created_at=datetime.now(UTC).isoformat(),
        # 新 session 不继承上一会话的 edit_diff always
        session_interaction_state=SessionInteractionState(),
    )


def apply_state(instance: Any, state: SessionRuntimeState) -> None:
    """把 SessionRuntimeState 写入 AgentSession 实例字段（含私有字段名）。"""
    instance.workspace_root = state.workspace_root
    instance.path_policy = state.path_policy
    instance.runs_root = state.runs_root
    instance.model_gateway = state.model_gateway
    instance.model_profile_name = state.model_profile_name
    instance.model_connection_id = state.model_connection_id
    instance.model_name = state.model_name
    instance.model_base_url = state.model_base_url
    instance.model_variant = state.model_variant
    instance.max_turns = state.max_turns
    instance.memory_extraction_enabled = state.memory_extraction_enabled
    instance.enable_web = state.enable_web
    instance._allowed_tools_override = state.allowed_tools_override
    instance._approval_allowed_tools_override = state.approval_allowed_tools_override
    instance._approved_tools_override = state.approved_tools_override
    instance._worker_context = state.worker_context
    instance._worker_permission_requester = state.worker_permission_requester
    instance.session_id = state.session_id
    instance.turn_count = state.turn_count
    instance._summaries = state.summaries
    instance._turn_records = list(state.turn_records)
    instance._manual_compaction_summary = state.manual_compaction_summary
    instance._manual_compaction_turn_count = state.manual_compaction_turn_count
    instance._next_turn_target_paths = state.next_turn_target_paths
    instance._last_user_image_attachments = state.last_user_image_attachments
    instance._image_attachment_history = state.image_attachment_history
    instance._historical_tool_compression_count = state.historical_tool_compression_count
    instance._working_state = state.working_state
    instance._task_ledger = state.task_ledger
    instance._current_cancellation_token = state.current_cancellation_token
    instance._mcp_settings = state.mcp_settings
    instance._mcp_runtime = state.mcp_runtime
    instance._owns_mcp_runtime = state.owns_mcp_runtime
    instance._mcp_tool_names = state.mcp_tool_names
    instance._tool_registry = state.tool_registry
    instance.session_path = state.session_path
    instance._created_at = state.created_at
    instance._session_interaction_state = state.session_interaction_state
