"""
src/haagent/runtime/session/lifecycle.py - AgentSession 创建/恢复/重置装配

SessionSnapshot：可序列化、版本化的 package 状态。
SessionResources：gateway/MCP/callback 等不可序列化运行资源。
AgentSession 只持有二者，不再逐字段镜像。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.settings import load_mcp_settings
from haagent.mcp.tool_adapter import mcp_tool_alias, mcp_tool_definitions
from haagent.models.types import ModelGateway
from haagent.models.model_ref import ModelRef
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.human_interaction_resolver import SessionInteractionState
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, load_path_policy
from haagent.runtime.session.attachments import ImageAttachment
from haagent.runtime.session.package import (
    ChatSessionError,
    new_session_id,
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

# 磁盘 session package 的逻辑 schema；迁移只经 SessionSnapshot 入口。
SESSION_SNAPSHOT_SCHEMA_VERSION = 1
# 无 session_snapshot_schema_version 字段的旧 package 视为 v0。
SESSION_SNAPSHOT_LEGACY_SCHEMA_VERSION = 0


def resolve_session_snapshot_schema_version(metadata: dict[str, object]) -> int:
    """从 session.json 读取 schema 版本；缺失=0，未来版本拒绝。"""
    raw = metadata.get("session_snapshot_schema_version")
    if raw is None:
        return SESSION_SNAPSHOT_LEGACY_SCHEMA_VERSION
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ChatSessionError(
            "invalid session.json: session_snapshot_schema_version must be an integer"
        )
    if raw < SESSION_SNAPSHOT_LEGACY_SCHEMA_VERSION:
        raise ChatSessionError(
            f"invalid session.json: session_snapshot_schema_version is invalid: {raw}"
        )
    if raw > SESSION_SNAPSHOT_SCHEMA_VERSION:
        raise ChatSessionError(
            f"unsupported session_snapshot_schema_version: {raw} "
            f"(supported up to {SESSION_SNAPSHOT_SCHEMA_VERSION})"
        )
    return raw


def migrate_session_snapshot_schema_version(disk_version: int) -> int:
    """把磁盘版本迁到当前逻辑版本；当前仅支持 0→1 与 1 保持。"""
    if disk_version == SESSION_SNAPSHOT_LEGACY_SCHEMA_VERSION:
        # 旧 package 字段布局已兼容 v1，显式升级而非假装磁盘本就是最新版。
        return SESSION_SNAPSHOT_SCHEMA_VERSION
    if disk_version == SESSION_SNAPSHOT_SCHEMA_VERSION:
        return disk_version
    raise ChatSessionError(
        f"unsupported session_snapshot_schema_version: {disk_version} "
        f"(supported up to {SESSION_SNAPSHOT_SCHEMA_VERSION})"
    )


@dataclass
class SessionSnapshot:
    """可持久化会话状态；不包含 gateway/MCP/callback。"""

    schema_version: int
    workspace_root: Path
    path_policy: PathPolicy
    runs_root: Path
    model_ref: ModelRef | None
    enable_web: bool
    session_id: str
    turn_count: int
    summaries: list[str]
    # 完整 turn 记录缓存；供 UI history 复用，避免 resume 后再读 turns.jsonl。
    turn_records: list[dict[str, object]]
    manual_compaction_summary: str | None
    manual_compaction_turn_count: int
    last_user_image_attachments: list[ImageAttachment]
    image_attachment_history: list[ImageAttachment]
    working_state: WorkingState
    task_ledger: TaskLedger
    session_path: Path
    created_at: str
    session_interaction_state: SessionInteractionState

    def clone(self) -> SessionSnapshot:
        """浅拷贝容器字段，避免 new/reload 共享可变 list。"""
        return replace(
            self,
            summaries=list(self.summaries),
            turn_records=list(self.turn_records),
            last_user_image_attachments=list(self.last_user_image_attachments),
            image_attachment_history=list(self.image_attachment_history),
        )


@dataclass
class SessionResources:
    """不可序列化运行资源与进程内注入项。"""

    model_gateway: ModelGateway | None
    max_turns: int | None
    memory_extraction_enabled: bool
    allowed_tools_override: list[str] | None
    approval_allowed_tools_override: list[str] | None
    approved_tools_override: list[str] | None
    worker_context: dict[str, object] | None
    worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None
    next_turn_target_paths: list[str]
    historical_tool_compression_count: int
    current_cancellation_token: CancellationToken | None
    mcp_settings: Any
    mcp_runtime: Any
    owns_mcp_runtime: bool
    mcp_tool_names: list[str]
    tool_registry: Any

    def clone(self) -> SessionResources:
        return replace(
            self,
            allowed_tools_override=(
                list(self.allowed_tools_override)
                if self.allowed_tools_override is not None
                else None
            ),
            approval_allowed_tools_override=(
                list(self.approval_allowed_tools_override)
                if self.approval_allowed_tools_override is not None
                else None
            ),
            approved_tools_override=(
                list(self.approved_tools_override)
                if self.approved_tools_override is not None
                else None
            ),
            worker_context=dict(self.worker_context) if self.worker_context is not None else None,
            next_turn_target_paths=list(self.next_turn_target_paths),
            mcp_tool_names=list(self.mcp_tool_names),
        )


@dataclass
class SessionRuntimeState:
    """create/resume/new 装配结果：snapshot + resources。"""

    snapshot: SessionSnapshot
    resources: SessionResources


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
    model_ref: ModelRef | None = None,
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
    snapshot = SessionSnapshot(
        schema_version=SESSION_SNAPSHOT_SCHEMA_VERSION,
        workspace_root=resolved_workspace,
        path_policy=default_path_policy(resolved_workspace),
        runs_root=runs_root,
        model_ref=model_ref,
        enable_web=enable_web,
        session_id=sid,
        turn_count=0,
        summaries=[],
        turn_records=[],
        manual_compaction_summary=None,
        manual_compaction_turn_count=0,
        last_user_image_attachments=[],
        image_attachment_history=[],
        working_state=empty_working_state(),
        task_ledger=empty_task_ledger(),
        session_path=runs_root / "sessions" / sid,
        created_at=datetime.now(UTC).isoformat(),
        session_interaction_state=SessionInteractionState(),
    )
    resources = SessionResources(
        model_gateway=model_gateway,
        max_turns=max_turns,
        memory_extraction_enabled=memory_extraction_enabled,
        allowed_tools_override=list(allowed_tools_override) if allowed_tools_override is not None else None,
        approval_allowed_tools_override=(
            list(approval_allowed_tools_override)
            if approval_allowed_tools_override is not None
            else None
        ),
        approved_tools_override=list(approved_tools_override) if approved_tools_override is not None else None,
        worker_context=dict(worker_context) if worker_context is not None else None,
        worker_permission_requester=worker_permission_requester,
        next_turn_target_paths=[],
        historical_tool_compression_count=0,
        current_cancellation_token=None,
        mcp_settings=mcp_settings,
        mcp_runtime=runtime,
        owns_mcp_runtime=owns,
        mcp_tool_names=mcp_tool_names,
        tool_registry=tool_registry,
    )
    return SessionRuntimeState(snapshot=snapshot, resources=resources)


def build_resume_state(
    session: str | Path,
    *,
    runs_root: Path | None = None,
    model_gateway: ModelGateway | None = None,
    model_ref: ModelRef | None = None,
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
    disk_schema_version = resolve_session_snapshot_schema_version(metadata)
    schema_version = migrate_session_snapshot_schema_version(disk_schema_version)
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
    snapshot = SessionSnapshot(
        schema_version=schema_version,
        workspace_root=workspace_root,
        path_policy=path_policy,
        runs_root=session_path.parent.parent,
        model_ref=(
            model_ref
            if model_ref is not None
            else ModelRef.from_dict(metadata["model_ref"])
            if isinstance(metadata.get("model_ref"), dict)
            else None
        ),
        enable_web=enable_web,
        session_id=str(metadata["session_id"]),
        turn_count=int(metadata["turn_count"]),
        summaries=[str(turn["summary"]) for turn in turns],
        turn_records=list(turns),
        manual_compaction_summary=compaction_summary,
        manual_compaction_turn_count=compacted_turn_count,
        last_user_image_attachments=read_session_image_attachments(metadata, session_path),
        image_attachment_history=read_image_attachment_history(metadata, session_path),
        working_state=working_state,
        task_ledger=task_ledger,
        session_path=session_path,
        created_at=str(metadata["created_at"]),
        # resume 同一 session：恢复 edit_diff 始终允许；缺失字段视为 False
        session_interaction_state=SessionInteractionState(
            edit_diff_session_always=bool(metadata.get("edit_diff_session_always", False)),
        ),
    )
    resources = SessionResources(
        model_gateway=model_gateway,
        max_turns=max_turns,
        memory_extraction_enabled=True,
        allowed_tools_override=None,
        approval_allowed_tools_override=None,
        approved_tools_override=None,
        worker_context=None,
        worker_permission_requester=None,
        next_turn_target_paths=[],
        historical_tool_compression_count=0,
        current_cancellation_token=None,
        mcp_settings=settings,
        mcp_runtime=runtime,
        owns_mcp_runtime=owns,
        mcp_tool_names=names,
        tool_registry=registry,
    )
    return SessionRuntimeState(snapshot=snapshot, resources=resources)


def build_new_package_state(state: SessionRuntimeState) -> SessionRuntimeState:
    """在保留 workspace/model/MCP 的前提下重置为新 session package。"""
    sid = new_session_id()
    snap = state.snapshot
    res = state.resources
    snapshot = SessionSnapshot(
        schema_version=SESSION_SNAPSHOT_SCHEMA_VERSION,
        workspace_root=snap.workspace_root,
        path_policy=default_path_policy(snap.workspace_root),
        runs_root=snap.runs_root,
        model_ref=snap.model_ref,
        enable_web=snap.enable_web,
        session_id=sid,
        turn_count=0,
        summaries=[],
        turn_records=[],
        manual_compaction_summary=None,
        manual_compaction_turn_count=0,
        last_user_image_attachments=[],
        image_attachment_history=[],
        working_state=empty_working_state(),
        task_ledger=empty_task_ledger(),
        session_path=snap.runs_root / "sessions" / sid,
        created_at=datetime.now(UTC).isoformat(),
        # 新 session 不继承上一会话的 edit_diff always
        session_interaction_state=SessionInteractionState(),
    )
    resources = SessionResources(
        model_gateway=res.model_gateway,
        max_turns=res.max_turns,
        memory_extraction_enabled=res.memory_extraction_enabled,
        allowed_tools_override=(
            list(res.allowed_tools_override) if res.allowed_tools_override is not None else None
        ),
        approval_allowed_tools_override=(
            list(res.approval_allowed_tools_override)
            if res.approval_allowed_tools_override is not None
            else None
        ),
        approved_tools_override=(
            list(res.approved_tools_override) if res.approved_tools_override is not None else None
        ),
        worker_context=dict(res.worker_context) if res.worker_context is not None else None,
        worker_permission_requester=res.worker_permission_requester,
        next_turn_target_paths=list(res.next_turn_target_paths),
        historical_tool_compression_count=res.historical_tool_compression_count,
        current_cancellation_token=res.current_cancellation_token,
        mcp_settings=res.mcp_settings,
        mcp_runtime=res.mcp_runtime,
        owns_mcp_runtime=res.owns_mcp_runtime,
        mcp_tool_names=list(res.mcp_tool_names),
        tool_registry=res.tool_registry,
    )
    return SessionRuntimeState(snapshot=snapshot, resources=resources)


def apply_state(instance: Any, state: SessionRuntimeState) -> None:
    """把 snapshot/resources 装入 AgentSession；不再逐字段镜像。"""
    instance._snapshot = state.snapshot
    instance._resources = state.resources
