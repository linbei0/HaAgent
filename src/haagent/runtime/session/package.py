"""
src/haagent/runtime/session/package.py - session package 磁盘读写

负责 session 目录下 metadata、turns、图片附件与手动压缩状态的读写，
以及会话列表/查找。不承担 AgentSession 运行时编排。
"""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from haagent.runtime.session.attachments import ImageAttachment
from haagent.runtime.execution.path_policy import PathPolicy, serialize_path_policy
from haagent.models.model_ref import ModelRef

ASSISTANT_DISPLAY_TEXT_CHAR_LIMIT = 4000


class ChatSessionError(RuntimeError):
    """Chat session package 损坏或无法恢复时抛出。"""


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
    assistant_display_text: str | None = None


def resolve_session_path(session: str | Path, runs_root: Path) -> Path:
    raw = Path(session)
    if raw.is_absolute() or raw.exists() or raw.name != str(session):
        return raw.resolve()
    sessions_root = runs_root / "sessions"
    locator = _session_locator_path(sessions_root, str(session))
    if not locator.exists():
        raise ChatSessionError(f"session not found: {session}")
    relative_path = locator.read_text(encoding="utf-8").strip()
    candidate = (sessions_root / relative_path).resolve()
    try:
        candidate.relative_to(sessions_root.resolve())
    except ValueError as error:
        raise ChatSessionError(f"invalid session locator: {locator}") from error
    if not candidate.is_dir():
        raise ChatSessionError(f"session path missing: {candidate}")
    return candidate


def peek_first_turn_request(session_path: Path) -> str:
    """只读 turns.jsonl 首行 request，避免列表热路径解析整文件。"""
    turns_path = session_path / "turns.jsonl"
    if not turns_path.exists():
        return "none"
    try:
        with turns_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ChatSessionError("invalid turns.jsonl line 1") from error
                if not isinstance(record, dict):
                    raise ChatSessionError("invalid turns.jsonl line 1: must contain an object")
                request = record.get("request")
                if not isinstance(request, str):
                    raise ChatSessionError("invalid turns.jsonl line 1: request must be a string")
                return request
    except OSError as error:
        raise ChatSessionError(f"unable to read turns.jsonl: {turns_path}") from error
    return "none"


def list_sessions(runs_root: Path, workspace_root: Path) -> list[SessionSummary]:
    """列出当前 workspace 下的 chat 会话摘要。"""
    sessions_root = runs_root / "sessions"
    if not sessions_root.exists():
        return []
    resolved_workspace = workspace_root.resolve()
    summaries: list[SessionSummary] = []
    for metadata_path in sessions_root.glob("*/*/*/*/session.json"):
        session_path = metadata_path.parent
        metadata = read_session_metadata(session_path)
        if Path(str(metadata["workspace_root"])).resolve() != resolved_workspace:
            continue
        # 优先 session.json 预览；缺省/"none" 且有轮次时再 peek（兼容旧 package）。
        raw_first = metadata.get("first_request")
        turn_count = int(metadata["turn_count"])
        if isinstance(raw_first, str) and raw_first and not (raw_first == "none" and turn_count > 0):
            first_request = raw_first
        else:
            first_request = peek_first_turn_request(session_path)
        summaries.append(
            SessionSummary(
                session_id=str(metadata["session_id"]),
                created_at=str(metadata["created_at"]),
                updated_at=str(metadata["updated_at"]),
                workspace_root=resolved_workspace,
                turn_count=turn_count,
                first_request=first_request,
                session_path=session_path.resolve(),
            ),
        )
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)


def find_latest_session(runs_root: Path, workspace_root: Path) -> SessionSummary | None:
    sessions = list_sessions(runs_root, workspace_root)
    return sessions[0] if sessions else None


def read_session_metadata(session_path: Path) -> dict[str, object]:
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


def read_session_image_attachments(
    metadata: dict[str, object],
    session_path: Path,
) -> list[ImageAttachment]:
    raw_attachments = metadata.get("last_user_image_attachments")
    if raw_attachments is None:
        return []
    if not isinstance(raw_attachments, list):
        raise ChatSessionError("invalid session.json: last_user_image_attachments must be a list")
    attachments: list[ImageAttachment] = []
    for index, raw_attachment in enumerate(raw_attachments, start=1):
        if not isinstance(raw_attachment, dict):
            raise ChatSessionError(
                f"invalid session.json: last_user_image_attachments[{index}] must be an object"
            )
        try:
            attachment = ImageAttachment.from_dict(raw_attachment).with_base_path(session_path)
        except ValueError as error:
            raise ChatSessionError(
                f"invalid session.json: last_user_image_attachments[{index}]: {error}"
            ) from error
        attachments.append(attachment)
    return attachments


def read_image_attachment_history(
    metadata: dict[str, object],
    session_path: Path,
) -> list[ImageAttachment]:
    raw_attachments = metadata.get("image_attachment_history")
    if raw_attachments is None:
        return list(read_session_image_attachments(metadata, session_path))
    if not isinstance(raw_attachments, list):
        raise ChatSessionError("invalid session.json: image_attachment_history must be a list")
    attachments: list[ImageAttachment] = []
    for index, raw_attachment in enumerate(raw_attachments, start=1):
        if not isinstance(raw_attachment, dict):
            raise ChatSessionError(
                f"invalid session.json: image_attachment_history[{index}] must be an object"
            )
        try:
            attachment = ImageAttachment.from_dict(raw_attachment).with_base_path(session_path)
        except ValueError as error:
            raise ChatSessionError(
                f"invalid session.json: image_attachment_history[{index}]: {error}"
            ) from error
        attachments.append(attachment)
    return attachments


def merge_image_attachment_history(
    existing: list[ImageAttachment],
    new_attachments: list[ImageAttachment],
) -> list[ImageAttachment]:
    by_id = {attachment.id: attachment for attachment in existing}
    for attachment in new_attachments:
        by_id[attachment.id] = attachment
    return list(by_id.values())


def read_session_turns(session_path: Path) -> list[dict[str, object]]:
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
        if "assistant_display_text" in record and not isinstance(record["assistant_display_text"], str):
            raise ChatSessionError(f"invalid turns.jsonl line {index}: assistant_display_text must be a string")
        turns.append(record)
    return turns


def read_manual_compaction_state(session_path: Path) -> tuple[str | None, int]:
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


def write_manual_compaction_state(
    session_path: Path,
    *,
    summary: str | None,
    compacted_turn_count: int,
) -> None:
    state_path = session_path / "session_memory.json"
    if summary is None:
        if state_path.exists():
            state_path.unlink()
        return
    session_path.mkdir(parents=True, exist_ok=True)
    state = {
        "summary": summary,
        "compacted_turn_count": compacted_turn_count,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_session_metadata(
    session_path: Path,
    *,
    session_id: str,
    workspace_root: Path,
    path_policy: PathPolicy,
    provider: str,
    model_ref: ModelRef | None,
    enable_web: bool,
    last_user_image_attachments: list[ImageAttachment],
    image_attachment_history: list[ImageAttachment],
    created_at: str,
    turn_count: int,
    edit_diff_session_always: bool = False,
    permission_rules: list[dict[str, str]] | None = None,
    first_request: str | None = None,
    session_snapshot_schema_version: int | None = None,
) -> str:
    """写入 session.json；返回实际保留的 created_at。"""
    # 延迟导入避免 package ↔ lifecycle 循环依赖。
    from haagent.runtime.session.lifecycle import SESSION_SNAPSHOT_SCHEMA_VERSION

    schema_version = (
        SESSION_SNAPSHOT_SCHEMA_VERSION
        if session_snapshot_schema_version is None
        else session_snapshot_schema_version
    )
    session_path.mkdir(parents=True, exist_ok=True)
    metadata_path = session_path / "session.json"
    effective_created_at = created_at
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
            effective_created_at = str(existing["created_at"])
    metadata = {
        "session_id": session_id,
        "workspace_root": str(workspace_root),
        "path_policy": serialize_path_policy(path_policy),
        # 仅布尔标志，不保存完整 diff；新 session 默认 False
        "edit_diff_session_always": bool(edit_diff_session_always),
        # 仅保存用户选择“始终允许”的结构化权限模式，不保存一次性批准。
        "permission_rules": list(permission_rules or []),
        # 持久化 SessionSnapshot 逻辑版本；resume 据此迁移/拒绝未知版本。
        "session_snapshot_schema_version": schema_version,
        "provider": provider,
        "model_ref": model_ref.to_dict() if model_ref is not None else None,
        "enable_web": enable_web,
        "last_user_image_attachments": [
            attachment.to_dict() for attachment in last_user_image_attachments
        ],
        "image_attachment_history": [
            attachment.to_dict() for attachment in image_attachment_history
        ],
        "created_at": effective_created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "turn_count": turn_count,
        # 列表预览字段；旧 package 无此键时 list_sessions 回退 peek turns。
        "first_request": first_request if first_request is not None else "none",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_session_locator(session_path, session_id)
    return effective_created_at


def append_turn_record(
    session_path: Path,
    *,
    turn_index: int,
    request: str,
    summary: str,
    status: str,
    episode_path: Path,
    verification_status: str,
    final_response: str,
) -> None:
    from haagent.runtime.session.turn import summary_value

    session_path.mkdir(parents=True, exist_ok=True)
    record = {
        "turn_index": turn_index,
        "request": summary_value(request, 300),
        "summary": summary,
        "status": status,
        "episode_path": str(episode_path),
        "verification_status": verification_status,
        "assistant_display_text": assistant_display_text(final_response),
    }
    with (session_path / "turns.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def session_turn_summary(record: dict[str, object]) -> SessionTurnSummary:
    assistant_text = record.get("assistant_display_text")
    return SessionTurnSummary(
        turn_index=int(record["turn_index"]),
        request=str(record["request"]),
        summary=str(record["summary"]),
        status=str(record["status"]),
        episode_path=Path(str(record["episode_path"])),
        verification_status=str(record["verification_status"]),
        assistant_display_text=assistant_text if isinstance(assistant_text, str) else None,
    )


def assistant_display_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(normalized) <= ASSISTANT_DISPLAY_TEXT_CHAR_LIMIT:
        return normalized
    return normalized[:ASSISTANT_DISPLAY_TEXT_CHAR_LIMIT] + "... [truncated]"


def optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:8]


def _write_session_locator(session_path: Path, session_id: str) -> None:
    sessions_root = _sessions_root_for_path(session_path)
    locator = _session_locator_path(sessions_root, session_id)
    if locator.exists():
        return
    locator.parent.mkdir(parents=True, exist_ok=True)
    locator.write_text(str(session_path.resolve().relative_to(sessions_root.resolve())), encoding="utf-8")


def _session_locator_path(sessions_root: Path, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return sessions_root / "by-id" / digest[:2] / f"{session_id}.json"


def _sessions_root_for_path(session_path: Path) -> Path:
    for parent in session_path.parents:
        if parent.name == "sessions":
            return parent
    raise ChatSessionError(f"invalid session path outside runs/sessions: {session_path}")


def manual_compaction_summary_text(messages: list[dict[str, object]]) -> str | None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith("Full Compact Summary:"):
            return content
    return None
