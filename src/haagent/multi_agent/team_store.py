"""
src/haagent/multi_agent/team_store.py - 用户级 team 与 mailbox 存储

以 UTF-8 JSON/JSONL 记录 worker 状态、消息和完成通知，供 TUI、inspect 与后续恢复审计使用。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any, Literal

from haagent.multi_agent.messages import WorkerMessage, WorkerPermissionRequest


WorkerStatus = Literal[
    "queued",
    "running",
    "idle",
    "awaiting_approval",
    "completed",
    "failed",
    "stopped",
    "interrupted",
]
_VISIBLE_NOTIFICATION_STATUSES = frozenset(
    {"completed", "failed", "stopped", "interrupted", "awaiting_approval"},
)
_NOTIFICATION_CURSOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkerRecord:
    agent_id: str
    task_id: str
    subagent_type: str
    description: str
    status: WorkerStatus
    session_id: str = ""
    episode_path: str = ""
    restart_count: int = 0
    status_note: str = ""
    profile: str = ""
    model_profile: str = ""
    parent_step_id: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    workspace_root: str
    leader_session_id: str
    active: bool = True
    agents: list[WorkerRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class TeamStore:
    _locks_guard = threading.Lock()
    _root_locks: dict[str, threading.RLock] = {}

    def __init__(self, root: Path) -> None:
        self.root = root
        self._message_sequence = count()
        key = str(root.resolve())
        with self._locks_guard:
            self._lock = self._root_locks.setdefault(key, threading.RLock())

    def ensure_team(
        self,
        *,
        team_id: str,
        workspace_root: Path,
        leader_session_id: str,
    ) -> TeamRecord:
        with self._lock:
            existing = self.load_team(team_id)
            if existing is not None:
                resolved_workspace = str(workspace_root.resolve())
                if (
                    existing.leader_session_id != leader_session_id
                    or existing.workspace_root != resolved_workspace
                ):
                    raise ValueError(f"team belongs to another session or workspace: {team_id}")
                if existing.active:
                    return existing
                reactivated = TeamRecord(
                    team_id=existing.team_id,
                    workspace_root=existing.workspace_root,
                    leader_session_id=existing.leader_session_id,
                    active=True,
                    agents=existing.agents,
                    created_at=existing.created_at,
                    updated_at=datetime.now(UTC).isoformat(),
                )
                self._write_team(reactivated)
                return reactivated
            team = TeamRecord(
                team_id=team_id,
                workspace_root=str(workspace_root.resolve()),
                leader_session_id=leader_session_id,
            )
            self._write_team(team)
            return team

    def load_team(self, team_id: str) -> TeamRecord | None:
        with self._lock:
            path = self._team_file(team_id)
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            return TeamRecord(
                team_id=raw["team_id"],
                workspace_root=raw["workspace_root"],
                leader_session_id=raw["leader_session_id"],
                active=bool(raw.get("active", True)),
                agents=[
                    WorkerRecord(
                        agent_id=item["agent_id"],
                        task_id=item["task_id"],
                        subagent_type=item["subagent_type"],
                        description=item["description"],
                        status=item["status"],
                        session_id=item.get("session_id", ""),
                        episode_path=item.get("episode_path", ""),
                        restart_count=int(item.get("restart_count", 0)),
                        status_note=item.get("status_note", ""),
                        profile=item.get("profile", ""),
                        model_profile=item.get("model_profile", ""),
                        parent_step_id=item.get("parent_step_id", ""),
                        updated_at=item.get("updated_at", ""),
                    )
                    for item in raw.get("agents", [])
                ],
                created_at=raw.get("created_at", ""),
                updated_at=raw.get("updated_at", ""),
            )

    def upsert_worker(self, team_id: str, worker: WorkerRecord) -> None:
        with self._lock:
            team = self._require_team(team_id)
            agents = [item for item in team.agents if item.agent_id != worker.agent_id]
            agents.append(worker)
            self._write_team(
                TeamRecord(
                    team_id=team.team_id,
                    workspace_root=team.workspace_root,
                    leader_session_id=team.leader_session_id,
                    active=team.active,
                    agents=agents,
                    created_at=team.created_at,
                    updated_at=datetime.now(UTC).isoformat(),
                ),
            )

    def update_worker_status(
        self,
        team_id: str,
        agent_id: str,
        status: WorkerStatus,
        *,
        episode_path: str = "",
        session_id: str = "",
        restart_count: int | None = None,
        status_note: str | None = None,
    ) -> None:
        with self._lock:
            self._update_worker_status_locked(
                team_id,
                agent_id,
                status,
                episode_path=episode_path,
                session_id=session_id,
                restart_count=restart_count,
                status_note=status_note,
            )

    def _update_worker_status_locked(
        self,
        team_id: str,
        agent_id: str,
        status: WorkerStatus,
        *,
        episode_path: str,
        session_id: str,
        restart_count: int | None,
        status_note: str | None,
    ) -> None:
        team = self._require_team(team_id)
        agents: list[WorkerRecord] = []
        found = False
        for worker in team.agents:
            if worker.agent_id != agent_id:
                agents.append(worker)
                continue
            found = True
            next_episode_path = episode_path or worker.episode_path
            agents.append(
                WorkerRecord(
                    agent_id=worker.agent_id,
                    task_id=worker.task_id,
                    subagent_type=worker.subagent_type,
                    description=worker.description,
                    status=status,
                    session_id=session_id or worker.session_id,
                    episode_path=next_episode_path,
                    restart_count=worker.restart_count if restart_count is None else restart_count,
                    status_note=worker.status_note if status_note is None else status_note,
                    profile=worker.profile,
                    model_profile=worker.model_profile,
                    parent_step_id=worker.parent_step_id,
                ),
            )
        if not found:
            raise ValueError(f"worker not found: {agent_id}")
        self._write_team(
            TeamRecord(
                team_id=team.team_id,
                workspace_root=team.workspace_root,
                leader_session_id=team.leader_session_id,
                active=team.active,
                agents=agents,
                created_at=team.created_at,
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )

    def mark_inactive(self, team_id: str) -> None:
        with self._lock:
            team = self._require_team(team_id)
            self._write_team(
                TeamRecord(
                    team_id=team.team_id,
                    workspace_root=team.workspace_root,
                    leader_session_id=team.leader_session_id,
                    active=False,
                    agents=team.agents,
                    created_at=team.created_at,
                    updated_at=datetime.now(UTC).isoformat(),
                ),
            )

    def write_worker_message(self, team_id: str, agent_id: str, message: WorkerMessage) -> Path:
        payload = message.to_dict()
        inbox = self._team_dir(team_id) / "agents" / _safe_id(agent_id) / "messages"
        inbox.mkdir(parents=True, exist_ok=True)
        # Windows 上连续创建消息可能得到相同时间戳，序号保留实际写入顺序。
        sequence = next(self._message_sequence)
        path = inbox / f"{payload['created_at']}-{sequence:020d}-{payload['message_id']}.json"
        return _atomic_write_json(path, payload)

    def read_worker_messages(self, team_id: str, agent_id: str) -> list[WorkerMessage]:
        inbox = self._team_dir(team_id) / "agents" / _safe_id(agent_id) / "messages"
        if not inbox.exists():
            return []
        messages = []
        for path in sorted(inbox.glob("*.json")):
            messages.append(WorkerMessage.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return messages

    def consume_worker_messages(self, team_id: str, agent_id: str) -> list[WorkerMessage]:
        inbox = self._team_dir(team_id) / "agents" / _safe_id(agent_id) / "messages"
        if not inbox.exists():
            return []
        messages: list[WorkerMessage] = []
        for path in sorted(inbox.glob("*.json")):
            messages.append(WorkerMessage.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            path.unlink(missing_ok=True)
        return messages

    def write_permission_request(self, request: WorkerPermissionRequest) -> Path:
        directory = self._team_dir(request.team_id) / "permissions" / request.status
        directory.mkdir(parents=True, exist_ok=True)
        return _atomic_write_json(directory / f"{_safe_id(request.request_id)}.json", request.to_dict())

    def read_permission_requests(
        self,
        team_id: str,
        *,
        status: str = "pending",
    ) -> list[WorkerPermissionRequest]:
        directory = self._team_dir(team_id) / "permissions" / _safe_id(status)
        if not directory.exists():
            return []
        requests = []
        for path in sorted(directory.glob("*.json")):
            requests.append(WorkerPermissionRequest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return requests

    def get_permission_request(
        self,
        request_id: str,
        *,
        statuses: tuple[str, ...] = ("pending",),
    ) -> WorkerPermissionRequest | None:
        with self._lock:
            found = self._find_permission_request(request_id, statuses=statuses)
            return found[1] if found is not None else None

    def resolve_permission_request(
        self,
        request_id: str,
        *,
        approved: bool,
        response_message: str = "",
    ) -> WorkerPermissionRequest:
        with self._lock:
            found = self._find_permission_request(request_id, statuses=("pending",))
            if found is None:
                raise ValueError(f"pending permission request not found: {request_id}")
            path, request = found
            resolved = WorkerPermissionRequest(
                request_id=request.request_id,
                team_id=request.team_id,
                agent_id=request.agent_id,
                task_id=request.task_id,
                tool_name=request.tool_name,
                tool_args_summary=request.tool_args_summary,
                reason=request.reason,
                status="approved" if approved else "rejected",
                response_message=response_message,
            )
            path.unlink(missing_ok=True)
            self.write_permission_request(resolved)
            return resolved

    def consume_permission_request(self, request_id: str) -> WorkerPermissionRequest:
        with self._lock:
            found = self._find_permission_request(request_id, statuses=("approved", "rejected"))
            if found is None:
                raise ValueError(f"resolved permission request not found: {request_id}")
            path, request = found
            consumed = WorkerPermissionRequest(
                request_id=request.request_id,
                team_id=request.team_id,
                agent_id=request.agent_id,
                task_id=request.task_id,
                tool_name=request.tool_name,
                tool_args_summary=request.tool_args_summary,
                reason=request.reason,
                status="consumed",
                response_message=request.response_message,
            )
            path.unlink(missing_ok=True)
            self.write_permission_request(consumed)
            return consumed

    def cancel_pending_permissions_for_task(
        self,
        team_id: str,
        task_id: str,
        *,
        reason: str,
    ) -> int:
        """终结任务时消费其待审批请求，防止停止后的 worker 被旧 request 重新启动。"""
        cancelled = 0
        with self._lock:
            for request in self.read_permission_requests(team_id, status="pending"):
                if request.task_id != task_id:
                    continue
                found = self._find_permission_request(request.request_id, statuses=("pending",))
                if found is None:
                    continue
                path, pending = found
                path.unlink(missing_ok=True)
                self.write_permission_request(
                    WorkerPermissionRequest(
                        request_id=pending.request_id,
                        team_id=pending.team_id,
                        agent_id=pending.agent_id,
                        task_id=pending.task_id,
                        tool_name=pending.tool_name,
                        tool_args_summary=pending.tool_args_summary,
                        reason=pending.reason,
                        status="consumed",
                        response_message=reason,
                    ),
                )
                cancelled += 1
        return cancelled

    def update_worker_status_and_notify(
        self,
        team_id: str,
        agent_id: str,
        status: WorkerStatus,
        notification: dict[str, Any],
        *,
        episode_path: str = "",
        session_id: str = "",
        restart_count: int | None = None,
        status_note: str | None = None,
    ) -> None:
        """在同一进程锁内结算 worker 状态并追加通知，避免观察到半完成状态。"""
        with self._lock:
            self._update_worker_status_locked(
                team_id,
                agent_id,
                status,
                episode_path=episode_path,
                session_id=session_id,
                restart_count=restart_count,
                status_note=status_note,
            )
            self._append_notification_locked(team_id, notification)

    def append_notification(self, team_id: str, notification: dict[str, Any]) -> None:
        with self._lock:
            self._append_notification_locked(team_id, notification)

    def _append_notification_locked(self, team_id: str, notification: dict[str, Any]) -> None:
        path = self._team_dir(team_id) / "notifications.jsonl"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        line = json.dumps(notification, ensure_ascii=False, sort_keys=True)
        _atomic_write_text(path, f"{existing}{line}\n")

    def read_notifications(self, team_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            notifications = self._read_all_notifications_locked(team_id)
            bounded_limit = max(0, limit)
            return notifications[-bounded_limit:] if bounded_limit else []

    def latest_notification_for_task(
        self,
        team_id: str,
        task_id: str,
        *,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            for item in reversed(self._read_all_notifications_locked(team_id)):
                if str(item.get("task_id", "")) != task_id:
                    continue
                if status is not None and str(item.get("status", "")) != status:
                    continue
                return item
        return None

    def read_unread_notifications(
        self,
        team_id: str,
        *,
        consumer: Literal["model", "ui"],
        limit: int = 10,
        max_chars: int = 4000,
    ) -> dict[str, Any]:
        """读取有界未读批次；调用方成功投递后再用 next_cursor 确认。"""
        with self._lock:
            visible = [
                item
                for item in self._read_all_notifications_locked(team_id)
                if str(item.get("status", "")) in _VISIBLE_NOTIFICATION_STATUSES
            ]
            cursor = self._read_notification_cursors_locked(team_id).get(consumer, "")
            start = 0
            if cursor:
                for index, item in enumerate(visible):
                    if str(item.get("notification_id", "")) == cursor:
                        start = index + 1
                        break
            selected: list[dict[str, Any]] = []
            chars = 0
            bounded_limit = max(1, limit)
            bounded_chars = max(1, max_chars)
            for item in visible[start:]:
                if len(selected) >= bounded_limit:
                    break
                bounded_item = _bounded_notification(item, max_chars=bounded_chars - chars)
                encoded_chars = len(json.dumps(bounded_item, ensure_ascii=False, default=str))
                if chars + encoded_chars > bounded_chars:
                    break
                selected.append(bounded_item)
                chars += encoded_chars
            next_cursor = str(selected[-1].get("notification_id", "")) if selected else cursor
            return {
                "notifications": selected,
                "next_cursor": next_cursor,
                "remaining": max(0, len(visible) - start - len(selected)),
            }

    def acknowledge_notifications(
        self,
        team_id: str,
        *,
        consumer: Literal["model", "ui"],
        cursor: str,
    ) -> None:
        if not cursor:
            return
        with self._lock:
            known = {
                str(item.get("notification_id", ""))
                for item in self._read_all_notifications_locked(team_id)
            }
            if cursor not in known:
                raise ValueError(f"unknown notification cursor: {cursor}")
            cursors = self._read_notification_cursors_locked(team_id)
            cursors[consumer] = cursor
            _atomic_write_json(
                self._team_dir(team_id) / "notification-cursors.json",
                {
                    "schema_version": _NOTIFICATION_CURSOR_SCHEMA_VERSION,
                    "consumers": cursors,
                },
            )

    def _read_all_notifications_locked(self, team_id: str) -> list[dict[str, Any]]:
        path = self._team_dir(team_id) / "notifications.jsonl"
        if not path.exists():
            return []
        notifications: list[dict[str, Any]] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("notification_id"):
                # 兼容已有 TeamStore 日志：确定性 ID 只用于游标，不改写历史行。
                digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]
                item["notification_id"] = f"legacy-{index}-{digest}"
            item.setdefault("created_at", "")
            notifications.append(item)
        return notifications

    def _read_notification_cursors_locked(self, team_id: str) -> dict[str, str]:
        path = self._team_dir(team_id) / "notification-cursors.json"
        if not path.exists():
            return {"model": "", "ui": ""}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != _NOTIFICATION_CURSOR_SCHEMA_VERSION:
            raise ValueError(f"unsupported notification cursor schema: {raw.get('schema_version')}")
        consumers = raw.get("consumers")
        if not isinstance(consumers, dict):
            raise ValueError("invalid notification cursor consumers")
        return {
            "model": str(consumers.get("model", "")),
            "ui": str(consumers.get("ui", "")),
        }

    def list_teams_for_leader(self, leader_session_id: str) -> list[TeamRecord]:
        if not self.root.exists():
            return []
        teams = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            team = self.load_team(path.name)
            if team is not None and team.leader_session_id == leader_session_id:
                teams.append(team)
        return sorted(teams, key=lambda item: item.updated_at)

    def _require_team(self, team_id: str) -> TeamRecord:
        team = self.load_team(team_id)
        if team is None:
            raise ValueError(f"team not found: {team_id}")
        return team

    def _team_dir(self, team_id: str) -> Path:
        return self.root / _safe_id(team_id)

    def _team_file(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "team.json"

    def _write_team(self, team: TeamRecord) -> None:
        payload = asdict(team)
        _atomic_write_json(self._team_file(team.team_id), payload)

    def _find_permission_request(
        self,
        request_id: str,
        *,
        statuses: tuple[str, ...],
    ) -> tuple[Path, WorkerPermissionRequest] | None:
        safe_request_id = _safe_id(request_id)
        if not self.root.exists():
            return None
        for team_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for status in statuses:
                path = team_dir / "permissions" / _safe_id(status) / f"{safe_request_id}.json"
                if path.exists():
                    return path, WorkerPermissionRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return None


def _bounded_notification(notification: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    """收件箱返回有界投影视图；完整输出与长证据继续留在磁盘原记录。"""
    budget = max(0, max_chars - 700)
    summary = str(notification.get("summary", ""))[: min(500, budget)]
    budget = max(0, budget - len(summary))
    return {
        "notification_id": str(notification.get("notification_id", ""))[:200],
        "created_at": str(notification.get("created_at", ""))[:100],
        "team_id": str(notification.get("team_id", ""))[:200],
        "agent_id": str(notification.get("agent_id", ""))[:200],
        "task_id": str(notification.get("task_id", ""))[:200],
        "status": str(notification.get("status", ""))[:50],
        "summary": summary,
        "episode_path": str(notification.get("episode_path", ""))[: min(500, budget)],
        "needs_attention": bool(notification.get("needs_attention", False)),
        "request_id": str(notification.get("request_id", ""))[:200],
        "parent_step_id": str(notification.get("parent_step_id", ""))[:200],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "default"
