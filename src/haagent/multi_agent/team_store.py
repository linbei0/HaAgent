"""
src/haagent/multi_agent/team_store.py - 用户级 team 与 mailbox 存储

以 UTF-8 JSON/JSONL 记录 worker 状态、消息和完成通知，供 TUI、inspect 与后续恢复审计使用。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


WorkerStatus = Literal["queued", "running", "idle", "completed", "failed", "stopped"]
MessageType = Literal["user_message", "shutdown_request"]


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


@dataclass(frozen=True)
class MailboxMessage:
    id: str
    type: MessageType
    sender: str
    recipient: str
    payload: dict[str, Any]
    timestamp: float
    read: bool = False

    @classmethod
    def user_message(cls, *, sender: str, recipient: str, content: str) -> "MailboxMessage":
        return cls(
            id=uuid.uuid4().hex,
            type="user_message",
            sender=sender,
            recipient=recipient,
            payload={"content": content},
            timestamp=time.time(),
        )


class TeamStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def ensure_team(
        self,
        *,
        team_id: str,
        workspace_root: Path,
        leader_session_id: str,
    ) -> TeamRecord:
        existing = self.load_team(team_id)
        if existing is not None:
            return existing
        team = TeamRecord(
            team_id=team_id,
            workspace_root=str(workspace_root.resolve()),
            leader_session_id=leader_session_id,
        )
        self._write_team(team)
        return team

    def load_team(self, team_id: str) -> TeamRecord | None:
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
                    updated_at=item.get("updated_at", ""),
                )
                for item in raw.get("agents", [])
            ],
            created_at=raw.get("created_at", ""),
            updated_at=raw.get("updated_at", ""),
        )

    def upsert_worker(self, team_id: str, worker: WorkerRecord) -> None:
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
        team = self._require_team(team_id)
        agents: list[WorkerRecord] = []
        for worker in team.agents:
            if worker.agent_id != agent_id:
                agents.append(worker)
                continue
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
                ),
            )
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

    def write_mailbox(self, team_id: str, agent_id: str, message: MailboxMessage) -> Path:
        inbox = self._team_dir(team_id) / "agents" / agent_id / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        filename = f"{message.timestamp:.6f}_{message.id}.json"
        return _atomic_write_json(inbox / filename, asdict(message))

    def append_notification(self, team_id: str, notification: dict[str, Any]) -> None:
        path = self._team_dir(team_id) / "notifications.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(notification, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def read_notifications(self, team_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        path = self._team_dir(team_id) / "notifications.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "default"
