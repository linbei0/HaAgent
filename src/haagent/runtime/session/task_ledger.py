"""
src/haagent/runtime/session/task_ledger.py - 长任务账本状态

为 session package 保存结构化长任务进度，并向模型提供有界摘要。
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK_LEDGER_SCHEMA_VERSION = 1
TASK_LEDGER_TEXT_FIELD_LIMIT = 240
TASK_LEDGER_MODEL_CHAR_LIMIT = 2400
TASK_LEDGER_RECENT_STEP_LIMIT = 5

LEDGER_STATUSES = {"planning", "running", "blocked", "failed", "completed", "cancelled"}
STEP_STATUSES = {"pending", "running", "completed", "blocked", "failed", "skipped"}
STEP_KINDS = {"plan", "research", "read", "edit", "verify", "summarize", "ask_user", "delegate", "recover"}
STEP_OWNERS = {"main", "worker"}
TERMINAL_LEDGER_STATUSES = {"completed", "failed", "cancelled"}


class TaskLedgerError(Exception):
    """task-ledger 文件损坏或结构不合法时抛出。"""


@dataclass
class TaskStep:
    id: str
    title: str
    kind: str
    owner: str
    status: str
    worker_id: str | None = None
    parent_step_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    blocker: dict[str, object] | None = None
    retry_count: int = 0
    updated_turn: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "owner": self.owner,
            "worker_id": self.worker_id,
            "parent_step_id": self.parent_step_id,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "checkpoint_ids": list(self.checkpoint_ids),
            "blocker": dict(self.blocker) if self.blocker is not None else None,
            "retry_count": self.retry_count,
            "updated_turn": self.updated_turn,
        }


@dataclass
class TaskCheckpoint:
    id: str
    step_id: str
    turn_index: int
    episode_path: str
    tool_call_ids: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    verification_refs: list[str] = field(default_factory=list)
    state_digest: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "turn_index": self.turn_index,
            "episode_path": self.episode_path,
            "tool_call_ids": list(self.tool_call_ids),
            "changed_paths": list(self.changed_paths),
            "verification_refs": list(self.verification_refs),
            "state_digest": self.state_digest,
            "created_at": self.created_at,
        }


@dataclass
class TaskLedger:
    goal: str
    status: str
    current_step_id: str | None = None
    steps: list[TaskStep] = field(default_factory=list)
    checkpoints: list[TaskCheckpoint] = field(default_factory=list)
    budgets: dict[str, object] = field(default_factory=dict)
    updated_turn: int = 0
    schema_version: int = TASK_LEDGER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "steps": [step.to_dict() for step in self.steps],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "budgets": dict(self.budgets),
            "updated_turn": self.updated_turn,
        }

    def is_empty(self) -> bool:
        return (
            not self.goal
            and self.status == "planning"
            and self.current_step_id is None
            and not self.steps
            and not self.checkpoints
            and not self.budgets
            and self.updated_turn == 0
        )

    def active_step(self) -> TaskStep | None:
        if self.current_step_id is None:
            return None
        for step in self.steps:
            if step.id == self.current_step_id:
                return step
        return None

    def status_summary(self) -> dict[str, object]:
        return {
            "exists": not self.is_empty(),
            "goal": self.goal,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "step_count": len(self.steps),
            "checkpoint_count": len(self.checkpoints),
            "updated_turn": self.updated_turn,
        }


def empty_task_ledger(goal: str = "") -> TaskLedger:
    return TaskLedger(
        goal=_bounded_text(goal, TASK_LEDGER_TEXT_FIELD_LIMIT),
        status="planning",
    )


def load_task_ledger(path: Path) -> TaskLedger:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskLedgerError(f"cannot load task ledger: {path}") from exc
    return task_ledger_from_dict(raw)


def write_task_ledger(path: Path, ledger: TaskLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ledger.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def begin_task_ledger_turn(ledger: TaskLedger, *, prompt: str, turn_index: int) -> TaskLedger:
    should_start_new = (
        ledger.is_empty()
        or ledger.status in TERMINAL_LEDGER_STATUSES
        or (ledger.status == "planning" and not ledger.steps)
    )
    source = empty_task_ledger(prompt) if should_start_new else ledger
    steps = [task_step_from_dict(step.to_dict()) for step in source.steps]
    checkpoints = [task_checkpoint_from_dict(checkpoint.to_dict()) for checkpoint in source.checkpoints]
    current_step_id = source.current_step_id
    if not steps:
        steps.append(_new_turn_step(prompt, turn_index))
        current_step_id = "step-001"
    elif current_step_id is None:
        current_step_id = steps[-1].id
    active = _find_or_default_step(steps, current_step_id)
    if active is not None and active.status in {"pending", "blocked", "failed", "skipped"}:
        active.status = "running"
        active.blocker = None
        active.updated_turn = turn_index
    return TaskLedger(
        goal=source.goal or _bounded_text(prompt, TASK_LEDGER_TEXT_FIELD_LIMIT),
        status="running",
        current_step_id=current_step_id,
        steps=steps,
        checkpoints=checkpoints,
        budgets={**source.budgets, "turns_used": turn_index},
        updated_turn=turn_index,
    )


def update_task_ledger(
    ledger: TaskLedger,
    *,
    prompt: str,
    turn_index: int,
    result_status: str,
    episode_path: Path,
    runtime_events: list[dict[str, object]],
) -> TaskLedger:
    current = begin_task_ledger_turn(ledger, prompt=prompt, turn_index=turn_index)
    steps = [task_step_from_dict(step.to_dict()) for step in current.steps]
    checkpoints = [task_checkpoint_from_dict(checkpoint.to_dict()) for checkpoint in current.checkpoints]
    goal = current.goal
    status = current.status
    current_step_id = current.current_step_id

    for event in runtime_events:
        event_type = str(event.get("event_type", ""))
        if event_type == "tool_finished":
            step = _find_or_default_step(steps, current_step_id)
            if step is not None:
                step.evidence_refs = _bounded_string_list([
                    *step.evidence_refs,
                    _tool_evidence(event),
                ])
                step.updated_turn = turn_index
        elif event_type == "task_step_finished":
            step_id = str(event.get("step_id", current_step_id or "step-001"))
            step = _ensure_step(steps, step_id, str(event.get("title", step_id)), str(event.get("owner", "main")))
            step.status = "completed"
            step.updated_turn = turn_index
            step.evidence_refs = _bounded_string_list([
                *step.evidence_refs,
                f"episode={episode_path}",
            ])
            current_step_id = step.id
        elif event_type == "task_step_blocked":
            step_id = str(event.get("step_id", current_step_id or "step-001"))
            step = _ensure_step(steps, step_id, str(event.get("title", step_id)), str(event.get("owner", "main")))
            step.status = "blocked"
            step.blocker = {
                "category": str(event.get("category", "blocked")),
                "reason": f"reason_chars={event.get('reason_chars', 0)}",
            }
            step.updated_turn = turn_index
            status = "blocked"
            current_step_id = step.id
        elif event_type == "worker_failed":
            worker_id = str(event.get("agent_id", "worker"))
            parent_step_id = str(event.get("parent_step_id", "") or "")
            reason = _bounded_text(str(event.get("reason", event.get("summary", ""))))
            evidence_refs = _worker_evidence_refs(event, worker_id, "worker_failed")
            if parent_step_id:
                parent = _ensure_step(steps, parent_step_id, str(event.get("description", parent_step_id)), "main")
                parent.kind = "delegate"
                parent.status = "blocked"
                parent.blocker = {"category": "worker_failure", "reason": reason}
                parent.evidence_refs = _bounded_string_list([*parent.evidence_refs, *evidence_refs])
                parent.updated_turn = turn_index
                current_step_id = parent.id
            step = _ensure_step(steps, worker_id, f"Worker {worker_id}", "worker")
            step.kind = "delegate"
            step.worker_id = worker_id
            step.status = "blocked"
            step.blocker = {"category": "worker_failure", "reason": reason}
            step.evidence_refs = _bounded_string_list([*step.evidence_refs, *evidence_refs])
            step.updated_turn = turn_index
            status = "blocked"
            if not parent_step_id:
                current_step_id = step.id
        elif event_type == "worker_completed":
            worker_id = str(event.get("agent_id", "worker"))
            parent_step_id = str(event.get("parent_step_id", "") or "")
            evidence_refs = _worker_evidence_refs(event, worker_id, "worker_completed")
            if parent_step_id:
                parent = _ensure_step(steps, parent_step_id, str(event.get("description", parent_step_id)), "main")
                parent.kind = "delegate"
                parent.status = "completed"
                parent.evidence_refs = _bounded_string_list([*parent.evidence_refs, *evidence_refs])
                parent.updated_turn = turn_index
                current_step_id = parent.id
            step = _ensure_step(steps, worker_id, f"Worker {worker_id}", "worker")
            step.kind = "delegate"
            step.worker_id = worker_id
            step.status = "completed"
            step.evidence_refs = _bounded_string_list([*step.evidence_refs, *evidence_refs])
            step.updated_turn = turn_index
            if not parent_step_id:
                current_step_id = step.id

    _apply_result_status_to_active_step(
        steps,
        current_step_id=current_step_id,
        result_status=result_status,
        episode_path=episode_path,
    )
    checkpoints.append(_checkpoint_for_turn(checkpoints, steps, current_step_id, turn_index, episode_path))
    if status != "blocked" and result_status in TERMINAL_LEDGER_STATUSES:
        status = result_status
    return TaskLedger(
        goal=goal,
        status=status,
        current_step_id=current_step_id,
        steps=steps,
        checkpoints=checkpoints,
        budgets={**current.budgets, "turns_used": turn_index},
        updated_turn=turn_index,
    )


def task_ledger_from_dict(raw: object) -> TaskLedger:
    if not isinstance(raw, dict):
        raise TaskLedgerError("task ledger must be an object")
    if raw.get("schema_version") != TASK_LEDGER_SCHEMA_VERSION:
        raise TaskLedgerError("unsupported task ledger schema version")
    goal = _required_string(raw, "goal")
    status = _required_choice(raw, "status", LEDGER_STATUSES)
    current_step_id = _optional_string(raw.get("current_step_id"), "current_step_id")
    steps = _required_list(raw, "steps")
    checkpoints = _required_list(raw, "checkpoints")
    budgets = raw.get("budgets")
    if not isinstance(budgets, dict):
        raise TaskLedgerError("budgets must be an object")
    updated_turn = _required_int(raw, "updated_turn")
    return TaskLedger(
        schema_version=TASK_LEDGER_SCHEMA_VERSION,
        goal=_bounded_text(goal, TASK_LEDGER_TEXT_FIELD_LIMIT),
        status=status,
        current_step_id=current_step_id,
        steps=[task_step_from_dict(item) for item in steps],
        checkpoints=[task_checkpoint_from_dict(item) for item in checkpoints],
        budgets=dict(budgets),
        updated_turn=updated_turn,
    )


def format_task_ledger_for_model(value: object) -> str:
    ledger = _ledger_from_value(value)
    if ledger.is_empty():
        return ""
    lines = [
        f"task_goal: {_bounded_text(ledger.goal)}",
        f"task_status: {ledger.status}",
    ]
    active_step = ledger.active_step()
    if active_step is not None:
        lines.append(
            "active_step: "
            f"id={active_step.id} status={active_step.status} owner={active_step.owner} "
            f"kind={active_step.kind} title={_bounded_text(active_step.title)} "
            f"evidence_count={len(active_step.evidence_refs)} retry_count={active_step.retry_count}",
        )
        if active_step.worker_id:
            lines.append(f"active_worker: {active_step.worker_id}")
        if active_step.blocker:
            category = _bounded_text(str(active_step.blocker.get("category", "unknown")), 80)
            reason = str(active_step.blocker.get("reason", ""))
            lines.append(f"blocker: category={category} reason_chars={len(reason)}")
    completed = [step for step in ledger.steps if step.status == "completed"][-TASK_LEDGER_RECENT_STEP_LIMIT:]
    if completed:
        lines.append("completed_steps:")
        for step in completed:
            lines.append(
                f"- {step.id} owner={step.owner} title={_bounded_text(step.title, 120)} "
                f"evidence_count={len(step.evidence_refs)} checkpoints={len(step.checkpoint_ids)}",
            )
    warnings = _budget_warnings(ledger.budgets)
    if warnings:
        lines.append("budget_warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    text = "\n".join(lines)
    if len(text) > TASK_LEDGER_MODEL_CHAR_LIMIT:
        return text[: TASK_LEDGER_MODEL_CHAR_LIMIT - 1] + "…"
    return text


def _ledger_from_value(value: object) -> TaskLedger:
    if isinstance(value, TaskLedger):
        return value
    return task_ledger_from_dict(value)


def task_step_from_dict(raw: object) -> TaskStep:
    if not isinstance(raw, dict):
        raise TaskLedgerError("task step must be an object")
    evidence_refs = _required_string_list(raw, "evidence_refs")
    checkpoint_ids = _required_string_list(raw, "checkpoint_ids")
    blocker = raw.get("blocker")
    if blocker is not None and not isinstance(blocker, dict):
        raise TaskLedgerError("blocker must be an object or null")
    return TaskStep(
        id=_required_string(raw, "id"),
        title=_bounded_text(_required_string(raw, "title"), TASK_LEDGER_TEXT_FIELD_LIMIT),
        kind=_required_choice(raw, "kind", STEP_KINDS),
        owner=_required_choice(raw, "owner", STEP_OWNERS),
        worker_id=_optional_string(raw.get("worker_id"), "worker_id"),
        parent_step_id=_optional_string(raw.get("parent_step_id"), "parent_step_id"),
        status=_required_choice(raw, "status", STEP_STATUSES),
        evidence_refs=_bounded_string_list(evidence_refs),
        checkpoint_ids=_bounded_string_list(checkpoint_ids),
        blocker=dict(blocker) if blocker is not None else None,
        retry_count=_required_int(raw, "retry_count"),
        updated_turn=_required_int(raw, "updated_turn"),
    )


def task_checkpoint_from_dict(raw: object) -> TaskCheckpoint:
    if not isinstance(raw, dict):
        raise TaskLedgerError("task checkpoint must be an object")
    return TaskCheckpoint(
        id=_required_string(raw, "id"),
        step_id=_required_string(raw, "step_id"),
        turn_index=_required_int(raw, "turn_index"),
        episode_path=_required_string(raw, "episode_path"),
        tool_call_ids=_bounded_string_list(_required_string_list(raw, "tool_call_ids")),
        changed_paths=_bounded_string_list(_required_string_list(raw, "changed_paths")),
        verification_refs=_bounded_string_list(_required_string_list(raw, "verification_refs")),
        state_digest=_bounded_text(_required_string(raw, "state_digest"), 160),
        created_at=_bounded_text(_required_string(raw, "created_at"), 80),
    )


def _required_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise TaskLedgerError(f"{key} must be a string")
    return value


def _optional_string(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskLedgerError(f"{key} must be a string or null")
    return value


def _required_choice(raw: dict[str, object], key: str, choices: set[str]) -> str:
    value = _required_string(raw, key)
    if value not in choices:
        raise TaskLedgerError(f"{key} is invalid")
    return value


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TaskLedgerError(f"{key} must be an integer")
    return value


def _required_list(raw: dict[str, object], key: str) -> list[object]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TaskLedgerError(f"{key} must be a list")
    return value


def _required_string_list(raw: dict[str, object], key: str) -> list[str]:
    values = _required_list(raw, key)
    if not all(isinstance(item, str) for item in values):
        raise TaskLedgerError(f"{key} must be a list of strings")
    return list(values)


def _bounded_string_list(items: list[str]) -> list[str]:
    return [_bounded_text(item, TASK_LEDGER_TEXT_FIELD_LIMIT) for item in items[:TASK_LEDGER_RECENT_STEP_LIMIT]]


def _bounded_text(value: str, limit: int = TASK_LEDGER_TEXT_FIELD_LIMIT) -> str:
    text = value.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _budget_warnings(budgets: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    for key in ("turns_used", "tool_calls", "retry_count"):
        value = budgets.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            warnings.append(f"{key}={value}")
    return warnings


def _find_or_default_step(steps: list[TaskStep], step_id: str | None) -> TaskStep | None:
    if step_id is not None:
        for step in steps:
            if step.id == step_id:
                return step
    return steps[-1] if steps else None


def _new_turn_step(prompt: str, turn_index: int) -> TaskStep:
    return TaskStep(
        id="step-001",
        title=_bounded_text(prompt or "Process task"),
        kind="plan",
        owner="main",
        status="running",
        updated_turn=turn_index,
    )


def _ensure_step(steps: list[TaskStep], step_id: str, title: str, owner: str) -> TaskStep:
    for step in steps:
        if step.id == step_id:
            return step
    step = TaskStep(
        id=_bounded_text(step_id, 80),
        title=_bounded_text(title),
        kind="delegate" if owner == "worker" else "plan",
        owner=owner if owner in STEP_OWNERS else "main",
        status="running",
    )
    steps.append(step)
    return step


def _apply_result_status_to_active_step(
    steps: list[TaskStep],
    *,
    current_step_id: str | None,
    result_status: str,
    episode_path: Path,
) -> None:
    if result_status not in TERMINAL_LEDGER_STATUSES:
        return
    step = _find_or_default_step(steps, current_step_id)
    if step is None or step.status == "blocked":
        return
    if result_status == "completed":
        step.status = "completed"
    elif result_status == "failed":
        step.status = "failed"
        step.blocker = {"category": "turn_failed", "reason": "episode failed"}
    elif result_status == "cancelled":
        step.status = "skipped"
        step.blocker = {"category": "turn_cancelled", "reason": "user cancelled current run"}
    step.evidence_refs = _bounded_string_list([
        *step.evidence_refs,
        f"episode={episode_path}",
    ])


def _tool_evidence(event: dict[str, object]) -> str:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    path = result.get("path") if isinstance(result.get("path"), str) else ""
    suffix = f" path={path}" if path else ""
    return f"tool={event.get('tool_name', 'unknown')} status={result.get('status', 'unknown')}{suffix}"


def _worker_evidence_refs(event: dict[str, object], worker_id: str, fallback: str) -> list[str]:
    refs = event.get("evidence_refs")
    evidence_refs = [str(item) for item in refs] if isinstance(refs, list | tuple) else []
    if not evidence_refs:
        evidence_refs = [f"{fallback}={worker_id}"]
    episode_path = str(event.get("episode_path", "") or "")
    if episode_path:
        evidence_refs.append(f"episode={episode_path}")
    task_id = str(event.get("task_id", "") or "")
    if task_id:
        evidence_refs.append(f"task={task_id}")
    return _bounded_string_list(evidence_refs)


def _checkpoint_for_turn(
    existing: list[TaskCheckpoint],
    steps: list[TaskStep],
    current_step_id: str | None,
    turn_index: int,
    episode_path: Path,
) -> TaskCheckpoint:
    step_id = current_step_id or (steps[-1].id if steps else "step-001")
    checkpoint_id = f"ckpt-{len(existing) + 1:04d}"
    changed_paths: list[str] = []
    for step in steps:
        for evidence in step.evidence_refs:
            if " path=" in evidence:
                changed_paths.append(evidence.split(" path=", 1)[1])
    digest_source = f"{step_id}|{turn_index}|{episode_path}|{'|'.join(changed_paths)}"
    return TaskCheckpoint(
        id=checkpoint_id,
        step_id=step_id,
        turn_index=turn_index,
        episode_path=str(episode_path),
        tool_call_ids=[],
        changed_paths=_bounded_string_list(changed_paths),
        verification_refs=[],
        state_digest="sha256:" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
        created_at=datetime.now(UTC).isoformat(),
    )
