"""
src/haagent/runtime/session/task_ledger.py - session 级 Todo 事实源

保存四态 Todo、运行时阻塞/证据元数据与检查点，并提供原子整体替换。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable


TASK_LEDGER_SCHEMA_VERSION = 2
TASK_LEDGER_MAX_ITEMS = 20
TASK_LEDGER_ID_LIMIT = 80
TASK_LEDGER_TEXT_FIELD_LIMIT = 240
TASK_LEDGER_MODEL_CHAR_LIMIT = 6000
TASK_LEDGER_RECENT_REF_LIMIT = 5

TODO_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
TERMINAL_TODO_STATUSES = {"completed", "cancelled"}


class TaskLedgerError(RuntimeError):
    """task-ledger 文件损坏或 Todo 更新不合法时抛出。"""


@dataclass(frozen=True)
class TodoItem:
    id: str
    content: str
    status: str
    source_plan_id: str | None = None
    source_plan_revision: int | None = None
    source_plan_step_id: str | None = None
    blocker: dict[str, object] | None = None
    evidence_refs: list[str] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    updated_turn: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "source_plan_id": self.source_plan_id,
            "source_plan_revision": self.source_plan_revision,
            "source_plan_step_id": self.source_plan_step_id,
            "blocker": dict(self.blocker) if self.blocker is not None else None,
            "evidence_refs": list(self.evidence_refs),
            "checkpoint_ids": list(self.checkpoint_ids),
            "updated_turn": self.updated_turn,
        }


@dataclass(frozen=True)
class TaskCheckpoint:
    id: str
    todo_id: str
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
            "todo_id": self.todo_id,
            "turn_index": self.turn_index,
            "episode_path": self.episode_path,
            "tool_call_ids": list(self.tool_call_ids),
            "changed_paths": list(self.changed_paths),
            "verification_refs": list(self.verification_refs),
            "state_digest": self.state_digest,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskLedger:
    goal: str
    todos: list[TodoItem] = field(default_factory=list)
    checkpoints: list[TaskCheckpoint] = field(default_factory=list)
    budgets: dict[str, object] = field(default_factory=dict)
    updated_turn: int = 0
    schema_version: int = TASK_LEDGER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "todos": [item.to_dict() for item in self.todos],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "budgets": dict(self.budgets),
            "updated_turn": self.updated_turn,
        }

    def is_empty(self) -> bool:
        return (
            not self.goal
            and not self.todos
            and not self.checkpoints
            and not self.budgets
            and self.updated_turn == 0
        )

    def active_todo(self) -> TodoItem | None:
        return next((item for item in self.todos if item.status == "in_progress"), None)

    def has_active_todos(self) -> bool:
        return any(item.status in {"pending", "in_progress"} for item in self.todos)

    def all_terminal(self) -> bool:
        return bool(self.todos) and all(item.status in TERMINAL_TODO_STATUSES for item in self.todos)

    def has_blocker(self) -> bool:
        active = self.active_todo()
        return active is not None and active.blocker is not None

    def status_counts(self) -> dict[str, int]:
        return {status: sum(item.status == status for item in self.todos) for status in sorted(TODO_STATUSES)}

    def status_summary(self) -> dict[str, object]:
        active = self.active_todo()
        return {
            "exists": not self.is_empty(),
            "goal": self.goal,
            "has_active_todos": self.has_active_todos(),
            "all_terminal": self.all_terminal(),
            "has_blocker": self.has_blocker(),
            "active_todo_id": active.id if active is not None else None,
            "todo_count": len(self.todos),
            "counts": self.status_counts(),
            "updated_turn": self.updated_turn,
        }


def empty_task_ledger(goal: str = "") -> TaskLedger:
    _validate_text(goal, "goal", TASK_LEDGER_TEXT_FIELD_LIMIT, allow_empty=True)
    return TaskLedger(goal=goal.strip())


def load_task_ledger(path: Path) -> TaskLedger:
    if not path.exists():
        raise TaskLedgerError(f"session package missing required file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskLedgerError(f"cannot load task ledger: {path}") from exc
    return task_ledger_from_dict(raw)


def write_task_ledger(path: Path, ledger: TaskLedger) -> None:
    validated = task_ledger_from_dict(ledger.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(validated.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def task_ledger_to_markdown(ledger: TaskLedger) -> str:
    """将 TaskLedger 转换为模型可读的 markdown 文本。

    模型可通过 file_read 读取此文件回顾完整 Todo 状态，
    实现 Manus 式的 attention recitation（文件操作自然出现在上下文末尾）。
    """
    # 兼容 dict 输入：builder 中 task_ledger 可能以 dict 形式传递。
    if not isinstance(ledger, TaskLedger):
        ledger = task_ledger_from_dict(ledger)
    if ledger.is_empty():
        return ""
    lines = ["# Task Ledger", ""]
    if ledger.goal:
        lines.append(f"**Goal:** {ledger.goal}")
    lines.append(f"**Updated at turn:** {ledger.updated_turn}")
    counts = ledger.status_counts()
    parts = []
    for status in ("pending", "in_progress", "completed", "cancelled"):
        if counts.get(status, 0):
            parts.append(f"{counts[status]} {status}")
    if parts:
        lines.append(f"**Progress:** {', '.join(parts)}")
    lines.append("")
    lines.append("## Todos")
    markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "cancelled": "[-]"}
    for item in ledger.todos:
        lines.append(f"- {markers[item.status]} **{item.id}**: {item.content}")
        if item.status == "in_progress" and item.blocker is not None:
            lines.append(
                f"  - blocker: {item.blocker.get('category', 'unknown')} — "
                f"{item.blocker.get('suggested_action', '')}",
            )
    budgets = ledger.budgets
    if budgets:
        lines.append("")
        lines.append("## Budget")
        for key in ("turns_used", "tool_calls", "model_attempts"):
            value = budgets.get(key)
            if isinstance(value, int) and value > 0:
                lines.append(f"- {key}: {value}")
    text = "\n".join(lines)
    return text if len(text) <= TASK_LEDGER_MODEL_CHAR_LIMIT else text[: TASK_LEDGER_MODEL_CHAR_LIMIT - 1] + "…"


def write_task_ledger_markdown(path: Path, ledger: TaskLedger) -> None:
    """将 TaskLedger 以 markdown 格式写入文件，供模型 file_read 读取。"""
    content = task_ledger_to_markdown(ledger)
    if not content.strip():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def task_ledger_from_dict(raw: object) -> TaskLedger:
    if not isinstance(raw, dict):
        raise TaskLedgerError("task ledger must be an object")
    if raw.get("schema_version") != TASK_LEDGER_SCHEMA_VERSION:
        raise TaskLedgerError("unsupported task ledger schema version")
    required = {"schema_version", "goal", "todos", "checkpoints", "budgets", "updated_turn"}
    if set(raw) != required:
        raise TaskLedgerError("task ledger fields do not match v2 schema")
    goal = _required_string(raw, "goal")
    _validate_text(goal, "goal", TASK_LEDGER_TEXT_FIELD_LIMIT, allow_empty=True)
    todos_raw = _required_list(raw, "todos")
    checkpoints_raw = _required_list(raw, "checkpoints")
    budgets = raw.get("budgets")
    if not isinstance(budgets, dict):
        raise TaskLedgerError("budgets must be an object")
    updated_turn = _required_int(raw, "updated_turn")
    todos = [todo_item_from_dict(item) for item in todos_raw]
    _validate_todo_collection(todos)
    return TaskLedger(
        schema_version=TASK_LEDGER_SCHEMA_VERSION,
        goal=goal.strip(),
        todos=todos,
        checkpoints=[task_checkpoint_from_dict(item) for item in checkpoints_raw],
        budgets=dict(budgets),
        updated_turn=updated_turn,
    )


def todo_item_from_dict(raw: object) -> TodoItem:
    if not isinstance(raw, dict):
        raise TaskLedgerError("todo item must be an object")
    required = {
        "id",
        "content",
        "status",
        "source_plan_id",
        "source_plan_revision",
        "source_plan_step_id",
        "blocker",
        "evidence_refs",
        "checkpoint_ids",
        "updated_turn",
    }
    if set(raw) != required:
        raise TaskLedgerError("todo item fields do not match v2 schema")
    item_id = _required_string(raw, "id")
    content = _required_string(raw, "content")
    _validate_text(item_id, "id", TASK_LEDGER_ID_LIMIT)
    _validate_text(content, "content", TASK_LEDGER_TEXT_FIELD_LIMIT)
    status = _required_string(raw, "status")
    if status not in TODO_STATUSES:
        raise TaskLedgerError("status is invalid")
    blocker = raw.get("blocker")
    if blocker is not None and not isinstance(blocker, dict):
        raise TaskLedgerError("blocker must be an object or null")
    source_revision = raw.get("source_plan_revision")
    if source_revision is not None and (not isinstance(source_revision, int) or isinstance(source_revision, bool)):
        raise TaskLedgerError("source_plan_revision must be an integer or null")
    return TodoItem(
        id=item_id.strip(),
        content=content.strip(),
        status=status,
        source_plan_id=_optional_string(raw.get("source_plan_id"), "source_plan_id"),
        source_plan_revision=source_revision,
        source_plan_step_id=_optional_string(raw.get("source_plan_step_id"), "source_plan_step_id"),
        blocker=dict(blocker) if blocker is not None else None,
        evidence_refs=_bounded_refs(_required_string_list(raw, "evidence_refs")),
        checkpoint_ids=_bounded_refs(_required_string_list(raw, "checkpoint_ids")),
        updated_turn=_required_int(raw, "updated_turn"),
    )


def task_checkpoint_from_dict(raw: object) -> TaskCheckpoint:
    if not isinstance(raw, dict):
        raise TaskLedgerError("task checkpoint must be an object")
    required = {
        "id",
        "todo_id",
        "turn_index",
        "episode_path",
        "tool_call_ids",
        "changed_paths",
        "verification_refs",
        "state_digest",
        "created_at",
    }
    if set(raw) != required:
        raise TaskLedgerError("task checkpoint fields are invalid")
    return TaskCheckpoint(
        id=_required_string(raw, "id"),
        todo_id=_required_string(raw, "todo_id"),
        turn_index=_required_int(raw, "turn_index"),
        episode_path=_required_string(raw, "episode_path"),
        tool_call_ids=_bounded_refs(_required_string_list(raw, "tool_call_ids")),
        changed_paths=_bounded_refs(_required_string_list(raw, "changed_paths")),
        verification_refs=_bounded_refs(_required_string_list(raw, "verification_refs")),
        state_digest=_required_string(raw, "state_digest"),
        created_at=_required_string(raw, "created_at"),
    )


def replace_todos(
    ledger: TaskLedger,
    *,
    items: Iterable[TodoItem | dict[str, object]],
    turn_index: int,
    goal: str | None = None,
) -> TaskLedger:
    """校验完整新清单后原子替换；任一失败都不改变旧 ledger。"""

    proposed = [_coerce_todo_update_item(item, turn_index=turn_index) for item in items]
    _validate_todo_collection(proposed)
    if not proposed and ledger.todos and not ledger.all_terminal():
        raise TaskLedgerError("empty todo list is only allowed after all existing items are terminal")
    proposed_ids = {item.id for item in proposed}
    missing_active = [
        item.id
        for item in ledger.todos
        if item.status in {"pending", "in_progress"} and item.id not in proposed_ids
    ]
    if missing_active:
        raise TaskLedgerError(f"unfinished todo items cannot disappear: {', '.join(missing_active)}")
    old_by_id = {item.id: item for item in ledger.todos}
    merged: list[TodoItem] = []
    for item in proposed:
        previous = old_by_id.get(item.id)
        if previous is None:
            merged.append(item)
            continue
        merged.append(
            replace(
                item,
                source_plan_id=previous.source_plan_id,
                source_plan_revision=previous.source_plan_revision,
                source_plan_step_id=previous.source_plan_step_id,
                blocker=previous.blocker if item.status == "in_progress" else None,
                evidence_refs=list(previous.evidence_refs),
                checkpoint_ids=list(previous.checkpoint_ids),
            ),
        )
    next_goal = ledger.goal if goal is None else goal.strip()
    _validate_text(next_goal, "goal", TASK_LEDGER_TEXT_FIELD_LIMIT, allow_empty=True)
    return TaskLedger(
        goal=next_goal,
        todos=merged,
        checkpoints=list(ledger.checkpoints),
        budgets=dict(ledger.budgets),
        updated_turn=turn_index,
    )


def cancel_active_todos(ledger: TaskLedger, *, turn_index: int) -> TaskLedger:
    todos = [
        replace(item, status="cancelled", blocker=None, updated_turn=turn_index)
        if item.status in {"pending", "in_progress"}
        else item
        for item in ledger.todos
    ]
    return replace(ledger, todos=todos, updated_turn=turn_index)


def initialize_todos_from_plan(
    *,
    goal: str,
    plan_id: str,
    revision: int,
    steps: Iterable[tuple[str, str]],
    verification: str | None,
    turn_index: int,
) -> TaskLedger:
    """把批准 Plan 确定性转换为 Todo；第一项立即进入执行态。"""

    items: list[TodoItem] = []
    for index, (step_id, content) in enumerate(steps):
        items.append(
            TodoItem(
                id=f"todo-{index + 1:03d}",
                content=content,
                status="in_progress" if index == 0 else "pending",
                source_plan_id=plan_id,
                source_plan_revision=revision,
                source_plan_step_id=step_id,
                updated_turn=turn_index,
            ),
        )
    if verification is not None:
        items.append(
            TodoItem(
                id=f"todo-{len(items) + 1:03d}",
                content=verification,
                status="pending" if items else "in_progress",
                source_plan_id=plan_id,
                source_plan_revision=revision,
                source_plan_step_id=None,
                updated_turn=turn_index,
            ),
        )
    _validate_todo_collection(items)
    return TaskLedger(goal=goal.strip(), todos=items, updated_turn=turn_index)


def update_task_ledger_runtime(
    ledger: TaskLedger,
    *,
    turn_index: int,
    episode_path: Path,
    runtime_events: list[object],
) -> TaskLedger:
    """只更新 blocker/evidence/checkpoint/budget，绝不隐式改变 Todo 状态。"""

    if not ledger.todos:
        return ledger
    from haagent.runtime.events.bus import bus_event_to_dict, coerce_bus_event

    active = ledger.active_todo()
    if active is None:
        return replace(ledger, updated_turn=turn_index)
    active_index = next(index for index, item in enumerate(ledger.todos) if item.id == active.id)
    current = active
    checkpoints = list(ledger.checkpoints)
    budgets = dict(ledger.budgets)
    tool_calls = int(budgets.get("tool_calls", 0) or 0)
    model_attempts = int(budgets.get("model_attempts", 0) or 0)
    for raw_event in runtime_events:
        event = bus_event_to_dict(coerce_bus_event(raw_event))
        event_type = str(event.get("event_type", ""))
        if event_type == "tool_finished":
            tool_calls += 1
            tool_name = str(event.get("tool_name", "unknown"))
            current = replace(
                current,
                evidence_refs=_bounded_refs([*current.evidence_refs, f"tool={tool_name} episode={episode_path}"]),
                updated_turn=turn_index,
            )
        elif event_type in {"task_recovery_suggested", "task_step_blocked", "worker_failed"}:
            # 阻塞是运行时元数据；Todo 仍保持 in_progress，便于恢复后继续。
            current = replace(
                current,
                blocker={
                    "category": str(event.get("category", event_type)),
                    "reason": str(event.get("reason", event.get("summary", "")))[:TASK_LEDGER_TEXT_FIELD_LIMIT],
                    "suggested_action": str(event.get("suggested_action", ""))[:TASK_LEDGER_TEXT_FIELD_LIMIT],
                },
                updated_turn=turn_index,
            )
        elif event_type == "task_checkpoint_saved":
            checkpoint = _checkpoint_for_active(current, checkpoints, turn_index, episode_path)
            checkpoints.append(checkpoint)
            current = replace(
                current,
                checkpoint_ids=_bounded_refs([*current.checkpoint_ids, checkpoint.id]),
                updated_turn=turn_index,
            )
        elif event_type == "task_step_progress" and event.get("category") == "model_turn_started":
            model_attempts += 1
    todos = list(ledger.todos)
    todos[active_index] = current
    budgets.update({"turns_used": turn_index, "tool_calls": tool_calls, "model_attempts": model_attempts})
    return replace(
        ledger,
        todos=todos,
        checkpoints=checkpoints[-TASK_LEDGER_MAX_ITEMS:],
        budgets=budgets,
        updated_turn=turn_index,
    )


def format_task_ledger_for_model(value: object) -> str:
    ledger = value if isinstance(value, TaskLedger) else task_ledger_from_dict(value)
    if not ledger.has_active_todos():
        return ""
    lines = [f"todo_goal: {ledger.goal or 'none'}", "todos:"]
    markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "cancelled": "[-]"}
    for item in ledger.todos:
        lines.append(f"- {markers[item.status]} id={item.id} status={item.status} content={item.content}")
        if item.status == "in_progress" and item.blocker is not None:
            lines.append(
                "  blocker: "
                f"category={item.blocker.get('category', 'unknown')} "
                f"suggested_action={item.blocker.get('suggested_action', '')}",
            )
    lines.append(
        "规则：Todo 是完整 session 清单；开始前保持一个 in_progress，完成里程碑后立即调用 todo_update。",
    )
    lines.append(
        "提示：Todo 状态每轮更新到 episode 目录的 task-ledger.md，长任务中可用 file_read 回顾完整进展。",
    )
    text = "\n".join(lines)
    return text if len(text) <= TASK_LEDGER_MODEL_CHAR_LIMIT else text[: TASK_LEDGER_MODEL_CHAR_LIMIT - 1] + "…"


def _coerce_todo_update_item(item: TodoItem | dict[str, object], *, turn_index: int) -> TodoItem:
    if isinstance(item, TodoItem):
        candidate = replace(item, updated_turn=turn_index)
    elif isinstance(item, dict):
        allowed = {"id", "content", "status"}
        if set(item) != allowed:
            raise TaskLedgerError("todo_update item only accepts id, content, and status")
        candidate = TodoItem(
            id=_required_string(item, "id").strip(),
            content=_required_string(item, "content").strip(),
            status=_required_string(item, "status"),
            updated_turn=turn_index,
        )
    else:
        raise TaskLedgerError("todo_update item must be an object")
    _validate_text(candidate.id, "id", TASK_LEDGER_ID_LIMIT)
    _validate_text(candidate.content, "content", TASK_LEDGER_TEXT_FIELD_LIMIT)
    if candidate.status not in TODO_STATUSES:
        raise TaskLedgerError("status is invalid")
    return candidate


def _validate_todo_collection(items: list[TodoItem]) -> None:
    if len(items) > TASK_LEDGER_MAX_ITEMS:
        raise TaskLedgerError(f"todo list cannot exceed {TASK_LEDGER_MAX_ITEMS} items")
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise TaskLedgerError("todo ids must be unique")
    if sum(item.status == "in_progress" for item in items) > 1:
        raise TaskLedgerError("todo list can contain at most one in_progress item")


def _checkpoint_for_active(
    item: TodoItem,
    existing: list[TaskCheckpoint],
    turn_index: int,
    episode_path: Path,
) -> TaskCheckpoint:
    checkpoint_id = f"checkpoint-{len(existing) + 1:03d}"
    digest = sha256(f"{item.id}|{turn_index}|{episode_path}".encode("utf-8")).hexdigest()[:16]
    return TaskCheckpoint(
        id=checkpoint_id,
        todo_id=item.id,
        turn_index=turn_index,
        episode_path=str(episode_path),
        state_digest=digest,
        created_at=datetime.now(UTC).isoformat(),
    )


def _validate_text(value: str, field_name: str, limit: int, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TaskLedgerError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise TaskLedgerError(f"{field_name} must be non-empty")
    if len(value.strip()) > limit:
        raise TaskLedgerError(f"{field_name} cannot exceed {limit} characters")


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


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TaskLedgerError(f"{key} must be a non-negative integer")
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


def _bounded_refs(items: list[str]) -> list[str]:
    return [item[:TASK_LEDGER_TEXT_FIELD_LIMIT] for item in items[-TASK_LEDGER_RECENT_REF_LIMIT:]]
