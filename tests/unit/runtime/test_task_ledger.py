"""
tests/unit/runtime/test_task_ledger.py - TaskLedger v2 与 Todo 原子更新测试

验证四态清单、持久化、Plan 初始化和运行时元数据边界。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.session.task_ledger import (
    TASK_LEDGER_SCHEMA_VERSION,
    TaskLedger,
    TaskLedgerError,
    TodoItem,
    cancel_active_todos,
    empty_task_ledger,
    format_task_ledger_for_model,
    initialize_todos_from_plan,
    replace_todos,
    task_ledger_from_dict,
    task_ledger_to_markdown,
    update_task_ledger_runtime,
    write_task_ledger_markdown,
)


def _item(item_id: str, status: str, content: str | None = None) -> TodoItem:
    return TodoItem(id=item_id, content=content or item_id, status=status)


def test_task_ledger_v2_persists_only_derived_todo_state() -> None:
    ledger = TaskLedger(
        goal="实现 Plan Mode",
        todos=[_item("a", "completed"), _item("b", "in_progress"), _item("c", "pending")],
        updated_turn=3,
    )

    raw = ledger.to_dict()
    restored = task_ledger_from_dict(raw)

    assert raw["schema_version"] == TASK_LEDGER_SCHEMA_VERSION
    assert "status" not in raw
    assert "current_todo_id" not in raw
    assert "steps" not in raw
    assert restored.active_todo().id == "b"
    assert restored.has_active_todos() is True
    assert restored.all_terminal() is False
    assert restored.status_summary()["counts"] == {
        "cancelled": 0,
        "completed": 1,
        "in_progress": 1,
        "pending": 1,
    }


@pytest.mark.parametrize(
    "items",
    [
        [_item("same", "pending"), _item("same", "completed")],
        [_item("a", "in_progress"), _item("b", "in_progress")],
        [_item("", "pending")],
        [_item("a", "unknown")],
        [TodoItem(id="a", content="", status="pending")],
        [_item(str(index), "pending") for index in range(21)],
    ],
)
def test_replace_todos_rejects_invalid_complete_list(items: list[TodoItem]) -> None:
    with pytest.raises(TaskLedgerError):
        replace_todos(empty_task_ledger("目标"), items=items, turn_index=1)


def test_replace_todos_is_atomic_and_unfinished_items_cannot_disappear() -> None:
    original = TaskLedger(
        goal="目标",
        todos=[_item("keep", "in_progress"), _item("later", "pending")],
        updated_turn=1,
    )

    with pytest.raises(TaskLedgerError, match="cannot disappear"):
        replace_todos(original, items=[_item("keep", "completed")], turn_index=2)

    assert [item.status for item in original.todos] == ["in_progress", "pending"]


def test_completed_and_cancelled_items_may_be_omitted() -> None:
    original = TaskLedger(
        goal="目标",
        todos=[_item("done", "completed"), _item("skip", "cancelled"), _item("next", "pending")],
    )

    updated = replace_todos(original, items=[_item("next", "in_progress")], turn_index=2)

    assert [item.id for item in updated.todos] == ["next"]
    assert updated.active_todo().id == "next"


def test_empty_list_only_allowed_after_all_existing_items_terminal() -> None:
    active = TaskLedger(goal="目标", todos=[_item("a", "pending")])
    terminal = TaskLedger(goal="目标", todos=[_item("a", "completed"), _item("b", "cancelled")])

    with pytest.raises(TaskLedgerError):
        replace_todos(active, items=[], turn_index=2)
    assert replace_todos(terminal, items=[], turn_index=2).todos == []


def test_runtime_metadata_does_not_change_todo_status(tmp_path) -> None:
    ledger = TaskLedger(goal="目标", todos=[_item("a", "in_progress")])

    updated = update_task_ledger_runtime(
        ledger,
        turn_index=2,
        episode_path=tmp_path / "episode",
        runtime_events=[
            {"event_type": "tool_finished", "tool_name": "file_read", "result": {"status": "success"}},
            {
                "event_type": "task_step_blocked",
                "category": "waiting_input",
                "reason": "需要用户确认",
                "suggested_action": "ask_user",
            },
        ],
    )

    assert updated.active_todo().status == "in_progress"
    assert updated.active_todo().blocker["category"] == "waiting_input"
    assert updated.active_todo().evidence_refs


def test_cancel_marks_all_active_items_cancelled() -> None:
    ledger = TaskLedger(
        goal="目标",
        todos=[_item("a", "completed"), _item("b", "in_progress"), _item("c", "pending")],
    )

    cancelled = cancel_active_todos(ledger, turn_index=4)

    assert [item.status for item in cancelled.todos] == ["completed", "cancelled", "cancelled"]
    assert cancelled.all_terminal() is True


def test_plan_initialization_sets_first_active_and_verification_last() -> None:
    ledger = initialize_todos_from_plan(
        goal="完成开发",
        plan_id="plan-1",
        revision=2,
        steps=[("step-2-1", "实现状态机"), ("step-2-2", "接入 TUI")],
        verification="运行完整测试",
        turn_index=3,
    )

    assert [item.status for item in ledger.todos] == ["in_progress", "pending", "pending"]
    assert ledger.todos[-1].content == "运行完整测试"
    assert ledger.todos[0].source_plan_revision == 2


def test_terminal_todos_are_not_injected_into_model_context() -> None:
    ledger = TaskLedger(goal="目标", todos=[_item("a", "completed"), _item("b", "cancelled")])
    assert format_task_ledger_for_model(ledger) == ""


def test_active_todos_are_fully_injected_without_evidence() -> None:
    secret = "SECRET_EVIDENCE"
    ledger = TaskLedger(
        goal="目标",
        todos=[
            TodoItem(id="a", content="第一步", status="completed", evidence_refs=[secret]),
            TodoItem(id="b", content="第二步", status="in_progress", evidence_refs=[secret]),
        ],
    )

    text = format_task_ledger_for_model(ledger)

    assert "id=a status=completed content=第一步" in text
    assert "id=b status=in_progress content=第二步" in text
    assert secret not in text


def test_task_ledger_v1_is_explicitly_rejected() -> None:
    with pytest.raises(TaskLedgerError, match="unsupported"):
        task_ledger_from_dict(
            {
                "schema_version": 1,
                "goal": "旧目标",
                "status": "running",
                "current_step_id": "step-001",
                "steps": [],
                "checkpoints": [],
                "budgets": {},
                "updated_turn": 1,
            },
        )


def test_persisted_json_contains_no_duplicate_summary_fields() -> None:
    ledger = TaskLedger(goal="目标", todos=[_item("a", "in_progress")])
    raw = json.dumps(ledger.to_dict(), ensure_ascii=False)
    assert '"status": "in_progress"' in raw
    assert '"current_todo_id"' not in raw
    assert '"has_active_todos"' not in raw


def test_markdown_for_empty_ledger_returns_empty() -> None:
    assert task_ledger_to_markdown(empty_task_ledger()) == ""


def test_markdown_includes_goal_todos_and_progress() -> None:
    ledger = TaskLedger(
        goal="实现功能 X",
        todos=[
            _item("todo-001", "completed", "分析需求"),
            _item("todo-002", "in_progress", "编写实现"),
            _item("todo-003", "pending", "运行测试"),
        ],
        updated_turn=5,
    )
    md = task_ledger_to_markdown(ledger)

    assert "# Task Ledger" in md
    assert "实现功能 X" in md
    assert "Updated at turn:** 5" in md
    assert "1 completed" in md
    assert "1 in_progress" in md
    assert "1 pending" in md
    assert "[x] **todo-001**: 分析需求" in md
    assert "[>] **todo-002**: 编写实现" in md
    assert "[ ] **todo-003**: 运行测试" in md


def test_markdown_includes_blocker_for_in_progress_item() -> None:
    ledger = TaskLedger(
        goal="目标",
        todos=[
            TodoItem(
                id="todo-001",
                content="受阻任务",
                status="in_progress",
                blocker={
                    "category": "waiting_input",
                    "reason": "需要确认",
                    "suggested_action": "ask_user",
                },
            ),
        ],
        updated_turn=2,
    )
    md = task_ledger_to_markdown(ledger)

    assert "blocker: waiting_input" in md
    assert "ask_user" in md


def test_markdown_includes_budget_when_nonzero() -> None:
    ledger = TaskLedger(
        goal="目标",
        todos=[_item("a", "in_progress")],
        budgets={"turns_used": 3, "tool_calls": 7, "model_attempts": 4},
        updated_turn=3,
    )
    md = task_ledger_to_markdown(ledger)

    assert "## Budget" in md
    assert "turns_used: 3" in md
    assert "tool_calls: 7" in md
    assert "model_attempts: 4" in md


def test_write_task_ledger_markdown_creates_file(tmp_path) -> None:
    ledger = TaskLedger(
        goal="测试目标",
        todos=[_item("a", "in_progress")],
        updated_turn=1,
    )
    md_path = tmp_path / "task-ledger.md"

    write_task_ledger_markdown(md_path, ledger)

    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "# Task Ledger" in content
    assert "测试目标" in content


def test_write_task_ledger_markdown_skips_empty_ledger(tmp_path) -> None:
    md_path = tmp_path / "task-ledger.md"

    write_task_ledger_markdown(md_path, empty_task_ledger())

    assert not md_path.exists()


def test_format_for_model_includes_markdown_hint() -> None:
    ledger = TaskLedger(goal="目标", todos=[_item("a", "in_progress")])
    text = format_task_ledger_for_model(ledger)

    assert "task-ledger.md" in text
    assert "file_read" in text
