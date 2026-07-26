"""
tests/unit/tools/test_plan_todo_tools.py - Plan/Todo 工具安全边界测试

验证 Router 双重拒绝、main-only 与结构化 Plan 确认结果。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.human_interaction import HumanInteractionResponse
from haagent.tools.router import ToolRouter


def _writer(tmp_path: Path) -> EpisodeWriter:
    task = tmp_path / "task.yaml"
    task.write_text("goal: test\nworkspace_root: .\n", encoding="utf-8")
    return EpisodeWriter.create(tmp_path / ".runs", task)


def test_plan_mode_router_denies_side_effects_and_dynamic_mcp(tmp_path: Path) -> None:
    router = ToolRouter(
        ["file_write", "shell", "todo_update", "mcp__demo__read"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        planning_status="planning",
    )

    for name, args in [
        ("file_write", {"path": "x.txt", "content": "x"}),
        ("shell", {"command": "echo x"}),
        ("todo_update", {"items": []}),
        ("mcp__demo__read", {}),
    ]:
        result = router.dispatch(name, args)
        assert result["error"]["type"] == "plan_mode_tool_denied"
    assert not (tmp_path / "x.txt").exists()


def test_todo_update_is_main_only_and_uses_atomic_sink(tmp_path: Path) -> None:
    calls: list[tuple[list[dict[str, object]], str, int | None]] = []

    def sink(items, explanation, turn):
        calls.append((items, explanation, turn))
        return {"items": items, "counts": {"pending": 1}, "all_terminal": False}

    worker = ToolRouter(
        ["todo_update"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        actor_role="worker",
        todo_state_sink=sink,
    )
    denied = worker.dispatch(
        "todo_update",
        {"items": [{"id": "a", "content": "任务", "status": "pending"}]},
        turn=2,
    )
    assert denied["error"]["type"] == "tool_actor_denied"
    assert calls == []

    main = ToolRouter(
        ["todo_update"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        todo_state_sink=sink,
    )
    result = main.dispatch(
        "todo_update",
        {"explanation": "初始化", "items": [{"id": "a", "content": "任务", "status": "pending"}]},
        turn=2,
    )
    assert result["status"] == "success"
    assert calls[-1][1:] == ("初始化", 2)


def test_submit_plan_revision_feedback_stays_in_same_tool_call(tmp_path: Path) -> None:
    actions: list[str] = []

    def state_handler(action: str, payload: dict[str, object], turn: int) -> dict[str, object]:
        del payload, turn
        actions.append(action)
        return {
            "status": "awaiting_confirmation" if action == "submit" else "planning",
            "plan_id": "plan-1",
            "revision": 1,
            "proposal": _proposal(),
        }

    router = ToolRouter(
        ["submit_plan"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        planning_status="planning",
        planning_state_handler=state_handler,
    )
    result = router.dispatch(
        "submit_plan",
        _proposal(),
        interaction_handler=lambda request: HumanInteractionResponse(
            approved=False,
            answer="请补充回滚边界",
            plan_outcome="revision_requested",
        ),
        turn=1,
    )

    assert result["outcome"] == "revision_requested"
    assert result["feedback"] == "请补充回滚边界"
    assert actions == ["submit", "feedback"]


def test_submit_plan_approval_returns_end_turn_control(tmp_path: Path) -> None:
    def state_handler(action: str, payload: dict[str, object], turn: int) -> dict[str, object]:
        del payload, turn
        base = {
            "plan_id": "plan-1",
            "revision": 1,
            "proposal": _proposal(),
        }
        return {**base, "status": "approved_pending_execution", "execution_id": "execution-1"} if action == "approve" else {**base, "status": "awaiting_confirmation"}

    router = ToolRouter(
        ["submit_plan"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        planning_status="planning",
        planning_state_handler=state_handler,
    )
    result = router.dispatch(
        "submit_plan",
        _proposal(),
        interaction_handler=lambda request: HumanInteractionResponse(
            approved=True,
            answer="approve",
            plan_outcome="approved",
        ),
        turn=1,
    )

    assert result["outcome"] == "approved"
    assert result["control"] == "end_turn"
    assert result["execution_id"] == "execution-1"


def _proposal() -> dict[str, object]:
    return {
        "goal": "实现功能",
        "summary": "结构化方案",
        "steps": [{"content": "实现状态", "completion_condition": "测试通过"}],
        "verification": {"required": True, "description": "运行 pytest"},
        "assumptions": [],
    }
