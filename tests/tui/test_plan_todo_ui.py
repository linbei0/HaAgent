"""
tests/tui/test_plan_todo_ui.py - Plan 确认与 Todo 面板交互测试

验证 Plan 审阅卡片、显式批准和 Todo 的直接点击展开。
"""

from __future__ import annotations

import asyncio

from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.state import PendingInteraction
from haagent.tui.widgets import PlanConfirmationPanel, PromptInput, TodoPanel
from tests.tui.support import FakeAssistantService


def _request() -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="plan_confirmation",
        tool_name="submit_plan",
        plan_id="plan-1",
        plan_revision=1,
        plan_proposal={
            "goal": "实现 Plan Mode",
            "summary": "结构化确认",
            "steps": [{"id": "step-1", "content": "实现状态", "completion_condition": "测试通过"}],
            "verification": {"required": True, "description": "运行 pytest"},
            "assumptions": [],
        },
    )


def test_plan_confirmation_defaults_to_feedback_and_escape_only_minimizes(tmp_path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            pending = PendingInteraction(_request())
            app._begin_interaction(pending)
            await pilot.pause()

            panel = app.query_one("#plan-confirmation", PlanConfirmationPanel)
            assert panel.feedback_input.has_focus
            assert app.query_one("#prompt-input", PromptInput).display is False
            assert "请确认方案" in str(app.query_one("#plan-confirmation-title").render())
            assert "修改意见（可选）" in str(app.query_one("#plan-feedback-label").render())
            assert "批准并开始执行" in str(app.query_one("#approve-plan").label)
            assert "完整方案已显示在上方对话中" in str(app.query_one("#plan-confirmation-content").render())
            assert "实现状态" not in str(app.query_one("#plan-confirmation-content").render())
            assert panel.size.width > 100

            app._restore_prompt_focus()
            assert panel.feedback_input.has_focus

            await pilot.press("escape")
            assert panel.minimized is True
            assert pending.done.is_set() is False
            assert app.query_one("#prompt-input", PromptInput).display is False

            panel.expand()
            assert panel.minimized is False

    asyncio.run(run())


def test_reopening_same_plan_preserves_feedback_draft_after_click(tmp_path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            pending = PendingInteraction(_request())
            app._begin_interaction(pending)
            await pilot.pause()
            panel = app.query_one("#plan-confirmation", PlanConfirmationPanel)
            panel.feedback_input.load_text("补充搜索范围")

            await pilot.click("#plan-feedback")
            await pilot.pause()
            app._input_dock().open_plan_confirmation(_request())

            assert panel.feedback_input.text == "补充搜索范围"
            assert panel.feedback_input.has_focus

    asyncio.run(run())


def test_plan_feedback_and_explicit_approval_resolve_once(tmp_path) -> None:
    async def run_feedback() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            pending = PendingInteraction(_request())
            app._begin_interaction(pending)
            await pilot.pause()
            await pilot.press(*list("补充回滚边界"), "enter")
            assert pending.done.wait(timeout=0.2)
            assert pending.response.plan_outcome == "revision_requested"
            assert pending.response.answer == "补充回滚边界"

    async def run_approval() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            pending = PendingInteraction(_request())
            app._begin_interaction(pending)
            await pilot.pause()
            await pilot.click("#approve-plan")
            assert pending.done.wait(timeout=0.2)
            assert pending.response.plan_outcome == "approved"
            assert pending.response.approved is True

    asyncio.run(run_feedback())
    asyncio.run(run_approval())


def test_todo_panel_expands_by_click_and_resizes_input_dock(tmp_path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(80, 24)) as pilot:
            panel = app.query_one("#todo-panel", TodoPanel)
            panel.update_state(
                (
                    {"id": "a", "content": "完成状态机", "status": "completed"},
                    {"id": "b", "content": "接入 TUI", "status": "in_progress"},
                    {"id": "c", "content": "运行测试", "status": "pending"},
                    {"id": "d", "content": "补全用户文档", "status": "pending"},
                    {"id": "e", "content": "执行回归验证", "status": "pending"},
                    {"id": "f", "content": "交付最终结果", "status": "pending"},
                ),
            )
            await pilot.pause()
            assert "已完成 1/6" in str(panel.render())
            assert "Enter 展开" not in str(panel.render())
            collapsed_height = app.query_one("#input-panel").size.height

            await pilot.click(panel)
            await pilot.pause()
            assert panel.expanded is True
            assert "✓ 完成状态机 · 已完成" in str(panel.render())
            assert "▶ 接入 TUI · 进行中" in str(panel.render())
            assert "○ 交付最终结果 · 待处理" in str(panel.render())
            assert "还有" not in str(panel.render())
            assert panel.size.height >= 7
            assert app.query_one("#input-panel").size.height > collapsed_height

            await pilot.click(panel)
            await pilot.pause()
            assert panel.expanded is False

    asyncio.run(run())
