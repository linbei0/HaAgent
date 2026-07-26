"""
src/haagent/tui/widgets/plan_confirmation.py - Plan 内联确认面板

提供修改意见输入、显式批准按钮、最小化和防重复提交。
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static, TextArea


class PlanFeedbackInput(TextArea):
    """Enter 提交修改意见，Ctrl+Enter 插入换行。"""

    BINDINGS = [
        Binding("enter", "submit_feedback", "提交修改", priority=True),
        Binding("ctrl+enter", "insert_feedback_newline", "换行", priority=True),
        Binding("escape", "minimize_plan", "最小化", priority=True),
        Binding("ctrl+x", "cancel_plan_task", "取消任务", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit_feedback(self) -> None:
        value = self.text.strip()
        if value:
            self.post_message(self.Submitted(value))

    def action_insert_feedback_newline(self) -> None:
        self.insert("\n")

    def action_minimize_plan(self) -> None:
        panel = self.parent
        if isinstance(panel, PlanConfirmationPanel):
            panel.minimize()

    def action_cancel_plan_task(self) -> None:
        self.app.action_cancel_current_task()


class PlanConfirmationPanel(Vertical):
    """Plan 的紧凑确认区；完整方案只保留在上方对话流。"""

    can_focus = True
    DEFAULT_CSS = """
    PlanConfirmationPanel {
        width: 1fr;
        height: auto;
        background: $surface;
        border-left: thick $primary;
        padding: 0 1;
    }
    PlanConfirmationPanel #plan-confirmation-title { height: 1; color: $text; text-style: bold; }
    PlanConfirmationPanel #plan-confirmation-guidance { height: 1; color: $text-muted; }
    PlanConfirmationPanel #plan-confirmation-content {
        height: 1;
        color: $text-muted;
    }
    PlanConfirmationPanel #plan-feedback-label { height: 1; margin-top: 1; color: $text; }
    PlanConfirmationPanel PlanFeedbackInput {
        height: 2;
        border: none;
        background: $surface-lighten-1;
    }
    PlanConfirmationPanel #approve-plan {
        width: 1fr;
        min-width: 0;
        height: 3;
        margin-top: 1;
        color: $text;
        text-style: bold;
        content-align: center middle;
    }
    """

    class FeedbackSubmitted(Message):
        def __init__(self, feedback: str) -> None:
            super().__init__()
            self.feedback = feedback

    class Approved(Message):
        pass

    class Minimized(Message):
        pass

    def __init__(self, plan_id: str, revision: int, proposal: dict[str, object], **kwargs) -> None:
        super().__init__(**kwargs)
        self.plan_id = plan_id
        self.revision = revision
        self.proposal = dict(proposal)
        self.minimized = False
        self.processing = False

    def compose(self) -> ComposeResult:
        yield Static(self._title_text(), id="plan-confirmation-title")
        yield Static("上方是完整方案；确认后开始执行，或输入意见重新规划。", id="plan-confirmation-guidance")
        yield Static(self._content_text(), id="plan-confirmation-content")
        yield Static("修改意见（可选）", id="plan-feedback-label")
        yield PlanFeedbackInput(id="plan-feedback", show_line_numbers=False)
        yield Button("批准并开始执行", id="approve-plan", variant="primary")

    def on_mount(self) -> None:
        # 子组件在父容器 Mount 事件前已就绪，直接聚焦可避免后续无关 refresh 抢占默认焦点。
        self.feedback_input.focus()

    @property
    def feedback_input(self) -> PlanFeedbackInput:
        return self.query_one("#plan-feedback", PlanFeedbackInput)

    def update_plan(self, plan_id: str, revision: int, proposal: dict[str, object]) -> None:
        is_new_revision = (plan_id, revision) != (self.plan_id, self.revision)
        self.plan_id = plan_id
        self.revision = revision
        self.proposal = dict(proposal)
        self.processing = False
        self.minimized = False
        self.query_one("#plan-confirmation-title", Static).update(self._title_text())
        self.query_one("#plan-confirmation-content", Static).update(self._content_text())
        self.feedback_input.disabled = False
        # 同一 Plan 可能因焦点恢复或重复 runtime 事件再次打开；不能清掉用户正在写的意见。
        if is_new_revision:
            self.feedback_input.load_text("")
        self.query_one("#approve-plan", Button).disabled = False
        self._apply_visibility()
        self.feedback_input.focus()

    def set_processing(self, label: str) -> None:
        self.processing = True
        self.feedback_input.disabled = True
        self.query_one("#approve-plan", Button).disabled = True
        self.query_one("#plan-confirmation-title", Static).update(label)

    def minimize(self) -> None:
        if self.processing:
            return
        self.minimized = True
        self._apply_visibility()
        self.focus()
        self.post_message(self.Minimized())

    def expand(self) -> None:
        self.minimized = False
        self._apply_visibility()
        self.feedback_input.focus()

    def on_key(self, event: events.Key) -> None:
        if self.minimized and event.key == "enter":
            event.stop()
            event.prevent_default()
            self.expand()

    def on_plan_feedback_input_submitted(self, event: PlanFeedbackInput.Submitted) -> None:
        if self.processing:
            return
        self.post_message(self.FeedbackSubmitted(event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve-plan" and not self.processing:
            self.post_message(self.Approved())

    def _apply_visibility(self) -> None:
        self.query_one("#plan-confirmation-guidance", Static).display = not self.minimized
        self.query_one("#plan-confirmation-content", Static).display = not self.minimized
        self.query_one("#plan-feedback-label", Static).display = not self.minimized
        self.feedback_input.display = not self.minimized
        self.query_one("#approve-plan", Button).display = not self.minimized
        self.query_one("#plan-confirmation-title", Static).update(
            "Plan 等待确认 · Enter 展开" if self.minimized else self._title_text(),
        )

    def _title_text(self) -> str:
        return f"请确认方案 · 第 {self.revision} 版"

    def _content_text(self) -> str:
        steps = self.proposal.get("steps")
        items = steps if isinstance(steps, list) else []
        verification = self.proposal.get("verification")
        verification_text = " · 含验证步骤" if isinstance(verification, dict) and verification.get("description") else ""
        return f"完整方案已显示在上方对话中 · 共 {len(items)} 步{verification_text}"


def format_plan_for_timeline(revision: int, proposal: dict[str, object]) -> str:
    lines = [f"Plan v{revision}", "", f"目标：{proposal.get('goal', '')}", "", str(proposal.get("summary", "")), "", "步骤："]
    steps = proposal.get("steps")
    if isinstance(steps, list):
        for index, item in enumerate(steps, start=1):
            if isinstance(item, dict):
                lines.append(
                    f"{index}. {item.get('content', '')}\n   完成条件：{item.get('completion_condition', '')}",
                )
    verification = proposal.get("verification")
    if isinstance(verification, dict):
        lines.extend(["", f"验证：{verification.get('description', '')}"])
    assumptions = proposal.get("assumptions")
    if isinstance(assumptions, list) and assumptions:
        lines.extend(["", "假设：", *[f"- {item}" for item in assumptions]])
    return "\n".join(lines)
