"""
haagent/tui/overlays/skill_picker.py - Skill 选择界面

提供可搜索、可键盘选择的本地 skill 列表，用于 /skill 空参数时先选 skill。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.tui.design.copy import EMPTY_LABELS
from haagent.tui.design.screen_helpers import safe_dismiss
from haagent.tui.design.utils import safe_summary


VISIBLE_SKILL_COUNT = 8


@dataclass(frozen=True)
class SkillPickerState:
    skills: list[dict[str, object]]
    blocked_project_skill_roots: list[str] = ()
    title: str = "选择 Skill"
    instruction: str = "输入过滤，↑/↓ 移动，Enter 选择，Esc 关闭"
    footer: str = ""
    query: str = ""
    selected_index: int = 0
    scroll_offset: int = 0

    @property
    def visible_skills(self) -> list[dict[str, object]]:
        needle = self.query.casefold()
        if not needle:
            return self.skills
        return [
            skill
            for skill in self.skills
            if needle in _skill_text(skill).casefold()
        ]

    @property
    def selected_skill(self) -> dict[str, object] | None:
        visible = self.visible_skills
        if not visible:
            return None
        return visible[min(max(self.selected_index, 0), len(visible) - 1)]

    def with_query(self, query: str) -> SkillPickerState:
        return replace(self, query=query, selected_index=0, scroll_offset=0)

    def move(self, delta: int) -> SkillPickerState:
        visible = self.visible_skills
        if not visible:
            return replace(self, selected_index=0, scroll_offset=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        scroll_offset = self.scroll_offset
        if next_index < scroll_offset:
            scroll_offset = next_index
        elif next_index >= scroll_offset + VISIBLE_SKILL_COUNT:
            scroll_offset = next_index - VISIBLE_SKILL_COUNT + 1
        return replace(self, selected_index=next_index, scroll_offset=scroll_offset)

    def render(self) -> str:
        visible = self.visible_skills
        lines = [
            self.title,
            f"{len(visible)} skills · {self.instruction}",
        ]
        if self.blocked_project_skill_roots:
            lines.append("项目 skills 未信任 · 输入 /skills trust 信任当前 workspace")
        lines.extend(["", f"搜索: {self.query or '-'}", ""])
        if not visible:
            lines.append(EMPTY_LABELS.get("no_matching_skills", "没有匹配的 skills。"))
        scroll_offset = min(self.scroll_offset, max(len(visible) - VISIBLE_SKILL_COUNT, 0))
        visible_window = visible[scroll_offset : scroll_offset + VISIBLE_SKILL_COUNT]
        for offset, skill in enumerate(visible_window):
            index = scroll_offset + offset
            marker = ">" if index == min(self.selected_index, len(visible) - 1) else " "
            command_name = str(skill.get("command_name") or skill.get("name") or "unknown")
            source = str(skill.get("source") or "unknown")
            description = safe_summary(str(skill.get("description") or ""), 56)
            flags = " user-only" if bool(skill.get("disable_model_invocation")) else ""
            lines.append(f"{marker} {command_name:<18} · {source}{flags} · {description}".rstrip())
        if self.footer:
            lines.extend(["", self.footer])
        return "\n".join(lines)


class SkillPickerOverlay(ModalScreen[dict[str, object] | None]):
    def __init__(
        self,
        skills: list[dict[str, object]],
        blocked_project_skill_roots: list[str] | None = None,
        *,
        title: str = "选择 Skill",
        instruction: str = "输入过滤，↑/↓ 移动，Enter 选择，Esc 关闭",
        footer: str = "",
        select_on_enter: bool = True,
    ) -> None:
        super().__init__()
        self.select_on_enter = select_on_enter
        self.state = SkillPickerState(
            skills=skills,
            blocked_project_skill_roots=list(blocked_project_skill_roots or []),
            title=title,
            instruction=instruction,
            footer=footer,
        )

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="skill-picker-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            safe_dismiss(self, None)
            return
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if key == "backspace":
            event.stop()
            self._set_state(self.state.with_query(self.state.query[:-1]))
            return
        if key == "enter":
            event.stop()
            if not self.select_on_enter:
                return
            selected = self.state.selected_skill
            if selected is not None:
                safe_dismiss(self, selected)
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.with_query(self.state.query + event.character))

    def _set_state(self, state: SkillPickerState) -> None:
        self.state = state
        self.query_one("#skill-picker-dialog", Static).update(state.render())


def _skill_text(skill: dict[str, object]) -> str:
    return "\n".join(
        [
            str(skill.get("name") or ""),
            str(skill.get("command_name") or ""),
            str(skill.get("description") or ""),
            str(skill.get("source") or ""),
        ],
    )
