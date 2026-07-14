"""
haagent/tui/overlays/workspace_picker.py - 渠道固定 workspace 路径选择器

在 TUI 中浏览目录或输入绝对路径，确认后返回目录 Path；不含 Agent 逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static


VISIBLE_ENTRY_COUNT = 10


def _list_directories(path: Path) -> list[Path]:
    try:
        entries = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return []
    return sorted(entries, key=lambda p: p.name.casefold())


@dataclass(frozen=True)
class WorkspacePickerState:
    """纯状态：当前目录列表、路径输入与确认结果。"""

    root: Path
    start_path: Path
    current_path: Path | None = None
    selected_index: int = 0
    scroll_offset: int = 0
    path_input: str = ""
    error: str = ""
    entries: tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # frozen dataclass：仅首次构造时填充 current/entries。
        if self.current_path is not None:
            return
        current = Path(self.start_path or self.root).resolve()
        entries = tuple(_list_directories(current))
        object.__setattr__(self, "current_path", current)
        object.__setattr__(self, "entries", entries)
        if not self.path_input:
            object.__setattr__(self, "path_input", str(current))

    @property
    def selected_entry(self) -> Path | None:
        if not self.entries:
            return None
        index = min(max(self.selected_index, 0), len(self.entries) - 1)
        return self.entries[index]

    def move(self, delta: int) -> WorkspacePickerState:
        if not self.entries:
            return replace(self, selected_index=0, scroll_offset=0)
        next_index = min(max(self.selected_index + delta, 0), len(self.entries) - 1)
        scroll = self.scroll_offset
        if next_index < scroll:
            scroll = next_index
        elif next_index >= scroll + VISIBLE_ENTRY_COUNT:
            scroll = next_index - VISIBLE_ENTRY_COUNT + 1
        return replace(self, selected_index=next_index, scroll_offset=scroll, error="")

    def enter_selected(self) -> WorkspacePickerState | None:
        selected = self.selected_entry
        if selected is None:
            return None
        entries = tuple(_list_directories(selected))
        return replace(
            self,
            current_path=selected.resolve(),
            entries=entries,
            selected_index=0,
            scroll_offset=0,
            path_input=str(selected.resolve()),
            error="",
        )

    def go_parent(self) -> WorkspacePickerState:
        current = Path(self.current_path or self.root).resolve()
        parent = current.parent
        if parent == current:
            return replace(self, error="已在磁盘根目录")
        entries = tuple(_list_directories(parent))
        return replace(
            self,
            current_path=parent,
            entries=entries,
            selected_index=0,
            scroll_offset=0,
            path_input=str(parent),
            error="",
        )

    def append_path_char(self, char: str) -> WorkspacePickerState:
        return replace(self, path_input=self.path_input + char, error="")

    def backspace_path(self) -> WorkspacePickerState:
        return replace(self, path_input=self.path_input[:-1], error="")

    def confirm_path(self) -> Path | None:
        """确认 path_input 为存在的目录；失败时写入 error，返回 None。"""
        raw = (self.path_input or "").strip().strip('"')
        if not raw:
            object.__setattr__(self, "error", "路径不能为空")
            return None
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            object.__setattr__(self, "error", "路径无效")
            return None
        if not path.exists():
            object.__setattr__(self, "error", f"不存在：{path}")
            return None
        if not path.is_dir():
            object.__setattr__(self, "error", f"不是目录：{path}")
            return None
        return path

    def confirm_current(self) -> Path | None:
        current = Path(self.current_path or self.start_path).resolve()
        if current.is_dir():
            return current
        object.__setattr__(self, "error", f"不是目录：{current}")
        return None

    def render(self) -> str:
        current = Path(self.current_path or self.start_path).resolve()
        lines = [
            "选择 workspace",
            f"当前：{current}",
            f"路径：{self.path_input or '-'}",
            "",
        ]
        if not self.entries:
            lines.append("（无子目录）")
        else:
            scroll = min(self.scroll_offset, max(len(self.entries) - VISIBLE_ENTRY_COUNT, 0))
            window = self.entries[scroll : scroll + VISIBLE_ENTRY_COUNT]
            for offset, entry in enumerate(window):
                index = scroll + offset
                marker = ">" if index == min(self.selected_index, len(self.entries) - 1) else " "
                lines.append(f"{marker} {entry.name}/")
        lines.extend(
            [
                "",
                "↑/↓ 移动  Enter 进入  Backspace 上级  c 选用当前目录",
                "直接输入/编辑路径后按 y 确认  Esc 取消",
            ]
        )
        if self.error:
            lines.extend(["", f"错误：{self.error}"])
        return "\n".join(lines)


class WorkspacePickerOverlay(ModalScreen[str | None]):
    """目录浏览 + 路径输入；dismiss 绝对路径字符串或 None。"""

    def __init__(self, start_path: Path, *, title: str = "选择 workspace") -> None:
        super().__init__()
        del title
        root = Path(start_path).resolve()
        # 用盘符/根作为浏览上限参考；Windows 上为盘符根。
        self.state = WorkspacePickerState(root=root.anchor and Path(root.anchor) or root, start_path=root)

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="workspace-picker-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if key == "enter":
            event.stop()
            entered = self.state.enter_selected()
            if entered is not None:
                self._set_state(entered)
            return
        if key == "backspace":
            event.stop()
            # 路径输入非空时先删字符；否则返回上级目录。
            if self.state.path_input and self.state.path_input != str(self.state.current_path):
                self._set_state(self.state.backspace_path())
            else:
                self._set_state(self.state.go_parent())
            return
        if key == "c":
            event.stop()
            path = self.state.confirm_current()
            if path is not None:
                self.dismiss(str(path))
            else:
                self._set_state(self.state)
            return
        if key == "y":
            event.stop()
            path = self.state.confirm_path()
            if path is not None:
                self.dismiss(str(path))
            else:
                self._set_state(self.state)
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.append_path_char(event.character))

    def _set_state(self, state: WorkspacePickerState) -> None:
        self.state = state
        self.query_one("#workspace-picker-dialog", Static).update(state.render())
