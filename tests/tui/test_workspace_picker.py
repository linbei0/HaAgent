"""
tests/tui/test_workspace_picker.py - 渠道 workspace 路径选择器

验证可浏览目录、输入绝对路径、确认后返回路径；不暴露 secrets。
"""

from __future__ import annotations

from pathlib import Path

from haagent.tui.overlays.workspace_picker import WorkspacePickerState


def test_workspace_picker_lists_directories(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    state = WorkspacePickerState(root=tmp_path, start_path=tmp_path)
    text = state.render()
    assert "alpha" in text
    assert "beta" in text
    assert "file.txt" not in text
    assert "选择 workspace" in text or "workspace" in text.lower()


def test_workspace_picker_enter_descends_and_parent(tmp_path: Path) -> None:
    child = tmp_path / "nested"
    child.mkdir()
    (child / "deep").mkdir()
    state = WorkspacePickerState(root=tmp_path, start_path=tmp_path)
    # 选中 nested（通常排序后第一项可能是 .. 或 nested）
    state = state.ensure_selection_on("nested")
    entered = state.enter_selected()
    assert entered is not None
    assert entered.current_path == child.resolve()
    up = entered.go_parent()
    assert up.current_path == tmp_path.resolve()


def test_workspace_picker_path_input_and_confirm(tmp_path: Path) -> None:
    target = tmp_path / "chosen"
    target.mkdir()
    state = WorkspacePickerState(root=tmp_path, start_path=tmp_path)
    state = state.with_path_input(str(target))
    confirmed = state.confirm_path()
    assert confirmed == target.resolve()


def test_workspace_picker_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not-dir.txt"
    file_path.write_text("x", encoding="utf-8")
    state = WorkspacePickerState(root=tmp_path, start_path=tmp_path)
    state = state.with_path_input(str(file_path))
    assert state.confirm_path() is None
    assert state.error
