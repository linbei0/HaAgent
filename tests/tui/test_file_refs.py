"""
tests/tui/test_file_refs.py - HaAgent TUI file_refs 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, fuzzy_file_matches, path_reference_token
from haagent.tui.widgets import PromptInput
from textual.screen import Screen

from tests.tui.support import FakeAssistantService, _all_text

def test_tui_file_reference_fuzzy_search_stays_inside_workspace(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "Project Plan.md"
    target.write_text("plan", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    outside = tmp_path.parent / "outside-haagent-ref.txt"
    outside.write_text("outside", encoding="utf-8")

    matches = fuzzy_file_matches(tmp_path, "plan")
    no_matches = fuzzy_file_matches(tmp_path, "missing")
    token = path_reference_token(tmp_path, target)

    assert [match.display_path for match in matches] == ["docs/Project Plan.md"]
    assert no_matches == []
    assert token == '@file("docs/Project Plan.md")'

def test_tui_file_reference_python_fallback_when_rg_missing(tmp_path: Path, monkeypatch) -> None:
    # rg 缺失时的纯 Python 扫描是有意保留的环境兼容路径（见 refs.py 注释），
    # 这里显式覆盖 shutil.which -> None 强制走 fallback，确认仍能在 workspace 内找到文件。
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Project Plan.md").write_text("plan", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr("haagent.tui.files.refs.shutil.which", lambda name: None)

    index = build_file_reference_index(tmp_path)

    display_paths = {item.display_path for item in index.files}
    assert "docs/Project Plan.md" in display_paths
    assert "README.md" in display_paths
    assert [item.display_path for item in index.matches("plan")] == ["docs/Project Plan.md"]

def test_tui_file_reference_index_uses_fast_file_walker(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Project Plan.md").write_text("plan", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")
    candidates = [
        FileReferenceMatch(path=docs / "Project Plan.md", display_path="docs/Project Plan.md"),
        FileReferenceMatch(path=src / "main.py", display_path="src/main.py"),
    ]

    monkeypatch.setattr("haagent.tui.files.refs._iter_file_reference_candidates", lambda root: iter(candidates))

    index = build_file_reference_index(tmp_path)
    assert [item.display_path for item in index.matches("plan")] == ["docs/Project Plan.md"]
    assert [item.display_path for item in index.matches("main")] == ["src/main.py"]

def test_tui_file_reference_overlay_scrolls_selected_match_into_view(tmp_path: Path) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    overlay = FileReferenceOverlay(tmp_path, "")
    overlay.index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=tuple(
            FileReferenceMatch(path=tmp_path / f"file-{index:02}.txt", display_path=f"file-{index:02}.txt")
            for index in range(12)
        ),
    )
    overlay.loading = False
    overlay._reload()

    for _ in range(4):
        overlay._move(1)

    body = overlay._body()
    assert "> file-04.txt" in body
    assert "  file-00.txt" not in body
    assert "  file-03.txt" in body

def test_tui_file_reference_overlay_uses_preloaded_index_without_loading(tmp_path: Path) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=(FileReferenceMatch(path=tmp_path / "README.md", display_path="README.md"),),
    )
    overlay = FileReferenceOverlay(tmp_path, "", index)
    overlay.on_mount()

    assert overlay.loading is False
    assert "正在搜索文件" not in overlay._body()
    assert "README.md" in overlay._body()

def test_tui_file_reference_overlay_filters_loaded_index_without_rescanning(tmp_path: Path, monkeypatch) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    def fail_rglob(self, pattern):
        raise AssertionError("query updates should not rescan workspace")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    overlay = FileReferenceOverlay(tmp_path, "")
    overlay.index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=(
            FileReferenceMatch(path=tmp_path / "README.md", display_path="README.md"),
            FileReferenceMatch(path=tmp_path / "docs" / "Project Plan.md", display_path="docs/Project Plan.md"),
        ),
    )

    overlay.update_query("plan")

    assert [item.display_path for item in overlay.matches] == ["docs/Project Plan.md"]

def test_tui_file_reference_overlay_ignores_index_after_unmount(tmp_path: Path, monkeypatch) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    overlay = FileReferenceOverlay(tmp_path, "")
    index = build_file_reference_index(tmp_path)
    overlay.on_unmount()

    def fail_reload():
        raise AssertionError("unmounted overlay should not refresh stale worker results")

    monkeypatch.setattr(overlay, "_reload", fail_reload)

    overlay._handle_index_ready(index)

    assert overlay.index is None

def test_tui_file_reference_overlay_selects_workspace_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Project Plan.md").write_text("plan", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "Read "
            await pilot.press("@")
            # 文件索引在线程 worker 中构建；等待公开 worker 生命周期，避免按机器速度猜测延时。
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert input_widget.value == "Read @"
            assert isinstance(app.screen, Screen)
            assert app.query_one("#prompt-input", PromptInput) is input_widget
            assert "文件引用" in _all_text(app)
            assert "docs/Project Plan.md" in _all_text(app)
            assert "README.md" in _all_text(app)
            assert "> README.md" in _all_text(app)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> README.md" not in _all_text(app)
            await pilot.press("p", "l", "a")
            await pilot.pause(0.1)
            assert "搜索: pla" in _all_text(app)
            assert "README.md" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == 'Read @file("docs/Project Plan.md")'

            input_widget.value = "Read @missing"
            await pilot.press("@")
            await pilot.pause(0.1)
            assert "无匹配文件" in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "文件引用" not in _all_text(app)
            assert app.query_one("#prompt-input", PromptInput) is input_widget
            assert input_widget.has_focus

    asyncio.run(run())

