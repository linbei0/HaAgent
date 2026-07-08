"""
tests/unit/tui/test_tui_package_layout.py - TUI 包结构测试

防止 TUI 顶层重新堆叠业务文件，要求界面职责进入子包。
"""

from pathlib import Path


def test_tui_top_level_contains_only_package_entrypoint() -> None:
    tui_root = Path(__file__).parents[3] / "src" / "haagent" / "tui"

    top_level_files = sorted(path.name for path in tui_root.iterdir() if path.is_file())

    assert top_level_files == ["__init__.py"]


def test_tui_top_level_packages_are_named_by_responsibility() -> None:
    tui_root = Path(__file__).parents[3] / "src" / "haagent" / "tui"

    package_names = sorted(
        path.name
        for path in tui_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )

    assert package_names == [
        "application",
        "assets",
        "commands",
        "design",
        "files",
        "flows",
        "memory",
        "overlays",
        "presentation",
        "state",
        "typography",
        "widgets",
    ]
