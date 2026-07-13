"""
tests/unit/runtime/test_runtime_package_layout.py - Runtime 包结构测试

防止 runtime 顶层重新堆叠业务文件，要求职责进入子包。
"""

from pathlib import Path


def test_runtime_top_level_contains_only_package_entrypoint() -> None:
    runtime_root = Path(__file__).parents[3] / "src" / "haagent" / "runtime"

    top_level_modules = sorted(path.name for path in runtime_root.glob("*.py"))

    # performance.py 是交互延迟轨迹的 runtime 级 DTO，允许保留在顶层。
    assert top_level_modules == ["__init__.py", "performance.py"]


def test_runtime_top_level_packages_are_named_by_responsibility() -> None:
    runtime_root = Path(__file__).parents[3] / "src" / "haagent" / "runtime"

    package_names = sorted(
        path.name
        for path in runtime_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )

    assert package_names == [
        "contracts",
        "episodes",
        "evaluation",
        "events",
        "execution",
        "orchestration",
        "sandbox",
        "session",
        "settings",
    ]
