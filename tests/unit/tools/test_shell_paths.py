"""
tests/unit/tools/test_shell_paths.py - shell 命令路径扫描测试

验证常见 Bash/PowerShell 文件命令中的路径在执行前可被权限层识别。
"""

from __future__ import annotations

from pathlib import Path

from haagent.tools.shell_paths import collect_shell_paths


def test_collect_shell_paths_expands_powershell_environment_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))

    paths = collect_shell_paths(
        r"Get-Content $env:USERPROFILE\.config\settings.json",
        cwd=tmp_path,
    )

    assert paths == [(home / ".config" / "settings.json").resolve()]


def test_collect_shell_paths_handles_multiple_bash_file_commands(tmp_path: Path) -> None:
    external = tmp_path.parent / "external"

    paths = collect_shell_paths(
        f'cat "{external / "input.txt"}" && cp "{external / "input.txt"}" output.txt',
        cwd=tmp_path,
    )

    assert (external / "input.txt").resolve() in paths
    assert (tmp_path / "output.txt").resolve() in paths


def test_collect_shell_paths_does_not_guess_paths_inside_arbitrary_python(tmp_path: Path) -> None:
    paths = collect_shell_paths(
        'python -c "from pathlib import Path; Path.home().joinpath(\'secret\').read_text()"',
        cwd=tmp_path,
    )

    assert paths == []
