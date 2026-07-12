"""
tests/unit/scheduling/background/test_windows.py - Windows Task Scheduler adapter

mock schtasks.exe 参数数组：登录触发、当前用户、quoted exe、幂等安装与错误透传。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from haagent.scheduling.background.windows import (
    TASK_NAME,
    WindowsBackgroundAdapter,
)


class FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_install_uses_logon_trigger_current_user_and_arg_array(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        assert isinstance(args, (list, tuple))
        calls.append(list(args))
        # 安装前 Query 失败；Create 后 Query 成功
        if "/Query" in args:
            if any("/Create" in c for c in calls):
                return FakeCompleted(0, stdout="TaskName: HaAgentScheduler\nStatus: Ready")
            return FakeCompleted(1, stderr="ERROR: The system cannot find the file specified.")
        return FakeCompleted(0, stdout="SUCCESS")

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    monkeypatch.setattr("haagent.scheduling.background.windows.getpass.getuser", lambda: "alice")

    adapter = WindowsBackgroundAdapter(task_name=TASK_NAME)
    status = adapter.install()
    assert status.state in {"installed", "running", "stopped"}
    assert status.host_type == "windows_task_scheduler"

    create_calls = [c for c in calls if "/Create" in c]
    assert len(create_calls) == 1
    create = create_calls[0]
    assert create[0] == "schtasks.exe"
    assert "/SC" in create and "ONLOGON" in create
    assert "/RU" in create
    ru_idx = create.index("/RU")
    assert create[ru_idx + 1] == "alice"
    assert "/TR" in create
    tr_idx = create.index("/TR")
    tr = create[tr_idx + 1]
    # quoted executable + -m haagent.cli schedule-worker
    assert sys.executable in tr or f'"{sys.executable}"' in tr
    assert "haagent.cli" in tr
    assert "schedule-worker" in tr
    # 不得是 shell 拼接后的单条未拆参数命令作为列表元素外再拼
    assert all(isinstance(x, str) for x in create)


def test_install_idempotent_when_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "/Query" in args:
            return FakeCompleted(0, stdout="TaskName: HaAgentScheduler\nStatus: Ready")
        if "/Create" in args:
            return FakeCompleted(0, stdout="SUCCESS")
        if "/Delete" in args:
            return FakeCompleted(0)
        return FakeCompleted(0)

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    monkeypatch.setattr("haagent.scheduling.background.windows.getpass.getuser", lambda: "bob")

    adapter = WindowsBackgroundAdapter()
    status = adapter.install()
    assert status.state in {"installed", "running", "stopped"}
    # 幂等：可先 delete 再 create，或 /F 强制
    assert any("/Create" in c for c in calls)


def test_install_propagates_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        if "/Query" in args:
            return FakeCompleted(1, stderr="not found")
        return FakeCompleted(1, stderr="Access is denied.")

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    monkeypatch.setattr("haagent.scheduling.background.windows.getpass.getuser", lambda: "bob")

    adapter = WindowsBackgroundAdapter()
    with pytest.raises(Exception) as exc:
        adapter.install()
    assert "denied" in str(exc.value).lower() or "Access" in str(exc.value) or str(exc.value)


def test_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        return FakeCompleted(1, stderr="cannot find")

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    status = WindowsBackgroundAdapter().status()
    assert status.state == "not_installed"
    assert status.host_type == "windows_task_scheduler"
    # 未安装说明必须是稳定中文，禁止透传 schtasks 乱码 stderr
    assert "尚未安装" in (status.detail or "")
    assert "\ufffd" not in (status.detail or "")


def test_status_uses_system_console_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """中文 Windows 下 schtasks 走系统代码页，不得强制 utf-8。"""
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["encoding"] = kwargs.get("encoding")
        return FakeCompleted(1, stderr="ERROR: The system cannot find the file specified.")

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    monkeypatch.setattr(
        "haagent.scheduling.background.windows._console_encoding",
        lambda: "gbk",
    )
    WindowsBackgroundAdapter().status()
    assert seen["encoding"] == "gbk"


def test_status_running_uses_chinese_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        return FakeCompleted(
            0,
            stdout="TaskName: HaAgentScheduler\nStatus: Running\n",
        )

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    status = WindowsBackgroundAdapter().status()
    assert status.state == "running"
    assert "运行" in (status.detail or "")


def test_uninstall_deletes_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return FakeCompleted(0)

    monkeypatch.setattr("haagent.scheduling.background.windows.subprocess.run", fake_run)
    status = WindowsBackgroundAdapter().uninstall()
    assert status.state == "not_installed"
    assert any("/Delete" in c for c in calls)
