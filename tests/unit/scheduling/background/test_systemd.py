"""
tests/unit/scheduling/background/test_systemd.py - systemd user service adapter

快照 unit 内容：Restart=on-failure、绝对路径、daemon-reload 与 enable/disable 顺序。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from haagent.scheduling.background.systemd import (
    SERVICE_NAME,
    SystemdBackgroundAdapter,
)


class FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_install_writes_unit_with_restart_and_worker_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit_dir = tmp_path / "systemd" / "user"
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return FakeCompleted(0, stdout="active")

    monkeypatch.setattr("haagent.scheduling.background.systemd.subprocess.run", fake_run)

    adapter = SystemdBackgroundAdapter(unit_dir=unit_dir)
    status = adapter.install()
    assert status.host_type == "systemd_user"
    assert status.state in {"installed", "running", "stopped"}

    unit_path = unit_dir / SERVICE_NAME
    assert unit_path.exists()
    content = unit_path.read_text(encoding="utf-8")
    assert "Restart=on-failure" in content
    assert "WantedBy=default.target" in content
    assert sys.executable in content
    assert "-m" in content
    assert "haagent.cli" in content
    assert "schedule-worker" in content

    # daemon-reload 后 enable --now 或 enable+start
    assert any("daemon-reload" in c for c in calls)
    enable_idx = next(i for i, c in enumerate(calls) if "enable" in c)
    reload_idx = next(i for i, c in enumerate(calls) if "daemon-reload" in c)
    assert reload_idx < enable_idx
    # 参数数组，无 shell 拼接
    for c in calls:
        assert isinstance(c, list)
        assert c[0] in {"systemctl", "systemctl.exe"} or c[0].endswith("systemctl")


def test_uninstall_disable_then_remove(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    unit_path = unit_dir / SERVICE_NAME
    unit_path.write_text("[Unit]\nDescription=x\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return FakeCompleted(0)

    monkeypatch.setattr("haagent.scheduling.background.systemd.subprocess.run", fake_run)
    adapter = SystemdBackgroundAdapter(unit_dir=unit_dir)
    status = adapter.uninstall()
    assert status.state == "not_installed"
    assert any("disable" in c for c in calls)
    assert not unit_path.exists()
    assert any("daemon-reload" in c for c in calls)


def test_status_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit_dir = tmp_path / "empty"
    unit_dir.mkdir()
    monkeypatch.setattr(
        "haagent.scheduling.background.systemd.subprocess.run",
        lambda *a, **k: FakeCompleted(1),
    )
    status = SystemdBackgroundAdapter(unit_dir=unit_dir).status()
    assert status.state == "not_installed"
