"""
tests/unit/scheduling/background/test_launchd.py - launchd agent adapter

plistlib 解析：label、ProgramArguments、RunAtLoad、KeepAlive。
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

import haagent.scheduling.background.launchd as launchd_module
from haagent.scheduling.background.base import (
    BackgroundServiceError,
    BackgroundServiceUnsupported,
)
from haagent.scheduling.background.factory import create_background_adapter
from haagent.scheduling.background.launchd import (
    LABEL,
    PLIST_NAME,
    LaunchdBackgroundAdapter,
)


class FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LaunchdBackgroundAdapter:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(launchd_module.os, "getuid", lambda: 1000, raising=False)
    return LaunchdBackgroundAdapter()


def test_install_writes_plist_with_required_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agents = tmp_path / "Library" / "LaunchAgents"
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return FakeCompleted(0)

    monkeypatch.setattr("haagent.scheduling.background.launchd.subprocess.run", fake_run)

    adapter = _adapter(monkeypatch, tmp_path)
    status = adapter.install()
    assert status.host_type == "launchd"
    assert status.state in {"installed", "running", "stopped"}

    plist_path = agents / PLIST_NAME
    assert plist_path.exists()
    with plist_path.open("rb") as fh:
        data = plistlib.load(fh)
    assert data["Label"] == LABEL
    args = data["ProgramArguments"]
    assert isinstance(args, list)
    assert args[0] == sys.executable
    assert args[1:4] == ["-m", "haagent.cli", "schedule-worker"]
    assert data["RunAtLoad"] is True
    # KeepAlive 仅在异常退出时恢复：dict 或 True 均可，但不能静默缺失
    assert "KeepAlive" in data
    keep = data["KeepAlive"]
    assert keep is True or (isinstance(keep, dict) and keep.get("SuccessfulExit") is False)

    assert any("load" in c or "bootstrap" in c for c in calls)


def test_uninstall_unloads_and_removes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    plist_path = agents / PLIST_NAME
    with plist_path.open("wb") as fh:
        plistlib.dump({"Label": LABEL, "ProgramArguments": [sys.executable]}, fh)
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return FakeCompleted(0)

    monkeypatch.setattr("haagent.scheduling.background.launchd.subprocess.run", fake_run)
    status = _adapter(monkeypatch, tmp_path).uninstall()
    assert status.state == "not_installed"
    assert not plist_path.exists()
    assert any("unload" in c or "bootout" in c for c in calls)


@pytest.mark.parametrize(
    ("print_stderr", "print_stdout", "should_succeed"),
    [
        ("launchctl print failed: access denied", "", False),
        ('Could not find service "com.haagent.scheduler" in domain', "", True),
        ("", "state = running\npid = 1\n", False),
    ],
)
def test_uninstall_requires_proof_service_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    print_stderr: str,
    print_stdout: str,
    should_succeed: bool,
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / PLIST_NAME
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("x", encoding="utf-8")

    def fake_run(args, **kwargs):
        if "print" in args:
            return FakeCompleted(
                1 if print_stderr else 0,
                stdout=print_stdout,
                stderr=print_stderr,
            )
        return FakeCompleted(1, stderr="bootout failed")

    monkeypatch.setattr("haagent.scheduling.background.launchd.subprocess.run", fake_run)
    if should_succeed:
        assert adapter.uninstall().state == "not_installed"
        assert not plist_path.exists()
    else:
        with pytest.raises(BackgroundServiceError):
            adapter.uninstall()
        assert plist_path.exists()


def test_status_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status = _adapter(monkeypatch, tmp_path).status()
    assert status.state == "not_installed"


def test_status_exposes_launchctl_access_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)
    plist_path = tmp_path / "Library" / "LaunchAgents" / PLIST_NAME
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "haagent.scheduling.background.launchd.subprocess.run",
        lambda *args, **kwargs: FakeCompleted(1, stderr="access denied"),
    )
    with pytest.raises(BackgroundServiceError, match="access denied"):
        adapter.status()


def test_unsupported_platform_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("haagent.scheduling.background.factory.sys.platform", "aix")
    adapter = create_background_adapter()
    with pytest.raises(BackgroundServiceUnsupported):
        adapter.install()
    status = adapter.status()
    assert status.state == "unsupported"
