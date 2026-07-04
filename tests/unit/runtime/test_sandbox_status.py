"""
tests/unit/runtime/test_sandbox_status.py - 沙箱用户状态服务测试

验证沙箱状态、Docker 诊断，以及显式开启/关闭配置的用户可见行为。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from haagent.runtime.sandbox.status import (
    disable_sandbox,
    enable_docker_sandbox,
    sandbox_doctor_report,
    sandbox_user_status,
)
from haagent.runtime.settings import load_runtime_settings


def test_sandbox_user_status_defaults_to_local(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"

    status = sandbox_user_status(config_path=config_path)

    assert status.backend == "local_subprocess"
    assert status.isolation_level == "weak"
    assert status.network_policy == "unrestricted"
    assert status.credential_policy == "inherit_environment"
    assert status.degraded is True
    assert "haagent sandbox enable docker" in status.recommendation
    assert status.config_path == config_path


def test_enable_docker_sandbox_preserves_existing_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"interactive_max_turns": 80, "active_profile": "local"}), encoding="utf-8")

    status = enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    settings = load_runtime_settings(config_path=config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    assert status.backend == "docker"
    assert status.degraded is False
    assert settings.interactive_max_turns == 80
    assert settings.sandbox.enabled is True
    assert settings.sandbox.backend == "docker"
    assert settings.sandbox.fail_if_unavailable is True
    assert raw["active_profile"] == "local"
    assert raw["sandbox"]["docker"]["image"] == "haagent-sandbox:py311"


def test_enable_docker_sandbox_can_allow_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"

    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=False)
    settings = load_runtime_settings(config_path=config_path)

    assert settings.sandbox.enabled is True
    assert settings.sandbox.backend == "docker"
    assert settings.sandbox.fail_if_unavailable is False


def test_disable_sandbox_preserves_unrelated_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"
    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["active_profile"] = "local"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    status = disable_sandbox(config_path=config_path)
    settings = load_runtime_settings(config_path=config_path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert status.backend == "local_subprocess"
    assert settings.sandbox.enabled is False
    assert settings.sandbox.backend == "local_subprocess"
    assert saved["active_profile"] == "local"


def test_sandbox_doctor_reports_missing_docker_cli(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "settings.json"
    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    monkeypatch.setattr("haagent.runtime.sandbox.status.shutil.which", lambda name: None)

    report = sandbox_doctor_report(config_path=config_path)

    assert report.backend == "docker"
    assert report.ready is False
    assert report.docker_cli == "missing"
    assert report.docker_daemon == "not_checked"
    assert "Install Docker Desktop" in report.next_action


def test_sandbox_doctor_reports_daemon_failure(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "settings.json"
    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    monkeypatch.setattr("haagent.runtime.sandbox.status.shutil.which", lambda name: "docker")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="daemon unavailable")

    monkeypatch.setattr("haagent.runtime.sandbox.status.subprocess.run", fake_run)

    report = sandbox_doctor_report(config_path=config_path)

    assert report.ready is False
    assert report.docker_cli == "found"
    assert report.docker_daemon == "unavailable"
    assert "daemon unavailable" in report.reason


def test_sandbox_doctor_reports_missing_image_as_auto_build_ready(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "settings.json"
    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    monkeypatch.setattr("haagent.runtime.sandbox.status.shutil.which", lambda name: "docker")
    monkeypatch.setattr(
        "haagent.runtime.sandbox.status.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr("haagent.runtime.sandbox.status.image_exists", lambda image: False)

    report = sandbox_doctor_report(config_path=config_path)

    assert report.ready is True
    assert report.image == "missing"
    assert report.auto_build_image is True
    assert "first sandbox run will build" in report.next_action


def test_sandbox_doctor_reports_ready_when_image_exists(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "settings.json"
    enable_docker_sandbox(config_path=config_path, fail_if_unavailable=True)
    monkeypatch.setattr("haagent.runtime.sandbox.status.shutil.which", lambda name: "docker")
    monkeypatch.setattr(
        "haagent.runtime.sandbox.status.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr("haagent.runtime.sandbox.status.image_exists", lambda image: True)

    report = sandbox_doctor_report(config_path=config_path)

    assert report.ready is True
    assert report.docker_daemon == "running"
    assert report.image == "present"
    assert report.next_action == "Docker sandbox is ready."
