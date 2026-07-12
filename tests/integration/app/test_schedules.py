"""
tests/integration/app/test_schedules.py - AssistantSchedules 应用层 API

验证 create/preview/edit/pause/resume/archive/delete/run-now/inbox 与错误映射。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from haagent.app.assistant_service import AssistantService
from haagent.app.assistant_types import (
    AssistantServiceError,
    ScheduleCreateRequest,
    SchedulePreviewRequest,
    ScheduleUpdateRequest,
    RunQuery,
)
from haagent.models.types import ModelResponse
from haagent.scheduling.models import RetryPolicy


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _write_connection(home: Path, *, name: str = "local", model: str = "m1") -> None:
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": name,
                        "name": name,
                        "provider_id": name,
                        "provider_name": name,
                        "gateway_provider": "openai-chat",
                        "base_url": "https://example.test",
                        "api_key_env": "HAAGENT_TEST_KEY",
                        "credential_source": "env",
                    }
                ],
                "custom_models": [],
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": name, "model": model}}),
        encoding="utf-8",
    )


class RecordingGateway:
    provider_name = "openai-chat"

    def generate(self, messages, tool_schemas):
        return ModelResponse("ok", [])


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> AssistantService:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")
    db = tmp_path / "schedules.sqlite3"
    return AssistantService(
        workspace_root=ws,
        runs_root=tmp_path / "runs",
        environ={"HAAGENT_TEST_KEY": "test-key"},
        gateway_factory=lambda profile, **_kw: RecordingGateway(),
        schedule_db_path=db,
        **kwargs,
    )


def _create_req(ws: Path, **overrides) -> ScheduleCreateRequest:
    base = dict(
        name="daily notes",
        prompt="整理今日笔记",
        workspace_root=ws,
        destination_kind="new_session",
        destination_session_path=None,
        connection_id="local",
        model="m1",
        web_enabled=False,
        allowed_tools=("file_read",),
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule="FREQ=DAILY",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=RetryPolicy(max_attempts=2),
    )
    base.update(overrides)
    return ScheduleCreateRequest(**base)  # type: ignore[arg-type]


def test_assistant_service_exposes_schedules_module(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    assert hasattr(service, "schedules")
    assert service.schedules._context is service.sessions._context


def test_create_list_get_preview(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    created = service.schedules.create(_create_req(ws))
    assert created.id
    assert created.name == "daily notes"
    assert created.status == "active"
    assert created.revision == 1
    assert created.next_run_at_utc is not None

    items = service.schedules.list()
    assert len(items) == 1
    assert items[0].id == created.id

    got = service.schedules.get(created.id)
    assert got.prompt == "整理今日笔记"

    preview = service.schedules.preview(
        SchedulePreviewRequest(
            dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
            timezone="UTC",
            rrule="FREQ=DAILY",
        ),
        count=3,
    )
    assert len(preview) == 3
    assert all(p.tzinfo is not None for p in preview)


def test_update_pause_resume_archive_delete(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    created = service.schedules.create(_create_req(ws))

    updated = service.schedules.update(
        created.id,
        ScheduleUpdateRequest(name="renamed", expected_revision=1),
    )
    assert updated.name == "renamed"
    assert updated.revision == 2

    paused = service.schedules.pause(created.id)
    assert paused.status == "paused"

    resumed = service.schedules.resume(created.id, now=_utc(2026, 7, 13, 10, 0, 0))
    assert resumed.status == "active"
    assert resumed.next_run_at_utc is not None

    archived = service.schedules.archive(created.id)
    assert archived.status == "archived"
    assert service.schedules.list() == []

    service.schedules.delete(created.id)
    with pytest.raises(AssistantServiceError):
        service.schedules.get(created.id)


def test_run_now_and_inbox_read_cancel(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    created = service.schedules.create(_create_req(ws, rrule=None))

    run = service.schedules.run_now(created.id, request_id="req-1")
    assert run.id
    assert run.schedule_id == created.id
    assert run.trigger_kind == "manual"
    assert run.status in {"queued", "running", "succeeded", "failed", "needs_attention"}

    # 幂等 request_id
    again = service.schedules.run_now(created.id, request_id="req-1")
    assert again.id == run.id

    runs = service.schedules.list_runs(RunQuery())
    assert any(r.id == run.id for r in runs)

    service.schedules.mark_run_read(run.id)
    unread = service.schedules.list_runs(RunQuery(unread_only=True))
    assert all(r.id != run.id for r in unread)

    # 再造一个未读
    run2 = service.schedules.run_now(created.id, request_id="req-2")
    count = service.schedules.mark_all_runs_read()
    assert count >= 1

    # 取消 queued：若已执行完则可能 not cancellable
    run3 = service.schedules.run_now(created.id, request_id="req-3")
    try:
        cancelled = service.schedules.cancel_run(run3.id)
        assert cancelled.status in {"cancelled", "running", "succeeded", "failed", "needs_attention"}
    except AssistantServiceError:
        pass


def test_invalid_id_and_revision_conflict_chinese_errors(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    created = service.schedules.create(_create_req(ws))

    with pytest.raises(AssistantServiceError) as exc:
        service.schedules.get("missing-id")
    assert "不存在" in str(exc.value) or "找不到" in str(exc.value)

    with pytest.raises(AssistantServiceError) as exc2:
        service.schedules.update(
            created.id,
            ScheduleUpdateRequest(name="x", expected_revision=99),
        )
    assert "revision" in str(exc2.value).lower() or "冲突" in str(exc2.value)


def test_unknown_connection_rejected(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    with pytest.raises(AssistantServiceError) as exc:
        service.schedules.create(_create_req(ws, connection_id="no-such-conn"))
    msg = str(exc.value)
    assert "连接" in msg or "connection" in msg.lower() or "不存在" in msg


def test_missing_workspace_rejected(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    missing = tmp_path / "no-ws"
    with pytest.raises(AssistantServiceError) as exc:
        service.schedules.create(_create_req(missing))
    assert "workspace" in str(exc.value).lower() or "不存在" in str(exc.value) or "目录" in str(exc.value)


def test_background_status_and_install_stub(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    status = service.schedules.background_status()
    assert status.state in {"unsupported", "not_installed", "stopped", "running", "error"}
    # Task 6 可 stub：install 应返回状态或抛明确错误，不静默成功
    try:
        installed = service.schedules.install_background_service()
        assert installed is not None
    except AssistantServiceError as err:
        assert str(err)
    try:
        uninstalled = service.schedules.uninstall_background_service()
        assert uninstalled is not None
    except AssistantServiceError as err:
        assert str(err)


def test_validation_error_on_empty_name(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path, monkeypatch)
    ws = service._context.workspace_root
    with pytest.raises(AssistantServiceError):
        service.schedules.create(_create_req(ws, name="  "))
