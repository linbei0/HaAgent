"""
tests/unit/scheduling/test_review_p2.py - 计划任务 review P2 行为

permission_mode 合法值、duplicate API、列表窗口滚动。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from haagent.scheduling.models import PERMISSION_MODES, RetryPolicy
from haagent.tui.overlays.schedule_editor import (
    PERMISSION_MODE_ORDER,
    ScheduleEditorState,
)
from haagent.tui.overlays.schedules import SchedulesOverlayState, VISIBLE_SCHEDULE_COUNT


def test_permission_mode_cycle_uses_real_modes() -> None:
    modes = list(PERMISSION_MODE_ORDER)
    assert set(modes) == set(PERMISSION_MODES)
    assert "deny" not in modes
    assert "allow" not in modes
    state = ScheduleEditorState(
        name="n",
        prompt="p",
        permission_mode="request_approval",
    )
    # 模拟 k 键循环三次应回到起点且不含 deny/allow
    cur = state.permission_mode
    seen = []
    for _ in range(3):
        idx = modes.index(cur) if cur in modes else 0
        cur = modes[(idx + 1) % len(modes)]
        seen.append(cur)
    assert set(seen) == set(modes)


def test_schedules_list_window_scrolls() -> None:
    items = [
        SimpleNamespace(
            id=f"sch_{i}",
            name=f"plan-{i}",
            status="active",
            rrule="FREQ=DAILY",
            next_run_at_utc=None,
            last_run_at_utc=None,
            workspace_root=Path("."),
        )
        for i in range(VISIBLE_SCHEDULE_COUNT + 5)
    ]
    state = SchedulesOverlayState(schedules=items, runs=[], selected_index=0)
    # 初始窗口只渲染可见条数
    text = "\n".join(state._list_lines())
    assert "plan-0" in text
    assert f"plan-{VISIBLE_SCHEDULE_COUNT}" not in text
    # 向下移动越过窗口底部应滚动
    for _ in range(VISIBLE_SCHEDULE_COUNT + 1):
        state = state.move(1)
    text = "\n".join(state._list_lines())
    assert f"plan-{state.selected_index}" in text
    assert state.scroll_offset > 0


def test_duplicate_schedule_creates_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from haagent.app.assistant_service import AssistantService
    from haagent.app.assistant_types import ScheduleCreateRequest
    from haagent.models.types import ModelResponse

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    home.mkdir()
    cfg = home / ".haagent"
    cfg.mkdir()
    (cfg / "providers.json").write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "local",
                        "name": "local",
                        "provider_id": "local",
                        "provider_name": "local",
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
    (cfg / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": "local", "model": "m1"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "k")

    class G:
        provider_name = "openai-chat"

        def generate(self, messages, tool_schemas):
            return ModelResponse("ok", [])

    service = AssistantService(
        workspace_root=ws,
        runs_root=tmp_path / "runs",
        environ={"HAAGENT_TEST_KEY": "k"},
        gateway_factory=lambda profile, **_kw: G(),
        schedule_db_path=tmp_path / "schedules.sqlite3",
    )
    created = service.schedules.create(
        ScheduleCreateRequest(
            name="orig",
            prompt="do work",
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
            retry_policy=RetryPolicy(max_attempts=2),
        )
    )
    copy = service.schedules.duplicate(created.id)
    assert copy.id != created.id
    assert copy.name.startswith("orig")
    assert "副本" in copy.name or "copy" in copy.name.lower() or copy.name != created.name
    assert copy.prompt == created.prompt
    assert copy.rrule == created.rrule
    assert copy.revision == 1
    assert len(service.schedules.list()) == 2
