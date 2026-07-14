"""
tests/unit/scheduling/test_models.py - 计划任务领域类型与验证规则

验证 ScheduleDefinition / RetryPolicy / validate_schedule 的纯函数边界。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from haagent.scheduling.models import (
    RetryPolicy,
    ScheduleDefinition,
    ScheduleValidationError,
    merge_web_tools,
    validate_schedule,
)


def _valid_definition(**overrides: object) -> ScheduleDefinition:
    base: dict[str, object] = {
        "id": "sch_test",
        "name": "daily report",
        "prompt": "Summarize the workspace",
        "workspace_root": Path("E:/workspace/project"),
        "destination_kind": "new_session",
        "destination_session_path": None,
        "connection_id": "conn_local",
        "model": "gpt-test",
        "web_enabled": False,
        "allowed_tools": ("file_read", "file_list"),
        "approval_allowed_tools": ("file_write",),
        "approved_tools": (),
        "permission_mode": "request_approval",
        "dtstart_local": datetime(2026, 7, 13, 9, 0, 0),
        "timezone": "Asia/Shanghai",
        "rrule": "FREQ=DAILY",
        "status": "active",
        "misfire_policy": "latest",
        "overlap_policy": "skip",
        "retry_policy": RetryPolicy(),
        "revision": 1,
    }
    base.update(overrides)
    return ScheduleDefinition(**base)  # type: ignore[arg-type]


def test_validate_schedule_accepts_valid_definition() -> None:
    definition = _valid_definition()
    assert validate_schedule(definition) == definition


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"name": "   "}, "empty_name"),
        ({"prompt": "\n"}, "empty_prompt"),
        ({"workspace_root": Path("relative/path")}, "relative_workspace"),
        ({"destination_kind": "other"}, "invalid_destination"),  # type: ignore[dict-item]
        (
            {
                "destination_kind": "resume_session",
                "destination_session_path": None,
            },
            "resume_requires_session",
        ),
        (
            {
                "destination_kind": "resume_session",
                "destination_session_path": Path("E:/sessions/s1"),
                "overlap_policy": "parallel",
            },
            "resume_forbids_parallel",
        ),
        (
            {
                "overlap_policy": "parallel",
                "allowed_tools": ("file_read", "file_write"),
            },
            "parallel_forbids_side_effects",
        ),
        (
            {
                "approved_tools": ("file_write", "shell"),
                "approval_allowed_tools": ("file_write",),
            },
            "approved_not_subset",
        ),
        ({"retry_policy": RetryPolicy(max_attempts=0)}, "retry_max_attempts"),
        ({"retry_policy": RetryPolicy(initial_delay_seconds=-1)}, "retry_initial_delay"),
        ({"retry_policy": RetryPolicy(multiplier=0.5)}, "retry_multiplier"),
        ({"retry_policy": RetryPolicy(max_delay_seconds=0)}, "retry_max_delay"),
        (
            {
                "retry_policy": RetryPolicy(
                    initial_delay_seconds=100,
                    max_delay_seconds=50,
                )
            },
            "retry_delay_order",
        ),
        ({"timezone": ""}, "empty_timezone"),
        ({"status": "running"}, "invalid_status"),  # type: ignore[dict-item]
        ({"misfire_policy": "none"}, "invalid_misfire"),  # type: ignore[dict-item]
        ({"overlap_policy": "race"}, "invalid_overlap"),  # type: ignore[dict-item]
        ({"permission_mode": "unsafe"}, "invalid_permission_mode"),  # type: ignore[dict-item]
        ({"revision": 0}, "invalid_revision"),
    ],
)
def test_validate_schedule_rejects_invalid_fields(
    overrides: dict[str, object],
    code: str,
) -> None:
    definition = _valid_definition(**overrides)
    with pytest.raises(ScheduleValidationError) as exc_info:
        validate_schedule(definition)
    assert exc_info.value.code == code


def test_validate_schedule_rejects_new_session_with_session_path() -> None:
    definition = _valid_definition(
        destination_kind="new_session",
        destination_session_path=Path("E:/sessions/s1"),
    )
    with pytest.raises(ScheduleValidationError) as exc_info:
        validate_schedule(definition)
    assert exc_info.value.code == "new_session_forbids_path"


def test_retry_policy_defaults() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.initial_delay_seconds == 30
    assert policy.multiplier == 2.0
    assert policy.max_delay_seconds == 900


def test_schedule_definition_is_frozen() -> None:
    definition = _valid_definition()
    with pytest.raises(Exception):
        definition.name = "other"  # type: ignore[misc]


def test_merge_web_tools_respects_enablement_without_duplicates() -> None:
    assert merge_web_tools(("file_read", "file_list"), web_enabled=True) == (
        "file_read",
        "file_list",
        "web_search",
        "web_fetch",
    )
    assert merge_web_tools(("file_read", "web_search"), web_enabled=False) == (
        "file_read",
        "web_search",
    )
    assert merge_web_tools(
        ("file_read", "web_search", "web_fetch"), web_enabled=True
    ) == ("file_read", "web_search", "web_fetch")
