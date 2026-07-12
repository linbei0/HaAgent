"""
tests/unit/scheduling/test_recurrence.py - RRULE、IANA 时区与 DST 语义

验证 normalize/preview/iter_due 使用固定 UTC 期望值，不依赖开发机时区。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from haagent.scheduling.models import RetryPolicy, ScheduleDefinition
from haagent.scheduling.recurrence import (
    RecurrenceError,
    iter_due_occurrences,
    normalize_rrule,
    preview_occurrences,
)


def _def(
    *,
    dtstart_local: datetime,
    timezone_name: str,
    rrule: str | None,
    **overrides: object,
) -> ScheduleDefinition:
    base: dict[str, object] = {
        "id": "sch_rec",
        "name": "rec",
        "prompt": "do work",
        "workspace_root": Path("E:/ws"),
        "destination_kind": "new_session",
        "destination_session_path": None,
        "connection_id": "c1",
        "model": "m1",
        "web_enabled": False,
        "allowed_tools": ("file_read",),
        "approval_allowed_tools": (),
        "approved_tools": (),
        "permission_mode": "request_approval",
        "dtstart_local": dtstart_local,
        "timezone": timezone_name,
        "rrule": rrule,
        "status": "active",
        "misfire_policy": "latest",
        "overlap_policy": "skip",
        "retry_policy": RetryPolicy(),
        "revision": 1,
    }
    base.update(overrides)
    return ScheduleDefinition(**base)  # type: ignore[arg-type]


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def test_normalize_rrule_strips_prefix_and_whitespace() -> None:
    assert normalize_rrule(" RRULE:FREQ=DAILY;INTERVAL=1 ") == "FREQ=DAILY;INTERVAL=1"
    assert normalize_rrule(None) is None
    assert normalize_rrule("") is None


def test_once_uses_dtstart_only() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="Asia/Shanghai",
        rrule=None,
    )
    after = _utc(2026, 7, 12, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=3)
    assert preview == (_utc(2026, 7, 13, 1, 0, 0),)
    # after 恰为触发点时，未来预览为空
    assert preview_occurrences(definition, after=_utc(2026, 7, 13, 1, 0, 0), count=3) == ()


def test_minute_interval() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=MINUTELY;INTERVAL=30",
    )
    after = _utc(2026, 7, 13, 9, 0, 0)
    preview = preview_occurrences(definition, after=after, count=3)
    assert preview == (
        _utc(2026, 7, 13, 9, 30, 0),
        _utc(2026, 7, 13, 10, 0, 0),
        _utc(2026, 7, 13, 10, 30, 0),
    )


def test_daily_shanghai_no_dst() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="Asia/Shanghai",
        rrule="FREQ=DAILY",
    )
    after = _utc(2026, 7, 12, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=2)
    assert preview == (
        _utc(2026, 7, 13, 1, 0, 0),
        _utc(2026, 7, 14, 1, 0, 0),
    )


def test_weekday_rrule() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),  # Monday
        timezone_name="UTC",
        rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    )
    after = _utc(2026, 7, 12, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=6)
    assert preview[:5] == (
        _utc(2026, 7, 13, 9, 0, 0),
        _utc(2026, 7, 14, 9, 0, 0),
        _utc(2026, 7, 15, 9, 0, 0),
        _utc(2026, 7, 16, 9, 0, 0),
        _utc(2026, 7, 17, 9, 0, 0),
    )
    assert preview[5] == _utc(2026, 7, 20, 9, 0, 0)


def test_monthly_last_weekday() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 1, 1, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1",
    )
    after = _utc(2025, 12, 1, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=2)
    assert preview[0] == _utc(2026, 1, 30, 9, 0, 0)
    assert preview[1] == _utc(2026, 2, 27, 9, 0, 0)


def test_count_includes_dtstart() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=DAILY;COUNT=2",
    )
    after = _utc(2026, 7, 1, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=5)
    assert preview == (
        _utc(2026, 7, 13, 9, 0, 0),
        _utc(2026, 7, 14, 9, 0, 0),
    )


def test_until_exclusive_boundary() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=DAILY;UNTIL=20260715T090000Z",
    )
    after = _utc(2026, 7, 1, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=5)
    assert preview == (
        _utc(2026, 7, 13, 9, 0, 0),
        _utc(2026, 7, 14, 9, 0, 0),
        _utc(2026, 7, 15, 9, 0, 0),
    )


def test_new_york_spring_skips_nonexistent_local_time() -> None:
    # 2026-03-08 02:30 在 America/New_York 不存在（春季拨快）
    definition = _def(
        dtstart_local=datetime(2026, 3, 7, 2, 30, 0),
        timezone_name="America/New_York",
        rrule="FREQ=DAILY",
    )
    after = _utc(2026, 3, 6, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=3)
    # 3/7 EST, 3/8 跳过 2:30, 3/9 EDT
    assert preview[0] == datetime(2026, 3, 7, 7, 30, tzinfo=timezone.utc)
    assert preview[1] == datetime(2026, 3, 9, 6, 30, tzinfo=timezone.utc)


def test_new_york_fall_uses_first_occurrence_fold0() -> None:
    # 2026-11-01 01:30 在 America/New_York 出现两次；选择 fold=0（较早/EDT）
    definition = _def(
        dtstart_local=datetime(2026, 10, 31, 1, 30, 0),
        timezone_name="America/New_York",
        rrule="FREQ=DAILY",
    )
    after = _utc(2026, 10, 30, 0, 0, 0)
    due = list(
        iter_due_occurrences(
            definition,
            after=after,
            through=_utc(2026, 11, 2, 12, 0, 0),
        )
    )
    nov1 = [d for d in due if d.date() == datetime(2026, 11, 1).date() or (
        d.astimezone(ZoneInfo("America/New_York")).date() == datetime(2026, 11, 1).date()
    )]
    # 只产生一次 11/1 实例，对应 fold=0 -> 05:30 UTC
    nov1_local = [
        d for d in due if d.astimezone(ZoneInfo("America/New_York")).day == 1
        and d.astimezone(ZoneInfo("America/New_York")).month == 11
    ]
    assert len(nov1_local) == 1
    assert nov1_local[0] == datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)


def test_invalid_month_end_dates_skipped() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 1, 31, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=MONTHLY",
    )
    after = _utc(2026, 1, 1, 0, 0, 0)
    preview = preview_occurrences(definition, after=after, count=3)
    assert preview[0] == _utc(2026, 1, 31, 9, 0, 0)
    # Feb 31 不存在，跳到 3 月 31
    assert preview[1] == _utc(2026, 3, 31, 9, 0, 0)


def test_windows_tzdata_asia_shanghai_resolves() -> None:
    # 显式依赖 tzdata 时 ZoneInfo 在 Windows 可用
    definition = _def(
        dtstart_local=datetime(2026, 1, 1, 0, 0, 0),
        timezone_name="Asia/Shanghai",
        rrule="FREQ=DAILY;COUNT=1",
    )
    preview = preview_occurrences(definition, after=_utc(2025, 12, 1, 0, 0, 0), count=1)
    assert preview == (_utc(2025, 12, 31, 16, 0, 0),)


@pytest.mark.parametrize(
    "rrule",
    [
        "FREQ=SECONDLY",
        "FREQ=MINUTELY;INTERVAL=30;BYSECOND=15",
        "FREQ=MINUTELY;INTERVAL=0",
        "FREQ=HOURLY;INTERVAL=0",
        "FREQ=MINUTELY",  # 默认间隔 1 分钟允许；下面单独测低于 1 分钟
        "FREQ=UNKNOWN",
        "FREQ=DAILY;FOO=1",
        "FREQ=DAILY;FREQ=WEEKLY",
        "FREQ=DAILY;COUNT=3;UNTIL=20260720T000000Z",
    ],
)
def test_normalize_rejects_invalid_rules(rrule: str) -> None:
    if rrule == "FREQ=MINUTELY":
        # 一分钟间隔合法
        assert normalize_rrule(rrule) == "FREQ=MINUTELY"
        return
    with pytest.raises(RecurrenceError):
        normalize_rrule(rrule)


def test_reject_sub_minute_interval() -> None:
    with pytest.raises(RecurrenceError) as exc:
        normalize_rrule("FREQ=MINUTELY;INTERVAL=0")
    assert exc.value.code == "interval_too_small"


def test_reject_naive_after_through() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=DAILY",
    )
    with pytest.raises(RecurrenceError) as exc:
        preview_occurrences(definition, after=datetime(2026, 7, 13, 9, 0, 0), count=1)
    assert exc.value.code == "naive_datetime"


def test_iter_due_occurrences_range() -> None:
    definition = _def(
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone_name="UTC",
        rrule="FREQ=HOURLY;INTERVAL=1",
    )
    due = list(
        iter_due_occurrences(
            definition,
            after=_utc(2026, 7, 13, 10, 0, 0),
            through=_utc(2026, 7, 13, 12, 0, 0),
        )
    )
    assert due == (
        [
            _utc(2026, 7, 13, 10, 0, 0),
            _utc(2026, 7, 13, 11, 0, 0),
            _utc(2026, 7, 13, 12, 0, 0),
        ]
    )
