"""
haagent/scheduling/recurrence.py - RRULE 规范化、预览与到期展开

使用 dateutil.rrulestr 与 zoneinfo 计算 occurrence；HaAgent 只处理产品策略与 DST 边界。
所有比较与返回值使用 UTC aware datetime。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import rrulestr

from haagent.scheduling.models import ScheduleDefinition

# dateutil / RFC 5545 常见键；未知键在规范化阶段拒绝
_ALLOWED_RRULE_KEYS = frozenset(
    {
        "FREQ",
        "INTERVAL",
        "COUNT",
        "UNTIL",
        "BYSECOND",
        "BYMINUTE",
        "BYHOUR",
        "BYDAY",
        "BYMONTHDAY",
        "BYYEARDAY",
        "BYWEEKNO",
        "BYMONTH",
        "BYSETPOS",
        "WKST",
    }
)
_ALLOWED_FREQ = frozenset(
    {"MINUTELY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY", "YEARLY"}
)


class RecurrenceError(ValueError):
    """RRULE 或时区语义错误。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def normalize_rrule(value: str | None) -> str | None:
    """规范化 RRULE 字符串（去掉 RRULE: 前缀）；None/空表示一次性计划。"""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.upper().startswith("RRULE:"):
        text = text[6:].strip()
    if not text:
        return None

    parts = [p.strip() for p in text.split(";") if p.strip()]
    keys: list[str] = []
    parsed: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise RecurrenceError("invalid_rrule_part", f"无效 RRULE 片段: {part!r}")
        key, raw = part.split("=", 1)
        key_u = key.strip().upper()
        raw_v = raw.strip()
        if key_u not in _ALLOWED_RRULE_KEYS:
            raise RecurrenceError("unknown_rrule_key", f"未知 RRULE 键: {key_u}")
        if key_u in parsed:
            raise RecurrenceError("duplicate_rrule_key", f"重复 RRULE 键: {key_u}")
        keys.append(key_u)
        parsed[key_u] = raw_v

    if "FREQ" not in parsed:
        raise RecurrenceError("missing_freq", "RRULE 必须包含 FREQ")
    freq = parsed["FREQ"].upper()
    if freq not in _ALLOWED_FREQ:
        # 失败边界：拒绝秒级及未知频率
        raise RecurrenceError("invalid_freq", f"不支持的 FREQ: {freq}")
    parsed["FREQ"] = freq

    if "COUNT" in parsed and "UNTIL" in parsed:
        raise RecurrenceError("count_and_until", "COUNT 与 UNTIL 不能同时使用")

    if "BYSECOND" in parsed:
        raise RecurrenceError("second_precision", "不支持秒级调度 (BYSECOND)")

    interval = 1
    if "INTERVAL" in parsed:
        try:
            interval = int(parsed["INTERVAL"])
        except ValueError as exc:
            raise RecurrenceError("invalid_interval", "INTERVAL 必须是整数") from exc
        if interval < 1:
            raise RecurrenceError("interval_too_small", "INTERVAL 必须 >= 1")

    # 规范化顺序：FREQ 优先，其余按出现顺序
    ordered = ["FREQ"] + [k for k in keys if k != "FREQ"]
    normalized = ";".join(f"{k}={parsed[k]}" for k in ordered)
    # 用 dateutil 再解析一次，尽早暴露语法错误
    try:
        rrulestr(normalized, ignoretz=True)
    except (ValueError, TypeError) as exc:
        raise RecurrenceError("rrule_parse", f"无法解析 RRULE: {exc}") from exc
    return normalized


def _require_aware(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise RecurrenceError("naive_datetime", f"{field} 必须是 timezone-aware UTC datetime")
    return value.astimezone(timezone.utc)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise RecurrenceError("invalid_timezone", f"未知 IANA 时区: {name}") from exc


def _local_to_utc(local_naive: datetime, tz: ZoneInfo) -> datetime | None:
    """墙上本地时间转 UTC；不存在的本地时间返回 None，歧义时间取 fold=0。"""
    if local_naive.tzinfo is not None:
        raise RecurrenceError("aware_local", "dtstart_local 必须是 naive 本地墙上时间")
    # fold=0：秋季重复时间取较早实例
    candidate = local_naive.replace(tzinfo=tz, fold=0)
    # round-trip 检测春季不存在时间：转 UTC 再回本地，分钟/小时应一致
    as_utc = candidate.astimezone(timezone.utc)
    back = as_utc.astimezone(tz)
    if (
        back.year != local_naive.year
        or back.month != local_naive.month
        or back.day != local_naive.day
        or back.hour != local_naive.hour
        or back.minute != local_naive.minute
        or back.second != local_naive.second
    ):
        return None
    return as_utc


def _occurrences_utc(
    definition: ScheduleDefinition,
    *,
    after_utc: datetime,
    through_utc: datetime | None,
    count: int | None,
    include_after_equal: bool,
) -> list[datetime]:
    tz = _zone(definition.timezone)
    after_utc = _require_aware(after_utc, field="after")
    if through_utc is not None:
        through_utc = _require_aware(through_utc, field="through")

    rule = normalize_rrule(definition.rrule)
    results: list[datetime] = []

    if rule is None:
        start_utc = _local_to_utc(definition.dtstart_local, tz)
        if start_utc is None:
            return results
        if include_after_equal:
            ok = start_utc >= after_utc
        else:
            ok = start_utc > after_utc
        if ok and (through_utc is None or start_utc <= through_utc):
            results.append(start_utc)
        return results

    rr = rrulestr(rule, dtstart=definition.dtstart_local, ignoretz=True)
    # 从 after 前一天起扫，避免漏掉本地边界
    after_local = after_utc.astimezone(tz).replace(tzinfo=None) - timedelta(days=1)
    # dateutil 按本地 naive 墙上时间展开。
    candidates = rr.xafter(
        after_local,
        count=5000 if count is None else max(count * 20, 50),
    )

    seen: set[datetime] = set()
    for local_occ in candidates:
        if not isinstance(local_occ, datetime):
            continue
        if local_occ.tzinfo is not None:
            local_occ = local_occ.replace(tzinfo=None)
        utc_occ = _local_to_utc(local_occ, tz)
        if utc_occ is None:
            continue
        if utc_occ in seen:
            continue
        seen.add(utc_occ)
        if include_after_equal:
            if utc_occ < after_utc:
                continue
        else:
            if utc_occ <= after_utc:
                continue
        if through_utc is not None and utc_occ > through_utc:
            break
        results.append(utc_occ)
        if count is not None and len(results) >= count:
            break
        # 防止无界规则在 through 缺失时跑飞
        if through_utc is None and count is None and len(results) >= 1000:
            break
    return results


def preview_occurrences(
    definition: ScheduleDefinition,
    *,
    after: datetime,
    count: int,
) -> tuple[datetime, ...]:
    """返回严格晚于 after 的最多 count 次 UTC occurrence（未来预览）。"""
    if count < 1:
        raise RecurrenceError("invalid_count", "count 必须 >= 1")
    # 预览语义：after 表示“现在”，不包含当前时刻上的触发点
    items = _occurrences_utc(
        definition,
        after_utc=after,
        through_utc=None,
        count=count,
        include_after_equal=False,
    )
    return tuple(items)


def iter_due_occurrences(
    definition: ScheduleDefinition,
    *,
    after: datetime,
    through: datetime,
    limit: int | None = None,
) -> Iterator[datetime]:
    """展开 (after, through] 内的到期 occurrence；limit 限制条数供有界批处理。"""
    items = _occurrences_utc(
        definition,
        after_utc=after,
        through_utc=through,
        count=limit,
        include_after_equal=True,
    )
    # due 语义：after 之后到 through（含）；若 after 恰为触发点则包含
    yield from items
