"""
haagent/scheduling/store.py - 计划任务 SQLite 持久化

CRUD、不可变 revision 快照、租约与 run claim；所有写操作显式事务，禁止拼接 SQL。
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haagent.scheduling.migrations import apply_migrations
from haagent.scheduling.models import (
    DestinationKind,
    FailureCategory,
    MisfirePolicy,
    OverlapPolicy,
    RetryPolicy,
    RunStatus,
    ScheduleDefinition,
    ScheduleRun,
    ScheduleStatus,
    TriggerKind,
    validate_schedule,
)


class ScheduleStoreError(RuntimeError):
    """存储层失败；code 供应用层映射。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _dt_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ScheduleStoreError("naive_datetime", "存储时间必须是 aware UTC")
    return value.astimezone(timezone.utc).isoformat()


def _dt_from_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_str_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ScheduleStoreError("invalid_json", f"{field} 必须是 list[str]")
    return tuple(value)


def _retry_from_json(raw: str) -> RetryPolicy:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScheduleStoreError("invalid_json", "retry_policy_json 无效") from exc
    if not isinstance(data, dict):
        raise ScheduleStoreError("invalid_json", "retry_policy_json 必须是对象")
    try:
        return RetryPolicy(
            max_attempts=int(data["max_attempts"]),
            initial_delay_seconds=int(data["initial_delay_seconds"]),
            multiplier=float(data["multiplier"]),
            max_delay_seconds=int(data["max_delay_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScheduleStoreError("invalid_json", "retry_policy_json 字段不完整") from exc


def _definition_to_json(definition: ScheduleDefinition) -> str:
    payload = {
        "id": definition.id,
        "name": definition.name,
        "prompt": definition.prompt,
        "workspace_root": str(definition.workspace_root),
        "destination_kind": definition.destination_kind,
        "destination_session_path": (
            str(definition.destination_session_path)
            if definition.destination_session_path
            else None
        ),
        "connection_id": definition.connection_id,
        "model": definition.model,
        "web_enabled": definition.web_enabled,
        "allowed_tools": list(definition.allowed_tools),
        "approval_allowed_tools": list(definition.approval_allowed_tools),
        "approved_tools": list(definition.approved_tools),
        "permission_mode": definition.permission_mode,
        "dtstart_local": definition.dtstart_local.isoformat(),
        "timezone": definition.timezone,
        "rrule": definition.rrule,
        "status": definition.status,
        "misfire_policy": definition.misfire_policy,
        "overlap_policy": definition.overlap_policy,
        "retry_policy": asdict(definition.retry_policy),
        "revision": definition.revision,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _definition_from_json(raw: str) -> ScheduleDefinition:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScheduleStoreError("invalid_json", "definition_json 无效") from exc
    if not isinstance(data, dict):
        raise ScheduleStoreError("invalid_json", "definition_json 必须是对象")
    try:
        dest_path = data.get("destination_session_path")
        definition = ScheduleDefinition(
            id=str(data["id"]),
            name=str(data["name"]),
            prompt=str(data["prompt"]),
            workspace_root=Path(str(data["workspace_root"])),
            destination_kind=data["destination_kind"],  # type: ignore[arg-type]
            destination_session_path=Path(dest_path) if dest_path else None,
            connection_id=str(data["connection_id"]),
            model=str(data["model"]),
            web_enabled=bool(data["web_enabled"]),
            allowed_tools=_json_str_list(data["allowed_tools"], field="allowed_tools"),
            approval_allowed_tools=_json_str_list(
                data["approval_allowed_tools"], field="approval_allowed_tools"
            ),
            approved_tools=_json_str_list(data["approved_tools"], field="approved_tools"),
            permission_mode=data["permission_mode"],  # type: ignore[arg-type]
            dtstart_local=datetime.fromisoformat(str(data["dtstart_local"])),
            timezone=str(data["timezone"]),
            rrule=data.get("rrule"),
            status=data["status"],  # type: ignore[arg-type]
            misfire_policy=data["misfire_policy"],  # type: ignore[arg-type]
            overlap_policy=data["overlap_policy"],  # type: ignore[arg-type]
            retry_policy=RetryPolicy(**data["retry_policy"]),
            revision=int(data["revision"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScheduleStoreError("invalid_json", f"definition_json 无法还原: {exc}") from exc
    return validate_schedule(definition)


def _store_locked(method: Any) -> Any:
    """公开方法显式持锁；RLock 允许同线程嵌套调用。"""

    @functools.wraps(method)
    def wrapper(self: ScheduleStore, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _lock_public_methods(cls: type) -> type:
    # 类级一次性包装，替代 __getattribute__ 动态代理
    for name, attr in list(cls.__dict__.items()):
        if name.startswith("_"):
            continue
        if not callable(attr):
            continue
        setattr(cls, name, _store_locked(attr))
    return cls


@_lock_public_methods
class ScheduleStore:
    """用户级计划库连接包装。

    单连接 + check_same_thread=False 供 TUI 与 worker 线程共享；
    所有公开方法经 RLock 串行化，避免交错事务。
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 线程安全：公开 API 经 _lock 串行；禁止跨线程无锁共用事务
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        # 连接设置：WAL / FK / busy / 同步
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=FULL")
        try:
            apply_migrations(self._conn)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("unsupported_schema:"):
                self._conn.close()
                raise ScheduleStoreError("unsupported_schema", msg.split(":", 2)[-1]) from exc
            self._conn.close()
            raise

    def __enter__(self) -> ScheduleStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def create(
        self,
        definition: ScheduleDefinition,
        *,
        now: datetime,
        next_run_at_utc: datetime | None = None,
    ) -> ScheduleDefinition:
        definition = validate_schedule(definition)
        if definition.revision != 1:
            raise ScheduleStoreError("invalid_revision", "新建计划 revision 必须为 1")
        now_s = _dt_to_iso(now)
        next_s = _dt_to_iso(next_run_at_utc)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                INSERT INTO schedules (
                    id, name, prompt, workspace_root, destination_kind,
                    destination_session_path, connection_id, model, web_enabled,
                    allowed_tools_json, approval_allowed_tools_json, approved_tools_json,
                    permission_mode, dtstart_local, timezone, rrule, next_run_at_utc,
                    status, misfire_policy, overlap_policy, retry_policy_json,
                    created_at_utc, updated_at_utc, last_run_at_utc, revision
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?
                )
                """,
                (
                    definition.id,
                    definition.name,
                    definition.prompt,
                    str(definition.workspace_root),
                    definition.destination_kind,
                    (
                        str(definition.destination_session_path)
                        if definition.destination_session_path
                        else None
                    ),
                    definition.connection_id,
                    definition.model,
                    1 if definition.web_enabled else 0,
                    json.dumps(list(definition.allowed_tools)),
                    json.dumps(list(definition.approval_allowed_tools)),
                    json.dumps(list(definition.approved_tools)),
                    definition.permission_mode,
                    definition.dtstart_local.isoformat(),
                    definition.timezone,
                    definition.rrule,
                    next_s,
                    definition.status,
                    definition.misfire_policy,
                    definition.overlap_policy,
                    json.dumps(asdict(definition.retry_policy)),
                    now_s,
                    now_s,
                    definition.revision,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO schedule_revisions (
                    schedule_id, revision, definition_json, created_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (definition.id, definition.revision, _definition_to_json(definition), now_s),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise ScheduleStoreError("duplicate_id", f"计划已存在: {definition.id}") from exc
        except Exception:
            self._conn.rollback()
            raise
        return definition

    def get(self, schedule_id: str) -> ScheduleDefinition | None:
        row = self._conn.execute(
            "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_definition(row)

    def list_schedules(
        self,
        *,
        status: ScheduleStatus | None = None,
        include_archived: bool = False,
    ) -> list[ScheduleDefinition]:
        sql = "SELECT * FROM schedules WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        elif not include_archived:
            sql += " AND status != 'archived'"
        sql += " ORDER BY updated_at_utc DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_definition(r) for r in rows]

    def update(
        self,
        schedule_id: str,
        *,
        expected_revision: int,
        now: datetime,
        next_run_at_utc: datetime | None | object = ...,
        **fields: Any,
    ) -> ScheduleDefinition:
        current = self.get(schedule_id)
        if current is None:
            raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
        if current.revision != expected_revision:
            raise ScheduleStoreError(
                "revision_conflict",
                f"revision 冲突: 期望 {expected_revision}, 实际 {current.revision}",
            )
        data = asdict(current)
        # asdict 会把 Path 变成 str；重建时再包装
        data["workspace_root"] = current.workspace_root
        data["destination_session_path"] = current.destination_session_path
        data["allowed_tools"] = current.allowed_tools
        data["approval_allowed_tools"] = current.approval_allowed_tools
        data["approved_tools"] = current.approved_tools
        data["retry_policy"] = current.retry_policy
        for key, value in fields.items():
            if key in {"id", "revision"}:
                continue
            if key not in data:
                raise ScheduleStoreError("unknown_field", f"未知字段: {key}")
            data[key] = value
        data["revision"] = current.revision + 1
        updated = validate_schedule(ScheduleDefinition(**data))
        now_s = _dt_to_iso(now)
        # ... 哨兵：未传入时保留原 next_run；显式 None 表示清除
        update_next = next_run_at_utc is not ...
        next_s = _dt_to_iso(next_run_at_utc) if update_next else None  # type: ignore[arg-type]
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # 重新读 revision 做乐观锁
            row = self._conn.execute(
                "SELECT revision FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
            if row is None or int(row[0]) != expected_revision:
                self._conn.rollback()
                raise ScheduleStoreError("revision_conflict", "revision 冲突")
            sets = [
                "name = ?",
                "prompt = ?",
                "workspace_root = ?",
                "destination_kind = ?",
                "destination_session_path = ?",
                "connection_id = ?",
                "model = ?",
                "web_enabled = ?",
                "allowed_tools_json = ?",
                "approval_allowed_tools_json = ?",
                "approved_tools_json = ?",
                "permission_mode = ?",
                "dtstart_local = ?",
                "timezone = ?",
                "rrule = ?",
                "status = ?",
                "misfire_policy = ?",
                "overlap_policy = ?",
                "retry_policy_json = ?",
                "updated_at_utc = ?",
                "revision = ?",
            ]
            values: list[Any] = [
                updated.name,
                updated.prompt,
                str(updated.workspace_root),
                updated.destination_kind,
                (
                    str(updated.destination_session_path)
                    if updated.destination_session_path
                    else None
                ),
                updated.connection_id,
                updated.model,
                1 if updated.web_enabled else 0,
                json.dumps(list(updated.allowed_tools)),
                json.dumps(list(updated.approval_allowed_tools)),
                json.dumps(list(updated.approved_tools)),
                updated.permission_mode,
                updated.dtstart_local.isoformat(),
                updated.timezone,
                updated.rrule,
                updated.status,
                updated.misfire_policy,
                updated.overlap_policy,
                json.dumps(asdict(updated.retry_policy)),
                now_s,
                updated.revision,
            ]
            if update_next:
                sets.append("next_run_at_utc = ?")
                values.append(next_s)
            values.extend([schedule_id, expected_revision])
            self._conn.execute(
                f"UPDATE schedules SET {', '.join(sets)} WHERE id = ? AND revision = ?",
                values,
            )
            self._conn.execute(
                """
                INSERT INTO schedule_revisions (
                    schedule_id, revision, definition_json, created_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (schedule_id, updated.revision, _definition_to_json(updated), now_s),
            )
            self._conn.commit()
        except ScheduleStoreError:
            raise
        except Exception:
            self._conn.rollback()
            raise
        return updated

    def pause(self, schedule_id: str, *, now: datetime) -> ScheduleDefinition:
        current = self.get(schedule_id)
        if current is None:
            raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
        return self.update(
            schedule_id,
            expected_revision=current.revision,
            now=now,
            status="paused",
            next_run_at_utc=None,
        )

    def resume(
        self,
        schedule_id: str,
        *,
        now: datetime,
        next_run_at_utc: datetime | None,
    ) -> ScheduleDefinition:
        current = self.get(schedule_id)
        if current is None:
            raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
        return self.update(
            schedule_id,
            expected_revision=current.revision,
            now=now,
            status="active",
            next_run_at_utc=next_run_at_utc,
        )

    def archive(self, schedule_id: str, *, now: datetime) -> ScheduleDefinition:
        current = self.get(schedule_id)
        if current is None:
            raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
        return self.update(
            schedule_id,
            expected_revision=current.revision,
            now=now,
            status="archived",
            next_run_at_utc=None,
        )

    def delete(self, schedule_id: str) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
            if cur.rowcount == 0:
                self._conn.rollback()
                raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
            self._conn.commit()
        except ScheduleStoreError:
            raise
        except Exception:
            self._conn.rollback()
            raise

    def get_revision(
        self, schedule_id: str, revision: int
    ) -> ScheduleDefinition | None:
        row = self._conn.execute(
            """
            SELECT definition_json FROM schedule_revisions
            WHERE schedule_id = ? AND revision = ?
            """,
            (schedule_id, revision),
        ).fetchone()
        if row is None:
            return None
        return _definition_from_json(row["definition_json"])

    def set_next_run(
        self, schedule_id: str, *, next_run_at_utc: datetime | None, now: datetime
    ) -> None:
        self._conn.execute(
            """
            UPDATE schedules SET next_run_at_utc = ?, updated_at_utc = ?
            WHERE id = ?
            """,
            (_dt_to_iso(next_run_at_utc), _dt_to_iso(now), schedule_id),
        )

    def get_next_run_at_utc(self, schedule_id: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT next_run_at_utc FROM schedules WHERE id = ?",
            (schedule_id,),
        ).fetchone()
        if row is None:
            return None
        return _dt_from_iso(row["next_run_at_utc"])

    def get_last_run_at_utc(self, schedule_id: str) -> datetime | None:
        """公开 API：读取 last_run_at_utc，禁止调用方直接访问 _conn。"""
        row = self._conn.execute(
            "SELECT last_run_at_utc FROM schedules WHERE id = ?",
            (schedule_id,),
        ).fetchone()
        if row is None:
            return None
        return _dt_from_iso(row["last_run_at_utc"])

    def list_due_schedules(self, *, now: datetime) -> list[ScheduleDefinition]:
        now_s = _dt_to_iso(now)
        rows = self._conn.execute(
            """
            SELECT * FROM schedules
            WHERE status = 'active'
              AND next_run_at_utc IS NOT NULL
              AND next_run_at_utc <= ?
            ORDER BY next_run_at_utc ASC
            """,
            (now_s,),
        ).fetchall()
        return [self._row_to_definition(r) for r in rows]

    def create_run(
        self,
        *,
        schedule_id: str,
        schedule_revision: int,
        trigger_key: str,
        trigger_kind: TriggerKind,
        scheduled_for_utc: datetime,
        status: RunStatus = "queued",
        now: datetime,
        run_id: str | None = None,
        summary: str = "",
        unread: bool = True,
        failure_category: FailureCategory | None = None,
        failure_reason: str | None = None,
        needs_attention_reason: str | None = None,
    ) -> ScheduleRun:
        rid = run_id or f"run_{uuid.uuid4().hex}"
        finished = (
            _dt_to_iso(now)
            if status
            in {
                "succeeded",
                "failed",
                "needs_attention",
                "cancelled",
                "skipped",
            }
            else None
        )
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                INSERT INTO schedule_runs (
                    id, schedule_id, schedule_revision, trigger_key, trigger_kind,
                    scheduled_for_utc, status, attempt_count, summary, unread,
                    failure_category, failure_reason, needs_attention_reason,
                    finished_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    schedule_id,
                    schedule_revision,
                    trigger_key,
                    trigger_kind,
                    _dt_to_iso(scheduled_for_utc),
                    status,
                    summary,
                    1 if unread else 0,
                    failure_category,
                    failure_reason,
                    needs_attention_reason,
                    finished,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise ScheduleStoreError("duplicate_trigger", "trigger_key 已存在") from exc
        except Exception:
            self._conn.rollback()
            raise
        return self.get_run(rid)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> ScheduleRun | None:
        row = self._conn.execute(
            "SELECT * FROM schedule_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(
        self,
        *,
        schedule_id: str | None = None,
        unread_only: bool = False,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[ScheduleRun]:
        sql = "SELECT * FROM schedule_runs WHERE 1=1"
        params: list[Any] = []
        if schedule_id is not None:
            sql += " AND schedule_id = ?"
            params.append(schedule_id)
        if unread_only:
            sql += " AND unread = 1"
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY COALESCE(finished_at_utc, scheduled_for_utc) DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_run(r) for r in rows]

    def mark_run_read(self, run_id: str) -> None:
        cur = self._conn.execute(
            "UPDATE schedule_runs SET unread = 0 WHERE id = ?", (run_id,)
        )
        if cur.rowcount == 0:
            raise ScheduleStoreError("not_found", f"run 不存在: {run_id}")

    def mark_all_runs_read(self) -> int:
        cur = self._conn.execute(
            "UPDATE schedule_runs SET unread = 0 WHERE unread = 1"
        )
        return int(cur.rowcount)

    def request_cancel(self, run_id: str) -> ScheduleRun:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT * FROM schedule_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                self._conn.rollback()
                raise ScheduleStoreError("not_found", f"run 不存在: {run_id}")
            status = row["status"]
            now_s = _dt_to_iso(datetime.now(timezone.utc))
            if status == "queued":
                self._conn.execute(
                    """
                    UPDATE schedule_runs
                    SET status = 'cancelled', cancellation_requested = 1,
                        finished_at_utc = ?, unread = 1
                    WHERE id = ?
                    """,
                    (now_s, run_id),
                )
            elif status == "retry_wait":
                # retry_wait 直接终态取消，避免 claim 永久排除导致卡死
                self._conn.execute(
                    """
                    UPDATE schedule_runs
                    SET status = 'cancelled', cancellation_requested = 1,
                        finished_at_utc = ?, unread = 1, retry_at_utc = NULL
                    WHERE id = ?
                    """,
                    (now_s, run_id),
                )
            elif status == "running":
                self._conn.execute(
                    "UPDATE schedule_runs SET cancellation_requested = 1 WHERE id = ?",
                    (run_id,),
                )
            else:
                self._conn.rollback()
                raise ScheduleStoreError(
                    "not_cancellable",
                    f"run 状态 {status} 不可取消",
                )
            self._conn.commit()
        except ScheduleStoreError:
            raise
        except Exception:
            self._conn.rollback()
            raise
        return self.get_run(run_id)  # type: ignore[return-value]

    def claim_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime,
    ) -> ScheduleRun | None:
        """原子 claim：仅 queued 或已到点的 retry_wait。"""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT * FROM schedule_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                self._conn.rollback()
                return None
            status = row["status"]
            if status == "retry_wait":
                retry_at = _dt_from_iso(row["retry_at_utc"])
                if retry_at is not None and retry_at > now:
                    self._conn.rollback()
                    return None
            elif status != "queued":
                self._conn.rollback()
                return None
            if row["cancellation_requested"]:
                self._conn.execute(
                    """
                    UPDATE schedule_runs
                    SET status = 'cancelled', finished_at_utc = ?
                    WHERE id = ?
                    """,
                    (_dt_to_iso(now), run_id),
                )
                self._conn.commit()
                return None
            attempt = int(row["attempt_count"]) + 1
            self._conn.execute(
                """
                UPDATE schedule_runs SET
                    status = 'running',
                    worker_id = ?,
                    lease_expires_at_utc = ?,
                    started_at_utc = COALESCE(started_at_utc, ?),
                    attempt_count = ?,
                    retry_at_utc = NULL
                WHERE id = ?
                """,
                (
                    worker_id,
                    _dt_to_iso(lease_expires_at),
                    _dt_to_iso(now),
                    attempt,
                    run_id,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO schedule_run_attempts (
                    run_id, attempt_number, started_at_utc
                ) VALUES (?, ?, ?)
                """,
                (run_id, attempt, _dt_to_iso(now)),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return self.get_run(run_id)

    def renew_run_lease(
        self, run_id: str, *, worker_id: str, lease_expires_at: datetime
    ) -> bool:
        cur = self._conn.execute(
            """
            UPDATE schedule_runs
            SET lease_expires_at_utc = ?
            WHERE id = ? AND worker_id = ? AND status = 'running'
            """,
            (_dt_to_iso(lease_expires_at), run_id, worker_id),
        )
        return cur.rowcount > 0

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        now: datetime,
        summary: str = "",
        failure_category: FailureCategory | None = None,
        failure_reason: str | None = None,
        needs_attention_reason: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
        episode_path: str | None = None,
        retry_at_utc: datetime | None = None,
        expected_worker_id: str | None = None,
        expected_attempt: int | None = None,
    ) -> ScheduleRun:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT schedule_id, attempt_count, cancellation_requested,
                       status AS cur_status, worker_id
                FROM schedule_runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                self._conn.rollback()
                raise ScheduleStoreError("not_found", f"run 不存在: {run_id}")
            attempt = int(row["attempt_count"])
            cur_status = str(row["cur_status"] or "")
            cur_worker = row["worker_id"]
            # fencing：running 必须带 worker_id+attempt，防止过期 worker 覆盖新 claim
            if cur_status == "running":
                if expected_worker_id is None or expected_attempt is None:
                    self._conn.rollback()
                    raise ScheduleStoreError(
                        "missing_fence",
                        "finish_run 在 running 状态必须提供 expected_worker_id 与 expected_attempt",
                    )
                if cur_worker != expected_worker_id or attempt != int(expected_attempt):
                    self._conn.rollback()
                    raise ScheduleStoreError(
                        "stale_finish",
                        "finish_run 被拒绝：worker/attempt 与当前 running lease 不匹配",
                    )
            elif expected_worker_id is not None or expected_attempt is not None:
                # 非 running 若显式传入 fence，仍校验（recover 二次 finish 等）
                if expected_worker_id is not None and cur_worker not in {
                    None,
                    expected_worker_id,
                }:
                    self._conn.rollback()
                    raise ScheduleStoreError(
                        "stale_finish",
                        "finish_run 被拒绝：worker_id 不匹配",
                    )
                if expected_attempt is not None and attempt != int(expected_attempt):
                    self._conn.rollback()
                    raise ScheduleStoreError(
                        "stale_finish",
                        "finish_run 被拒绝：attempt 不匹配",
                    )
            # CAS：已请求取消时，禁止成功/失败覆盖为非取消终态
            final_status: RunStatus = status
            final_category = failure_category
            final_summary = summary
            if int(row["cancellation_requested"] or 0) and status in {
                "succeeded",
                "failed",
                "needs_attention",
                "retry_wait",
                "interrupted",
            }:
                final_status = "cancelled"
                final_category = "cancelled"
                final_summary = summary or "cancelled"
                retry_at_utc = None
            if cur_status == "running":
                cur = self._conn.execute(
                    """
                    UPDATE schedule_runs SET
                        status = ?,
                        finished_at_utc = ?,
                        summary = ?,
                        failure_category = ?,
                        failure_reason = ?,
                        needs_attention_reason = ?,
                        session_id = COALESCE(?, session_id),
                        session_path = COALESCE(?, session_path),
                        episode_path = COALESCE(?, episode_path),
                        retry_at_utc = ?,
                        lease_expires_at_utc = NULL,
                        unread = 1
                    WHERE id = ?
                      AND status = 'running'
                      AND worker_id = ?
                      AND attempt_count = ?
                    """,
                    (
                        final_status,
                        _dt_to_iso(now) if final_status != "retry_wait" else None,
                        final_summary,
                        final_category,
                        failure_reason if final_status != "cancelled" else final_summary,
                        needs_attention_reason if final_status != "cancelled" else None,
                        session_id,
                        session_path,
                        episode_path,
                        _dt_to_iso(retry_at_utc),
                        run_id,
                        expected_worker_id,
                        int(expected_attempt),  # type: ignore[arg-type]
                    ),
                )
                if cur.rowcount != 1:
                    self._conn.rollback()
                    raise ScheduleStoreError(
                        "stale_finish",
                        "finish_run CAS 失败：running lease 已变更",
                    )
            else:
                self._conn.execute(
                    """
                    UPDATE schedule_runs SET
                        status = ?,
                        finished_at_utc = ?,
                        summary = ?,
                        failure_category = ?,
                        failure_reason = ?,
                        needs_attention_reason = ?,
                        session_id = COALESCE(?, session_id),
                        session_path = COALESCE(?, session_path),
                        episode_path = COALESCE(?, episode_path),
                        retry_at_utc = ?,
                        lease_expires_at_utc = NULL,
                        unread = 1
                    WHERE id = ?
                    """,
                    (
                        final_status,
                        _dt_to_iso(now) if final_status != "retry_wait" else None,
                        final_summary,
                        final_category,
                        failure_reason if final_status != "cancelled" else final_summary,
                        needs_attention_reason if final_status != "cancelled" else None,
                        session_id,
                        session_path,
                        episode_path,
                        _dt_to_iso(retry_at_utc),
                        run_id,
                    ),
                )
            status = final_status
            failure_category = final_category
            summary = final_summary
            if status != "retry_wait":
                self._conn.execute(
                    """
                    UPDATE schedule_run_attempts SET
                        finished_at_utc = ?,
                        outcome = ?,
                        failure_category = ?,
                        failure_reason = ?
                    WHERE run_id = ? AND attempt_number = ?
                    """,
                    (
                        _dt_to_iso(now),
                        status,
                        failure_category,
                        failure_reason,
                        run_id,
                        attempt,
                    ),
                )
            # run 终结与计划最近执行时间必须同事务提交，避免 UI 读到半完成状态。
            self._conn.execute(
                """
                UPDATE schedules
                SET last_run_at_utc = ?, updated_at_utc = ?
                WHERE id = ?
                """,
                (_dt_to_iso(now), _dt_to_iso(now), row["schedule_id"]),
            )
            self._conn.commit()
        except ScheduleStoreError:
            raise
        except Exception:
            self._conn.rollback()
            raise
        return self.get_run(run_id)  # type: ignore[return-value]

    def list_claimable_runs(self, *, now: datetime, limit: int = 10) -> list[ScheduleRun]:
        """按 overlap_policy 过滤可 claim runs。

        - parallel：允许同计划多 running；不因已有 running 阻塞
        - queue/skip：同计划有 running 则不放行；queue 还要求只放行队首
        - queue：未来 retry_wait 仍占据队首，较晚 queued 不可越队
        """
        now_s = _dt_to_iso(now)
        rows = self._conn.execute(
            """
            SELECT r.* FROM schedule_runs r
            LEFT JOIN schedules s ON s.id = r.schedule_id
            WHERE (
                r.status = 'queued'
                OR (r.status = 'retry_wait' AND (r.retry_at_utc IS NULL OR r.retry_at_utc <= ?))
            )
            AND r.cancellation_requested = 0
            AND (
                -- parallel 不因 running 阻塞；其余策略同计划有 running 则不可 claim
                COALESCE(s.overlap_policy, 'skip') = 'parallel'
                OR NOT EXISTS (
                    SELECT 1 FROM schedule_runs active
                    WHERE active.schedule_id = r.schedule_id
                      AND active.status = 'running'
                )
            )
            AND (
                -- queue：队首按 scheduled_for 串行；含尚未到点的 retry_wait
                COALESCE(s.overlap_policy, 'skip') != 'queue'
                OR r.id = (
                    SELECT r2.id FROM schedule_runs r2
                    WHERE r2.schedule_id = r.schedule_id
                      AND r2.cancellation_requested = 0
                      AND (
                        r2.status = 'queued'
                        OR r2.status = 'retry_wait'
                      )
                    ORDER BY r2.scheduled_for_utc ASC, r2.id ASC
                    LIMIT 1
                )
            )
            ORDER BY r.scheduled_for_utc ASC
            LIMIT ?
            """,
            (now_s, limit),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def list_expired_running(self, *, now: datetime) -> list[ScheduleRun]:
        now_s = _dt_to_iso(now)
        rows = self._conn.execute(
            """
            SELECT * FROM schedule_runs
            WHERE status = 'running'
              AND lease_expires_at_utc IS NOT NULL
              AND lease_expires_at_utc < ?
            """,
            (now_s,),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def list_active_runs_for_schedule(self, schedule_id: str) -> list[ScheduleRun]:
        rows = self._conn.execute(
            """
            SELECT * FROM schedule_runs
            WHERE schedule_id = ?
              AND status IN ('queued', 'running', 'retry_wait')
            """,
            (schedule_id,),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def acquire_lease(
        self,
        *,
        owner_id: str,
        now: datetime,
        ttl_seconds: int = 30,
    ) -> bool:
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc)
        now_s = _dt_to_iso(now)
        exp_s = _dt_to_iso(expires)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT owner_id, expires_at_utc FROM coordinator_lease WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO coordinator_lease (
                        singleton, owner_id, acquired_at_utc, heartbeat_at_utc, expires_at_utc
                    ) VALUES (1, ?, ?, ?, ?)
                    """,
                    (owner_id, now_s, now_s, exp_s),
                )
                self._conn.commit()
                return True
            current_owner = row["owner_id"]
            expires_at = _dt_from_iso(row["expires_at_utc"])
            if current_owner == owner_id or (
                expires_at is not None and expires_at <= now
            ):
                self._conn.execute(
                    """
                    UPDATE coordinator_lease SET
                        owner_id = ?,
                        acquired_at_utc = CASE WHEN owner_id = ? THEN acquired_at_utc ELSE ? END,
                        heartbeat_at_utc = ?,
                        expires_at_utc = ?
                    WHERE singleton = 1
                    """,
                    (owner_id, owner_id, now_s, now_s, exp_s),
                )
                self._conn.commit()
                return True
            self._conn.commit()
            return False
        except Exception:
            self._conn.rollback()
            raise

    def heartbeat_lease(
        self, *, owner_id: str, now: datetime, ttl_seconds: int = 30
    ) -> bool:
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc)
        cur = self._conn.execute(
            """
            UPDATE coordinator_lease SET
                heartbeat_at_utc = ?,
                expires_at_utc = ?
            WHERE singleton = 1 AND owner_id = ?
            """,
            (_dt_to_iso(now), _dt_to_iso(expires), owner_id),
        )
        return cur.rowcount > 0

    def release_lease(self, *, owner_id: str) -> None:
        self._conn.execute(
            "DELETE FROM coordinator_lease WHERE singleton = 1 AND owner_id = ?",
            (owner_id,),
        )

    def _row_to_definition(self, row: sqlite3.Row) -> ScheduleDefinition:
        try:
            allowed = json.loads(row["allowed_tools_json"])
            approval = json.loads(row["approval_allowed_tools_json"])
            approved = json.loads(row["approved_tools_json"])
        except json.JSONDecodeError as exc:
            raise ScheduleStoreError("invalid_json", "tools JSON 无效") from exc
        allowed_t = _json_str_list(allowed, field="allowed_tools")
        approval_t = _json_str_list(approval, field="approval_allowed_tools")
        approved_t = _json_str_list(approved, field="approved_tools")
        dest = row["destination_session_path"]
        definition = ScheduleDefinition(
            id=row["id"],
            name=row["name"],
            prompt=row["prompt"],
            workspace_root=Path(row["workspace_root"]),
            destination_kind=row["destination_kind"],  # type: ignore[arg-type]
            destination_session_path=Path(dest) if dest else None,
            connection_id=row["connection_id"],
            model=row["model"],
            web_enabled=bool(row["web_enabled"]),
            allowed_tools=allowed_t,
            approval_allowed_tools=approval_t,
            approved_tools=approved_t,
            permission_mode=row["permission_mode"],  # type: ignore[arg-type]
            dtstart_local=datetime.fromisoformat(row["dtstart_local"]),
            timezone=row["timezone"],
            rrule=row["rrule"],
            status=row["status"],  # type: ignore[arg-type]
            misfire_policy=row["misfire_policy"],  # type: ignore[arg-type]
            overlap_policy=row["overlap_policy"],  # type: ignore[arg-type]
            retry_policy=_retry_from_json(row["retry_policy_json"]),
            revision=int(row["revision"]),
        )
        return validate_schedule(definition)

    def _row_to_run(self, row: sqlite3.Row) -> ScheduleRun:
        return ScheduleRun(
            id=row["id"],
            schedule_id=row["schedule_id"],
            schedule_revision=int(row["schedule_revision"]),
            trigger_key=row["trigger_key"],
            trigger_kind=row["trigger_kind"],  # type: ignore[arg-type]
            scheduled_for_utc=_dt_from_iso(row["scheduled_for_utc"])  # type: ignore[arg-type]
            or datetime.now(timezone.utc),
            status=row["status"],  # type: ignore[arg-type]
            attempt_count=int(row["attempt_count"]),
            retry_at_utc=_dt_from_iso(row["retry_at_utc"]),
            worker_id=row["worker_id"],
            lease_expires_at_utc=_dt_from_iso(row["lease_expires_at_utc"]),
            started_at_utc=_dt_from_iso(row["started_at_utc"]),
            finished_at_utc=_dt_from_iso(row["finished_at_utc"]),
            session_id=row["session_id"],
            session_path=row["session_path"],
            episode_path=row["episode_path"],
            summary=row["summary"] or "",
            failure_category=row["failure_category"],  # type: ignore[arg-type]
            failure_reason=row["failure_reason"],
            needs_attention_reason=row["needs_attention_reason"],
            unread=bool(row["unread"]),
            cancellation_requested=bool(row["cancellation_requested"]),
        )
