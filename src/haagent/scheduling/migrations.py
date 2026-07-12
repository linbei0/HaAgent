"""
haagent/scheduling/migrations.py - 计划任务 SQLite schema 与版本迁移

在 BEGIN IMMEDIATE 事务中创建/升级 schema；版本高于支持范围时显式失败。
"""

from __future__ import annotations

import sqlite3

CURRENT_SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    workspace_root TEXT NOT NULL,
    destination_kind TEXT NOT NULL CHECK(destination_kind IN ('new_session', 'resume_session')),
    destination_session_path TEXT,
    connection_id TEXT NOT NULL,
    model TEXT NOT NULL,
    web_enabled INTEGER NOT NULL CHECK(web_enabled IN (0, 1)),
    allowed_tools_json TEXT NOT NULL,
    approval_allowed_tools_json TEXT NOT NULL,
    approved_tools_json TEXT NOT NULL,
    permission_mode TEXT NOT NULL CHECK(permission_mode IN ('request_approval', 'auto_approve', 'full_access')),
    dtstart_local TEXT NOT NULL,
    timezone TEXT NOT NULL,
    rrule TEXT,
    next_run_at_utc TEXT,
    status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'completed', 'error', 'archived')),
    misfire_policy TEXT NOT NULL CHECK(misfire_policy IN ('skip', 'latest', 'all')),
    overlap_policy TEXT NOT NULL CHECK(overlap_policy IN ('skip', 'queue', 'parallel')),
    retry_policy_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    last_run_at_utc TEXT,
    revision INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS schedules_due_idx ON schedules(status, next_run_at_utc);

CREATE TABLE IF NOT EXISTS schedule_revisions (
    schedule_id TEXT NOT NULL REFERENCES schedules(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    definition_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY(schedule_id, revision)
);

CREATE TABLE IF NOT EXISTS schedule_runs (
    id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL REFERENCES schedules(id) ON DELETE CASCADE,
    schedule_revision INTEGER NOT NULL,
    trigger_key TEXT NOT NULL,
    trigger_kind TEXT NOT NULL CHECK(trigger_kind IN ('scheduled', 'manual')),
    scheduled_for_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'queued', 'running', 'retry_wait', 'succeeded', 'failed',
        'needs_attention', 'cancelled', 'skipped', 'interrupted'
    )),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    retry_at_utc TEXT,
    worker_id TEXT,
    lease_expires_at_utc TEXT,
    started_at_utc TEXT,
    finished_at_utc TEXT,
    session_id TEXT,
    session_path TEXT,
    episode_path TEXT,
    summary TEXT NOT NULL DEFAULT '',
    failure_category TEXT,
    failure_reason TEXT,
    needs_attention_reason TEXT,
    unread INTEGER NOT NULL DEFAULT 1 CHECK(unread IN (0, 1)),
    cancellation_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancellation_requested IN (0, 1)),
    UNIQUE(schedule_id, trigger_key),
    FOREIGN KEY(schedule_id, schedule_revision)
        REFERENCES schedule_revisions(schedule_id, revision)
);

CREATE INDEX IF NOT EXISTS schedule_runs_queue_idx
    ON schedule_runs(status, retry_at_utc, scheduled_for_utc);
CREATE INDEX IF NOT EXISTS schedule_runs_inbox_idx
    ON schedule_runs(unread, finished_at_utc DESC);

CREATE TABLE IF NOT EXISTS schedule_run_attempts (
    run_id TEXT NOT NULL REFERENCES schedule_runs(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    outcome TEXT,
    failure_category TEXT,
    failure_reason TEXT,
    PRIMARY KEY(run_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS coordinator_lease (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    owner_id TEXT NOT NULL,
    acquired_at_utc TEXT NOT NULL,
    heartbeat_at_utc TEXT NOT NULL,
    expires_at_utc TEXT NOT NULL
);
"""


def apply_migrations(conn: sqlite3.Connection) -> int:
    """应用迁移并返回当前 schema 版本。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            conn.executescript(SCHEMA_SQL)
            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
            conn.commit()
            return CURRENT_SCHEMA_VERSION

        version = int(row[0])
        if version > CURRENT_SCHEMA_VERSION:
            conn.rollback()
            raise ValueError(
                f"unsupported_schema:{version}:数据库 schema 版本 {version} 高于当前支持的 {CURRENT_SCHEMA_VERSION}"
            )
        if version < CURRENT_SCHEMA_VERSION:
            # 未来迁移链挂在这里；v1 无中间步
            conn.execute("UPDATE schema_version SET version = ?", (CURRENT_SCHEMA_VERSION,))
        # 确保表存在（幂等）
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        return CURRENT_SCHEMA_VERSION
    except Exception:
        conn.rollback()
        raise
