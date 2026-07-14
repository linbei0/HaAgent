"""
haagent/scheduling/coordinator.py - 计划到期展开、租约与 claim 状态机

只在持有 coordinator lease 时展开 occurrence 并创建 logical run；
Agent 执行在事务外由 executor 完成。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from haagent.scheduling.models import ScheduleDefinition
from haagent.scheduling.recurrence import (
    iter_due_occurrences,
    preview_occurrences,
)
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError

LEASE_TTL_SECONDS = 30
RUN_LEASE_TTL_SECONDS = 45
MISFIRE_BATCH_LIMIT = 50
# skip：超过此秒数的过期 occurrence 视为 misfire 整批跳过；之内视为 tick 抖动仍执行
MISFIRE_GRACE_SECONDS = 60

# 可重试基础设施错误
RETRYABLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "model_transient",
        "worker_interrupted",
        "internal_error",
    }
)


def _trigger_key_for_occurrence(occurrence_utc: datetime) -> str:
    return f"occ:{occurrence_utc.astimezone(timezone.utc).isoformat()}"


def _compute_retry_delay(policy, attempt_count: int) -> int:
    # attempt_count 为刚失败的 attempt 编号（从 1 起）
    delay = policy.initial_delay_seconds * (policy.multiplier ** max(0, attempt_count - 1))
    return int(min(delay, policy.max_delay_seconds))


class ScheduleCoordinator:
    def __init__(
        self,
        store: ScheduleStore,
        *,
        owner_id: str,
    ) -> None:
        self._store = store
        self._owner_id = owner_id

    def heartbeat(self, *, now: datetime) -> bool:
        if self._store.heartbeat_lease(
            owner_id=self._owner_id, now=now, ttl_seconds=LEASE_TTL_SECONDS
        ):
            return True
        return self._store.acquire_lease(
            owner_id=self._owner_id, now=now, ttl_seconds=LEASE_TTL_SECONDS
        )

    def release(self) -> None:
        self._store.release_lease(owner_id=self._owner_id)

    def _recover_expired_runs(self, *, now: datetime) -> None:
        expired = self._store.list_expired_running(now=now)
        for run in expired:
            fence_worker = run.worker_id
            fence_attempt = run.attempt_count
            # 先 interrupted 证据，再按 retry policy；必须带 fencing 防 stale 覆盖
            try:
                self._store.finish_run(
                    run.id,
                    status="interrupted",
                    now=now,
                    summary="worker lease expired",
                    failure_category="worker_interrupted",
                    failure_reason="run lease expired; worker may have crashed",
                    expected_worker_id=fence_worker,
                    expected_attempt=fence_attempt,
                )
            except ScheduleStoreError as error:
                if error.code == "stale_finish":
                    # 已被其他 worker 接管。
                    continue
                raise
            definition = self._store.get_revision(run.schedule_id, run.schedule_revision)
            if definition is None:
                # run 通过复合外键绑定 revision；缺失说明持久化已损坏，必须暴露。
                raise ScheduleStoreError(
                    "revision_not_found",
                    f"run {run.id} 的计划 revision 不存在",
                )
            if run.attempt_count < definition.retry_policy.max_attempts:
                delay = _compute_retry_delay(definition.retry_policy, run.attempt_count)
                retry_at = now + timedelta(seconds=delay)
                try:
                    self._store.finish_run(
                        run.id,
                        status="retry_wait",
                        now=now,
                        summary="retry after worker interrupt",
                        failure_category="worker_interrupted",
                        failure_reason="run lease expired; scheduled retry",
                        retry_at_utc=retry_at,
                        expected_worker_id=fence_worker,
                        expected_attempt=fence_attempt,
                    )
                except ScheduleStoreError as error:
                    if error.code == "stale_finish":
                        continue
                    raise

    def tick(self, *, now: datetime) -> bool:
        # 仅持有 lease 时展开 due 与 recover；不 claim、不执行 Agent
        if not self.heartbeat(now=now):
            return False
        self._recover_expired_runs(now=now)
        self._expand_due(now=now)
        return True

    def _expand_due(self, *, now: datetime) -> None:
        due_schedules = self._store.list_due_schedules(now=now)
        for schedule in due_schedules:
            self._expand_schedule(schedule, now=now)

    def _expand_schedule(self, schedule: ScheduleDefinition, *, now: datetime) -> None:
        next_cached = self._store.get_next_run_at_utc(schedule.id)
        if next_cached is None:
            return
        # 从上一次 next 到 now 展开；after 取 next 前一微秒以包含 next 本身
        after = next_cached - timedelta(microseconds=1)
        # 多取 1 条用于判断 all 是否仍有积压；避免无界 list 物化
        fetch_limit = MISFIRE_BATCH_LIMIT + 1
        occurrences = list(
            iter_due_occurrences(
                schedule, after=after, through=now, limit=fetch_limit
            )
        )
        if not occurrences:
            # 重算 next
            self._advance_next(schedule, after=now, now=now)
            return

        policy = schedule.misfire_policy
        if policy == "skip":
            # 规格：跳过所有真正过期 occurrence；grace 内视为 tick 抖动仍执行
            grace = timedelta(seconds=MISFIRE_GRACE_SECONDS)
            on_time = [o for o in occurrences if o + grace >= now]
            if not on_time:
                self._advance_next(schedule, after=now, now=now)
                return
            target = on_time
        elif policy == "latest":
            target = [occurrences[-1]]
        else:
            # all：单次 tick 有界批处理，剩余留给下一 tick，禁止推进到 now 丢弃积压
            target = occurrences[:MISFIRE_BATCH_LIMIT]

        for occ in target:
            self._create_occurrence_run(schedule, occ, now=now)

        if policy == "all" and len(occurrences) > MISFIRE_BATCH_LIMIT:
            # 从本批最后一条之后继续，不跳到 now
            self._advance_next(schedule, after=target[-1], now=now)
        else:
            self._advance_next(schedule, after=now, now=now)

    def _create_occurrence_run(
        self,
        schedule: ScheduleDefinition,
        occurrence: datetime,
        *,
        now: datetime,
    ) -> None:
        trigger_key = _trigger_key_for_occurrence(occurrence)
        active = self._store.list_active_runs_for_schedule(schedule.id)

        status = "queued"
        if schedule.overlap_policy == "skip" and active:
            status = "skipped"

        try:
            self._store.create_run(
                schedule_id=schedule.id,
                schedule_revision=schedule.revision,
                trigger_key=trigger_key,
                trigger_kind="scheduled",
                scheduled_for_utc=occurrence,
                status=status,  # type: ignore[arg-type]
                now=now,
                summary="skipped due to overlap" if status == "skipped" else "",
            )
        except ScheduleStoreError as exc:
            if exc.code == "duplicate_trigger":
                return
            raise

    def _advance_next(
        self,
        schedule: ScheduleDefinition,
        *,
        after: datetime,
        now: datetime,
    ) -> None:
        preview = preview_occurrences(schedule, after=after, count=1)
        next_run = preview[0] if preview else None
        if next_run is None:
            # once 或 COUNT/UNTIL 等有限 RRULE 耗尽 → completed，避免僵尸 active
            current = self._store.get(schedule.id)
            if current is not None and current.status == "active":
                self._store.update(
                    schedule.id,
                    expected_revision=current.revision,
                    now=now,
                    status="completed",
                    next_run_at_utc=None,
                )
            return
        self._store.set_next_run(schedule.id, next_run_at_utc=next_run, now=now)
