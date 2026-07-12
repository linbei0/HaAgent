"""
haagent/scheduling/worker.py - 计划任务前台 worker 循环

负责信号停止、租约 heartbeat、到期展开与 claim 执行；不实现 Agent loop。
派发与执行解耦：tick 只展开/恢复，执行在独立线程池，heartbeat 线程持续续租。
"""

from __future__ import annotations

import logging
import signal
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Protocol

from haagent.scheduling.coordinator import (
    LEASE_TTL_SECONDS,
    RUN_LEASE_TTL_SECONDS,
    ScheduleCoordinator,
)
from haagent.scheduling.models import RunClaim
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError

HEARTBEAT_INTERVAL_SECONDS = 10.0
MAX_IN_FLIGHT = 2

logger = logging.getLogger(__name__)


class RunExecutor(Protocol):
    def execute(self, claim: RunClaim) -> object: ...

    def request_cancel(self, run_id: str) -> None: ...


def _default_sleep(seconds: float) -> None:
    threading.Event().wait(timeout=seconds)


class ScheduleWorker:
    """前台/一次性调度 worker；signal handler 只设置 stop_event。"""

    def __init__(
        self,
        store: ScheduleStore,
        *,
        owner_id: str | None = None,
        executor: RunExecutor | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        max_in_flight: int = MAX_IN_FLIGHT,
    ) -> None:
        self._store = store
        self._owner_id = owner_id or f"worker-{uuid.uuid4().hex[:12]}"
        self._executor = executor
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or _default_sleep
        self.stop_event = threading.Event()
        # tick 只展开/恢复，不在 coordinator 内同步执行 Agent
        self._coordinator = ScheduleCoordinator(
            store,
            owner_id=self._owner_id,
        )
        self._max_in_flight = max(1, int(max_in_flight))
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_in_flight,
            thread_name_prefix="haagent-sched-exec",
        )
        self._in_flight: dict[str, Future[object]] = {}
        self._in_flight_lock = threading.Lock()
        # 结构化失败：供 TUI host_status / 进程退出诊断
        self._last_error: str | None = None
        self._fatal = False

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def fatal(self) -> bool:
        return self._fatal

    def install_signal_handlers(self) -> None:
        # 仅置位停止事件；不在 handler 内做 IO 或释放租约。
        def _handler(_signum: int, _frame: object) -> None:
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # 非主线程或平台不支持时跳过
                pass

    def run_once(self) -> int:
        """展开到期、恢复崩溃、claim/执行可运行项后退出。"""
        exit_code = 0
        try:
            now = self._clock()
            tick = self._coordinator.tick(now=now)
            # 仅持有 coordinator lease 的进程可派发，避免 TUI/系统 worker 双消费
            if tick.lease_held:
                self._dispatch_claimable(now=now)
            self._wait_in_flight()
            if self._fatal:
                exit_code = 1
        except ScheduleStoreError as error:
            self._record_error(f"store:{error.code}:{error}")
            exit_code = 1
        except Exception as error:
            self._record_error(f"worker:{type(error).__name__}:{error}")
            exit_code = 1
        finally:
            try:
                self._coordinator.release()
            except Exception as error:
                logger.warning("release coordinator lease failed: %s", error)
            self._shutdown_pool()
        return exit_code

    def run_forever(self) -> int:
        """循环 tick + 动态 sleep；执行在线程池，主循环持续 heartbeat。"""
        exit_code = 0
        try:
            while not self.stop_event.is_set():
                now = self._clock()
                try:
                    tick = self._coordinator.tick(now=now)
                    if tick.lease_held:
                        self._dispatch_claimable(now=now)
                    self._reap_finished()
                    if self._fatal:
                        exit_code = 1
                        break
                except ScheduleStoreError as error:
                    # 核心 Store 错误：非零退出，禁止装作健康
                    self._record_error(f"store:{error.code}:{error}", fatal=True)
                    exit_code = 1
                    break
                except Exception as error:
                    self._record_error(f"tick:{type(error).__name__}:{error}", fatal=True)
                    exit_code = 1
                    break
                if self.stop_event.is_set():
                    break
                delay = self._next_wakeup_delay(now=self._clock())
                if self._sleep is not _default_sleep:
                    self._sleep(delay)
                else:
                    self.stop_event.wait(timeout=delay)
        finally:
            self.stop_event.set()
            self._wait_in_flight(timeout=30.0)
            try:
                self._coordinator.release()
            except Exception as error:
                logger.warning("release coordinator lease failed: %s", error)
            self._shutdown_pool()
        return exit_code

    def request_cancel(self, run_id: str) -> None:
        """同进程取消：通知执行器中断当前 Agent。"""
        if self._executor is not None and hasattr(self._executor, "request_cancel"):
            try:
                self._executor.request_cancel(run_id)
            except Exception as error:
                logger.warning("request_cancel failed run=%s: %s", run_id, error)

    def _record_error(self, message: str, *, fatal: bool = False) -> None:
        self._last_error = message[:500]
        if fatal:
            self._fatal = True
        logger.error("schedule worker: %s", message)

    def _dispatch_claimable(self, *, now: datetime) -> None:
        if self._executor is None:
            return
        with self._in_flight_lock:
            capacity = self._max_in_flight - len(self._in_flight)
        if capacity <= 0:
            return
        try:
            claimable = self._store.list_claimable_runs(now=now, limit=capacity)
        except ScheduleStoreError as error:
            self._record_error(f"list_claimable:{error.code}:{error}", fatal=True)
            return
        except Exception as error:
            self._record_error(f"list_claimable:{type(error).__name__}:{error}", fatal=True)
            return
        for run in claimable:
            with self._in_flight_lock:
                if run.id in self._in_flight:
                    continue
                if len(self._in_flight) >= self._max_in_flight:
                    break
            lease_exp = now + timedelta(seconds=RUN_LEASE_TTL_SECONDS)
            try:
                claimed = self._store.claim_run(
                    run.id,
                    worker_id=self._owner_id,
                    lease_expires_at=lease_exp,
                    now=now,
                )
            except ScheduleStoreError as error:
                self._record_error(f"claim:{error.code}:{error}", fatal=True)
                return
            if claimed is None:
                continue
            # claim 时固化不可变 token
            claim = RunClaim(
                run_id=claimed.id,
                worker_id=self._owner_id,
                attempt=int(claimed.attempt_count or 0),
            )
            future = self._pool.submit(self._execute_with_leases, claim)
            with self._in_flight_lock:
                self._in_flight[claim.run_id] = future

    def _execute_with_leases(self, claim: RunClaim) -> object:
        """在独立线程执行 Agent，同时续租 coordinator 与 run lease。"""
        run_id = claim.run_id
        stop_hb = threading.Event()
        lease_lost = threading.Event()

        def _request_cancel() -> None:
            if self._executor is not None and hasattr(self._executor, "request_cancel"):
                try:
                    self._executor.request_cancel(run_id)
                except Exception as error:
                    logger.warning("executor cancel failed run=%s: %s", run_id, error)

        def _heartbeat_loop() -> None:
            while not stop_hb.wait(HEARTBEAT_INTERVAL_SECONDS):
                hb_now = self._clock()
                try:
                    self._coordinator.heartbeat(now=hb_now)
                except Exception as error:
                    logger.warning("coordinator heartbeat failed: %s", error)
                renewed = False
                try:
                    renewed = bool(
                        self._store.renew_run_lease(
                            run_id,
                            worker_id=claim.worker_id,
                            lease_expires_at=hb_now
                            + timedelta(seconds=RUN_LEASE_TTL_SECONDS),
                        )
                    )
                except Exception as error:
                    logger.warning("renew_run_lease failed run=%s: %s", run_id, error)
                    renewed = False
                # 续租失败：通知 executor 取消，避免双 Agent 副作用
                if not renewed and not lease_lost.is_set():
                    lease_lost.set()
                    try:
                        self._store.request_cancel(run_id)
                    except Exception as error:
                        logger.warning("request_cancel store failed run=%s: %s", run_id, error)
                    _request_cancel()
                # 跨线程取消：轮询 DB 标志
                try:
                    current = self._store.get_run(run_id)
                    if current is not None and current.cancellation_requested:
                        _request_cancel()
                except Exception as error:
                    logger.warning("poll cancel flag failed run=%s: %s", run_id, error)

        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"haagent-sched-hb-{run_id[:8]}",
            daemon=True,
        )
        hb_thread.start()
        try:
            assert self._executor is not None
            return self._executor.execute(claim)
        finally:
            stop_hb.set()
            hb_thread.join(timeout=2.0)

    def _reap_finished(self) -> None:
        with self._in_flight_lock:
            done = [rid for rid, fut in self._in_flight.items() if fut.done()]
            for rid in done:
                fut = self._in_flight.pop(rid)
                try:
                    fut.result()
                except ScheduleStoreError as error:
                    # stale_execute 等可预期；记录但不立刻 fatal
                    if error.code == "stale_execute":
                        logger.info("stale execute reaped run=%s: %s", rid, error)
                    else:
                        self._record_error(f"executor:{error.code}:{error}")
                except Exception as error:
                    # executor 崩溃必须可见，不能静默
                    self._record_error(
                        f"executor_crash:{rid}:{type(error).__name__}:{error}",
                        fatal=True,
                    )

    def _wait_in_flight(self, timeout: float | None = None) -> None:
        with self._in_flight_lock:
            items = list(self._in_flight.items())
        for rid, fut in items:
            try:
                fut.result(timeout=timeout)
            except ScheduleStoreError as error:
                if error.code != "stale_execute":
                    self._record_error(f"executor:{error.code}:{error}")
            except Exception as error:
                self._record_error(
                    f"executor_crash:{rid}:{type(error).__name__}:{error}",
                    fatal=True,
                )
        with self._in_flight_lock:
            self._in_flight.clear()

    def _shutdown_pool(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._pool.shutdown(wait=False)
        except Exception as error:
            logger.warning("pool shutdown failed: %s", error)

    def _next_wakeup_delay(self, *, now: datetime) -> float:
        # 有界轮询：默认 1s，最长 5s；查询最近 next_run
        try:
            due = self._store.list_due_schedules(now=now)
            if due:
                return 0.2
            # 无到期：短轮询，避免饿死 retry_wait / 新计划
            return 5.0
        except Exception:
            return 1.0


def run_schedule_worker(
    *,
    db_path: object | None = None,
    once: bool = False,
    owner_id: str | None = None,
    install_signals: bool = True,
    executor: RunExecutor | None = None,
) -> int:
    """CLI/系统服务入口：打开 store，运行 once 或 forever。"""
    from pathlib import Path

    from haagent.models.model_connections import user_config_dir
    from haagent.scheduling.executor import ScheduledRunExecutor

    path = Path(db_path) if db_path is not None else user_config_dir() / "schedules.sqlite3"
    try:
        store = ScheduleStore(path)
    except Exception as error:
        logger.error("open schedule store failed: %s", error)
        return 1
    try:
        run_executor = executor if executor is not None else ScheduledRunExecutor(store)
        worker = ScheduleWorker(
            store,
            owner_id=owner_id,
            executor=run_executor,
        )
        if install_signals:
            worker.install_signal_handlers()
        if once:
            return worker.run_once()
        return worker.run_forever()
    except Exception as error:
        logger.error("schedule worker failed: %s", error)
        return 1
    finally:
        try:
            store.close()
        except Exception:
            pass
