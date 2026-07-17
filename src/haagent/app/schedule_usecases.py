"""
haagent/app/schedule_usecases.py - 计划任务应用用例

为 TUI/CLI 提供计划 CRUD、预览、立即运行、收件箱与后台服务状态；
错误转为简体中文 AssistantServiceError，不伪造空成功。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantSchedule,
    AssistantScheduleRun,
    AssistantScheduleSummary,
    AssistantServiceError,
    BackgroundServiceStatus,
    RunQuery,
    ScheduleCreateRequest,
    ScheduleHostStatus,
    SchedulePreviewRequest,
    ScheduleUpdateRequest,
)
from haagent.models.config.connections import (
    ProviderProfileError,
    user_config_dir,
)
from haagent.scheduling.models import (
    RetryPolicy,
    ScheduleDefinition,
    ScheduleRun,
    ScheduleStatus,
    ScheduleValidationError,
    merge_web_tools,
    validate_schedule,
)
from haagent.scheduling.background.base import (
    BackgroundServiceAdapter,
    BackgroundServiceError,
    BackgroundServiceUnsupported,
)
from haagent.scheduling.recurrence import RecurrenceError, normalize_rrule, preview_occurrences
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError

if TYPE_CHECKING:
    from haagent.scheduling.worker import ScheduleWorker


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _map_store_error(error: ScheduleStoreError) -> AssistantServiceError:
    messages = {
        "not_found": "计划或运行记录不存在",
        "revision_conflict": "计划已被其他操作修改，请刷新后重试（revision 冲突）",
        "duplicate_id": "计划 ID 已存在",
        "duplicate_trigger": "相同触发已存在",
        "not_cancellable": "当前运行状态不可取消",
        "unsupported_schema": "计划数据库版本不受支持，请升级 HaAgent",
        "invalid_json": "计划数据损坏，无法读取",
    }
    text = messages.get(error.code, str(error))
    if error.code == "not_found" and "run" in str(error).lower():
        text = "运行记录不存在"
    return AssistantServiceError(text)


def _to_schedule(
    definition: ScheduleDefinition,
    *,
    next_run_at_utc: datetime | None = None,
    last_run_at_utc: datetime | None = None,
) -> AssistantSchedule:
    return AssistantSchedule(
        id=definition.id,
        name=definition.name,
        prompt=definition.prompt,
        workspace_root=definition.workspace_root,
        destination_kind=definition.destination_kind,
        destination_session_path=definition.destination_session_path,
        connection_id=definition.connection_id,
        model=definition.model,
        web_enabled=definition.web_enabled,
        allowed_tools=definition.allowed_tools,
        approval_allowed_tools=definition.approval_allowed_tools,
        approved_tools=definition.approved_tools,
        permission_mode=definition.permission_mode,
        dtstart_local=definition.dtstart_local,
        timezone=definition.timezone,
        rrule=definition.rrule,
        status=definition.status,
        misfire_policy=definition.misfire_policy,
        overlap_policy=definition.overlap_policy,
        retry_policy=definition.retry_policy,
        revision=definition.revision,
        next_run_at_utc=next_run_at_utc,
        last_run_at_utc=last_run_at_utc,
    )


def _to_summary(
    definition: ScheduleDefinition,
    *,
    next_run_at_utc: datetime | None,
    last_run_at_utc: datetime | None,
) -> AssistantScheduleSummary:
    return AssistantScheduleSummary(
        id=definition.id,
        name=definition.name,
        status=definition.status,
        timezone=definition.timezone,
        rrule=definition.rrule,
        next_run_at_utc=next_run_at_utc,
        last_run_at_utc=last_run_at_utc,
        workspace_root=definition.workspace_root,
        revision=definition.revision,
        model=definition.model,
        connection_id=definition.connection_id,
    )


def _to_run(run: ScheduleRun) -> AssistantScheduleRun:
    return AssistantScheduleRun(
        id=run.id,
        schedule_id=run.schedule_id,
        schedule_revision=run.schedule_revision,
        trigger_key=run.trigger_key,
        trigger_kind=run.trigger_kind,
        scheduled_for_utc=run.scheduled_for_utc,
        status=run.status,
        attempt_count=run.attempt_count,
        summary=run.summary,
        unread=run.unread,
        session_id=run.session_id,
        session_path=run.session_path,
        episode_path=run.episode_path,
        failure_category=run.failure_category,
        failure_reason=run.failure_reason,
        needs_attention_reason=run.needs_attention_reason,
        started_at_utc=run.started_at_utc,
        finished_at_utc=run.finished_at_utc,
        cancellation_requested=run.cancellation_requested,
    )


class AssistantSchedules:
    """TUI/CLI 共用的计划任务用例。"""

    def __init__(self, context: AssistantContext) -> None:
        self._context = context
        self._store: ScheduleStore | None = None
        self._background: BackgroundServiceAdapter | None = None
        self._clock = _now_utc
        # TUI 内嵌 host：由 start_host/stop_host 管理，TUI 不得直接碰 worker/store
        self._host_worker: ScheduleWorker | None = None
        self._host_thread: threading.Thread | None = None
        self._host_lock = threading.RLock()

    def _get_background(self) -> BackgroundServiceAdapter:
        if self._background is not None:
            return self._background
        from haagent.scheduling.background.factory import create_background_adapter

        self._background = create_background_adapter()
        return self._background

    def _get_store(self) -> ScheduleStore:
        if self._store is not None:
            return self._store
        db = user_config_dir() / "schedules.sqlite3"
        self._store = ScheduleStore(db)
        return self._store

    def create(self, request: ScheduleCreateRequest) -> AssistantSchedule:
        now = self._clock()
        self._validate_workspace(request.workspace_root)
        self._validate_connection(request.connection_id)
        try:
            rrule = normalize_rrule(request.rrule)
        except RecurrenceError as error:
            raise AssistantServiceError(str(error)) from error

        schedule_id = f"sch_{uuid.uuid4().hex[:16]}"
        definition = ScheduleDefinition(
            id=schedule_id,
            name=request.name,
            prompt=request.prompt,
            workspace_root=Path(request.workspace_root).resolve(),
            destination_kind=request.destination_kind,
            destination_session_path=(
                Path(request.destination_session_path).resolve()
                if request.destination_session_path is not None
                else None
            ),
            connection_id=request.connection_id,
            model=request.model,
            web_enabled=request.web_enabled,
            allowed_tools=merge_web_tools(
                request.allowed_tools, web_enabled=request.web_enabled
            ),
            approval_allowed_tools=tuple(request.approval_allowed_tools),
            approved_tools=tuple(request.approved_tools),
            permission_mode=request.permission_mode,
            dtstart_local=request.dtstart_local,
            timezone=request.timezone,
            rrule=rrule,
            status="active",
            misfire_policy=request.misfire_policy,
            overlap_policy=request.overlap_policy,
            retry_policy=request.retry_policy or RetryPolicy(),
            revision=1,
        )
        try:
            validate_schedule(definition)
        except ScheduleValidationError as error:
            raise AssistantServiceError(str(error)) from error

        next_run = self._compute_next(definition, after=now)
        store = self._get_store()
        try:
            created = store.create(definition, now=now, next_run_at_utc=next_run)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        except ScheduleValidationError as error:
            raise AssistantServiceError(str(error)) from error
        return _to_schedule(created, next_run_at_utc=next_run)

    def duplicate(self, schedule_id: str) -> AssistantSchedule:
        """复制计划为新 active 定义；名称加「副本」后缀，revision 从 1 起。"""
        source = self.get(schedule_id)
        base_name = str(source.name or "").strip() or "计划"
        copy_name = f"{base_name} 副本"
        return self.create(
            ScheduleCreateRequest(
                name=copy_name,
                prompt=source.prompt,
                workspace_root=source.workspace_root,
                destination_kind=source.destination_kind,
                destination_session_path=source.destination_session_path,
                connection_id=source.connection_id,
                model=source.model,
                web_enabled=source.web_enabled,
                allowed_tools=tuple(source.allowed_tools),
                approval_allowed_tools=tuple(source.approval_allowed_tools),
                approved_tools=tuple(source.approved_tools),
                permission_mode=source.permission_mode,
                dtstart_local=source.dtstart_local,
                timezone=source.timezone,
                rrule=source.rrule,
                misfire_policy=source.misfire_policy,
                overlap_policy=source.overlap_policy,
                retry_policy=source.retry_policy or RetryPolicy(),
            )
        )

    def update(self, schedule_id: str, request: ScheduleUpdateRequest) -> AssistantSchedule:
        now = self._clock()
        store = self._get_store()
        fields: dict = {}
        if request.name is not None:
            fields["name"] = request.name
        if request.prompt is not None:
            fields["prompt"] = request.prompt
        if request.workspace_root is not None:
            self._validate_workspace(request.workspace_root)
            fields["workspace_root"] = Path(request.workspace_root).resolve()
        if request.destination_kind is not None:
            fields["destination_kind"] = request.destination_kind
        if request.destination_session_path is not ...:
            path = request.destination_session_path
            fields["destination_session_path"] = (
                Path(path).resolve() if path is not None else None  # type: ignore[arg-type]
            )
        if request.connection_id is not None:
            self._validate_connection(request.connection_id)
            fields["connection_id"] = request.connection_id
        if request.model is not None:
            fields["model"] = request.model
        if request.web_enabled is not None:
            fields["web_enabled"] = request.web_enabled
        if request.allowed_tools is not None:
            fields["allowed_tools"] = tuple(request.allowed_tools)
        if request.approval_allowed_tools is not None:
            fields["approval_allowed_tools"] = tuple(request.approval_allowed_tools)
        if request.approved_tools is not None:
            fields["approved_tools"] = tuple(request.approved_tools)
        if request.permission_mode is not None:
            fields["permission_mode"] = request.permission_mode
        if request.dtstart_local is not None:
            fields["dtstart_local"] = request.dtstart_local
        if request.timezone is not None:
            fields["timezone"] = request.timezone
        if request.rrule is not ...:
            try:
                fields["rrule"] = normalize_rrule(request.rrule)  # type: ignore[arg-type]
            except RecurrenceError as error:
                raise AssistantServiceError(str(error)) from error
        if request.misfire_policy is not None:
            fields["misfire_policy"] = request.misfire_policy
        if request.overlap_policy is not None:
            fields["overlap_policy"] = request.overlap_policy
        if request.retry_policy is not None:
            fields["retry_policy"] = request.retry_policy

        try:
            current = store.get(schedule_id)
            if current is None:
                raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
            # web_enabled 与 allowed_tools 合并后再写入，避免只开联网却无 web 工具
            merged_web = fields.get("web_enabled", current.web_enabled)
            merged_tools = fields.get("allowed_tools", current.allowed_tools)
            fields["allowed_tools"] = merge_web_tools(
                merged_tools, web_enabled=bool(merged_web)
            )
            # 合并后算 next
            merged_data = {**current.__dict__, **fields}
            probe = ScheduleDefinition(**merged_data)
            probe = validate_schedule(
                ScheduleDefinition(
                    **{
                        **probe.__dict__,
                        "revision": current.revision + 1,
                    }
                )
            )
            next_run = self._compute_next(probe, after=now) if probe.status == "active" else None
            updated = store.update(
                schedule_id,
                expected_revision=request.expected_revision,
                now=now,
                next_run_at_utc=next_run,
                **fields,
            )
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        except ScheduleValidationError as error:
            raise AssistantServiceError(str(error)) from error
        return _to_schedule(
            updated,
            next_run_at_utc=store.get_next_run_at_utc(schedule_id),
            last_run_at_utc=self._last_run(store, schedule_id),
        )

    def list(self, status: ScheduleStatus | None = None) -> list[AssistantScheduleSummary]:
        store = self._get_store()
        try:
            items = store.list_schedules(status=status)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        return [
            _to_summary(
                item,
                next_run_at_utc=store.get_next_run_at_utc(item.id),
                last_run_at_utc=self._last_run(store, item.id),
            )
            for item in items
        ]

    def get(self, schedule_id: str) -> AssistantSchedule:
        store = self._get_store()
        try:
            definition = store.get(schedule_id)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        if definition is None:
            raise AssistantServiceError(f"计划不存在：{schedule_id}")
        return _to_schedule(
            definition,
            next_run_at_utc=store.get_next_run_at_utc(schedule_id),
            last_run_at_utc=self._last_run(store, schedule_id),
        )

    def pause(self, schedule_id: str) -> AssistantSchedule:
        now = self._clock()
        store = self._get_store()
        try:
            paused = store.pause(schedule_id, now=now)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        return _to_schedule(paused, next_run_at_utc=None, last_run_at_utc=self._last_run(store, schedule_id))

    def resume(self, schedule_id: str, *, now: datetime) -> AssistantSchedule:
        store = self._get_store()
        try:
            current = store.get(schedule_id)
            if current is None:
                raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
            next_run = self._compute_next(current, after=now)
            resumed = store.resume(schedule_id, now=now, next_run_at_utc=next_run)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        except RecurrenceError as error:
            raise AssistantServiceError(str(error)) from error
        return _to_schedule(
            resumed,
            next_run_at_utc=store.get_next_run_at_utc(schedule_id),
            last_run_at_utc=self._last_run(store, schedule_id),
        )

    def archive(self, schedule_id: str) -> AssistantSchedule:
        now = self._clock()
        store = self._get_store()
        try:
            archived = store.archive(schedule_id, now=now)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        return _to_schedule(archived, next_run_at_utc=None, last_run_at_utc=self._last_run(store, schedule_id))

    def delete(self, schedule_id: str) -> None:
        store = self._get_store()
        try:
            store.delete(schedule_id)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error

    def run_now(self, schedule_id: str, *, request_id: str) -> AssistantScheduleRun:
        now = self._clock()
        store = self._get_store()
        try:
            definition = store.get(schedule_id)
            if definition is None:
                raise ScheduleStoreError("not_found", f"计划不存在: {schedule_id}")
            trigger_key = f"manual:{request_id}"
            # 幂等：相同 request_id 返回已有 run
            existing = [
                r
                for r in store.list_runs(schedule_id=schedule_id, limit=50)
                if r.trigger_key == trigger_key
            ]
            if existing:
                return _to_run(existing[0])
            run = store.create_run(
                schedule_id=schedule_id,
                schedule_revision=definition.revision,
                trigger_key=trigger_key,
                trigger_kind="manual",
                scheduled_for_utc=now,
                status="queued",
                now=now,
            )
        except ScheduleStoreError as error:
            if error.code == "duplicate_trigger":
                # 并发下再查一次
                for r in store.list_runs(schedule_id=schedule_id, limit=50):
                    if r.trigger_key == f"manual:{request_id}":
                        return _to_run(r)
            raise _map_store_error(error) from error
        return _to_run(run)

    def list_runs(self, query: RunQuery) -> list[AssistantScheduleRun]:
        store = self._get_store()
        try:
            runs = store.list_runs(
                schedule_id=query.schedule_id,
                unread_only=query.unread_only,
                status=query.status,
                limit=query.limit,
            )
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        return [_to_run(r) for r in runs]

    def get_run(self, run_id: str) -> AssistantScheduleRun:
        """按 id 取单条 run；打开会话等路径禁止只靠 list 截断查找。"""
        store = self._get_store()
        try:
            run = store.get_run(run_id)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        if run is None:
            raise AssistantServiceError("运行记录不存在")
        return _to_run(run)

    def mark_run_read(self, run_id: str) -> None:
        store = self._get_store()
        try:
            store.mark_run_read(run_id)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error

    def mark_all_runs_read(self) -> int:
        store = self._get_store()
        try:
            return store.mark_all_runs_read()
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error

    def cancel_run(self, run_id: str) -> AssistantScheduleRun:
        store = self._get_store()
        try:
            run = store.request_cancel(run_id)
        except ScheduleStoreError as error:
            raise _map_store_error(error) from error
        # 同进程 host worker/executor 若在途，立即传播取消
        worker = self._host_worker
        if worker is not None:
            worker.request_cancel(run_id)
        return _to_run(run)

    def start_host(self) -> ScheduleHostStatus:
        """启动 TUI 内嵌 ScheduleWorker；已在跑则幂等返回状态。"""
        with self._host_lock:
            if self._host_thread is not None and self._host_thread.is_alive():
                return self.host_status()
            store = self._get_store()
            from haagent.scheduling.executor import ScheduledRunExecutor
            from haagent.scheduling.worker import ScheduleWorker

            run_executor = ScheduledRunExecutor(store)
            worker = ScheduleWorker(
                store,
                owner_id=f"tui-{uuid.uuid4().hex[:12]}",
                executor=run_executor,
                clock=self._clock,
            )
            self._host_worker = worker

            thread = threading.Thread(
                target=worker.run_forever,
                name="haagent-schedule-host",
                daemon=True,
            )
            self._host_thread = thread
            thread.start()
            return self.host_status()

    def stop_host(self) -> ScheduleHostStatus:
        """停止 TUI host 并释放 coordinator 租约。"""
        with self._host_lock:
            worker = self._host_worker
            thread = self._host_thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        # join 超时必须继续暴露 running；不能先清状态伪装已停止。
        if thread is None or not thread.is_alive():
            with self._host_lock:
                if self._host_thread is thread:
                    self._host_worker = None
                    self._host_thread = None
        return self.host_status()

    def host_status(self) -> ScheduleHostStatus:
        """返回内嵌 host 可展示状态（供后台页与诊断）。"""
        worker = self._host_worker
        thread = self._host_thread
        running = bool(thread is not None and thread.is_alive())
        owner_id = None
        last_error = None
        fatal = False
        if worker is not None:
            owner_id = worker.owner_id
            last_error = worker.last_error
            fatal = worker.fatal
            # 线程已死但 worker 仍有错误：视为未运行且需关注
            if not running and last_error:
                fatal = True
        return ScheduleHostStatus(
            running=running,
            owner_id=str(owner_id) if owner_id else None,
            last_error=str(last_error) if last_error else None,
            fatal=fatal,
        )

    def preview(
        self, request: SchedulePreviewRequest, *, count: int = 3
    ) -> tuple[datetime, ...]:
        after = request.after or self._clock()
        if after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)
        try:
            rrule = normalize_rrule(request.rrule)
            probe = ScheduleDefinition(
                id="preview",
                name="preview",
                prompt="preview",
                workspace_root=Path.cwd().resolve(),
                destination_kind="new_session",
                destination_session_path=None,
                connection_id="preview",
                model="preview",
                web_enabled=False,
                allowed_tools=(),
                approval_allowed_tools=(),
                approved_tools=(),
                permission_mode="request_approval",
                dtstart_local=request.dtstart_local,
                timezone=request.timezone,
                rrule=rrule,
                status="active",
                misfire_policy="latest",
                overlap_policy="skip",
                retry_policy=RetryPolicy(),
                revision=1,
            )
            return preview_occurrences(probe, after=after, count=count)
        except (RecurrenceError, ScheduleValidationError) as error:
            raise AssistantServiceError(str(error)) from error

    def background_status(self) -> BackgroundServiceStatus:
        return self._get_background().status()

    def install_background_service(self) -> BackgroundServiceStatus:
        try:
            return self._get_background().install()
        except Exception as error:
            if isinstance(error, (BackgroundServiceUnsupported, BackgroundServiceError)):
                raise AssistantServiceError(str(error)) from error
            if isinstance(error, AssistantServiceError):
                raise
            raise AssistantServiceError(str(error) or "后台服务安装失败") from error

    def uninstall_background_service(self) -> BackgroundServiceStatus:
        try:
            return self._get_background().uninstall()
        except Exception as error:
            if isinstance(error, (BackgroundServiceUnsupported, BackgroundServiceError)):
                raise AssistantServiceError(str(error)) from error
            if isinstance(error, AssistantServiceError):
                raise
            raise AssistantServiceError(str(error) or "后台服务卸载失败") from error

    def _validate_workspace(self, workspace_root: Path) -> None:
        root = Path(workspace_root)
        if not root.is_absolute():
            raise AssistantServiceError("workspace 必须是绝对路径")
        resolved = root.resolve()
        if not resolved.exists():
            raise AssistantServiceError(f"workspace 不存在：{resolved}")
        if not resolved.is_dir():
            raise AssistantServiceError(f"workspace 必须是目录：{resolved}")

    def _validate_connection(self, connection_id: str) -> None:
        try:
            assert self._context.model_runtime is not None
            self._context.model_runtime.connection(connection_id)
        except ProviderProfileError as error:
            raise AssistantServiceError(f"模型连接不存在或不可用：{connection_id}") from error

    def _compute_next(
        self, definition: ScheduleDefinition, *, after: datetime
    ) -> datetime | None:
        if after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)
        try:
            # 包含恰好 after 的触发：用 after-epsilon 取下一次
            items = preview_occurrences(
                definition, after=after - timedelta(seconds=1), count=1
            )
            if not items:
                items = preview_occurrences(definition, after=after, count=1)
            if not items and definition.rrule is None:
                early = after - timedelta(days=3650)
                items = preview_occurrences(definition, after=early, count=1)
                if items and items[0] < after:
                    return None
            return items[0] if items else None
        except RecurrenceError as error:
            raise AssistantServiceError(str(error)) from error

    def _last_run(self, store: ScheduleStore, schedule_id: str) -> datetime | None:
        return store.get_last_run_at_utc(schedule_id)
