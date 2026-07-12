"""
haagent/tui/application/schedule_flow.py - 计划任务 TUI 流程

打开 /schedules overlay，处理创建编辑、运行收件箱、后台服务与未读 badge；
DB 调用经 Textual worker，不阻塞 UI 线程。TUI 内嵌 ScheduleCoordinator host：
应用打开期间可派发到期任务，退出时释放租约。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haagent.app.assistant_types import (
    RunQuery,
    ScheduleCreateRequest,
    SchedulePreviewRequest,
    ScheduleUpdateRequest,
)
from haagent.scheduling.models import RetryPolicy
from haagent.tui.overlays.modals import ConfirmModal
from haagent.tui.overlays.schedule_background import ScheduleBackgroundState
from haagent.tui.overlays.schedule_editor import ScheduleEditorOverlay, ScheduleEditorState
from haagent.tui.overlays.schedule_runs import ScheduleRunsState
from haagent.tui.overlays.schedules import (
    SchedulesOverlay,
    SchedulesOverlayResult,
    SchedulesOverlayState,
    SchedulesTab,
)


class ScheduleFlow:
    """封装计划任务交互；所有写操作经 AssistantService.schedules。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.unread_count = 0
        self._badge_timer = None
        self._last_tab: SchedulesTab = "plans"
        self._pending_open_tab: SchedulesTab = "plans"

    @property
    def host_worker_running(self) -> bool:
        schedules = getattr(getattr(self._app, "service", None), "schedules", None)
        if schedules is None or not hasattr(schedules, "host_status"):
            return False
        try:
            return bool(schedules.host_status().running)
        except Exception:
            return False

    def start_background_polling(self) -> None:
        self.refresh_badge()
        try:
            self._badge_timer = self._app.set_interval(2.0, self.refresh_badge)
        except Exception:
            self._badge_timer = None
        # 真实 AssistantSchedules 才启 host；Fake 无 start_host
        self._maybe_start_host_worker()

    def stop_background_polling(self) -> None:
        timer = self._badge_timer
        self._badge_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._stop_host_worker()

    def start_coordinator_host(
        self,
        *,
        executor: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        owner_id: str | None = None,
        store: Any | None = None,
    ) -> None:
        """启动 TUI 内嵌 host（经 AssistantSchedules.start_host，可注入 executor 供测试）。"""
        del store  # store 仅由 AssistantSchedules 持有，TUI 不得注入私有连接
        schedules = getattr(getattr(self._app, "service", None), "schedules", None)
        if schedules is None or not hasattr(schedules, "start_host"):
            return
        try:
            schedules.start_host(
                executor=executor,
                clock=clock,
                sleep=sleep,
                owner_id=owner_id,
            )
        except Exception:
            return

    def _maybe_start_host_worker(self) -> None:
        schedules = getattr(getattr(self._app, "service", None), "schedules", None)
        if schedules is None or not hasattr(schedules, "start_host"):
            return
        try:
            self.start_coordinator_host()
        except Exception:
            return

    def _stop_host_worker(self) -> None:
        schedules = getattr(getattr(self._app, "service", None), "schedules", None)
        if schedules is None or not hasattr(schedules, "stop_host"):
            return
        try:
            schedules.stop_host()
        except Exception:
            pass

    def refresh_badge(self) -> None:
        # set_interval 在 UI 线程触发；DB 读放到 worker/线程，避免卡主事件循环
        schedules = getattr(self._app.service, "schedules", None)
        if schedules is None:
            self._apply_badge_count(0)
            return
        try:
            # Textual run_worker(thread=True) 优先；测试/无 worker 时同步回退
            run_worker = getattr(self._app, "run_worker", None)
            if callable(run_worker):
                run_worker(
                    lambda: self._poll_badge_count(schedules),
                    thread=True,
                    exclusive=True,
                    name="schedule-badge",
                    group="schedule-badge",
                )
                return
        except Exception:
            pass
        self._poll_badge_count(schedules)

    def _poll_badge_count(self, schedules: Any) -> None:
        try:
            if hasattr(schedules, "unread_count"):
                count = int(schedules.unread_count())
            else:
                runs = schedules.list_runs(RunQuery(unread_only=True, limit=200))
                count = len(runs)
        except Exception:
            return
        apply = getattr(self._app, "call_from_thread", None)
        if callable(apply):
            try:
                apply(self._apply_badge_count, count)
                return
            except Exception:
                pass
        self._apply_badge_count(count)

    def _apply_badge_count(self, count: int) -> None:
        # 未读数未变时不触发全量 _refresh，减轻 UI 压力
        if count == self.unread_count:
            return
        self.unread_count = count
        try:
            if self._app.is_mounted:
                self._app._refresh()
        except Exception:
            pass

    def open_schedules(self, *, initial_tab: SchedulesTab | None = None) -> None:
        if self._app._prompt_has_pending_text():
            return
        tab = initial_tab or self._last_tab
        self._last_tab = tab
        self._pending_open_tab = tab
        # 同步打开路径：Fake/测试与轻量 list 可直接走主线程；重 IO 由 app worker 包装。
        try:
            data = self._load_overlay_data()
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"打开失败：{error}")
            self._app._refresh()
            return
        self._push_overlay(data, tab)

    def open_schedules_async(self, *, initial_tab: SchedulesTab | None = None) -> None:
        """由 app @work 线程加载后回调。"""
        tab = initial_tab or self._last_tab
        self._last_tab = tab
        self._pending_open_tab = tab
        try:
            data = self._load_overlay_data()
        except Exception as error:
            self._app.call_from_thread(self._open_failed, str(error))
            return
        self._app.call_from_thread(self._push_overlay, data, tab)

    def _load_overlay_data(self) -> dict[str, Any]:
        schedules_api = self._app.service.schedules
        items = schedules_api.list()
        runs = schedules_api.list_runs(RunQuery(limit=100))
        detail = None
        if items:
            try:
                detail = schedules_api.get(items[0].id)
            except Exception:
                detail = items[0]
        background = None
        try:
            background = schedules_api.background_status()
        except Exception:
            background = None
        host = None
        try:
            if hasattr(schedules_api, "host_status"):
                host = schedules_api.host_status()
        except Exception:
            host = None
        return {
            "schedules": items,
            "runs": runs,
            "detail": detail,
            "background": background,
            "host": host,
        }

    def _open_failed(self, message: str) -> None:
        self._app._conversation.append_block("计划任务", f"打开失败：{message}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def _push_overlay(self, data: dict[str, Any], tab: SchedulesTab) -> None:
        width = getattr(self._app.size, "width", 120) or 120
        wide = width >= 100
        run_state = ScheduleRunsState(runs=list(data.get("runs") or []))
        bg_state = ScheduleBackgroundState(
            status=data.get("background"),
            host=data.get("host"),
        )
        state = SchedulesOverlayState(
            schedules=list(data.get("schedules") or []),
            runs=list(data.get("runs") or []),
            selected_index=0,
            wide=wide,
            tab=tab,
            detail=data.get("detail"),
            run_state=run_state,
            background=bg_state,
        )
        self._app.push_screen(SchedulesOverlay(state), self.handle_schedules_result)

    def handle_schedules_result(self, result: SchedulesOverlayResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            self.refresh_badge()
            return
        action = result.action
        try:
            if action == "switch_tab" and result.tab:
                self.open_schedules(initial_tab=result.tab)
                return
            if action == "create":
                self._open_editor(None)
                return
            if action == "edit" and result.schedule_id:
                self._open_editor(result.schedule_id)
                return
            if action == "duplicate" and result.schedule_id:
                self._app.service.schedules.duplicate(result.schedule_id)
                self._notify("已复制计划")
                self.open_schedules()
                return
            if action == "pause" and result.schedule_id:
                self._app.service.schedules.pause(result.schedule_id)
                self._notify("已暂停")
                self.open_schedules()
                return
            if action == "resume" and result.schedule_id:
                self._app.service.schedules.resume(
                    result.schedule_id, now=datetime.now(timezone.utc)
                )
                self._notify("已恢复")
                self.open_schedules()
                return
            if action == "archive" and result.schedule_id:
                self._app.service.schedules.archive(result.schedule_id)
                self._notify("已归档")
                self.open_schedules()
                return
            if action == "delete" and result.schedule_id:
                self._confirm_delete(result.schedule_id)
                return
            if action == "run_now" and result.schedule_id:
                self._app.service.schedules.run_now(
                    result.schedule_id, request_id=uuid.uuid4().hex
                )
                self._notify("已排队立即运行")
                self.open_schedules()
                return
            if action == "open_run" and result.run_id:
                try:
                    self._app.service.schedules.mark_run_read(result.run_id)
                except Exception:
                    pass
                self.refresh_badge()
                self.open_schedules(initial_tab="runs")
                return
            if action == "mark_run_read" and result.run_id:
                self._app.service.schedules.mark_run_read(result.run_id)
                self._notify("已标为已读")
                self.refresh_badge()
                self.open_schedules(initial_tab="runs")
                return
            if action == "mark_all_read":
                self._app.service.schedules.mark_all_runs_read()
                self._notify("全部已读")
                self.refresh_badge()
                self.open_schedules(initial_tab="runs")
                return
            if action == "cancel_run" and result.run_id:
                self._app.service.schedules.cancel_run(result.run_id)
                self._notify("已请求取消")
                self.open_schedules(initial_tab="runs")
                return
            if action == "rerun" and result.schedule_id:
                self._app.service.schedules.run_now(
                    result.schedule_id, request_id=uuid.uuid4().hex
                )
                self._notify("已重新排队")
                self.open_schedules(initial_tab="runs")
                return
            if action == "open_session" and result.run_id:
                self._open_run_session(result.run_id)
                return
            if action == "install_background":
                self._confirm_background_install()
                return
            if action == "uninstall_background":
                self._confirm_background_uninstall()
                return
            if action == "refresh_background":
                self.open_schedules(initial_tab="background")
                return
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"操作失败：{error}")
            self._app._refresh()
        self.open_schedules(initial_tab=self._last_tab)

    def _notify(self, message: str) -> None:
        self._app._conversation.append_line(f"计划任务：{message}")
        self._app._refresh()

    def _confirm_delete(self, schedule_id: str) -> None:
        self._app.push_screen(
            ConfirmModal("删除计划", f"将永久删除计划 {schedule_id} 及其运行索引。确认？"),
            lambda confirmed, sid=schedule_id: self._handle_delete_confirm(sid, confirmed),
        )

    def _handle_delete_confirm(self, schedule_id: str, confirmed: bool | None) -> None:
        if not confirmed:
            self.open_schedules()
            return
        try:
            self._app.service.schedules.delete(schedule_id)
            self._notify(f"已删除 {schedule_id}")
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"删除失败：{error}")
            self._app._refresh()
        self.open_schedules()

    def _open_run_session(self, run_id: str) -> None:
        # 优先 session_path 恢复；resume 后必须装载历史，否则主界面像“没反应”
        try:
            schedules = self._app.service.schedules
            match = None
            getter = getattr(schedules, "get_run", None)
            if callable(getter):
                try:
                    match = getter(run_id)
                except Exception:
                    match = None
            if match is None:
                runs = schedules.list_runs(RunQuery(limit=500))
                match = next((r for r in runs if r.id == run_id), None)
            if match is None:
                self._notify("运行记录不存在")
                self.open_schedules(initial_tab="runs")
                return
            session_path = getattr(match, "session_path", None)
            session_id = getattr(match, "session_id", None)
            target = session_path or session_id
            if not target:
                self._notify("该运行没有关联会话")
                self.open_schedules(initial_tab="runs")
                return
            status = self._app.service.sessions.resume(str(target))
            session_flow = getattr(self._app, "session_flow", None)
            if session_flow is not None and hasattr(session_flow, "show_session_history"):
                session_flow.show_session_history(status, prefix="已打开计划会话")
            else:
                self._notify(f"已打开会话：{getattr(status, 'session_id', target)}")
            self._app._refresh()
            self._app._defer_prompt_focus()
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"打开会话失败：{error}")
            self._app._refresh()
            self.open_schedules(initial_tab="runs")

    def _confirm_background_install(self) -> None:
        self._app.push_screen(
            ConfirmModal(
                "安装后台服务",
                "将向操作系统注册用户级 schedule-worker（Windows 任务计划/systemd/launchd）。\n"
                "这是高影响外部系统操作。确认安装？",
            ),
            self._handle_install_confirm,
        )

    def _handle_install_confirm(self, confirmed: bool | None) -> None:
        if not confirmed:
            self.open_schedules(initial_tab="background")
            return
        try:
            status = self._app.service.schedules.install_background_service()
            detail = getattr(status, "detail", "") or getattr(status, "state", "")
            self._notify(f"安装结果：{detail}")
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"安装失败：{error}")
            self._app._refresh()
        self.open_schedules(initial_tab="background")

    def _confirm_background_uninstall(self) -> None:
        self._app.push_screen(
            ConfirmModal(
                "卸载后台服务",
                "将移除操作系统中的 schedule-worker 注册。确认卸载？",
            ),
            self._handle_uninstall_confirm,
        )

    def _handle_uninstall_confirm(self, confirmed: bool | None) -> None:
        if not confirmed:
            self.open_schedules(initial_tab="background")
            return
        try:
            status = self._app.service.schedules.uninstall_background_service()
            detail = getattr(status, "detail", "") or getattr(status, "state", "")
            self._notify(f"卸载结果：{detail}")
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"卸载失败：{error}")
            self._app._refresh()
        self.open_schedules(initial_tab="background")

    def _open_editor(self, schedule_id: str | None) -> None:
        try:
            state = self._build_editor_state(schedule_id)
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"打开编辑器失败：{error}")
            self._app._refresh()
            self.open_schedules()
            return
        self._app.push_screen(ScheduleEditorOverlay(state), self.handle_editor_result)

    def _build_editor_state(self, schedule_id: str | None) -> ScheduleEditorState:
        status = self._app.service.workspace.status()
        ws = str(status.workspace_root)
        connection = status.profile_name or "local"
        model = status.model or "default"
        base = ScheduleEditorState(
            workspace_root=ws,
            connection_id=connection,
            model=model,
            timezone="Asia/Shanghai",
        )
        if not schedule_id:
            return base
        item = self._app.service.schedules.get(schedule_id)
        return ScheduleEditorState.from_schedule(item, defaults=base)

    def handle_editor_result(self, state: ScheduleEditorState | None) -> None:
        if state is None:
            self.open_schedules()
            return
        run_now = state.message == "run_now_test"
        try:
            schedule_id = self._save_editor_state(state)
            if run_now:
                self._app.service.schedules.run_now(
                    schedule_id, request_id=uuid.uuid4().hex
                )
                self._notify("已保存并触发测试运行")
            else:
                self._notify("计划已保存")
        except Exception as error:
            self._app._conversation.append_block("计划任务", f"保存失败：{error}")
            self._app._refresh()
        self.open_schedules()

    def _save_editor_state(self, state: ScheduleEditorState) -> str:
        request = state.to_create_request()
        status = self._app.service.workspace.status()
        ws = request.workspace_root
        try:
            if not Path(ws).exists():
                ws = Path(status.workspace_root)
        except Exception:
            ws = Path(status.workspace_root)
        request = ScheduleCreateRequest(
            name=request.name,
            prompt=request.prompt,
            workspace_root=ws,
            destination_kind=request.destination_kind,
            destination_session_path=request.destination_session_path,
            connection_id=state.connection_id or status.profile_name or "local",
            model=state.model or status.model or "default",
            web_enabled=request.web_enabled,
            allowed_tools=request.allowed_tools,
            approval_allowed_tools=request.approval_allowed_tools,
            approved_tools=request.approved_tools,
            permission_mode=request.permission_mode,
            dtstart_local=request.dtstart_local,
            timezone=request.timezone,
            rrule=request.rrule,
            misfire_policy=request.misfire_policy,
            overlap_policy=request.overlap_policy,
            retry_policy=request.retry_policy or RetryPolicy(),
        )
        if state.editing_id and state.expected_revision is not None:
            updated = self._app.service.schedules.update(
                state.editing_id,
                ScheduleUpdateRequest(
                    expected_revision=state.expected_revision,
                    name=request.name,
                    prompt=request.prompt,
                    workspace_root=request.workspace_root,
                    destination_kind=request.destination_kind,
                    destination_session_path=request.destination_session_path,
                    connection_id=request.connection_id,
                    model=request.model,
                    web_enabled=request.web_enabled,
                    allowed_tools=request.allowed_tools,
                    approval_allowed_tools=request.approval_allowed_tools,
                    approved_tools=request.approved_tools,
                    permission_mode=request.permission_mode,
                    dtstart_local=request.dtstart_local,
                    timezone=request.timezone,
                    rrule=request.rrule,
                    misfire_policy=request.misfire_policy,
                    overlap_policy=request.overlap_policy,
                    retry_policy=request.retry_policy,
                ),
            )
            return updated.id
        created = self._app.service.schedules.create(request)
        return created.id

    def preview_editor_state(self, editor: ScheduleEditorOverlay) -> None:
        state = editor.state
        try:
            previews = self._app.service.schedules.preview(
                SchedulePreviewRequest(
                    dtstart_local=state.parse_dtstart(),
                    timezone=state.timezone,
                    rrule=state.build_rrule(),
                ),
                count=3,
            )
            editor.apply_previews(previews, "已更新预览")
        except Exception as error:
            editor.apply_previews((), f"预览失败：{error}")
