"""
haagent/tui/application/schedule_flow.py - 计划任务 TUI 流程

打开 /schedules overlay，处理创建编辑、运行收件箱、后台服务、状态轮询与未读 badge；
DB 调用经 Textual worker，不阻塞 UI 线程。TUI 内嵌 ScheduleCoordinator host：
应用打开期间可派发到期任务，退出时释放租约。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from haagent.app.assistant_types import (
    AssistantSchedule,
    AssistantScheduleRun,
    AssistantScheduleSummary,
    BackgroundServiceStatus,
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
        self._badge_error: str | None = None
        self._last_tab: SchedulesTab = "plans"

    def start_background_polling(self) -> None:
        self.refresh_badge()
        self._badge_timer = self._app.set_interval(2.0, self.refresh_badge)
        try:
            self._app.service.schedules.start_host()
        except Exception as error:
            # Host 启动失败不能阻断普通 TUI，但必须在主对话中显式暴露。
            self._app._conversation.append_block("计划任务", f"TUI Host 启动失败：{error}")
            self._app._refresh()

    def stop_background_polling(self) -> None:
        timer = self._badge_timer
        self._badge_timer = None
        if timer is not None:
            timer.stop()
        # stop_host 负责停止线程和释放租约；异常必须向 Textual 生命周期传播。
        self._app.service.schedules.stop_host()

    def refresh_badge(self) -> None:
        # set_interval 在 UI 线程触发；DB 读放到 worker/线程，避免卡主事件循环
        schedules = self._app.service.schedules
        screen = getattr(self._app, "screen", None)
        target_overlay = screen if isinstance(screen, SchedulesOverlay) else None
        selected = target_overlay.state.selected_schedule if target_overlay is not None else None
        selected_schedule_id = selected.id if selected is not None else None
        self._app.run_worker(
            lambda: self._poll_schedule_state(
                schedules,
                target_overlay=target_overlay,
                selected_schedule_id=selected_schedule_id,
            ),
            thread=True,
            exclusive=True,
            name="schedule-badge",
            group="schedule-badge",
        )

    def _poll_schedule_state(
        self,
        schedules: Any,
        *,
        target_overlay: SchedulesOverlay | None = None,
        selected_schedule_id: str | None = None,
    ) -> None:
        try:
            unread_runs = schedules.list_runs(RunQuery(unread_only=True, limit=200))
            count = len(unread_runs)
        except Exception as error:
            self._app.call_from_thread(self._apply_badge_error, str(error))
            if target_overlay is not None:
                self._app.call_from_thread(
                    self._apply_overlay_refresh_error,
                    target_overlay,
                    str(error),
                )
            return
        self._app.call_from_thread(self._apply_badge_count, count)
        if target_overlay is None:
            return
        try:
            all_runs = schedules.list_runs(RunQuery(limit=100))
            items = schedules.list()
            available_ids = {item.id for item in items}
            detail_id = selected_schedule_id if selected_schedule_id in available_ids else None
            if detail_id is None and items:
                detail_id = items[0].id
            detail = schedules.get(detail_id) if detail_id is not None else None
        except Exception as error:
            self._app.call_from_thread(
                self._apply_overlay_refresh_error,
                target_overlay,
                str(error),
            )
            return
        # 批量回到 UI 线程，减少运行列表和计划详情分步重绘。
        self._app.call_from_thread(
            self._apply_overlay_refresh,
            target_overlay,
            items,
            all_runs,
            detail,
        )

    def _apply_overlay_refresh(
        self,
        target_overlay: SchedulesOverlay,
        schedules: list[AssistantScheduleSummary],
        runs: list[AssistantScheduleRun],
        detail: AssistantSchedule | AssistantScheduleSummary | None,
    ) -> None:
        screen = getattr(self._app, "screen", None)
        if screen is target_overlay and target_overlay.is_mounted:
            target_overlay.apply_refresh(schedules, runs, detail)

    def _apply_overlay_refresh_error(
        self,
        target_overlay: SchedulesOverlay,
        message: str,
    ) -> None:
        screen = getattr(self._app, "screen", None)
        if screen is target_overlay and target_overlay.is_mounted:
            target_overlay.apply_refresh_error(message)

    def _apply_badge_error(self, message: str) -> None:
        if message == self._badge_error or not self._app.is_mounted:
            return
        self._badge_error = message
        self._app._conversation.append_block("计划任务", f"未读状态刷新失败：{message}")
        self._app._refresh()

    def _apply_badge_count(self, count: int) -> None:
        # 未读数未变时不触发全量 _refresh，减轻 UI 压力
        self._badge_error = None
        if count == self.unread_count:
            return
        previous_count = self.unread_count
        self.unread_count = count
        if self._app.is_mounted:
            # 未读计划结果属于低频通知，不再挤占只承载当前上下文的顶部状态栏。
            if count > previous_count and count > 0:
                self._app.notify(f"有 {count} 条计划任务结果", title="计划任务")
            self._app._refresh()

    def open_schedules(self, *, initial_tab: SchedulesTab | None = None) -> None:
        if self._app._prompt_has_pending_text():
            return
        tab = initial_tab or self._last_tab
        self._last_tab = tab
        self._app._load_schedules_overlay_worker(tab)

    def load_schedules_overlay(self, tab: SchedulesTab) -> None:
        """仅由 App 的 thread worker 调用，完成 DB 读取后回到 UI 线程挂载 overlay。"""
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
        detail = schedules_api.get(items[0].id) if items else None
        try:
            background = schedules_api.background_status()
        except Exception as error:
            # 系统后台状态失败不应遮蔽 plans/runs，但必须显式呈现为 error。
            background = BackgroundServiceStatus(
                state="error",
                host_type="none",
                detail=f"状态读取失败：{error}",
            )
        host = schedules_api.host_status()
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
        width = self._app.size.width or 120
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
                self._app.service.schedules.mark_run_read(result.run_id)
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
            match = schedules.get_run(run_id)
            session_path = match.session_path
            session_id = match.session_id
            target = session_path or session_id
            if not target:
                self._notify("该运行没有关联会话")
                self.open_schedules(initial_tab="runs")
                return
            status = self._app.service.sessions.resume(str(target))
            self._app.session_flow.show_session_history(status, prefix="已打开计划会话")
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
            detail = status.detail or status.state
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
            detail = status.detail or status.state
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
        request = ScheduleCreateRequest(
            name=request.name,
            prompt=request.prompt,
            workspace_root=request.workspace_root,
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
            from haagent.scheduling.draft import FieldPatch

            set_ = FieldPatch.set
            updated = self._app.service.schedules.update(
                state.editing_id,
                ScheduleUpdateRequest(
                    expected_revision=state.expected_revision,
                    name=set_(request.name),
                    prompt=set_(request.prompt),
                    workspace_root=set_(request.workspace_root),
                    destination_kind=set_(request.destination_kind),
                    destination_session_path=(
                        set_(request.destination_session_path)
                        if request.destination_session_path is not None
                        else FieldPatch.clear()
                    ),
                    connection_id=set_(request.connection_id),
                    model=set_(request.model),
                    web_enabled=set_(request.web_enabled),
                    allowed_tools=set_(request.allowed_tools),
                    approval_allowed_tools=set_(request.approval_allowed_tools),
                    approved_tools=set_(request.approved_tools),
                    permission_mode=set_(request.permission_mode),
                    dtstart_local=set_(request.dtstart_local),
                    timezone=set_(request.timezone),
                    rrule=set_(request.rrule) if request.rrule is not None else FieldPatch.clear(),
                    misfire_policy=set_(request.misfire_policy),
                    overlap_policy=set_(request.overlap_policy),
                    retry_policy=set_(request.retry_policy),
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
