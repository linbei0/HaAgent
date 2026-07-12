"""
haagent/tui/overlays/schedule_editor.py - 计划创建/编辑四步向导

任务 / 计划 / 执行 / 确认；频率表单生成 RRULE；预览与确认摘要。
名称与 prompt 通过真实 Input 编辑，不使用按键追加假字。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from haagent.app.assistant_types import ScheduleCreateRequest
from haagent.scheduling.models import (
    RetryPolicy,
    SCHEDULE_WEB_TOOLS,
    merge_web_tools,
)
from haagent.tui.design.utils import safe_summary

FrequencyKind = Literal["once", "interval", "daily", "weekly", "monthly", "custom"]
EditorPage = Literal[0, 1, 2, 3]
InputMode = Literal[
    "name",
    "prompt",
    "custom_rrule",
    "interval_value",
    "byday",
    "bymonthday",
    "connection_id",
    "model",
    "session_path",
    "workspace_root",
    "custom_tools",
    "approval_tools",
    "approved_tools",
    "retry_max",
    "retry_initial",
    "retry_multiplier",
    "retry_max_delay",
    None,
]

PAGE_TITLES = ("任务", "计划", "执行", "确认")

TOOL_PRESETS: dict[str, tuple[str, ...]] = {
    "readonly": ("file_list", "grep", "file_read", "skill_list", "skill_read"),
    "workspace_write": (
        "file_list",
        "grep",
        "file_read",
        "file_write",
        "apply_patch",
        "skill_list",
        "skill_read",
    ),
    "custom": (),
}

WEB_TOOLS: tuple[str, ...] = SCHEDULE_WEB_TOOLS

# 与 runtime PermissionMode 一致；k 键循环顺序
PERMISSION_MODE_ORDER: tuple[str, ...] = (
    "request_approval",
    "auto_approve",
    "full_access",
)


def parse_rrule_fields(rrule: str | None) -> dict[str, Any]:
    """从 RRULE 还原编辑器频率字段，供 load→save round-trip。"""
    if not rrule or not str(rrule).strip():
        return {
            "frequency": "once",
            "custom_rrule": "",
            "interval_value": 30,
            "interval_unit": "minutes",
            "byday": "MO",
            "bymonthday": 1,
        }
    text = str(rrule).strip()
    if text.upper().startswith("RRULE:"):
        text = text[6:].strip()
    parts: dict[str, str] = {}
    for segment in text.split(";"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        parts[key.strip().upper()] = value.strip()
    freq = (parts.get("FREQ") or "").upper()
    interval = 1
    if "INTERVAL" in parts:
        try:
            interval = max(1, int(parts["INTERVAL"]))
        except ValueError:
            interval = 1
    byday = parts.get("BYDAY") or "MO"
    bymonthday = 1
    if "BYMONTHDAY" in parts:
        try:
            bymonthday = int(parts["BYMONTHDAY"].split(",")[0])
        except ValueError:
            bymonthday = 1
    # 含 BYSETPOS 或其它复杂限定 → custom，避免丢字段
    complex_keys = {"BYSETPOS", "BYYEARDAY", "BYWEEKNO", "BYMONTH", "COUNT", "UNTIL", "WKST"}
    if complex_keys.intersection(parts) or (
        freq == "WEEKLY" and "," in byday and "BYSETPOS" in parts
    ):
        return {
            "frequency": "custom",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "minutes",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    if freq == "MINUTELY":
        return {
            "frequency": "interval",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "minutes",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    if freq == "HOURLY":
        return {
            "frequency": "interval",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "hours",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    if freq == "DAILY" and "INTERVAL" not in parts and "BYDAY" not in parts:
        return {
            "frequency": "daily",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "minutes",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    if freq == "WEEKLY" and set(parts) <= {"FREQ", "BYDAY", "INTERVAL"}:
        return {
            "frequency": "weekly",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "minutes",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    if freq == "MONTHLY" and set(parts) <= {"FREQ", "BYMONTHDAY", "INTERVAL"}:
        return {
            "frequency": "monthly",
            "custom_rrule": text,
            "interval_value": interval,
            "interval_unit": "minutes",
            "byday": byday,
            "bymonthday": bymonthday,
        }
    return {
        "frequency": "custom",
        "custom_rrule": text,
        "interval_value": interval,
        "interval_unit": "minutes",
        "byday": byday,
        "bymonthday": bymonthday,
    }


def infer_tool_preset(
    allowed_tools: tuple[str, ...] | list[str],
) -> tuple[str, tuple[str, ...]]:
    """识别只读/写权限预设；否则 custom 并保留完整工具列表。"""
    cleaned = tuple(
        t for t in allowed_tools if t not in SCHEDULE_WEB_TOOLS
    )
    for name, preset in TOOL_PRESETS.items():
        if name == "custom":
            continue
        if cleaned == preset:
            return name, cleaned
    return "custom", cleaned


@dataclass(frozen=True)
class ScheduleEditorState:
    page: int = 0
    # 任务
    name: str = ""
    prompt: str = ""
    destination_kind: str = "new_session"
    destination_session_path: str = ""
    workspace_root: str = ""
    # 计划
    frequency: FrequencyKind = "daily"
    interval_value: int = 30
    interval_unit: str = "minutes"
    byday: str = "MO"
    bymonthday: int = 1
    custom_rrule: str = ""
    timezone: str = "Asia/Shanghai"
    dtstart_local: str = ""
    misfire_policy: str = "latest"
    overlap_policy: str = "skip"
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: int = 30
    retry_multiplier: float = 2.0
    retry_max_delay_seconds: int = 900
    previews: tuple[datetime, ...] = ()
    # 执行
    connection_id: str = ""
    model: str = ""
    web_enabled: bool = False
    tool_preset: str = "readonly"
    custom_allowed_tools: tuple[str, ...] = ()
    approval_allowed_tools: tuple[str, ...] = ()
    approved_tools: tuple[str, ...] = ()
    permission_mode: str = "request_approval"
    # 元数据
    editing_id: str | None = None
    expected_revision: int | None = None
    message: str = ""
    field_focus: str = "name"

    @classmethod
    def from_schedule(
        cls,
        item: Any,
        *,
        defaults: ScheduleEditorState | None = None,
    ) -> ScheduleEditorState:
        """从 AssistantSchedule 完整还原编辑器状态。"""
        base = defaults or cls()
        fields = parse_rrule_fields(getattr(item, "rrule", None))
        preset, custom_tools = infer_tool_preset(
            tuple(getattr(item, "allowed_tools", ()) or ())
        )
        retry = getattr(item, "retry_policy", None) or RetryPolicy()
        return replace(
            base,
            name=str(getattr(item, "name", "") or ""),
            prompt=str(getattr(item, "prompt", "") or ""),
            destination_kind=str(getattr(item, "destination_kind", "new_session")),
            destination_session_path=str(
                getattr(item, "destination_session_path", None) or ""
            ),
            workspace_root=str(getattr(item, "workspace_root", "") or ""),
            frequency=fields["frequency"],  # type: ignore[arg-type]
            interval_value=int(fields["interval_value"]),
            interval_unit=str(fields["interval_unit"]),
            byday=str(fields["byday"]),
            bymonthday=int(fields["bymonthday"]),
            custom_rrule=str(fields["custom_rrule"]),
            timezone=str(getattr(item, "timezone", base.timezone) or base.timezone),
            dtstart_local=getattr(item, "dtstart_local").isoformat(timespec="minutes")
            if getattr(item, "dtstart_local", None) is not None
            else base.dtstart_local,
            misfire_policy=str(getattr(item, "misfire_policy", "latest")),
            overlap_policy=str(getattr(item, "overlap_policy", "skip")),
            retry_max_attempts=int(getattr(retry, "max_attempts", 3)),
            retry_initial_delay_seconds=int(
                getattr(retry, "initial_delay_seconds", 30)
            ),
            retry_multiplier=float(getattr(retry, "multiplier", 2.0)),
            retry_max_delay_seconds=int(getattr(retry, "max_delay_seconds", 900)),
            connection_id=str(getattr(item, "connection_id", "") or ""),
            model=str(getattr(item, "model", "") or ""),
            web_enabled=bool(getattr(item, "web_enabled", False)),
            tool_preset=preset,
            custom_allowed_tools=custom_tools,
            approval_allowed_tools=tuple(
                getattr(item, "approval_allowed_tools", ()) or ()
            ),
            approved_tools=tuple(getattr(item, "approved_tools", ()) or ()),
            permission_mode=str(
                getattr(item, "permission_mode", "request_approval")
                or "request_approval"
            ),
            editing_id=str(getattr(item, "id", "") or "") or None,
            expected_revision=int(getattr(item, "revision", 1)),
        )

    def with_page(self, page: int) -> ScheduleEditorState:
        return replace(self, page=max(0, min(3, page)), message="")

    def with_field(self, key: str, value: Any) -> ScheduleEditorState:
        if not hasattr(self, key):
            return self
        return replace(self, **{key: value})

    def with_previews(self, previews: tuple[datetime, ...]) -> ScheduleEditorState:
        return replace(self, previews=previews)

    def build_rrule(self) -> str | None:
        if self.frequency == "once":
            return None
        if self.frequency == "daily":
            n = max(1, int(self.interval_value or 1))
            if n > 1:
                return f"FREQ=DAILY;INTERVAL={n}"
            return "FREQ=DAILY"
        if self.frequency == "weekly":
            days = self.byday or "MO"
            n = max(1, int(self.interval_value or 1))
            if n > 1:
                return f"FREQ=WEEKLY;INTERVAL={n};BYDAY={days}"
            return f"FREQ=WEEKLY;BYDAY={days}"
        if self.frequency == "monthly":
            n = max(1, int(self.interval_value or 1))
            day = int(self.bymonthday)
            if n > 1:
                return f"FREQ=MONTHLY;INTERVAL={n};BYMONTHDAY={day}"
            return f"FREQ=MONTHLY;BYMONTHDAY={day}"
        if self.frequency == "interval":
            unit = self.interval_unit
            n = max(1, int(self.interval_value))
            if unit in {"minutes", "minute"}:
                return f"FREQ=MINUTELY;INTERVAL={n}"
            if unit in {"hours", "hour"}:
                return f"FREQ=HOURLY;INTERVAL={n}"
            return f"FREQ=MINUTELY;INTERVAL={n}"
        if self.frequency == "custom":
            text = (self.custom_rrule or "").strip()
            return text or None
        return "FREQ=DAILY"

    def parse_dtstart(self) -> datetime:
        raw = (self.dtstart_local or "").strip()
        if raw:
            try:
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is not None:
                    return parsed.replace(tzinfo=None)
                return parsed
            except ValueError:
                pass
        now = datetime.now()
        return now.replace(second=0, microsecond=0)

    def validate_for_save(self) -> str | None:
        """空名称/空 prompt 拒绝保存。"""
        if not self.name.strip():
            return "名称不能为空"
        if not self.prompt.strip():
            return "任务内容不能为空"
        return None

    def to_create_request(self) -> ScheduleCreateRequest:
        error = self.validate_for_save()
        if error:
            raise ValueError(error)
        if self.tool_preset == "custom":
            # 空自定义集合必须原样保留，禁止静默替换为 readonly
            tools = list(self.custom_allowed_tools)
        else:
            tools = list(TOOL_PRESETS.get(self.tool_preset, TOOL_PRESETS["readonly"]))
        # 开启联网时把 web 工具并入计划 allowed_tools（执行器/turn 也会兜底合并）
        tools = list(merge_web_tools(tools, web_enabled=self.web_enabled))
        ws = Path(self.workspace_root) if self.workspace_root else Path.cwd()
        dest_path = (
            Path(self.destination_session_path)
            if self.destination_kind == "resume_session" and self.destination_session_path
            else None
        )
        return ScheduleCreateRequest(
            name=self.name.strip(),
            prompt=self.prompt.strip(),
            workspace_root=ws,
            destination_kind=self.destination_kind,  # type: ignore[arg-type]
            destination_session_path=dest_path,
            connection_id=self.connection_id or "local",
            model=self.model or "default",
            web_enabled=self.web_enabled,
            allowed_tools=tuple(tools),
            approval_allowed_tools=tuple(self.approval_allowed_tools),
            approved_tools=tuple(self.approved_tools),
            permission_mode=self.permission_mode,  # type: ignore[arg-type]
            dtstart_local=self.parse_dtstart(),
            timezone=self.timezone or "UTC",
            rrule=self.build_rrule(),
            misfire_policy=self.misfire_policy,  # type: ignore[arg-type]
            overlap_policy=self.overlap_policy,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(
                max_attempts=max(1, int(self.retry_max_attempts)),
                # 允许 0：立即重试；禁止用 max(1, ...) 抬高用户配置
                initial_delay_seconds=max(0, int(self.retry_initial_delay_seconds)),
                multiplier=float(self.retry_multiplier),
                max_delay_seconds=max(1, int(self.retry_max_delay_seconds)),
            ),
        )

    def render(self) -> str:
        title = PAGE_TITLES[self.page] if 0 <= self.page < 4 else "?"
        lines = [
            f"计划编辑 — {title} ({self.page + 1}/4)",
            "",
        ]
        if self.page == 0:
            lines.extend(self._render_task())
        elif self.page == 1:
            lines.extend(self._render_plan())
        elif self.page == 2:
            lines.extend(self._render_exec())
        else:
            lines.extend(self._render_confirm())
        if self.message:
            lines.extend(["", self.message])
        lines.extend(
            [
                "",
                "Tab 下一页  Shift+Tab 上一页  v 预览触发  Enter 确认保存  Esc 取消",
            ]
        )
        return "\n".join(lines)

    def _render_task(self) -> list[str]:
        return [
            f"名称: {self.name or '(空)'}",
            f"Prompt: {safe_summary(self.prompt or '(空)', 60)}",
            f"Destination: {self.destination_kind}",
            f"续接会话路径: {self.destination_session_path or '-'}",
            f"Workspace: {self.workspace_root or '(当前)'}",
            "",
            "n 名称  p 任务  d 切换 destination  s 续接路径  e 工作区  w 用当前目录",
            "提示: 快捷键后在底部输入框编辑，Enter 确认，Esc 取消",
        ]

    def _render_plan(self) -> list[str]:
        misfire_help = {
            "skip": "跳过过期(grace 外)",
            "latest": "只跑最近一次",
            "all": "补跑全部积压(有界批)",
        }.get(self.misfire_policy, self.misfire_policy)
        overlap_help = {
            "skip": "有运行中则跳过新触发",
            "queue": "同计划串行排队",
            "parallel": "仅只读工具可并行",
        }.get(self.overlap_policy, self.overlap_policy)
        lines = [
            f"频率: {self.frequency}",
            f"RRULE: {self.build_rrule() or '(一次性 once)'}",
            f"时区: {self.timezone}",
            f"开始(本地): {self.dtstart_local or self.parse_dtstart().isoformat(timespec='minutes')}",
            f"misfire: {self.misfire_policy} ({misfire_help})",
            f"overlap: {self.overlap_policy} ({overlap_help})",
            (
                f"retry: max={self.retry_max_attempts} "
                f"initial={self.retry_initial_delay_seconds}s "
                f"×{self.retry_multiplier} cap={self.retry_max_delay_seconds}s"
            ),
            "",
            "f 频率  t 时区  m misfire  o overlap  y 重试参数",
            "未来 3 次预览 (v 刷新):",
        ]
        if self.previews:
            for item in self.previews[:3]:
                lines.append(f"  - {item.isoformat()}")
        else:
            lines.append("  （尚未预览）")
        if self.frequency == "interval":
            lines.append(f"间隔: {self.interval_value} {self.interval_unit}")
            lines.append("i 输入间隔  u 切换分钟/小时")
        if self.frequency in {"weekly", "monthly", "daily"} and self.frequency != "daily":
            lines.append(f"INTERVAL: {self.interval_value}")
            lines.append("i 输入 INTERVAL")
        if self.frequency == "weekly":
            lines.append(f"BYDAY: {self.byday}")
            lines.append("b 编辑 BYDAY (如 MO,WE)")
        if self.frequency == "monthly":
            lines.append(f"BYMONTHDAY: {self.bymonthday}")
            lines.append("d 编辑 BYMONTHDAY")
        if self.frequency == "custom":
            lines.append(f"自定义: {self.custom_rrule or '(空)'}")
            lines.append("r 输入自定义 RRULE")
        return lines

    def _render_exec(self) -> list[str]:
        if self.tool_preset == "custom":
            tools = self.custom_allowed_tools
            tools_label = ", ".join(tools) if tools else "(空集合)"
        else:
            tools = TOOL_PRESETS.get(self.tool_preset, ())
            tools_label = ", ".join(tools) if tools else "-"
        return [
            f"连接: {self.connection_id or '(默认)'}",
            f"模型: {self.model or '(默认)'}",
            f"联网: {'开' if self.web_enabled else '关'}",
            f"工具预设: {self.tool_preset}",
            f"工具: {tools_label}",
            f"批准工具: {', '.join(self.approval_allowed_tools) or '-'}",
            f"已批准: {', '.join(self.approved_tools) or '-'}",
            f"权限模式: {self.permission_mode}",
            f"联网工具: {', '.join(WEB_TOOLS) if self.web_enabled else '（关；按 w 开启）'}",
            "",
            "c 连接  m 模型  w 联网  p 工具预设  a 自定义工具  A 批准工具  P 已批准  k 权限",
            "高风险工具需写入批准工具列表，无人值守才可自动通过",
            "查天气等外网任务：必须 w 开联网",
        ]

    def _render_confirm(self) -> list[str]:
        # 确认页展示草稿；空名称/prompt 时仍可渲染，保存时再拒绝
        err = self.validate_for_save()
        try:
            if err:
                raise ValueError(err)
            req = self.to_create_request()
            name = req.name
            prompt = req.prompt
            ws = str(req.workspace_root)
            dest = req.destination_kind
            tz_rule = f"{req.timezone} / {req.rrule or 'once'}"
            model = f"{req.connection_id} / {req.model}"
            web = req.web_enabled
            tools = ", ".join(req.allowed_tools[:6])
            policy = (
                f"misfire={req.misfire_policy} overlap={req.overlap_policy} "
                f"retry={req.retry_policy.max_attempts}"
            )
        except ValueError:
            name = self.name.strip() or "(空 — 不可保存)"
            prompt = self.prompt.strip() or "(空 — 不可保存)"
            ws = self.workspace_root or "(当前)"
            dest = self.destination_kind
            tz_rule = f"{self.timezone} / {self.build_rrule() or 'once'}"
            model = f"{self.connection_id or '-'} / {self.model or '-'}"
            web = self.web_enabled
            tools = self.tool_preset
            policy = (
                f"misfire={self.misfire_policy} overlap={self.overlap_policy} "
                f"retry={self.retry_max_attempts}"
            )
        lines = [
            f"名称: {name}",
            f"Prompt: {safe_summary(prompt, 70)}",
            f"Workspace: {ws}",
            f"Destination: {dest}",
            f"时区/规则: {tz_rule}",
            f"模型: {model}",
            f"联网: {web}  工具: {tools}",
            policy,
            "",
            "未来 3 次:",
        ]
        if err:
            lines.append(f"⚠ {err}")
        if self.previews:
            for item in self.previews[:3]:
                lines.append(f"  - {item.isoformat()}")
        else:
            lines.append("  （按 v 预览）")
        lines.append("")
        lines.append("Enter 启用/保存计划  ! 立即测试一次")
        return lines


class ScheduleEditorOverlay(ModalScreen[ScheduleEditorState | None]):
    """四步编辑器；名称/prompt 用 Input 真编辑；确认页 Enter 返回最终 state。"""

    def __init__(self, state: ScheduleEditorState | None = None) -> None:
        super().__init__()
        self.state = state or ScheduleEditorState()
        self.input_mode: InputMode = None

    def compose(self) -> ComposeResult:
        with Vertical(id="schedule-editor-dialog"):
            yield Static(self.state.render(), id="schedule-editor-body")
            yield Input(
                placeholder="按 n 输入名称，或 p 输入任务内容…",
                id="schedule-editor-input",
            )

    def on_mount(self) -> None:
        self._hide_input()

    def on_key(self, event: events.Key) -> None:
        # 输入模式下由 Input 处理字符；只拦截 Esc 取消输入
        if self.input_mode is not None:
            if event.key == "escape":
                event.stop()
                self._cancel_input()
            return

        key = event.key
        if key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if key == "tab":
            event.stop()
            self._set_state(self.state.with_page(self.state.page + 1))
            return
        if key == "shift+tab":
            event.stop()
            self._set_state(self.state.with_page(self.state.page - 1))
            return
        if key == "v":
            event.stop()
            self._set_state(replace(self.state, message="正在预览…"))
            self.app.call_later(self._request_preview)
            return
        if key == "enter" and self.state.page == 3:
            event.stop()
            err = self.state.validate_for_save()
            if err:
                self._set_state(replace(self.state, message=err))
                return
            self.dismiss(self.state)
            return
        if key == "exclamation_mark" or event.character == "!":
            if self.state.page == 3:
                event.stop()
                err = self.state.validate_for_save()
                if err:
                    self._set_state(replace(self.state, message=err))
                    return
                self.dismiss(replace(self.state, message="run_now_test"))
                return
        if self.state.page == 0:
            self._handle_task_key(event)
            return
        if self.state.page == 1:
            self._handle_plan_key(event)
            return
        if self.state.page == 2:
            self._handle_exec_key(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "schedule-editor-input":
            return
        event.stop()
        self._commit_input(event.value)

    def _handle_task_key(self, event: events.Key) -> None:
        ch = event.character
        if ch == "n":
            event.stop()
            self._begin_input("name", self.state.name, "输入计划名称，Enter 确认")
        elif ch == "p":
            event.stop()
            self._begin_input("prompt", self.state.prompt, "输入任务内容（发给助手的指令），Enter 确认")
        elif ch == "d":
            event.stop()
            next_kind = (
                "resume_session"
                if self.state.destination_kind == "new_session"
                else "new_session"
            )
            self._set_state(self.state.with_field("destination_kind", next_kind))
        elif ch == "s":
            event.stop()
            self._begin_input(
                "session_path",
                self.state.destination_session_path,
                "输入续接 session 路径（resume_session 必填）",
            )
        elif ch == "e":
            event.stop()
            self._begin_input(
                "workspace_root",
                self.state.workspace_root,
                "输入 workspace 绝对路径",
            )
        elif ch == "w":
            event.stop()
            cwd = str(Path.cwd())
            self._set_state(
                replace(self.state, workspace_root=cwd, message=f"workspace={cwd}")
            )

    def _handle_plan_key(self, event: events.Key) -> None:
        ch = event.character
        if ch == "f":
            event.stop()
            order: list[FrequencyKind] = [
                "once",
                "interval",
                "daily",
                "weekly",
                "monthly",
                "custom",
            ]
            idx = order.index(self.state.frequency) if self.state.frequency in order else 0
            self._set_state(self.state.with_field("frequency", order[(idx + 1) % len(order)]))
        elif ch == "t":
            event.stop()
            zones = ["Asia/Shanghai", "UTC", "America/New_York", "Europe/Berlin"]
            idx = zones.index(self.state.timezone) if self.state.timezone in zones else 0
            self._set_state(self.state.with_field("timezone", zones[(idx + 1) % len(zones)]))
        elif ch == "m":
            event.stop()
            order = ["skip", "latest", "all"]
            idx = order.index(self.state.misfire_policy) if self.state.misfire_policy in order else 0
            self._set_state(self.state.with_field("misfire_policy", order[(idx + 1) % len(order)]))
        elif ch == "o":
            event.stop()
            order = ["skip", "queue", "parallel"]
            idx = order.index(self.state.overlap_policy) if self.state.overlap_policy in order else 0
            self._set_state(self.state.with_field("overlap_policy", order[(idx + 1) % len(order)]))
        elif ch == "i" and self.state.frequency in {
            "interval",
            "weekly",
            "monthly",
            "daily",
        }:
            event.stop()
            self._begin_input(
                "interval_value",
                str(self.state.interval_value),
                "输入 INTERVAL / 间隔数值",
            )
        elif ch == "u" and self.state.frequency == "interval":
            event.stop()
            unit = "hours" if self.state.interval_unit in {"minutes", "minute"} else "minutes"
            self._set_state(self.state.with_field("interval_unit", unit))
        elif ch == "b" and self.state.frequency == "weekly":
            event.stop()
            self._begin_input("byday", self.state.byday, "输入 BYDAY，如 MO 或 MO,WE,FR")
        elif ch == "d" and self.state.frequency == "monthly":
            event.stop()
            self._begin_input(
                "bymonthday",
                str(self.state.bymonthday),
                "输入 BYMONTHDAY (1-31 或 -1)",
            )
        elif ch == "y":
            event.stop()
            self._begin_input(
                "retry_max",
                str(self.state.retry_max_attempts),
                "输入 max_attempts，随后可继续 initial/multiplier/max_delay",
            )
        elif ch == "r" and self.state.frequency == "custom":
            event.stop()
            self._begin_input(
                "custom_rrule",
                self.state.custom_rrule,
                "输入 RRULE，例如 FREQ=WEEKLY;BYDAY=MO",
            )

    def _handle_exec_key(self, event: events.Key) -> None:
        ch = event.character
        if ch == "w":
            event.stop()
            self._set_state(self.state.with_field("web_enabled", not self.state.web_enabled))
        elif ch == "p":
            event.stop()
            order = ["readonly", "workspace_write", "custom"]
            idx = order.index(self.state.tool_preset) if self.state.tool_preset in order else 0
            self._set_state(self.state.with_field("tool_preset", order[(idx + 1) % len(order)]))
        elif ch == "c":
            event.stop()
            self._begin_input(
                "connection_id",
                self.state.connection_id,
                "输入 connection_id（providers.json 中的连接 id）",
            )
        elif ch == "m":
            event.stop()
            self._begin_input("model", self.state.model, "输入模型名")
        elif ch == "a":
            event.stop()
            current = ",".join(self.state.custom_allowed_tools)
            self._begin_input(
                "custom_tools",
                current,
                "输入自定义工具列表，逗号分隔（空=空集合）",
            )
            self._set_state(self.state.with_field("tool_preset", "custom"))
        elif ch == "A":
            event.stop()
            current = ",".join(self.state.approval_allowed_tools)
            self._begin_input(
                "approval_tools",
                current,
                "输入可走审批的工具列表，逗号分隔（空=无）",
            )
        elif ch == "P":
            event.stop()
            current = ",".join(self.state.approved_tools)
            self._begin_input(
                "approved_tools",
                current,
                "输入无人值守已批准工具列表，逗号分隔（空=无）",
            )
        elif ch == "k":
            event.stop()
            # 与 runtime PERMISSION_MODES 一致，禁止 deny/allow 伪值
            modes = list(PERMISSION_MODE_ORDER)
            cur = self.state.permission_mode
            idx = modes.index(cur) if cur in modes else 0
            self._set_state(self.state.with_field("permission_mode", modes[(idx + 1) % len(modes)]))

    def _begin_input(self, mode: InputMode, value: str, placeholder: str) -> None:
        # 进入文本编辑：显示 Input，焦点离开快捷键层
        self.input_mode = mode
        field = self.query_one("#schedule-editor-input", Input)
        field.display = True
        field.disabled = False
        field.placeholder = placeholder
        field.value = value or ""
        field.focus()
        self._set_state(replace(self.state, message=placeholder))

    def _commit_input(self, value: str) -> None:
        mode = self.input_mode
        text = (value or "").strip()
        follow: InputMode = None
        follow_value = ""
        follow_ph = ""
        if mode == "name":
            self.state = self.state.with_field("name", text)
        elif mode == "prompt":
            self.state = self.state.with_field("prompt", text)
        elif mode == "custom_rrule":
            self.state = self.state.with_field("custom_rrule", text)
        elif mode == "interval_value":
            try:
                n = max(1, int(text))
            except ValueError:
                n = self.state.interval_value
            self.state = self.state.with_field("interval_value", n)
        elif mode == "byday":
            self.state = self.state.with_field("byday", text.upper() or "MO")
        elif mode == "bymonthday":
            try:
                day = int(text)
            except ValueError:
                day = self.state.bymonthday
            self.state = self.state.with_field("bymonthday", day)
        elif mode == "connection_id":
            self.state = self.state.with_field("connection_id", text)
        elif mode == "model":
            self.state = self.state.with_field("model", text)
        elif mode == "session_path":
            self.state = self.state.with_field("destination_session_path", text)
        elif mode == "workspace_root":
            self.state = self.state.with_field("workspace_root", text)
        elif mode == "custom_tools":
            tools = tuple(t.strip() for t in text.split(",") if t.strip())
            self.state = replace(
                self.state,
                tool_preset="custom",
                custom_allowed_tools=tools,
            )
        elif mode == "approval_tools":
            tools = tuple(t.strip() for t in text.split(",") if t.strip())
            self.state = self.state.with_field("approval_allowed_tools", tools)
        elif mode == "approved_tools":
            tools = tuple(t.strip() for t in text.split(",") if t.strip())
            self.state = self.state.with_field("approved_tools", tools)
        elif mode == "retry_max":
            try:
                n = max(1, int(text))
            except ValueError:
                n = self.state.retry_max_attempts
            self.state = self.state.with_field("retry_max_attempts", n)
            follow = "retry_initial"
            follow_value = str(self.state.retry_initial_delay_seconds)
            follow_ph = "输入 initial_delay_seconds（可为 0）"
        elif mode == "retry_initial":
            try:
                n = max(0, int(text))
            except ValueError:
                n = self.state.retry_initial_delay_seconds
            self.state = self.state.with_field("retry_initial_delay_seconds", n)
            follow = "retry_multiplier"
            follow_value = str(self.state.retry_multiplier)
            follow_ph = "输入 multiplier（>=1）"
        elif mode == "retry_multiplier":
            try:
                n = max(1.0, float(text))
            except ValueError:
                n = self.state.retry_multiplier
            self.state = self.state.with_field("retry_multiplier", n)
            follow = "retry_max_delay"
            follow_value = str(self.state.retry_max_delay_seconds)
            follow_ph = "输入 max_delay_seconds"
        elif mode == "retry_max_delay":
            try:
                n = max(1, int(text))
            except ValueError:
                n = self.state.retry_max_delay_seconds
            self.state = self.state.with_field("retry_max_delay_seconds", n)
        self.input_mode = None
        self._hide_input()
        if follow is not None:
            self._begin_input(follow, follow_value, follow_ph)
            return
        self._set_state(replace(self.state, message="已保存字段"))

    def _cancel_input(self) -> None:
        self.input_mode = None
        self._hide_input()
        self._set_state(replace(self.state, message="已取消输入"))

    def _hide_input(self) -> None:
        try:
            field = self.query_one("#schedule-editor-input", Input)
        except Exception:
            return
        field.value = ""
        field.display = False
        field.disabled = True

    def _request_preview(self) -> None:
        flow = getattr(self.app, "schedule_flow", None)
        if flow is not None and hasattr(flow, "preview_editor_state"):
            flow.preview_editor_state(self)
            return
        self._set_state(replace(self.state, message="无法预览", previews=()))

    def apply_previews(self, previews: tuple[datetime, ...], message: str = "") -> None:
        self._set_state(self.state.with_previews(previews).with_field("message", message or "已更新预览"))

    def _set_state(self, state: ScheduleEditorState) -> None:
        self.state = state
        self.query_one("#schedule-editor-body", Static).update(state.render())
