"""
haagent/channels/presenter.py - Runtime 事件到渠道投递动作

把 RuntimeUiEvent 映射为平台无关的 SendText/SetTyping/FinalizeText；
不直接调用平台 SDK，也不处理审批（由 InteractionBroker 负责）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from haagent.runtime.events.types import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    RuntimeUiEvent,
    SessionLifecycleEvent,
    ToolActivityEvent,
)


@dataclass(frozen=True)
class SendText:
    text: str


@dataclass(frozen=True)
class SetTyping:
    active: bool


@dataclass(frozen=True)
class FinalizeText:
    text: str


@dataclass(frozen=True)
class SendInteractionPrompt:
    text: str
    nonce: str


ChannelDelivery = SendText | SetTyping | FinalizeText | SendInteractionPrompt


class ChannelPresenter:
    def __init__(
        self,
        *,
        tool_summary_limit: int = 160,
        tool_summary_silence_seconds: float = 8.0,
    ) -> None:
        self._tool_summary_limit = tool_summary_limit
        # 仅在长时间无用户可见输出后才发工具摘要，避免刷屏。
        self._tool_summary_silence_seconds = tool_summary_silence_seconds
        self._delta_buffer: list[str] = []
        self._typing_active = False
        self._finalized = False
        self._last_user_visible_at = time.monotonic()

    def reset(self) -> None:
        self._delta_buffer.clear()
        self._typing_active = False
        self._finalized = False
        self._last_user_visible_at = time.monotonic()

    def handle(self, event: RuntimeUiEvent) -> list[ChannelDelivery]:
        if isinstance(event, SessionLifecycleEvent):
            return self._handle_lifecycle(event)
        if isinstance(event, AssistantDeltaEvent):
            self._delta_buffer.append(event.delta)
            return []
        if isinstance(event, AssistantMessageEvent):
            return self._finalize(event.content)
        if isinstance(event, ToolActivityEvent):
            return self._handle_tool(event)
        if isinstance(event, FailureNoticeEvent):
            return self._handle_failure(event)
        # 审批/补充输入由 InteractionBroker 发送提示，Presenter 不重复发送。
        return []

    def _handle_lifecycle(self, event: SessionLifecycleEvent) -> list[ChannelDelivery]:
        if event.state == "turn_started":
            self.reset()
            self._typing_active = True
            return [SetTyping(True)]
        if event.state in {"turn_finished", "session_finished"}:
            actions: list[ChannelDelivery] = []
            if not self._finalized and self._delta_buffer:
                actions.extend(self._finalize("".join(self._delta_buffer)))
            if self._typing_active:
                self._typing_active = False
                actions.append(SetTyping(False))
            return actions
        return []

    def _handle_tool(self, event: ToolActivityEvent) -> list[ChannelDelivery]:
        if event.status != "started":
            return []
        now = time.monotonic()
        # 静默期未到：不发工具摘要。
        if now - self._last_user_visible_at < self._tool_summary_silence_seconds:
            return []
        summary = (event.summary or event.tool_name).strip()
        if len(summary) > self._tool_summary_limit:
            summary = summary[: self._tool_summary_limit] + "…"
        self._last_user_visible_at = now
        return [SendText(text=f"工具：{event.tool_name} — {summary}")]

    def _handle_failure(self, event: FailureNoticeEvent) -> list[ChannelDelivery]:
        actions: list[ChannelDelivery] = []
        if self._typing_active:
            self._typing_active = False
            actions.append(SetTyping(False))
        episode_id = Path(event.episode_path).name if event.episode_path else ""
        if not episode_id and event.episode_path:
            episode_id = Path(event.episode_path).parent.name
        # 只展示脱敏 episode 标识，不输出完整路径或 secret。
        text = f"失败：{event.failure_category} / {event.reason}"
        if episode_id:
            text += f"（{episode_id}）"
        actions.append(SendText(text=text))
        self._last_user_visible_at = time.monotonic()
        return actions

    def _finalize(self, content: str) -> list[ChannelDelivery]:
        self._finalized = True
        self._delta_buffer.clear()
        text = content.strip()
        if not text:
            return []
        self._last_user_visible_at = time.monotonic()
        return [FinalizeText(text=text)]
