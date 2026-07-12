"""
haagent/channels/session_actor.py - 每绑定一个顺序会话 Actor

独占 AssistantService，串行处理入站消息，把 RuntimeUiEvent 呈现并回投到 Adapter。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from haagent.channels.event_bridge import ChannelEventBridge
from haagent.channels.interactions import InteractionBroker
from haagent.channels.presenter import (
    ChannelDelivery,
    ChannelPresenter,
    FinalizeText,
    SendInteractionPrompt,
    SendText,
    SetTyping,
)
from haagent.channels.settings import ChannelPermissionMode
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage, SendResult

_SENTINEL = object()


@dataclass(frozen=True)
class SubmitResult:
    status: Literal["accepted", "queued", "busy", "rejected"]
    reason: str = ""
    session_id: str | None = None


class ChannelSessionActor:
    """每个 binding 一个 mailbox：最多 1 active + 1 pending。"""

    def __init__(
        self,
        *,
        binding_key: str,
        address: ChannelAddress,
        owner_sender_id: str,
        workspace_root: Path,
        state: ChannelStateStore,
        adapter: Any,
        service: Any,
        broker: InteractionBroker,
        presenter: ChannelPresenter | None = None,
        permission_mode: ChannelPermissionMode = "request_approval",
    ) -> None:
        self.binding_key = binding_key
        self.address = address
        self.owner_sender_id = owner_sender_id
        self.workspace_root = Path(workspace_root)
        self._state = state
        self._adapter = adapter
        self._service = service
        self._broker = broker
        self._presenter = presenter or ChannelPresenter()
        self._bridge = ChannelEventBridge()
        self._lock = asyncio.Lock()
        self._active = False
        self._pending: tuple[InboundChannelMessage, ChannelPermissionMode] | None = None
        self._default_permission_mode = permission_mode
        self._worker: asyncio.Task[None] | None = None
        self._session_ready = False
        self._session_id: str | None = None
        self._closed = False
        self._current_reply_handle: ChannelReplyHandle | None = None
        self._interaction_watcher: asyncio.Task[None] | None = None
        # 最近一次出站失败原因；不落盘，供 manager status 可见。
        self.last_send_error: str = ""
        # drain 只在 sentinel/关闭时结束；空闲超时不得丢弃仍在生成的模型回复。
        self._drain_idle_timeout: float | None = None

    async def submit(
        self,
        message: InboundChannelMessage,
        *,
        permission_mode: str | None = None,
    ) -> SubmitResult:
        if self._closed:
            return SubmitResult(status="rejected", reason="actor_closed")
        # 首次接受前先确保 session/binding 已持久化，便于调用方立即读到绑定。
        await self._ensure_session()
        mode = _channel_permission_mode(permission_mode or self._default_permission_mode)
        async with self._lock:
            if not self._active and self._pending is None:
                self._pending = (message, mode)
                self._ensure_worker()
                return SubmitResult(status="accepted", session_id=self._session_id)
            if self._active and self._pending is None:
                self._pending = (message, mode)
                return SubmitResult(status="queued", session_id=self._session_id)
            # active + pending 已满：明确拒绝，不静默丢弃。
            busy_text = "当前任务繁忙，请等待或发送 /stop"
        await self._deliver_text(busy_text, reply_handle=message.reply_handle)
        return SubmitResult(status="busy", reason=busy_text, session_id=self._session_id)

    async def cancel_current_turn(self) -> bool:
        result = self._service.sessions.cancel_current_run()
        return getattr(result, "status", "") == "cancelled"

    async def close(self) -> None:
        self._closed = True
        await self.cancel_current_turn()
        if self._interaction_watcher is not None:
            self._interaction_watcher.cancel()
            try:
                await self._interaction_watcher
            except asyncio.CancelledError:
                pass
            self._interaction_watcher = None
        await self._bridge.stop()
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._worker, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker.cancel()
                try:
                    await self._worker
                except asyncio.CancelledError:
                    pass
            self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        await self._bridge.start()
        try:
            while not self._closed:
                async with self._lock:
                    pending = self._pending
                    self._pending = None
                    if pending is None:
                        self._active = False
                        break
                    message, permission_mode = pending
                    self._active = True
                try:
                    await self._process_message(message, permission_mode)
                except Exception as error:
                    # 失败边界：向用户发送脱敏错误，不泄露内部细节。
                    await self._deliver_text(f"处理失败：{type(error).__name__}")
                finally:
                    async with self._lock:
                        self._active = False
                        if self._pending is None:
                            break
        finally:
            await self._bridge.stop()
            restart = False
            async with self._lock:
                if self._worker is asyncio.current_task():
                    self._worker = None
                # bridge 已完整停止后才能启动继任 worker，避免并发重置同一事件桥。
                restart = not self._closed and self._pending is not None
            if restart:
                self._ensure_worker()

    async def _process_message(
        self,
        message: InboundChannelMessage,
        permission_mode: ChannelPermissionMode,
    ) -> None:
        await self._ensure_session()
        self._current_reply_handle = message.reply_handle
        self._presenter.reset()
        # 远程渠道只接受两种受限模式，禁止 full_access 等模式泄漏。
        self._lock_permission_mode(permission_mode)
        handler = self._broker.build_handler(
            address=self.address,
            owner_sender_id=self.owner_sender_id,
            binding_key=self.binding_key,
        )
        # 在 worker 线程阻塞期间，loop 侧监视 pending interaction 并发送提示。
        self._interaction_watcher = asyncio.create_task(self._watch_interactions())
        drain_task = asyncio.create_task(self._drain_bridge_until_done())
        try:
            await asyncio.to_thread(
                self._run_prompt_sync,
                message.text,
                handler,
                permission_mode,
            )
        finally:
            # 通知 drain 结束并等待排空。
            self._bridge.emit_from_thread(_SENTINEL)
            try:
                await asyncio.wait_for(drain_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                drain_task.cancel()
            if self._interaction_watcher is not None:
                self._interaction_watcher.cancel()
                try:
                    await self._interaction_watcher
                except asyncio.CancelledError:
                    pass
                self._interaction_watcher = None
            # 异常/取消路径也必须关闭 typing，避免指示器卡住。
            await self._force_typing_off()
            self._current_reply_handle = None

    def _lock_permission_mode(self, mode: str) -> None:
        mode = _channel_permission_mode(mode)
        sessions = getattr(self._service, "sessions", None)
        if sessions is None:
            return
        setter = getattr(sessions, "set_permission_mode", None)
        if callable(setter):
            setter(mode)
            return
        # 部分 service 把 mode 挂在 session 本体上。
        session = getattr(sessions, "current", None) or getattr(self._service, "session", None)
        if session is not None and hasattr(session, "set_permission_mode"):
            session.set_permission_mode(mode)

    async def _force_typing_off(self) -> None:
        if not getattr(self._adapter.capabilities, "typing", False):
            return
        try:
            await self._adapter.set_typing(
                self.address,
                False,
                reply_handle=self._current_reply_handle,
            )
        except Exception:
            # typing 失败不阻断主路径。
            pass

    def _run_prompt_sync(self, prompt: str, handler: Any, permission_mode: ChannelPermissionMode) -> None:
        # 每次 turn 前再锁一次，防止 resume 后 session 带有其他 mode。
        self._lock_permission_mode(permission_mode)
        self._service.sessions.run_prompt_events(
            prompt,
            event_sink=self._bridge.emit_from_thread,
            include_session_events=True,
            interaction_handler=handler,
        )
    async def _drain_bridge_until_done(self) -> None:
        # 关键路径：阻塞到 worker 发出 sentinel，不因模型长思考空闲而提前退出。
        # 测试可设 _drain_idle_timeout 复现旧 bug；生产必须为 None。
        idle = self._drain_idle_timeout
        while True:
            try:
                if idle is None:
                    event = await self._bridge.get()
                else:
                    event = await self._bridge.get(timeout=idle)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                # 仅测试/异常路径：空闲超时退出会丢后续回复，生产禁止。
                if idle is not None:
                    break
                continue
            if event is _SENTINEL or event is None:
                # 排空剩余非哨兵事件后结束。
                while True:
                    try:
                        event = await self._bridge.get(timeout=0.05)
                    except (asyncio.TimeoutError, asyncio.QueueEmpty):
                        return
                    if event is _SENTINEL or event is None:
                        continue
                    actions = self._presenter.handle(event)
                    await self._apply_actions(actions)
                return
            actions = self._presenter.handle(event)
            await self._apply_actions(actions)

    async def _ensure_session(self) -> None:
        if self._session_ready:
            return
        binding = self._state.get_binding(self.binding_key)
        # 空 session_id 或缺失 binding 时创建新会话，不 resume 无效 id。
        resumed = False
        if binding is not None and binding.session_id and binding.session_id not in {"", "__new__"}:
            try:
                status = self._service.sessions.resume(binding.session_id)
                self._session_id = status.session_id
                resumed = True
            except Exception:
                # session package 被删/损坏时回退 create，避免入站批处理永久卡死。
                resumed = False
        if not resumed:
            status = self._service.sessions.create()
            self._session_id = status.session_id
            self._state.upsert_binding(
                binding_key=self.binding_key,
                instance_id=self.address.instance_id,
                platform=self.address.platform,
                conversation_kind=self.address.conversation_kind,
                conversation_id=self.address.conversation_id,
                thread_id=self.address.thread_id,
                workspace_root=str(self.workspace_root),
                session_id=status.session_id,
                owner_sender_id=self.owner_sender_id,
            )
        self._session_ready = True

    async def _watch_interactions(self) -> None:
        """轮询 broker 的 pending，向渠道发送审批/补充输入提示。"""
        seen: set[str] = set()
        try:
            while True:
                # 按 binding 过滤 + 跳过已提示 nonce，避免多会话饥饿与空转。
                pending = await asyncio.to_thread(
                    self._broker.wait_for_pending,
                    timeout=0.5,
                    binding_key=self.binding_key,
                    exclude_nonces=seen,
                )
                if pending is None:
                    continue
                seen.add(pending.nonce)
                if pending.kind == "approval":
                    text = (
                        f"需要批准 [{pending.nonce}]\n"
                        f"工具：{pending.tool_name}\n"
                        f"{pending.question}\n\n"
                        f"回复：/approve {pending.nonce} 或 /deny {pending.nonce}"
                    )
                else:
                    text = (
                        f"需要补充输入 [{pending.nonce}]\n"
                        f"{pending.question}\n\n"
                        f"回复：/answer {pending.nonce} <内容>"
                    )
                await self._deliver_text(text)
        except asyncio.CancelledError:
            raise

    async def _apply_actions(self, actions: list[ChannelDelivery]) -> None:
        for action in actions:
            if isinstance(action, (SendText, FinalizeText, SendInteractionPrompt)):
                text = action.text if not isinstance(action, SendInteractionPrompt) else action.text
                await self._deliver_text(text)
            elif isinstance(action, SetTyping):
                if getattr(self._adapter.capabilities, "typing", False):
                    try:
                        await self._adapter.set_typing(
                            self.address,
                            action.active,
                            reply_handle=self._current_reply_handle,
                        )
                    except Exception:
                        # typing 失败不阻断主回复。
                        pass

    async def _deliver_text(
        self,
        text: str,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> SendResult | None:
        handle = reply_handle if reply_handle is not None else self._current_reply_handle
        result = await self._adapter.send_text(
            self.address,
            text,
            reply_handle=handle,
        )
        # 发送失败必须显式记录，禁止“模型已回复但渠道无消息”静默发生。
        if result is not None and not getattr(result, "ok", True):
            self.last_send_error = str(getattr(result, "error", None) or "send_failed")
        return result


def _channel_permission_mode(value: str) -> ChannelPermissionMode:
    """聊天渠道失败关闭：禁止将 full_access 或未知值传给 session。"""
    if value == "auto_approve":
        return "auto_approve"
    return "request_approval"
