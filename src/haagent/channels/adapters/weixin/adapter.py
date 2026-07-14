"""
haagent/channels/adapters/weixin/adapter.py - 微信 iLink ChannelAdapter

把协议消息映射为渠道合同；负责 poll 生命周期、typing ticket 缓存与文本切分。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Literal

from haagent.channels.adapter import InboundMessageHandler
from haagent.channels.adapters.weixin.rendering import split_weixin_text
from haagent.channels.adapters.weixin.types import (
    WeixinAuthenticationExpired,
    WeixinInboundMessage,
    WeixinProtocolError,
    WeixinRateLimited,
)
from haagent.channels.types import (
    ChannelAddress,
    ChannelReplyHandle,
    InboundChannelMessage,
    SendResult,
)

AdapterState = Literal["connected", "reconnecting", "auth_expired", "failed", "stopped"]
CursorPersistCallback = Callable[[str], None]
# handler 返回这些 outcome 时，本条消息算“可推进批次”
_CURSOR_OK_OUTCOMES = frozenset({"accepted", "queued", "control", "auth_reject", "pair", "duplicate", None})
_MAX_SESSION_RECOVERY_ATTEMPTS = 3


class WeixinAdapter:
    def __init__(
        self,
        *,
        instance_id: str,
        protocol: Any,
        poll_interval: float = 0.5,
        initial_cursor: str = "",
        on_cursor_persist: CursorPersistCallback | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.platform = "weixin"
        self._protocol = protocol
        self._poll_interval = poll_interval
        self._cursor = initial_cursor
        self._on_message: InboundMessageHandler | None = None
        self._task: asyncio.Task[None] | None = None
        self.state: AdapterState = "stopped"
        self.last_error: str = ""
        # typing ticket 仅内存 TTL 缓存，不落盘。
        self._typing_tickets: dict[str, tuple[str, float]] = {}
        self._ticket_ttl = 300.0
        # 批次成功后回调持久化 cursor（通常写 SQLite）。
        self._on_cursor_persist = on_cursor_persist
        self._session_recovery_attempts = 0
        self._server_session_ready = False

    async def start(self, on_message: InboundMessageHandler) -> None:
        self._on_message = on_message
        self.last_error = ""
        # iLink 上线通知及失败分类统一由 poll 状态机处理；成功 poll 前不宣称 connected。
        self._server_session_ready = False
        self.state = "reconnecting"
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self.state = "stopped"
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._notify_stop()
        await self._protocol.aclose()

    async def send_text(
        self,
        address: ChannelAddress,
        text: str,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> SendResult:
        payload = _payload(reply_handle)
        context_token = str(payload.get("context_token") or "")
        to_user_id = str(payload.get("to_user_id") or address.conversation_id)
        if not context_token:
            # 未拿到 context_token 时显式失败，不尝试只凭 user ID 发送。
            return SendResult(ok=False, error="context_token_required")
        chunks = split_weixin_text(text, limit=3000)
        if not chunks:
            return SendResult(ok=False, error="empty_text")
        for chunk in chunks:
            await self._protocol.send_text(
                to_user_id=to_user_id,
                text=chunk,
                context_token=context_token,
            )
        return SendResult(ok=True)

    async def set_typing(
        self,
        address: ChannelAddress,
        active: bool,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> None:
        del address
        payload = _payload(reply_handle)
        context_token = str(payload.get("context_token") or "")
        if not context_token:
            return
        try:
            ticket = await self._get_ticket(context_token)
            await self._protocol.send_typing(
                typing_ticket=ticket,
                active=active,
                context_token=context_token,
            )
        except Exception as error:
            # typing 失败只记诊断，不伪造 turn 失败。
            self.last_error = f"typing:{type(error).__name__}"

    async def _get_ticket(self, context_token: str) -> str:
        now = time.time()
        cached = self._typing_tickets.get(context_token)
        if cached is not None and cached[1] > now:
            return cached[0]
        ticket = await self._protocol.get_typing_ticket(context_token=context_token)
        self._typing_tickets[context_token] = (ticket, now + self._ticket_ttl)
        return ticket

    async def _poll_loop(self) -> None:
        while self.state not in {"stopped", "auth_expired", "failed"}:
            if not self._server_session_ready:
                self.state = "reconnecting"
                try:
                    await self._notify_start()
                except WeixinAuthenticationExpired as error:
                    self._session_recovery_attempts += 1
                    if self._session_recovery_attempts > _MAX_SESSION_RECOVERY_ATTEMPTS:
                        self.state = "auth_expired"
                        self.last_error = str(error) or "auth_expired"
                        return
                    self.last_error = str(error) or "auth_expired"
                    await asyncio.sleep(max(self._poll_interval, 1.0))
                    continue
                except WeixinRateLimited as error:
                    self.last_error = str(error) or "rate_limited"
                    await asyncio.sleep(max(self._poll_interval, 1.0))
                    continue
                except WeixinProtocolError as error:
                    # 已完成协议级分类的未知错误不可通过盲重试掩盖。
                    self.state = "failed"
                    self.last_error = str(error) or "protocol_error"
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.last_error = f"notify_start:{type(error).__name__}"
                    await asyncio.sleep(max(self._poll_interval, 1.0))
                    continue
            try:
                updates = await self._protocol.get_updates(cursor=self._cursor)
            except WeixinAuthenticationExpired as error:
                self._session_recovery_attempts += 1
                if self._session_recovery_attempts > _MAX_SESSION_RECOVERY_ATTEMPTS:
                    self.state = "auth_expired"
                    self.last_error = str(error) or "auth_expired"
                    return
                # -14 既可能是 token 失效，也可能只是服务端长轮询 session 过期。
                # 先按官方实现重新 notifyStart；连续恢复失败后才要求用户扫码。
                self.state = "reconnecting"
                self.last_error = str(error) or "auth_expired"
                self._server_session_ready = False
                await asyncio.sleep(self._poll_interval)
                continue
            except WeixinRateLimited as error:
                self.state = "reconnecting"
                self.last_error = str(error) or "rate_limited"
                await asyncio.sleep(max(self._poll_interval, 1.0))
                continue
            except WeixinProtocolError as error:
                # 已完成协议级分类的未知错误不可通过盲重试掩盖。
                self.state = "failed"
                self.last_error = str(error) or "protocol_error"
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.state = "reconnecting"
                self.last_error = f"{type(error).__name__}"
                await asyncio.sleep(self._poll_interval)
                continue

            new_cursor = updates.cursor
            self._session_recovery_attempts = 0
            try:
                batch_ok = await self._dispatch_batch(list(updates.messages))
                # 空批次（心跳）或全部可接受 outcome 才推进 cursor。
                if batch_ok and new_cursor != self._cursor:
                    self._cursor = new_cursor
                    # 仅在 cursor 实际变化时落盘，避免心跳重复写。
                    if self._on_cursor_persist is not None:
                        self._on_cursor_persist(new_cursor)
                elif batch_ok:
                    self._cursor = new_cursor
                self.state = "connected"
                self.last_error = ""
            except Exception as error:
                # 部分失败：不推进 cursor。
                self.last_error = f"batch:{type(error).__name__}"
            await asyncio.sleep(self._poll_interval)

    async def _notify_start(self) -> None:
        await self._protocol.notify_start()
        self._server_session_ready = True

    async def _notify_stop(self) -> None:
        try:
            await self._protocol.notify_stop()
            self._server_session_ready = False
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 关闭通知失败不阻断本地资源释放，但必须留下脱敏诊断。
            self.last_error = f"notify_stop:{type(error).__name__}"

    async def _dispatch_batch(self, messages: list[WeixinInboundMessage]) -> bool:
        """
        投递一批入站消息。

        返回 True 表示可推进 cursor（无消息、或每条都返回可接受 outcome）。
        handler 抛错或返回 busy/rejected 时返回 False，不推进 cursor。
        """
        if self._on_message is None:
            return not messages
        deliverable = [inbound for msg in messages if (inbound := self._to_inbound(msg)) is not None]
        if messages and not deliverable:
            # 全是系统/空消息：可推进 cursor 跳过。
            return True
        if not deliverable:
            return True
        for inbound in deliverable:
            outcome = await self._invoke_handler(inbound)
            if outcome not in _CURSOR_OK_OUTCOMES:
                # busy/rejected：整批不推进，等待平台重投。
                return False
        return True

    async def _invoke_handler(self, inbound: InboundChannelMessage) -> str | None:
        assert self._on_message is not None
        result = await self._on_message(inbound)
        if result is None:
            return None
        return str(result)

    def _to_inbound(self, msg: WeixinInboundMessage) -> InboundChannelMessage | None:
        # 只接收用户私聊文本；缺 from_user_id 的系统消息忽略。
        if not msg.from_user_id or not msg.message_id:
            return None
        raw = msg.raw if isinstance(msg.raw, dict) else {}
        # 群聊 / 房间消息第一阶段不接。
        if raw.get("room_id") or raw.get("group_id") or raw.get("chatroom_id"):
            return None
        message_type = raw.get("message_type")
        if message_type is not None:
            try:
                mt = int(message_type)
            except (TypeError, ValueError):
                mt = -1
            # iLink 协议中 1=USER、2=BOT；只接收用户消息，避免机器人回复回流。
            if mt != 1:
                return None
        # 非文本 item（图片等）或空正文：不进模型。
        if not (msg.text or "").strip():
            return None
        if self._raw_has_non_text_items(raw):
            return None
        address = ChannelAddress(
            instance_id=self.instance_id,
            platform="weixin",
            conversation_kind="dm",
            conversation_id=msg.from_user_id,
        )
        # reply handle 仅内存；payload 含 context_token，不得落盘。
        handle = ChannelReplyHandle(
            platform="weixin",
            payload={
                "context_token": msg.context_token,
                "to_user_id": msg.from_user_id,
            },
        )
        return InboundChannelMessage(
            address=address,
            message_id=msg.message_id,
            sender_id=msg.from_user_id,
            text=(msg.text or "").strip(),
            reply_handle=handle,
        )

    @staticmethod
    def _raw_has_non_text_items(raw: dict[str, Any]) -> bool:
        items = raw.get("item_list") or []
        if not isinstance(items, list) or not items:
            return False
        for part in items:
            if not isinstance(part, dict):
                continue
            try:
                ptype = int(part.get("type") or 0)
            except (TypeError, ValueError):
                continue
            # type=1 文本；其他视为媒体/非文本。
            if ptype not in {0, 1}:
                return True
        return False


def _payload(reply_handle: ChannelReplyHandle | None) -> dict[str, Any]:
    if reply_handle is None:
        return {}
    raw = reply_handle.payload
    return raw if isinstance(raw, dict) else {}
