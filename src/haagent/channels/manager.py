"""
haagent/channels/manager.py - 渠道鉴权、去重、命令与 Actor 路由

入站消息经 pairing/owner 校验与 receipt 去重后，路由到 ChannelSessionActor；
控制命令不进入模型 prompt。
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from haagent.channels.adapter import ChannelAdapter
from haagent.channels.interactions import InteractionBroker, InteractionError
from haagent.channels.session_actor import ChannelSessionActor
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, load_channel_settings, save_channel_settings
from haagent.channels.state import ChannelStateStore, PairingError
from haagent.channels.types import ChannelAddress, InboundChannelMessage

ServiceFactory = Callable[[Path], Any]
AdapterState = Literal["connected", "reconnecting", "auth_expired", "failed", "stopped"]
# Actor 明确接受/入队后写 receipt；busy/rejected 不写，允许平台重投。
_ACCEPT_OUTCOMES = frozenset({"accepted", "queued"})


@dataclass
class AdapterRuntime:
    adapter: ChannelAdapter
    state: AdapterState = "stopped"
    last_send_error: str = ""
    last_error: str = ""


@dataclass(frozen=True)
class PendingPermissionConfirmation:
    nonce: str
    instance_id: str
    owner_sender_id: str
    workspace_root: str
    expires_at: datetime


class ChannelManager:
    def __init__(
        self,
        *,
        state: ChannelStateStore,
        default_workspace_root: Path,
        service_factory: ServiceFactory,
        broker: InteractionBroker | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._state = state
        self._default_workspace_root = Path(default_workspace_root)
        self._service_factory = service_factory
        self._broker = broker or InteractionBroker()
        self._config_path = Path(config_path) if config_path is not None else None
        self._adapters: dict[str, AdapterRuntime] = {}
        self._actors: dict[str, ChannelSessionActor] = {}
        # instance_id -> 配置中的固定 workspace
        self._instance_workspaces: dict[str, Path] = {}
        self._instance_permission_modes: dict[str, str] = {}
        self._permission_confirmations: dict[str, PendingPermissionConfirmation] = {}

    def register_instance_workspace(
        self,
        instance_id: str,
        workspace_root: Path,
        permission_mode: str = "request_approval",
    ) -> None:
        """注册渠道实例的 workspace，优先于 gateway 默认 cwd。"""
        self._instance_workspaces[instance_id] = Path(workspace_root)
        self._instance_permission_modes[instance_id] = _channel_permission_mode(permission_mode)

    async def attach_adapter(self, adapter: ChannelAdapter) -> None:
        instance_id = adapter.instance_id
        self._adapters[instance_id] = AdapterRuntime(adapter=adapter, state="connected")
        await adapter.start(self._on_message)

    async def mark_adapter_state(self, instance_id: str, state: AdapterState) -> None:
        runtime = self._adapters.get(instance_id)
        if runtime is None:
            return
        runtime.state = state

    async def sync_adapter_states(self) -> None:
        """从 adapter 对象同步 state（如 auth_expired），供 status/CLI 可见。"""
        for instance_id, runtime in self._adapters.items():
            adapter_state = getattr(runtime.adapter, "state", None)
            if adapter_state in {"connected", "reconnecting", "auth_expired", "failed", "stopped"}:
                runtime.state = adapter_state  # type: ignore[assignment]
            last_error = getattr(runtime.adapter, "last_error", "") or ""
            # Adapter 恢复后必须清掉旧错误，否则 CLI 会长期展示已经失效的诊断。
            runtime.last_error = str(last_error)
            # 同步 actor 侧发送失败
            for actor in self._actors.values():
                if actor.address.instance_id == instance_id and actor.last_send_error:
                    runtime.last_send_error = actor.last_send_error

    def last_send_error(self, instance_id: str) -> str:
        runtime = self._adapters.get(instance_id)
        if runtime is None:
            return ""
        return runtime.last_send_error

    def status(self) -> list[dict[str, str]]:
        # 读取前尽量同步 adapter 真实状态。
        for instance_id, runtime in self._adapters.items():
            adapter_state = getattr(runtime.adapter, "state", None)
            if adapter_state in {"connected", "reconnecting", "auth_expired", "failed", "stopped"}:
                runtime.state = adapter_state  # type: ignore[assignment]
            last_error = getattr(runtime.adapter, "last_error", "") or ""
            runtime.last_error = str(last_error)
            for actor in self._actors.values():
                if actor.address.instance_id == instance_id and actor.last_send_error:
                    runtime.last_send_error = actor.last_send_error
        rows: list[dict[str, str]] = []
        for instance_id, runtime in self._adapters.items():
            summary = self._state.instance_status_summary(instance_id)
            rows.append(
                {
                    "instance_id": instance_id,
                    "platform": runtime.adapter.platform,
                    "state": runtime.state,
                    "last_send_error": runtime.last_send_error,
                    "last_error": runtime.last_error,
                    "owner": summary.get("owner", "(unpaired)"),
                    "pairing": summary.get("pairing", "none"),
                    "cursor": summary.get("cursor", "empty"),
                }
            )
        return rows

    async def stop(self) -> list[str]:
        """停止全部 actor/adapter；adapter 失败写入返回列表，不静默吞掉。"""
        errors: list[str] = []
        for actor in list(self._actors.values()):
            try:
                await actor.close()
            except Exception as error:
                errors.append(f"actor:{type(error).__name__}:{error}")
        self._actors.clear()
        for runtime in list(self._adapters.values()):
            try:
                await runtime.adapter.stop()
            except Exception as error:
                # 显式记录，避免「看起来已停、实际关失败」不可见。
                msg = f"{runtime.adapter.instance_id}:stop:{error}"
                errors.append(msg)
                runtime.last_error = str(error)
            runtime.state = "stopped"
        return errors

    async def _on_message(self, message: InboundChannelMessage) -> str:
        """
        处理入站消息。

        返回 outcome 字符串供 Adapter 决定是否推进 cursor：
        - accepted/queued：已接受，应写 receipt（及可选 cursor）
        - control/auth_reject/pair：控制面已处理，写 receipt 但不推进 cursor（无 batch cursor）
        - busy/rejected/duplicate：不写 receipt
        """
        instance_id = message.address.instance_id
        # 1) 已处理则跳过（不重跑模型）。
        if self._state.has_receipt(instance_id, message.message_id):
            return "duplicate"
        text = (message.text or "").strip()
        # 2) 控制命令优先（部分在配对前也允许 /pair）。
        if text.startswith("/pair"):
            await self._handle_pair(message, text)
            self._commit_receipt_only(instance_id, message.message_id)
            return "pair"
        owner = self._state.get_owner(instance_id)
        if owner is None:
            await self._reply(message, "未配对：请先发送 /pair <配对码>")
            self._commit_receipt_only(instance_id, message.message_id)
            return "auth_reject"
        if message.sender_id != owner:
            await self._reply(message, "未授权：仅绑定 owner 可使用此助手")
            self._commit_receipt_only(instance_id, message.message_id)
            return "auth_reject"
        if text.startswith("/approve") or text.startswith("/deny"):
            await self._handle_approval(message, text)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        if text.startswith("/answer"):
            await self._handle_answer(message, text)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        if text == "/stop":
            await self._handle_stop(message)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        if text == "/status":
            await self._handle_status(message)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        if text == "/new":
            await self._handle_new(message)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        if re.match(r"^/permissions(?:\s|$)", text, flags=re.IGNORECASE):
            await self._handle_permissions(message, text)
            self._commit_receipt_only(instance_id, message.message_id)
            return "control"
        # 3) 普通文本进入 Actor；仅 accepted/queued 后写 receipt。
        actor = await self._get_or_create_actor(message)
        result = await actor.submit(
            message,
            permission_mode=self._effective_permission_mode(message),
        )
        if result.status in _ACCEPT_OUTCOMES:
            self._commit_receipt_only(instance_id, message.message_id)
            return result.status
        # busy/rejected：不写 receipt，允许重投。
        return result.status

    async def _handle_permissions(self, message: InboundChannelMessage, text: str) -> None:
        normalized = " ".join(text.split())
        if normalized.lower() == "/permissions":
            mode = self._effective_permission_mode(message)
            temporary = self._state.get_temporary_permission(
                message.address.instance_id,
                owner_sender_id=message.sender_id,
                workspace_root=str(self._workspace_for(message.address)),
            )
            if temporary is None:
                source = "永久" if mode == "auto_approve" else "默认"
                await self._reply(message, f"权限模式：{mode}（{source}）")
                return
            remaining = max(0, int((temporary.expires_at - datetime.now(timezone.utc)).total_seconds() // 60))
            await self._reply(message, f"权限模式：{mode}（临时，剩余约 {remaining} 分钟）")
            return
        if normalized.lower() == "/permissions safe":
            self._state.clear_temporary_permission(message.address.instance_id)
            if not self._set_permanent_permission(message.address.instance_id, "request_approval"):
                await self._reply(message, "恢复安全模式失败：渠道配置不可用")
                return
            self._clear_permission_confirmations(message.address.instance_id)
            await self._reply(message, "已恢复 request_approval")
            return
        if normalized.lower() == "/permissions auto permanent":
            workspace = str(self._workspace_for(message.address))
            nonce = secrets.token_urlsafe(12)
            self._clear_permission_confirmations(message.address.instance_id)
            self._permission_confirmations[nonce] = PendingPermissionConfirmation(
                nonce=nonce,
                instance_id=message.address.instance_id,
                owner_sender_id=message.sender_id,
                workspace_root=workspace,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            await self._reply(
                message,
                f"确认永久 auto_approve [{nonce}]\n回复：/permissions confirm {nonce}\n确认有效期 5 分钟",
            )
            return
        confirm = re.fullmatch(r"/permissions\s+confirm\s+(\S+)", normalized, flags=re.IGNORECASE)
        if confirm is not None:
            nonce = confirm.group(1)
            pending = self._permission_confirmations.get(nonce)
            if pending is None:
                await self._reply(message, "永久权限确认失败：nonce 不存在或已使用")
                return
            if datetime.now(timezone.utc) >= pending.expires_at:
                self._permission_confirmations.pop(nonce, None)
                await self._reply(message, "永久权限确认失败：nonce 已过期")
                return
            if (
                pending.instance_id != message.address.instance_id
                or pending.owner_sender_id != message.sender_id
                or pending.workspace_root != str(self._workspace_for(message.address))
            ):
                await self._reply(message, "永久权限确认失败：确认上下文不匹配")
                return
            if not self._set_permanent_permission(
                message.address.instance_id,
                "auto_approve",
                owner_sender_id=message.sender_id,
                workspace_root=pending.workspace_root,
            ):
                await self._reply(message, "永久权限确认失败：渠道配置不可用")
                return
            self._clear_permission_confirmations(message.address.instance_id)
            await self._reply(message, "已永久开启 auto_approve")
            return
        match = re.fullmatch(r"/permissions\s+auto\s+(\d+)([mh])", normalized, flags=re.IGNORECASE)
        if match is None:
            await self._reply(
                message,
                "用法：/permissions、/permissions safe、/permissions auto <1m-1440m|1h-24h>、/permissions auto permanent",
            )
            return
        value = int(match.group(1))
        unit = match.group(2).lower()
        minutes = value if unit == "m" else value * 60
        if value < 1 or (unit == "m" and value > 1440) or (unit == "h" and value > 24):
            await self._reply(message, "时长必须是 1m-1440m 或 1h-24h")
            return
        workspace = self._workspace_for(message.address)
        self._state.set_temporary_permission(
            instance_id=message.address.instance_id,
            owner_sender_id=message.sender_id,
            workspace_root=str(workspace),
            permission_mode="auto_approve",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=minutes),
        )
        await self._reply(message, f"已开启临时 auto_approve，持续 {minutes} 分钟")

    def _effective_permission_mode(self, message: InboundChannelMessage) -> str:
        """返回渠道允许的有效模式；未知状态一律失败关闭。"""
        temporary = self._state.get_temporary_permission(
            message.address.instance_id,
            owner_sender_id=message.sender_id,
            workspace_root=str(self._workspace_for(message.address)),
        )
        if temporary is not None and temporary.permission_mode == "auto_approve":
            return "auto_approve"
        configured = _channel_permission_mode(
            self._instance_permission_modes.get(message.address.instance_id, "request_approval")
        )
        if configured != "auto_approve":
            return "request_approval"
        permanent = self._state.get_permanent_permission_binding(
            message.address.instance_id,
            owner_sender_id=message.sender_id,
            workspace_root=str(self._workspace_for(message.address)),
        )
        return "auto_approve" if permanent is not None else "request_approval"

    def _set_permanent_permission(
        self,
        instance_id: str,
        mode: str,
        *,
        owner_sender_id: str | None = None,
        workspace_root: str | None = None,
    ) -> bool:
        """持久化实例模式；不接受 full_access 或未知权限。"""
        normalized = _channel_permission_mode(mode)
        if normalized != mode:
            return False
        if normalized == "auto_approve" and (not owner_sender_id or not workspace_root):
            return False
        if self._config_path is None:
            if normalized == "request_approval":
                self._state.clear_permanent_permission_binding(instance_id)
                self._instance_permission_modes[instance_id] = normalized
                return True
            return False
        try:
            if normalized == "auto_approve":
                self._state.set_permanent_permission_binding(
                    instance_id=instance_id,
                    owner_sender_id=owner_sender_id,
                    workspace_root=workspace_root,
                    permission_mode=normalized,
                )
            else:
                self._state.clear_permanent_permission_binding(instance_id)
            settings = load_channel_settings(self._config_path)
            found = False
            instances: list[ChannelInstanceConfig] = []
            for item in settings.instances:
                if item.id != instance_id:
                    instances.append(item)
                    continue
                found = True
                instances.append(
                    ChannelInstanceConfig(
                        id=item.id,
                        platform=item.platform,
                        enabled=item.enabled,
                        workspace_root=item.workspace_root,
                        credential_username=item.credential_username,
                        metadata=dict(item.metadata),
                        permission_mode=normalized,
                    )
                )
            if not found:
                # 绑定已先写入 SQLite；配置目标缺失时必须回滚，不能留下孤儿授权。
                if normalized == "auto_approve":
                    self._state.clear_permanent_permission_binding(instance_id)
                return False
            save_channel_settings(self._config_path, ChannelSettings(version=settings.version, instances=instances))
        except Exception:
            if normalized == "auto_approve":
                self._state.clear_permanent_permission_binding(instance_id)
            return False
        self._instance_permission_modes[instance_id] = normalized
        return True

    def _clear_permission_confirmations(self, instance_id: str) -> None:
        for nonce, pending in list(self._permission_confirmations.items()):
            if pending.instance_id == instance_id:
                del self._permission_confirmations[nonce]

    def _commit_receipt_only(self, instance_id: str, message_id: str) -> None:
        self._state.commit_accepted_batch(
            instance_id=instance_id,
            message_ids=[message_id],
        )

    def commit_transport_cursor(
        self,
        *,
        instance_id: str,
        cursor_name: str,
        cursor_value: str,
        message_ids: list[str] | None = None,
    ) -> None:
        """Adapter 在批次全部 accepted 后调用：同事务写 receipt + cursor。"""
        self._state.commit_accepted_batch(
            instance_id=instance_id,
            message_ids=list(message_ids or []),
            cursor_name=cursor_name,
            cursor_value=cursor_value,
        )

    async def _handle_pair(self, message: InboundChannelMessage, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await self._reply(message, "用法：/pair <配对码>")
            return
        code = parts[1].strip()
        existing = self._state.get_owner(message.address.instance_id)
        if existing is not None:
            await self._reply(message, "已配对，无需重复 /pair")
            return
        try:
            owner = self._state.consume_pairing_token(
                message.address.instance_id,
                code,
                sender_id=message.sender_id,
            )
        except PairingError as error:
            await self._reply(message, f"配对失败：{error}")
            return
        # 新 owner 不得继承旧 owner 的自动权限。
        self._set_permanent_permission(message.address.instance_id, "request_approval")
        await self._reply(message, f"配对成功，owner 已绑定为当前用户")
        del owner

    async def _handle_approval(self, message: InboundChannelMessage, text: str) -> None:
        match = re.match(r"^/(approve|deny)\s+(\S+)\s*$", text, flags=re.IGNORECASE)
        if match is None:
            await self._reply(message, "用法：/approve <nonce> 或 /deny <nonce>")
            return
        action = match.group(1).lower()
        nonce = match.group(2)
        approved = action == "approve"
        try:
            self._broker.resolve(
                nonce,
                approved=approved,
                sender_id=message.sender_id,
                binding_key=message.address.binding_key(),
            )
        except InteractionError as error:
            await self._reply(message, f"审批失败：{error}")
            return
        await self._reply(message, "已批准" if approved else "已拒绝")

    async def _handle_answer(self, message: InboundChannelMessage, text: str) -> None:
        match = re.match(r"^/answer\s+(\S+)\s+(.+)$", text, flags=re.DOTALL)
        if match is None:
            await self._reply(message, "用法：/answer <nonce> <内容>")
            return
        nonce = match.group(1)
        answer = match.group(2)
        try:
            self._broker.resolve_answer(
                nonce,
                answer=answer,
                sender_id=message.sender_id,
                binding_key=message.address.binding_key(),
            )
        except InteractionError as error:
            await self._reply(message, f"补充输入失败：{error}")
            return
        await self._reply(message, "已提交补充输入")

    async def _handle_stop(self, message: InboundChannelMessage) -> None:
        actor = self._actors.get(message.address.binding_key())
        if actor is None:
            await self._reply(message, "当前无运行中的任务")
            return
        cancelled = await actor.cancel_current_turn()
        await self._reply(message, "已请求停止" if cancelled else "当前无运行中的任务")

    async def _handle_status(self, message: InboundChannelMessage) -> None:
        binding = self._state.get_binding(message.address.binding_key())
        owner = self._state.get_owner(message.address.instance_id)
        session_id = binding.session_id if binding else "(none)"
        await self._reply(
            message,
            f"status: instance={message.address.instance_id} owner={owner or '(unpaired)'} session={session_id}",
        )

    async def _handle_new(self, message: InboundChannelMessage) -> None:
        key = message.address.binding_key()
        actor = self._actors.pop(key, None)
        if actor is not None:
            await actor.close()
        # 删除 binding 以便下次创建新 session；不删 HaAgent session package。
        binding = self._state.get_binding(key)
        if binding is not None:
            self._state.upsert_binding(
                binding_key=key,
                instance_id=binding.instance_id,
                platform=binding.platform,
                conversation_kind=binding.conversation_kind,
                conversation_id=binding.conversation_id,
                thread_id=binding.thread_id,
                workspace_root=binding.workspace_root,
                session_id="",  # 空 session 触发下次 create
                owner_sender_id=binding.owner_sender_id,
            )
        # 用空 session_id 不够干净：直接重建 actor 时 _ensure_session 会 create。
        # 若 binding.session_id 为空字符串，resume 会失败；改为删除后靠 create。
        # 简化：关闭旧 actor，写入占位后下次 create 覆盖。
        actor = await self._get_or_create_actor(message, force_new=True)
        del actor
        await self._reply(message, "已开启新会话")

    async def _get_or_create_actor(
        self,
        message: InboundChannelMessage,
        *,
        force_new: bool = False,
    ) -> ChannelSessionActor:
        key = message.address.binding_key()
        if not force_new and key in self._actors:
            return self._actors[key]
        if force_new and key in self._actors:
            await self._actors[key].close()
            del self._actors[key]
        workspace = self._workspace_for(message.address)
        service = self._service_factory(workspace)
        # force_new：删除 binding 后 actor 走 create，避免 resume 旧 session。
        if force_new:
            self._state.delete_binding(key)
        actor = ChannelSessionActor(
            binding_key=key,
            address=message.address,
            owner_sender_id=message.sender_id,
            workspace_root=workspace,
            state=self._state,
            adapter=self._adapters[message.address.instance_id].adapter,
            service=service,
            broker=self._broker,
        )
        self._actors[key] = actor
        return actor

    def _workspace_for(self, address: ChannelAddress) -> Path:
        # 优先 binding 中已持久化的 workspace，其次 instance 配置，最后 gateway 默认。
        binding = self._state.get_binding(address.binding_key())
        if binding is not None and binding.workspace_root:
            root = Path(binding.workspace_root)
            if root.exists() and root.is_dir():
                return root.resolve()
        configured = self._instance_workspaces.get(address.instance_id)
        if configured is not None:
            root = Path(configured)
            root.mkdir(parents=True, exist_ok=True)
            return root.resolve()
        root = self._default_workspace_root
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    async def _reply(self, message: InboundChannelMessage, text: str) -> None:
        runtime = self._adapters.get(message.address.instance_id)
        if runtime is None:
            return
        result = await runtime.adapter.send_text(
            message.address,
            text,
            reply_handle=message.reply_handle,
            reply_to_message_id=message.message_id,
        )
        # 发送失败显式记录，禁止静默丢弃。
        if result is not None and not getattr(result, "ok", True):
            runtime.last_send_error = str(getattr(result, "error", None) or "send_failed")


def _channel_permission_mode(value: str) -> Literal["request_approval", "auto_approve"]:
    """渠道配置失败关闭：不能把 full_access 传播到远程 session。"""
    if value == "auto_approve":
        return "auto_approve"
    return "request_approval"
