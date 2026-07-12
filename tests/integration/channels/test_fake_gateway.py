"""
tests/integration/channels/test_fake_gateway.py - ChannelManager + FakeAdapter 端到端
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from haagent.channels.adapters.fake import FakeAdapter
from haagent.channels.interactions import InteractionBroker
from haagent.channels.manager import ChannelManager
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, load_channel_settings, save_channel_settings
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage
from haagent.runtime.events.types import AssistantMessageEvent, SessionLifecycleEvent
from haagent.runtime.execution.human_interaction import HumanInteractionRequest


@dataclass
class FakeSessionStatus:
    session_id: str
    workspace_root: Path
    runs_root: Path
    session_path: Path
    turn_count: int = 0
    max_turns: int | None = 200
    provider: str = "openai-chat"


@dataclass
class FakeCancelResult:
    status: str
    reason: str


class FakeSessions:
    def __init__(self, service: "FakeAssistantService") -> None:
        self._service = service
        self.permission_modes: list[str] = []

    def create(self) -> FakeSessionStatus:
        return self._service._create()

    def resume(self, session: str | Path) -> FakeSessionStatus:
        return self._service._resume(str(session))

    def set_permission_mode(self, mode: str) -> None:
        self.permission_modes.append(mode)

    def run_prompt_events(
        self,
        prompt: str,
        *,
        event_sink: Callable[[object], None] | None = None,
        include_session_events: bool = True,
        interaction_handler=None,
        attachments=None,
    ):
        return self._service._run_prompt(
            prompt,
            event_sink=event_sink,
            include_session_events=include_session_events,
            interaction_handler=interaction_handler,
        )

    def cancel_current_run(self) -> FakeCancelResult:
        return self._service._cancel()


class FakeAssistantService:
    _id_seq = 0

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.runs_root = workspace_root / ".runs"
        self.sessions = FakeSessions(self)
        self.session_id: str | None = None
        self.prompts: list[str] = []
        self.created_count = 0
        self.model_calls = 0
        self._running = False
        self.block_on_interaction = False
        self.reply_text = "pong"

    def _create(self) -> FakeSessionStatus:
        FakeAssistantService._id_seq += 1
        self.session_id = f"sess-{FakeAssistantService._id_seq}"
        self.created_count += 1
        return FakeSessionStatus(
            session_id=self.session_id,
            workspace_root=self.workspace_root,
            runs_root=self.runs_root,
            session_path=self.runs_root / self.session_id,
        )

    def _resume(self, session: str) -> FakeSessionStatus:
        name = Path(session).name
        self.session_id = name
        return FakeSessionStatus(
            session_id=name,
            workspace_root=self.workspace_root,
            runs_root=self.runs_root,
            session_path=self.runs_root / name,
        )

    def _run_prompt(self, prompt: str, *, event_sink, include_session_events, interaction_handler):
        del include_session_events
        self.prompts.append(prompt)
        self.model_calls += 1
        self._running = True
        try:
            if event_sink is not None:
                event_sink(SessionLifecycleEvent(self.session_id or "s", 1, "turn_started", "start"))
            if self.block_on_interaction and interaction_handler is not None:
                interaction_handler(
                    HumanInteractionRequest(
                        interaction_type="approval",
                        tool_name="shell",
                        question="run tests?",
                        args_summary={"command": "pytest"},
                    )
                )
            if event_sink is not None:
                event_sink(AssistantMessageEvent(self.session_id or "s", 1, 1, self.reply_text))
                event_sink(SessionLifecycleEvent(self.session_id or "s", 1, "turn_finished", "end", status="ok"))
            return {"status": "ok"}
        finally:
            self._running = False

    def _cancel(self) -> FakeCancelResult:
        if not self._running:
            return FakeCancelResult(status="idle", reason="no_active_run")
        return FakeCancelResult(status="cancelled", reason="user_cancelled")


def _msg(
    text: str,
    *,
    message_id: str,
    sender_id: str = "owner-1",
    instance_id: str = "fake-1",
    conversation_id: str | None = None,
) -> InboundChannelMessage:
    conv = conversation_id or sender_id
    address = ChannelAddress(
        instance_id=instance_id,
        platform="fake",
        conversation_kind="dm",
        conversation_id=conv,
    )
    return InboundChannelMessage(
        address=address,
        message_id=message_id,
        sender_id=sender_id,
        text=text,
        received_at=datetime.now(timezone.utc),
        reply_handle=ChannelReplyHandle(platform="fake", payload={"token": "secret-token"}),
    )


def _make_manager(
    tmp_path: Path,
    *,
    service_factory: Callable[[Path], Any] | None = None,
    config_path: Path | None = None,
) -> tuple[ChannelManager, FakeAdapter, ChannelStateStore, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    adapter = FakeAdapter("fake-1", platform="fake")
    services: list[FakeAssistantService] = []

    def factory(root: Path) -> FakeAssistantService:
        svc = FakeAssistantService(root)
        services.append(svc)
        return svc

    kwargs: dict[str, Any] = {}
    if config_path is not None:
        kwargs["config_path"] = config_path
    manager = ChannelManager(
        state=state,
        default_workspace_root=workspace,
        service_factory=service_factory or factory,
        broker=InteractionBroker(timeout_seconds=5.0),
        **kwargs,
    )
    manager._test_services = services  # type: ignore[attr-defined]
    return manager, adapter, state, workspace


def test_unauthorized_sender_does_not_trigger_model(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("hello", message_id="m1", sender_id="stranger"))
        await asyncio.sleep(0.05)
        services = manager._test_services  # type: ignore[attr-defined]
        assert all(s.model_calls == 0 for s in services)
        assert any("未授权" in t or "拒绝" in t or "owner" in t.lower() for t in adapter.sent_texts)
        await manager.stop()

    asyncio.run(_run())


def test_pairing_only_correct_pair_establishes_owner(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.create_pairing_token("fake-1", "ABCD1234")
        await manager.attach_adapter(adapter)
        # 配对前普通消息拒绝
        await adapter.emit(_msg("hi", message_id="m0", sender_id="user-a"))
        await asyncio.sleep(0.05)
        assert state.get_owner("fake-1") is None
        services = manager._test_services  # type: ignore[attr-defined]
        assert all(s.model_calls == 0 for s in services)
        # 错误配对码
        await adapter.emit(_msg("/pair WRONGCODE", message_id="m1", sender_id="user-a"))
        await asyncio.sleep(0.05)
        assert state.get_owner("fake-1") is None
        # 正确配对
        await adapter.emit(_msg("/pair ABCD1234", message_id="m2", sender_id="user-a"))
        await asyncio.sleep(0.05)
        assert state.get_owner("fake-1") == "user-a"
        # 二维码元数据 user 不被直接信任：另一人仍拒绝
        await adapter.emit(_msg("after", message_id="m3", sender_id="qr-user-meta"))
        await asyncio.sleep(0.05)
        assert all(s.model_calls == 0 for s in services)
        await manager.stop()

    asyncio.run(_run())


def test_duplicate_message_id_runs_once(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        msg = _msg("once", message_id="dup-1")
        await adapter.emit(msg)
        deadline = time.time() + 2
        services = manager._test_services  # type: ignore[attr-defined]
        while time.time() < deadline and not any(s.model_calls >= 1 for s in services):
            await asyncio.sleep(0.01)
        await adapter.emit(msg)
        await asyncio.sleep(0.1)
        total = sum(s.model_calls for s in services)
        assert total == 1
        await manager.stop()

    asyncio.run(_run())


def test_control_commands_do_not_enter_model_prompt(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, workspace = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        # 先建立 binding/session
        await adapter.emit(_msg("bootstrap", message_id="b1"))
        deadline = time.time() + 2
        services = manager._test_services  # type: ignore[attr-defined]
        while time.time() < deadline and not any(s.model_calls >= 1 for s in services):
            await asyncio.sleep(0.01)
        baseline = sum(s.model_calls for s in services)
        for text, mid in [("/status", "c1"), ("/new", "c2"), ("/stop", "c3")]:
            await adapter.emit(_msg(text, message_id=mid))
            await asyncio.sleep(0.05)
        assert sum(s.model_calls for s in services) == baseline
        assert any("status" in t.lower() or "会话" in t or "session" in t.lower() for t in adapter.sent_texts)
        await manager.stop()

    asyncio.run(_run())


def test_owner_can_enable_temporary_auto_approve_from_chat(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)

        await adapter.emit(_msg("/permissions auto 30m", message_id="perm-1"))
        await asyncio.sleep(0.05)

        assert manager._effective_permission_mode(_msg("next", message_id="perm-2")) == "auto_approve"
        assert any("临时" in text and "auto_approve" in text for text in adapter.sent_texts)
        await adapter.emit(_msg("next", message_id="perm-3"))
        deadline = time.time() + 2
        services = manager._test_services  # type: ignore[attr-defined]
        while time.time() < deadline and not any(service.prompts for service in services):
            await asyncio.sleep(0.01)
        assert any(set(service.sessions.permission_modes) == {"auto_approve"} for service in services)
        assert all(service.model_calls <= 1 for service in services)
        await manager.stop()

    asyncio.run(_run())


def test_permanent_auto_approve_requires_owner_confirmation_and_safe_clears_it(tmp_path: Path) -> None:
    async def _run() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        config_path = tmp_path / "channels.json"
        save_channel_settings(
            config_path,
            ChannelSettings(
                instances=[
                    ChannelInstanceConfig(
                        id="fake-1",
                        platform="fake",
                        enabled=True,
                        workspace_root=workspace,
                        credential_username="channel:fake:fake-1:token",
                    )
                ]
            ),
        )
        manager, adapter, state, _ = _make_manager(tmp_path, config_path=config_path)
        state.set_owner("fake-1", "owner-1")
        manager.register_instance_workspace("fake-1", workspace)
        await manager.attach_adapter(adapter)

        await adapter.emit(_msg("/permissions auto permanent", message_id="perm-permanent-1"))
        await asyncio.sleep(0.05)
        reply = adapter.sent_texts[-1]
        nonce = reply.split("[", 1)[1].split("]", 1)[0]
        assert load_channel_settings(config_path).instances[0].permission_mode == "request_approval"

        await adapter.emit(_msg(f"/permissions confirm {nonce}", message_id="perm-permanent-2"))
        await asyncio.sleep(0.05)
        assert load_channel_settings(config_path).instances[0].permission_mode == "auto_approve"

        await adapter.emit(_msg("/permissions", message_id="perm-permanent-status"))
        await asyncio.sleep(0.05)
        assert "永久" in adapter.sent_texts[-1]

        state.set_owner("fake-1", "owner-2")
        assert (
            manager._effective_permission_mode(
                _msg("next", message_id="perm-permanent-owner-change", sender_id="owner-2")
            )
            == "request_approval"
        )
        state.set_owner("fake-1", "owner-1")

        await adapter.emit(_msg("/permissions safe", message_id="perm-permanent-3"))
        await asyncio.sleep(0.05)
        assert load_channel_settings(config_path).instances[0].permission_mode == "request_approval"
        await manager.stop()

    asyncio.run(_run())


def test_failed_permanent_permission_persistence_leaves_no_binding(tmp_path: Path) -> None:
    """配置里不存在实例时，确认失败不能遗留可复用的永久授权绑定。"""

    async def _run() -> None:
        config_path = tmp_path / "channels.json"
        save_channel_settings(config_path, ChannelSettings())
        manager, adapter, state, workspace = _make_manager(tmp_path, config_path=config_path)
        state.set_owner("fake-1", "owner-1")
        manager.register_instance_workspace("fake-1", workspace)
        await manager.attach_adapter(adapter)

        await adapter.emit(_msg("/permissions auto permanent", message_id="perm-missing-1"))
        await asyncio.sleep(0.05)
        nonce = adapter.sent_texts[-1].split("[", 1)[1].split("]", 1)[0]
        await adapter.emit(_msg(f"/permissions confirm {nonce}", message_id="perm-missing-2"))
        await asyncio.sleep(0.05)

        assert "失败" in adapter.sent_texts[-1]
        assert (
            state.get_permanent_permission_binding(
                "fake-1",
                owner_sender_id="owner-1",
                workspace_root=str(workspace.resolve()),
            )
            is None
        )
        await manager.stop()

    asyncio.run(_run())


def test_plain_text_goes_through_service_and_returns(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("hello agent", message_id="p1"))
        deadline = time.time() + 2
        services = manager._test_services  # type: ignore[attr-defined]
        while time.time() < deadline and not any("pong" in t for t in adapter.sent_texts):
            await asyncio.sleep(0.01)
        assert any(s.prompts == ["hello agent"] or "hello agent" in s.prompts for s in services)
        assert any("pong" in t for t in adapter.sent_texts)
        await manager.stop()

    asyncio.run(_run())


def test_high_risk_tool_triggers_remote_approval(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")

        def factory(root: Path) -> FakeAssistantService:
            svc = FakeAssistantService(root)
            svc.block_on_interaction = True
            return svc

        manager2, adapter2, state2, _ = _make_manager(tmp_path / "a2", service_factory=factory)
        state2.set_owner("fake-1", "owner-1")
        await manager2.attach_adapter(adapter2)
        task = asyncio.create_task(adapter2.emit(_msg("please shell", message_id="a1")))
        deadline = time.time() + 3
        nonce = None
        while time.time() < deadline:
            for text in adapter2.sent_texts:
                if "需要批准" in text or "/approve" in text:
                    # 提取 [NONCE]
                    if "[" in text and "]" in text:
                        nonce = text.split("[", 1)[1].split("]", 1)[0]
                        break
            if nonce:
                break
            await asyncio.sleep(0.02)
        assert nonce is not None
        await adapter2.emit(_msg(f"/approve {nonce}", message_id="a2"))
        await asyncio.wait_for(task, timeout=3)
        deadline = time.time() + 2
        while time.time() < deadline and not any("pong" in t for t in adapter2.sent_texts):
            await asyncio.sleep(0.01)
        assert any("pong" in t for t in adapter2.sent_texts)
        await manager2.stop()
        await manager.stop()

    asyncio.run(_run())


def test_adapter_disconnect_is_explicit(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        status = manager.status()
        assert any(item["instance_id"] == "fake-1" and item["state"] == "connected" for item in status)
        await adapter.stop()
        await manager.mark_adapter_state("fake-1", "stopped")
        status2 = manager.status()
        assert any(item["instance_id"] == "fake-1" and item["state"] == "stopped" for item in status2)
        await manager.stop()

    asyncio.run(_run())


def test_busy_message_does_not_write_receipt(tmp_path: Path) -> None:
    """Actor busy 时不得写 receipt，平台重投可再试。"""
    from tests.unit.channels.test_session_actor import FakeAssistantService as ActorFakeService

    async def _run() -> None:
        services_box: list[ActorFakeService] = []

        def factory(root: Path) -> ActorFakeService:
            svc = ActorFakeService(root)
            svc.hold_runs()
            services_box.append(svc)
            return svc

        manager, adapter, state, _ = _make_manager(tmp_path, service_factory=factory)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("one", message_id="busy-1"))
        await asyncio.sleep(0.05)
        await adapter.emit(_msg("two", message_id="busy-2"))
        await asyncio.sleep(0.05)
        await adapter.emit(_msg("three", message_id="busy-3"))
        await asyncio.sleep(0.1)
        # busy 的第三条不得落 receipt
        assert state.has_receipt("fake-1", "busy-3") is False
        assert any("繁忙" in t for t in adapter.sent_texts)
        for svc in services_box:
            svc.release_runs()
        await asyncio.sleep(0.1)
        await manager.stop()

    asyncio.run(_run())


def test_accepted_message_writes_receipt(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("hello", message_id="acc-1"))
        deadline = time.time() + 2
        while time.time() < deadline and not state.has_receipt("fake-1", "acc-1"):
            await asyncio.sleep(0.01)
        assert state.has_receipt("fake-1", "acc-1") is True
        await manager.stop()

    asyncio.run(_run())


def test_instance_workspace_root_honored(tmp_path: Path) -> None:
    async def _run() -> None:
        default_ws = tmp_path / "default-ws"
        default_ws.mkdir()
        instance_ws = tmp_path / "instance-ws"
        instance_ws.mkdir()
        state = ChannelStateStore(tmp_path / "channels.sqlite3")
        adapter = FakeAdapter("fake-1", platform="fake")
        services: list[FakeAssistantService] = []

        def factory(root: Path) -> FakeAssistantService:
            svc = FakeAssistantService(root)
            services.append(svc)
            return svc

        manager = ChannelManager(
            state=state,
            default_workspace_root=default_ws,
            service_factory=factory,
            broker=InteractionBroker(timeout_seconds=5.0),
        )
        manager.register_instance_workspace("fake-1", instance_ws)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("ws-check", message_id="ws-1"))
        deadline = time.time() + 2
        while time.time() < deadline and not services:
            await asyncio.sleep(0.01)
        assert services
        assert services[0].workspace_root.resolve() == instance_ws.resolve()
        await manager.stop()

    asyncio.run(_run())


def test_send_failure_is_surfaced(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        adapter.fail_next_sends = 99  # type: ignore[attr-defined]
        # 扩展 FakeAdapter 支持 fail
        original = adapter.send_text

        async def failing_send(*args, **kwargs):
            from haagent.channels.types import SendResult

            return SendResult(ok=False, error="send_failed")

        adapter.send_text = failing_send  # type: ignore[method-assign]
        await manager.attach_adapter(adapter)
        await adapter.emit(_msg("hi", message_id="sf-1", sender_id="stranger"))
        await asyncio.sleep(0.05)
        # 未授权路径也会 send；失败应记入 manager 可见状态
        status = manager.status()
        item = next(x for x in status if x["instance_id"] == "fake-1")
        assert item.get("last_send_error") or manager.last_send_error("fake-1")
        await manager.stop()

    asyncio.run(_run())


def test_auth_expired_visible_in_manager_status(tmp_path: Path) -> None:
    async def _run() -> None:
        manager, adapter, state, _ = _make_manager(tmp_path)
        state.set_owner("fake-1", "owner-1")
        await manager.attach_adapter(adapter)
        # 模拟 adapter 进入 auth_expired
        adapter.state = "auth_expired"  # type: ignore[attr-defined]
        await manager.sync_adapter_states()
        status = manager.status()
        assert any(item["instance_id"] == "fake-1" and item["state"] == "auth_expired" for item in status)
        await manager.stop()

    asyncio.run(_run())
