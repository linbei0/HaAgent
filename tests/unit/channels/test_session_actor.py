"""
tests/unit/channels/test_session_actor.py - ChannelSessionActor 会话绑定与并发测试
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from haagent.channels.interactions import InteractionBroker
from haagent.channels.session_actor import ChannelSessionActor, SubmitResult
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage
from haagent.runtime.events.types import AssistantMessageEvent, SessionLifecycleEvent
from haagent.runtime.execution.human_interaction import (
    HumanInteractionRequest,
    HumanInteractionResponse,
)
from tests.support.channel_adapter import FakeChannelAdapter as FakeAdapter


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
        self.permissions = self
        self.permission_modes: list[str] = []

    def create(self) -> FakeSessionStatus:
        return self._service._create()

    def resume(self, session: str | Path) -> FakeSessionStatus:
        return self._service._resume(str(session))

    def set_mode(self, mode: str) -> None:
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
    """轻量 service：记录 create/resume/run，不碰真实模型。"""

    _id_seq = 0

    def __init__(self, workspace_root: Path, runs_root: Path | None = None) -> None:
        self.workspace_root = workspace_root
        self.runs_root = runs_root or (workspace_root / ".runs")
        self.sessions = FakeSessions(self)
        self.session_id: str | None = None
        self.created_count = 0
        self.resumed: list[str] = []
        self.prompts: list[str] = []
        self.interaction_handlers: list[object] = []
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._running = False
        self._cancel_requested = False
        self.block_on_interaction = False
        self.interaction_result: HumanInteractionResponse | None = None
        self.turn_count = 0
        self.reply_text = "pong"
        self.emit_lifecycle = True

    def hold_runs(self) -> None:
        self._run_gate.clear()

    def release_runs(self) -> None:
        self._run_gate.set()

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
        self.session_id = Path(session).name if "/" in session.replace("\\", "/") or "\\" in session else session
        # 兼容传入 session_id 或 path
        name = Path(session).name
        self.session_id = name
        self.resumed.append(name)
        return FakeSessionStatus(
            session_id=name,
            workspace_root=self.workspace_root,
            runs_root=self.runs_root,
            session_path=self.runs_root / name,
        )

    def _run_prompt(
        self,
        prompt: str,
        *,
        event_sink,
        include_session_events: bool,
        interaction_handler,
    ):
        del include_session_events
        self.prompts.append(prompt)
        self.interaction_handlers.append(interaction_handler)
        self._running = True
        self._cancel_requested = False
        try:
            if self.emit_lifecycle and event_sink is not None:
                event_sink(SessionLifecycleEvent(self.session_id or "s", 1, "turn_started", "start"))
            if self.block_on_interaction and interaction_handler is not None:
                req = HumanInteractionRequest(
                    interaction_type="approval",
                    tool_name="shell",
                    question="run?",
                    args_summary={"command": "pytest"},
                )
                self.interaction_result = interaction_handler(req)
            # 阻塞直到 release，用于并发测试
            self._run_gate.wait(timeout=5.0)
            if self._cancel_requested:
                return {"status": "cancelled"}
            if event_sink is not None:
                event_sink(
                    AssistantMessageEvent(self.session_id or "s", 1, 1, self.reply_text)
                )
                if self.emit_lifecycle:
                    event_sink(
                        SessionLifecycleEvent(
                            self.session_id or "s", 1, "turn_finished", "end", status="ok"
                        )
                    )
            self.turn_count += 1
            return {"status": "ok", "reply": self.reply_text}
        finally:
            self._running = False

    def _cancel(self) -> FakeCancelResult:
        if not self._running:
            return FakeCancelResult(status="idle", reason="no_active_run")
        self._cancel_requested = True
        self._run_gate.set()
        return FakeCancelResult(status="cancelled", reason="user_cancelled")


def _address(
    *,
    instance_id: str = "wx-1",
    conversation_id: str = "owner-1",
    platform: str = "fake",
) -> ChannelAddress:
    return ChannelAddress(
        instance_id=instance_id,
        platform=platform,
        conversation_kind="dm",
        conversation_id=conversation_id,
    )


def _message(
    text: str,
    *,
    message_id: str = "m-1",
    address: ChannelAddress | None = None,
    sender_id: str = "owner-1",
) -> InboundChannelMessage:
    addr = address or _address()
    return InboundChannelMessage(
        address=addr,
        message_id=message_id,
        sender_id=sender_id,
        text=text,
        reply_handle=ChannelReplyHandle(platform=addr.platform, payload={"token": "hidden"}),
    )


def _make_actor(
    tmp_path: Path,
    *,
    address: ChannelAddress | None = None,
    service: FakeAssistantService | None = None,
    adapter: FakeAdapter | None = None,
    broker: InteractionBroker | None = None,
    owner_sender_id: str = "owner-1",
    permission_mode: str = "request_approval",
) -> tuple[ChannelSessionActor, FakeAssistantService, FakeAdapter, ChannelStateStore]:
    addr = address or _address()
    workspace = tmp_path / "ws" / addr.conversation_id
    workspace.mkdir(parents=True, exist_ok=True)
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    svc = service or FakeAssistantService(workspace)
    adp = adapter or FakeAdapter(addr.instance_id, platform=addr.platform)
    actor = ChannelSessionActor(
        binding_key=addr.binding_key(),
        address=addr,
        owner_sender_id=owner_sender_id,
        workspace_root=workspace,
        state=state,
        adapter=adp,
        service=svc,
        broker=broker or InteractionBroker(timeout_seconds=5.0),
        permission_mode=permission_mode,  # type: ignore[arg-type]
    )
    return actor, svc, adp, state


def test_submit_during_worker_teardown_starts_successor_worker(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, service, _adapter, state = _make_actor(tmp_path)
        entered_stop = asyncio.Event()
        release_stop = asyncio.Event()
        original_stop = actor._bridge.stop

        async def delayed_stop() -> None:
            entered_stop.set()
            await release_stop.wait()
            await original_stop()

        actor._bridge.stop = delayed_stop
        first = await actor.submit(_message("first", message_id="m-1"))
        assert first.status == "accepted"
        await asyncio.wait_for(entered_stop.wait(), timeout=2.0)

        second = await actor.submit(_message("second", message_id="m-2"))
        assert second.status == "accepted"
        release_stop.set()

        deadline = asyncio.get_running_loop().time() + 2.0
        while service.prompts != ["first", "second"] and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert service.prompts == ["first", "second"]
        assert actor._pending is None
        await actor.close()
        state.close()

    asyncio.run(_run())


def test_slow_model_reply_still_delivered_after_idle_gap(tmp_path: Path) -> None:
    """模型思考期间无事件时，drain 不得提前退出，最终回复必须投递。"""

    async def _run() -> None:
        actor, svc, adp, _ = _make_actor(tmp_path)

        def slow_run(prompt, *, event_sink, include_session_events, interaction_handler):
            del include_session_events, interaction_handler
            sid = svc.session_id or "s"
            if event_sink is not None:
                event_sink(SessionLifecycleEvent(sid, 1, "turn_started", "start"))
            # 模拟真实模型长思考：turn 中途长时间无事件。
            time.sleep(0.3)
            if event_sink is not None:
                event_sink(AssistantMessageEvent(sid, 1, 1, "late-pong"))
                event_sink(SessionLifecycleEvent(sid, 1, "turn_finished", "end", status="ok"))
            svc.prompts.append(prompt)
            svc.turn_count += 1
            return {"status": "ok"}

        svc._run_prompt = slow_run  # type: ignore[method-assign]
        await adp.start(lambda m: None)
        result = await actor.submit(_message("slow", message_id="slow-1"))
        assert result.status == "accepted"
        deadline = time.time() + 3
        while time.time() < deadline and not any("late-pong" in t for t in adp.sent_texts):
            await asyncio.sleep(0.02)
        assert any("late-pong" in t for t in adp.sent_texts), f"got {adp.sent_texts}"
        await actor.close()

    asyncio.run(_run())


def test_deliver_text_surfaces_send_failure(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, svc, adp, state = _make_actor(tmp_path)
        await adp.start(lambda m: None)

        async def fail_send(*args, **kwargs):
            from haagent.channels.types import SendResult

            return SendResult(ok=False, error="network_down")

        adp.send_text = fail_send  # type: ignore[method-assign]
        result = await actor._deliver_text("hello")
        assert result is not None
        assert result.ok is False
        assert actor.last_send_error == "network_down"

    asyncio.run(_run())


def test_first_message_creates_session_and_persists_binding(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, svc, adp, state = _make_actor(tmp_path)
        await adp.start(lambda m: None)
        result = await actor.submit(_message("hello", message_id="m-create"))
        assert result.status == "accepted"
        assert svc.created_count == 1
        assert svc.session_id is not None
        binding = state.get_binding(_address().binding_key())
        assert binding is not None
        assert binding.session_id == svc.session_id
        # 等待 turn 完成并投递回复
        await asyncio.sleep(0.05)
        await actor.close()
        assert "pong" in adp.sent_texts or any("pong" in t for t in adp.sent_texts)

    asyncio.run(_run())


def test_actor_applies_auto_approve_without_accepting_full_access(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, service, adapter, _ = _make_actor(tmp_path, permission_mode="auto_approve")
        await adapter.start(lambda message: None)
        await actor.submit(_message("auto", message_id="auto-1"))
        deadline = time.time() + 2
        while time.time() < deadline and not service.prompts:
            await asyncio.sleep(0.01)
        assert service.sessions.permission_modes
        assert set(service.sessions.permission_modes) == {"auto_approve"}
        await actor.close()
        await adapter.stop()

    asyncio.run(_run())


def test_second_message_resumes_same_session(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, svc, adp, state = _make_actor(tmp_path)
        await adp.start(lambda m: None)
        r1 = await actor.submit(_message("first", message_id="m1"))
        assert r1.status == "accepted"
        # 等第一轮完全空闲后再提交第二条
        deadline = time.time() + 2
        while (svc.turn_count < 1 or actor._active or actor._pending is not None) and time.time() < deadline:
            await asyncio.sleep(0.01)
        r2 = await actor.submit(_message("second", message_id="m2"))
        assert r2.status in {"accepted", "queued"}
        deadline = time.time() + 2
        while svc.turn_count < 2 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc.created_count == 1
        assert svc.resumed == [] or svc.session_id is not None
        binding = state.get_binding(_address().binding_key())
        assert binding is not None
        assert binding.session_id == svc.session_id
        assert svc.prompts == ["first", "second"]
        await actor.close()

    asyncio.run(_run())


def test_two_bindings_do_not_share_current_session(tmp_path: Path) -> None:
    async def _run() -> None:
        a1 = _address(conversation_id="u1")
        a2 = _address(conversation_id="u2")
        actor1, svc1, adp1, state = _make_actor(tmp_path, address=a1)
        actor2, svc2, adp2, _ = _make_actor(
            tmp_path,
            address=a2,
            adapter=FakeAdapter("wx-1", platform="fake"),
        )
        # 共享同一 state store
        actor2._state = state  # type: ignore[attr-defined]
        await adp1.start(lambda m: None)
        await adp2.start(lambda m: None)
        await actor1.submit(_message("hi-1", message_id="m1", address=a1, sender_id="u1"))
        await actor2.submit(_message("hi-2", message_id="m2", address=a2, sender_id="u2"))
        deadline = time.time() + 2
        while (svc1.turn_count < 1 or svc2.turn_count < 1) and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc1.session_id != svc2.session_id
        assert svc1.created_count == 1
        assert svc2.created_count == 1
        b1 = state.get_binding(a1.binding_key())
        b2 = state.get_binding(a2.binding_key())
        assert b1 is not None and b2 is not None
        assert b1.session_id != b2.session_id
        await actor1.close()
        await actor2.close()

    asyncio.run(_run())


def test_same_binding_serializes_turns_and_busy_rejects_third(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, svc, adp, _ = _make_actor(tmp_path)
        svc.hold_runs()
        await adp.start(lambda m: None)
        r1 = await actor.submit(_message("one", message_id="m1"))
        assert r1.status == "accepted"
        # 等待 active turn 真正启动
        deadline = time.time() + 2
        while not svc._running and time.time() < deadline:
            await asyncio.sleep(0.01)
        r2 = await actor.submit(_message("two", message_id="m2"))
        assert r2.status in {"queued", "accepted"}
        r3 = await actor.submit(_message("three", message_id="m3"))
        assert r3.status == "busy"
        assert "繁忙" in (r3.reason or "") or "busy" in (r3.reason or "").lower() or r3.status == "busy"
        # busy 时应向用户明确回复
        await asyncio.sleep(0.05)
        busy_texts = [t for t in adp.sent_texts if "繁忙" in t or "/stop" in t]
        assert busy_texts, f"expected busy reply, got {adp.sent_texts}"
        svc.release_runs()
        deadline = time.time() + 3
        while svc.turn_count < 2 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc.prompts == ["one", "two"]
        assert "three" not in svc.prompts
        await actor.close()

    asyncio.run(_run())


def test_stop_cancels_current_run(tmp_path: Path) -> None:
    async def _run() -> None:
        actor, svc, adp, _ = _make_actor(tmp_path)
        svc.hold_runs()
        await adp.start(lambda m: None)
        await actor.submit(_message("long", message_id="m1"))
        deadline = time.time() + 2
        while not svc._running and time.time() < deadline:
            await asyncio.sleep(0.01)
        cancelled = await actor.cancel_current_turn()
        assert cancelled is True
        deadline = time.time() + 2
        while svc._running and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert not svc._running
        await actor.close()

    asyncio.run(_run())


def test_interaction_broker_stays_in_same_turn(tmp_path: Path) -> None:
    async def _run() -> None:
        broker = InteractionBroker(timeout_seconds=5.0)
        actor, svc, adp, _ = _make_actor(tmp_path, broker=broker)
        svc.block_on_interaction = True
        await adp.start(lambda m: None)
        submit_task = asyncio.create_task(actor.submit(_message("need approve", message_id="m1")))
        # 等 worker 进入 interaction
        deadline = time.time() + 3
        pending = None
        while time.time() < deadline:
            pending = broker.wait_for_pending(timeout=0.05)
            if pending is not None:
                break
            await asyncio.sleep(0.01)
        assert pending is not None
        assert pending.kind == "approval"
        # 同一 turn 内 resolve，不应新开 prompt
        broker.resolve(
            pending.nonce,
            approved=True,
            sender_id="owner-1",
            binding_key=_address().binding_key(),
        )
        result = await submit_task
        assert result.status == "accepted"
        deadline = time.time() + 2
        while svc.turn_count < 1 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc.prompts == ["need approve"]
        assert svc.interaction_result is not None
        assert svc.interaction_result.approved is True
        assert len(svc.interaction_handlers) == 1
        await actor.close()

    asyncio.run(_run())


def test_actor_resume_from_persisted_binding(tmp_path: Path) -> None:
    """关闭后再开新 Actor，应从 state 恢复同一 session_id。"""

    async def _run() -> None:
        addr = _address()
        actor1, svc1, adp1, state = _make_actor(tmp_path, address=addr)
        await adp1.start(lambda m: None)
        await actor1.submit(_message("first", message_id="m1"))
        deadline = time.time() + 2
        while svc1.turn_count < 1 and time.time() < deadline:
            await asyncio.sleep(0.01)
        session_id = svc1.session_id
        assert session_id is not None
        await actor1.close()

        svc2 = FakeAssistantService(tmp_path / "ws" / addr.conversation_id)
        adp2 = FakeAdapter(addr.instance_id, platform=addr.platform)
        actor2 = ChannelSessionActor(
            binding_key=addr.binding_key(),
            address=addr,
            owner_sender_id="owner-1",
            workspace_root=tmp_path / "ws" / addr.conversation_id,
            state=state,
            adapter=adp2,
            service=svc2,
            broker=InteractionBroker(timeout_seconds=5.0),
        )
        await adp2.start(lambda m: None)
        await actor2.submit(_message("second", message_id="m2"))
        deadline = time.time() + 2
        while svc2.turn_count < 1 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc2.created_count == 0
        assert session_id in svc2.resumed or svc2.session_id == session_id
        assert svc2.prompts == ["second"]
        await actor2.close()

    asyncio.run(_run())


def test_missing_session_package_falls_back_to_create(tmp_path: Path) -> None:
    """binding 指向已删除的 session package 时，必须 create 新会话而不是整批卡死。"""

    async def _run() -> None:
        actor, svc, adp, state = _make_actor(tmp_path)
        addr = actor.address
        state.upsert_binding(
            binding_key=addr.binding_key(),
            instance_id=addr.instance_id,
            platform=addr.platform,
            conversation_kind=addr.conversation_kind,
            conversation_id=addr.conversation_id,
            thread_id=addr.thread_id,
            workspace_root=str(actor.workspace_root),
            session_id="session-missing",
            owner_sender_id="owner-1",
        )

        def boom_resume(session: str) -> FakeSessionStatus:
            raise RuntimeError(f"session package missing: {session}")

        svc._resume = boom_resume  # type: ignore[method-assign]
        await adp.start(lambda m: None)
        result = await actor.submit(_message("hello after wipe", message_id="m-miss"))
        assert result.status == "accepted"
        deadline = time.time() + 2
        while svc.turn_count < 1 and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert svc.created_count == 1
        assert svc.prompts == ["hello after wipe"]
        binding = state.get_binding(addr.binding_key())
        assert binding is not None
        assert binding.session_id == svc.session_id
        assert binding.session_id != "session-missing"
        await actor.close()
        state.close()

    asyncio.run(_run())
