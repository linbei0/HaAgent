"""
tests/integration/scheduling/test_executor.py - 隔离 ScheduledRunExecutor

验证计划运行走独立 AssistantService，关联 session/episode，映射失败分类。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.app.assistant_service import AssistantService
from haagent.models.types import ModelCallError, ModelFailureDetails, ModelResponse
from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.runtime.session.turn_completion import ChatTurnResult
from haagent.scheduling.executor import ScheduledRunExecutor
from haagent.scheduling.models import RunClaim, RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _write_connection(home: Path, *, name: str = "local", model: str = "m1") -> None:
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "version": 4,
                "connections": [
                    {
                        "id": name,
                        "name": name,
                        "provider_id": name,
                        "provider_name": name,
                        "gateway_provider": "openai-chat",
                        "base_url": "https://example.test",
                        "api_key_env": "HAAGENT_TEST_KEY",
                        "credential_source": "env",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": name, "model": model}}),
        encoding="utf-8",
    )


def _definition(
    workspace: Path,
    *,
    schedule_id: str = "sch_exec",
    destination: str = "new_session",
    session_path: Path | None = None,
    connection_id: str = "local",
    model: str = "m1",
    allowed: tuple[str, ...] = ("file_read",),
    approval_allowed: tuple[str, ...] = (),
    approved: tuple[str, ...] = (),
    prompt: str = "summarize workspace",
    web_enabled: bool = False,
    permission_mode: str = "request_approval",
    retry_policy: RetryPolicy | None = None,
) -> ScheduleDefinition:
    return ScheduleDefinition(
        id=schedule_id,
        name="exec-plan",
        prompt=prompt,
        workspace_root=workspace,
        destination_kind=destination,  # type: ignore[arg-type]
        destination_session_path=session_path,
        connection_id=connection_id,
        model=model,
        web_enabled=web_enabled,
        allowed_tools=allowed,
        approval_allowed_tools=approval_allowed,
        approved_tools=approved,
        permission_mode=permission_mode,  # type: ignore[arg-type]
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule=None,
        status="active",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=retry_policy or RetryPolicy(max_attempts=1),
        revision=1,
    )


class RecordingGateway:
    provider_name = "openai-chat"

    def __init__(self, text: str = "scheduled ok") -> None:
        self.text = text
        self.calls = 0

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        self.calls += 1
        return ModelResponse(self.text, [])


class PolicyDeniedGateway:
    provider_name = "openai-chat"

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        from haagent.models.types import ToolCall

        return ModelResponse("", [ToolCall(name="shell", args={"command": "echo hi"}, id="1")])


class TransientGateway:
    provider_name = "openai-chat"

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        raise ModelCallError(
            "rate limited for sk-abcdefghijklmnopqrstuvwxyz",
            details=ModelFailureDetails(category="rate_limited", retryable=True, status_code=429),
        )


class ToolPolicySessionMixin:
    def set_permission_mode(self, mode: str) -> None:
        self.permission_mode = mode

    def set_tool_overrides(
        self,
        *,
        allowed_tools: list[str],
        approval_allowed_tools: list[str],
        approved_tools: list[str],
    ) -> None:
        self._allowed_tools_override = allowed_tools
        self._approval_allowed_tools_override = approval_allowed_tools
        self._approved_tools_override = approved_tools


class InteractionTriggerSession(ToolPolicySessionMixin):
    """最小 session 替身：调用 interaction_handler 以触发无人值守失败。"""

    provider_name = "openai-chat"
    turn_count = 0

    def __init__(self, **kwargs) -> None:
        self.session_id = "sess-interaction"
        self.workspace_root = kwargs["workspace_root"]
        self.runs_root = kwargs["runs_root"]
        self.model_gateway = kwargs.get("model_gateway")
        self.model_ref = kwargs.get("model_ref")
        self.max_turns = kwargs.get("max_turns")
        self.enable_web = kwargs.get("enable_web", False)
        self.session_path = self.runs_root / "sessions" / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

    def run_prompt_events(self, prompt, **kwargs):
        handler = kwargs.get("interaction_handler")
        if handler is not None:
            req = HumanInteractionRequest(
                interaction_type="approval",
                tool_name="file_write",
                question="批准写入？",
            )
            handler(req) if callable(handler) else handler.request(req)
        ep = self.session_path / "episodes" / "ep1"
        ep.mkdir(parents=True, exist_ok=True)
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=1,
            status="failed",
            episode_path=ep,
            provider="openai-chat",
            final_response="",
            verification_status="not_run",
            failure_category="User Denied Failure",
            reason="interaction",
        )


class CancelAwareSession(ToolPolicySessionMixin):
    provider_name = "openai-chat"
    turn_count = 0

    def __init__(self, **kwargs) -> None:
        self.session_id = "sess-cancel"
        self.workspace_root = kwargs["workspace_root"]
        self.runs_root = kwargs["runs_root"]
        self.model_gateway = kwargs.get("model_gateway")
        self.model_ref = kwargs.get("model_ref")
        self.max_turns = kwargs.get("max_turns")
        self.enable_web = kwargs.get("enable_web", False)
        self.session_path = self.runs_root / "sessions" / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self._cancelled = False

    def cancel_current_run(self) -> bool:
        self._cancelled = True
        return True

    def run_prompt_events(self, prompt, **kwargs):
        if self._cancelled:
            ep = self.session_path / "episodes" / "ep-cancel"
            ep.mkdir(parents=True, exist_ok=True)
            return ChatTurnResult(
                session_id=self.session_id,
                turn_index=1,
                status="cancelled",
                episode_path=ep,
                provider="openai-chat",
                final_response="",
                verification_status="not_run",
                failure_category="cancelled",
                reason="user_cancelled",
            )
        ep = self.session_path / "episodes" / "ep-ok"
        ep.mkdir(parents=True, exist_ok=True)
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=1,
            status="completed",
            episode_path=ep,
            provider="openai-chat",
            final_response="ok",
            verification_status="not_run",
        )


def _seed_running_run(
    store: ScheduleStore,
    definition: ScheduleDefinition,
    *,
    now: datetime,
    worker_id: str = "worker-1",
) -> str:
    store.create(definition, now=now, next_run_at_utc=now)
    run = store.create_run(
        schedule_id=definition.id,
        schedule_revision=definition.revision,
        trigger_key=f"manual:{now.isoformat()}",
        trigger_kind="manual",
        scheduled_for_utc=now,
        status="queued",
        now=now,
    )
    claimed = store.claim_run(
        run.id,
        worker_id=worker_id,
        lease_expires_at=_utc(2026, 7, 13, 12, 5, 0),
        now=now,
    )
    assert claimed is not None
    return claimed.id


def test_execute_rejects_stale_attempt_without_mutating_current_claim(tmp_path: Path) -> None:
    now = _utc(2026, 7, 13, 12, 0, 0)
    with ScheduleStore(tmp_path / "stale.db") as store:
        run_id = _seed_running_run(store, _definition(tmp_path / "ws"), now=now)
        with pytest.raises(ScheduleStoreError, match="claim token") as exc_info:
            ScheduledRunExecutor(store).execute(
                RunClaim(run_id=run_id, worker_id="worker-1", attempt=2)
            )
        current = store.get_run(run_id)

    assert exc_info.value.code == "stale_execute"
    assert current is not None
    assert current.status == "running"
    assert current.worker_id == "worker-1"
    assert current.attempt_count == 1


def test_execute_rejects_unclaimed_run_without_finishing_it(tmp_path: Path) -> None:
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(tmp_path / "ws")
    with ScheduleStore(tmp_path / "queued.db") as store:
        store.create(definition, now=now)
        run = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="manual:queued",
            trigger_kind="manual",
            scheduled_for_utc=now,
            now=now,
        )
        with pytest.raises(ScheduleStoreError) as exc_info:
            ScheduledRunExecutor(store).execute(
                RunClaim(run_id=run.id, worker_id="worker-1", attempt=1)
            )
        current = store.get_run(run.id)

    assert exc_info.value.code == "stale_execute"
    assert current is not None and current.status == "queued"


def test_execute_new_session_success_records_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = home / ".haagent" / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(ws)
    gateway = RecordingGateway("hello from schedule")
    captured: dict[str, Path] = {}

    def service_factory(**kwargs) -> AssistantService:
        captured["runs_root"] = kwargs["runs_root"]
        return AssistantService(
            workspace_root=kwargs["workspace_root"],
            runs_root=kwargs["runs_root"],
            environ=kwargs.get("environ"),
            gateway_factory=lambda profile, **_kw: gateway,
        )

    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=service_factory,
            clock=lambda: now,
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "succeeded"
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.session_id
        assert finished.session_path
        assert finished.episode_path
        assert "hello" in finished.summary or finished.summary
        # 不得把 episode 全文塞进 schedule DB
        assert "transcript" not in (finished.summary or "").lower()
        session_path = Path(finished.session_path)
        assert session_path.exists()
        assert Path(finished.episode_path).exists()
        assert gateway.calls >= 1
        assert captured["runs_root"] == runs


def test_execute_resume_session_reuses_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    gateway = RecordingGateway("resume turn")
    service = AssistantService(
        workspace_root=ws,
        runs_root=runs,
        environ={"HAAGENT_TEST_KEY": "test-key"},
        gateway_factory=lambda profile, **_kw: gateway,
    )
    status = service.sessions.create()
    service.sessions.run_prompt_events("first turn")
    session_path = status.session_path
    turn_before = service.sessions._context.session.turn_count  # type: ignore[union-attr]

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(
        ws,
        destination="resume_session",
        session_path=session_path,
        prompt="second scheduled turn",
    )
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: gateway,
            ),
            runs_root=runs,
            clock=lambda: now,
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "succeeded"
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.session_id == status.session_id
        assert Path(finished.session_path).resolve() == session_path.resolve()
        # schedule DB 不复制 episode 内容
        episode_text = Path(finished.episode_path).read_text(encoding="utf-8") if Path(finished.episode_path).is_file() else ""
        assert "first turn" not in (finished.summary or "")
        # 续接后 turn 增加
        resumed = AssistantService(
            workspace_root=ws,
            runs_root=runs,
            environ={"HAAGENT_TEST_KEY": "test-key"},
            gateway_factory=lambda profile, **_kw: gateway,
        )
        resumed.sessions.resume(session_path)
        assert resumed.sessions._context.session is not None
        assert resumed.sessions._context.session.turn_count > turn_before


def test_execute_interaction_required_needs_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(ws)
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: RecordingGateway(),
                session_cls=InteractionTriggerSession,  # type: ignore[arg-type]
            ),
            runs_root=runs,
            clock=lambda: now,
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "needs_attention"
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.status == "needs_attention"
        assert finished.failure_category == "interaction_required"


def test_execute_workspace_missing_needs_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    missing = tmp_path / "gone-ws"
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    # store 校验不检查磁盘存在；运行时才检查
    definition = _definition(missing)
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            runs_root=runs,
            clock=lambda: now,
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "needs_attention"
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.failure_category == "workspace_unavailable"


def test_execute_credential_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    # 故意不设置 HAAGENT_TEST_KEY

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(ws)
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            runs_root=runs,
            clock=lambda: now,
            environ={},
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "needs_attention"
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.failure_category in {
            "credential_unavailable",
            "profile_unavailable",
        }


def test_execute_model_transient_maps_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(
        ws,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=30,
            multiplier=2.0,
            max_delay_seconds=900,
        ),
    )
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: TransientGateway(),
            ),
            runs_root=runs,
            clock=lambda: now,
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.failure_category == "model_transient"
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in finished.summary
        assert "REDACTED" in finished.summary
        assert finished.status == "retry_wait"
        assert finished.retry_at_utc == now + timedelta(seconds=30)
        assert result.status == "retry_wait"

        later = now + timedelta(seconds=30)
        claimed_again = store.claim_run(
            run_id,
            worker_id="worker-1",
            lease_expires_at=later + timedelta(seconds=45),
            now=later,
        )
        assert claimed_again is not None and claimed_again.attempt_count == 2
        exhausted = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: TransientGateway(),
            ),
            runs_root=runs,
            clock=lambda: later,
        ).execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=2))
        assert exhausted.status == "failed"


def test_execute_uses_isolated_service_not_shared_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    shared = AssistantService(
        workspace_root=ws,
        runs_root=runs,
        environ={"HAAGENT_TEST_KEY": "test-key"},
        gateway_factory=lambda profile, **_kw: RecordingGateway("shared"),
    )
    shared.sessions.create()
    shared_session_id = shared.sessions._context.session.session_id  # type: ignore[union-attr]

    created_services: list[AssistantService] = []

    def factory(**kwargs):
        svc = AssistantService(
            workspace_root=kwargs["workspace_root"],
            runs_root=kwargs.get("runs_root", runs),
            environ=kwargs.get("environ"),
            gateway_factory=lambda profile, **_kw: RecordingGateway("isolated"),
        )
        created_services.append(svc)
        return svc

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(ws)
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=factory,
            runs_root=runs,
            clock=lambda: now,
        )
        executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert created_services
        # 不得复用 shared 的 session
        finished = store.get_run(run_id)
        assert finished is not None
        assert finished.session_id != shared_session_id


def test_passes_tool_overrides_into_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    runs = tmp_path / "runs"
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    captured: dict[str, object] = {}

    class CaptureSession(ToolPolicySessionMixin):
        provider_name = "openai-chat"
        turn_count = 0

        def __init__(self, **kwargs) -> None:
            self.session_id = "sess-tools"
            self.workspace_root = kwargs["workspace_root"]
            self.runs_root = kwargs["runs_root"]
            self.model_gateway = kwargs.get("model_gateway")
            self.model_ref = kwargs.get("model_ref")
            self.max_turns = kwargs.get("max_turns")
            self.enable_web = kwargs.get("enable_web", False)
            self.session_path = self.runs_root / "sessions" / self.session_id
            self.session_path.mkdir(parents=True, exist_ok=True)

        def run_prompt_events(self, prompt, **kwargs):
            captured["allowed"] = self._allowed_tools_override
            captured["approval"] = self._approval_allowed_tools_override
            captured["approved"] = self._approved_tools_override
            captured["permission_mode"] = self.permission_mode
            captured["enable_web"] = self.enable_web
            captured["handler"] = kwargs.get("interaction_handler")
            ep = self.session_path / "episodes" / "ep"
            ep.mkdir(parents=True, exist_ok=True)
            return ChatTurnResult(
                session_id=self.session_id,
                turn_index=1,
                status="completed",
                episode_path=ep,
                provider="openai-chat",
                final_response="ok",
                verification_status="not_run",
            )

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    definition = _definition(
        ws,
        allowed=("file_read", "file_write"),
        approval_allowed=("file_write",),
        approved=("file_write",),
        permission_mode="auto_approve",
    )
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: RecordingGateway(),
                session_cls=CaptureSession,  # type: ignore[arg-type]
            ),
            runs_root=runs,
            clock=lambda: now,
        )
        executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert captured.get("allowed") == ["file_read", "file_write"] or captured.get(
            "allowed"
        ) == ("file_read", "file_write")
        assert captured.get("permission_mode") == "auto_approve"
        assert captured.get("handler") is not None


def test_web_enabled_real_session_writes_web_tools_to_task_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 AgentSession：DB 仅存只读工具 + web_enabled 时，task.yaml 仍含 web 工具。"""
    import yaml

    from haagent.runtime.session.agent import AgentSession

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    runs = tmp_path / "runs"
    ws.mkdir()
    _write_connection(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HAAGENT_TEST_KEY", "test-key")

    db = tmp_path / "schedules.db"
    now = _utc(2026, 7, 13, 12, 0, 0)
    # 模拟旧计划：web_enabled=True 但 allowed_tools 无 web_*
    definition = _definition(
        ws,
        allowed=("file_list", "grep", "file_read", "skill_list", "skill_read"),
        web_enabled=True,
        prompt="天气预报",
    )
    with ScheduleStore(db) as store:
        run_id = _seed_running_run(store, definition, now=now)
        executor = ScheduledRunExecutor(
            store,
            service_factory=lambda **kwargs: AssistantService(
                workspace_root=kwargs["workspace_root"],
                runs_root=kwargs.get("runs_root", runs),
                environ=kwargs.get("environ"),
                gateway_factory=lambda profile, **_kw: RecordingGateway("晴天"),
                session_cls=AgentSession,
                enable_web=kwargs.get("enable_web", False),
            ),
            runs_root=runs,
            clock=lambda: now,
            config_dir=home / ".haagent",
        )
        result = executor.execute(RunClaim(run_id=run_id, worker_id="worker-1", attempt=1))
        assert result.status == "succeeded"
        assert result.episode_path
        task_path = Path(result.episode_path) / "task.yaml"
        assert task_path.exists()
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        allowed = task["allowed_tools"]
        assert "web_search" in allowed
        assert "web_fetch" in allowed
        assert "file_list" in allowed
