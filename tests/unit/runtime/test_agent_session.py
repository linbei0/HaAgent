"""
tests/unit/runtime/test_agent_session.py - AgentSession 生命周期测试

验证会话运行失败时仍会释放运行态资源，避免 TUI 误判任务仍在取消中。
"""

import json
from pathlib import Path

import pytest

from haagent.models.types import ModelCallError, ModelResponse
from haagent.models.model_ref import ModelRef
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.turn_completion import ChatTurnResult
from haagent.runtime.session.task_ledger import load_task_ledger
from haagent.runtime.session.turn import ChatTurnRunner
from haagent.runtime.orchestration.recorder import RunResult
from haagent.runtime.orchestration.state import RunStatus


class _FinalAnswerGateway:
    provider_name = "fake"

    def generate(self, invocation, **kwargs):
        del invocation, kwargs
        return ModelResponse(content="done", tool_calls=[])


class _FailingGateway:
    provider_name = "fake"

    def generate(self, invocation, **kwargs):
        del invocation, kwargs
        raise ModelCallError("rate limit")


def test_agent_session_clears_cancellation_token_after_run_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self, request):
        del self, request
        raise RuntimeError("boom")

    monkeypatch.setattr(ChatTurnRunner, "run", _raise)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )

    with pytest.raises(RuntimeError, match="boom"):
        session.run_prompt_events("hello")

    assert session.cancel_current_run() is False


def test_agent_session_records_bounded_assistant_display_text(tmp_path: Path) -> None:
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    result = ChatTurnResult(
        session_id=session.session_id,
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-1",
        provider="fake",
        final_response="答" * 4100,
        verification_status="success",
    )

    session._record_turn("搜索今天的新闻", result, "summary")

    [record] = [
        json.loads(line)
        for line in (session.session_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    display_text = record["assistant_display_text"]
    assert display_text.endswith("... [truncated]")
    assert len(display_text) == 4000 + len("... [truncated]")

    [turn] = session.turn_summaries()
    assert turn.assistant_display_text == display_text


def test_agent_session_assistant_display_text_preserves_markdown_newlines(tmp_path: Path) -> None:
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    markdown_table = "\n".join(
        [
            "下面是结果：",
            "",
            "| 类别 | 标题 |",
            "| --- | --- |",
            "| 国际 | 新闻 A |",
        ],
    )
    result = ChatTurnResult(
        session_id=session.session_id,
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-markdown",
        provider="fake",
        final_response=markdown_table,
        verification_status="success",
    )

    session._record_turn("搜索今日新闻，使用表格展示", result, "summary")

    [turn] = session.turn_summaries()
    assert turn.assistant_display_text == markdown_table


def test_agent_session_turn_summaries_keep_legacy_records_compatible(tmp_path: Path) -> None:
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    record = {
        "turn_index": 1,
        "request": "旧问题",
        "summary": "旧摘要",
        "status": "completed",
        "episode_path": str(tmp_path / ".runs" / "episodes" / "episode-legacy"),
        "verification_status": "success",
    }
    (session.session_path / "turns.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    resumed = AgentSession.resume(session.session_path, model_gateway=None)
    [turn] = resumed.turn_summaries()

    assert turn.request == "旧问题"
    assert turn.assistant_display_text is None


def test_agent_session_writes_and_resumes_task_ledger(tmp_path: Path) -> None:
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    ledger_path = session.session_path / "task-ledger.json"

    assert ledger_path.exists()
    ledger = load_task_ledger(ledger_path)
    assert ledger.status == "planning"

    resumed = AgentSession.resume(
        session.session_path,
        model_gateway=None,
    )

    assert resumed.status()["task_ledger"]["status"] == "planning"
    assert resumed.status()["task_ledger"]["step_count"] == 0


def test_agent_session_passes_task_ledger_to_turn_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def _run(self, request):
        del self
        captured["task_ledger"] = request.task_ledger
        return RunResult(
            status=RunStatus.COMPLETED,
            state_history=[RunStatus.EXECUTING, RunStatus.COMPLETED],
            episode_path=tmp_path / ".runs" / "episodes" / "episode-1",
        )

    def _build_result(self, prompt, result):
        del prompt, result
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=1,
            status="completed",
            episode_path=tmp_path / ".runs" / "episodes" / "episode-1",
            provider="fake",
            final_response="ok",
            verification_status="not_run",
        )

    monkeypatch.setattr(ChatTurnRunner, "run", _run)
    monkeypatch.setattr(AgentSession, "_build_turn_result", _build_result)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )

    session.run_prompt_events("完成一个需要多步骤恢复的长任务")

    assert captured["task_ledger"]["status"] == "running"
    assert captured["task_ledger"]["goal"] == "完成一个需要多步骤恢复的长任务"
    persisted = load_task_ledger(session.session_path / "task-ledger.json")
    assert persisted.goal == "完成一个需要多步骤恢复的长任务"
    assert persisted.status == "completed"
    assert persisted.updated_turn == 1
    assert persisted.checkpoints


def test_agent_session_records_in_band_shell_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run(self, request):
        request.event_sink(
            {
                "event_type": "tool_finished",
                "turn": 1,
                "tool_name": "shell",
                "args": {"command": "uv run pytest tests/extended/test_cli_inspect.py -q --run-extended"},
                "result": {"status": "success", "exit_code": 0},
            },
        )
        return RunResult(
            status=RunStatus.COMPLETED,
            state_history=[RunStatus.EXECUTING, RunStatus.COMPLETED],
            episode_path=tmp_path / ".runs" / "episodes" / "episode-verify",
        )

    def _build_result(self, prompt, result):
        del prompt, result
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=1,
            status="completed",
            episode_path=tmp_path / ".runs" / "episodes" / "episode-verify",
            provider="fake",
            final_response="ok",
            verification_status="not_run",
        )

    monkeypatch.setattr(ChatTurnRunner, "run", _run)
    monkeypatch.setattr(AgentSession, "_build_turn_result", _build_result)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )

    result = session.run_prompt_events("修复并验证 inspect 状态同步")

    assert result.verification_status == "success"
    turns = (session.session_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(turns[-1])["verification_status"] == "success"


def test_agent_session_emits_task_progress_for_turn_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run(self, request):
        request.event_sink(
            {
                "event_type": "tool_finished",
                "turn": 1,
                "tool_name": "shell",
                "args": {"command": "uv run pytest tests/unit/runtime/test_agent_session.py -q"},
                "result": {"status": "success", "exit_code": 0},
            },
        )
        return RunResult(
            status=RunStatus.COMPLETED,
            state_history=[RunStatus.EXECUTING, RunStatus.COMPLETED],
            episode_path=tmp_path / ".runs" / "episodes" / "episode-progress",
        )

    def _build_result(self, prompt, result):
        del prompt, result
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=1,
            status="completed",
            episode_path=tmp_path / ".runs" / "episodes" / "episode-progress",
            provider="fake",
            final_response="ok",
            verification_status="not_run",
        )

    events: list[dict[str, object]] = []
    monkeypatch.setattr(ChatTurnRunner, "run", _run)
    monkeypatch.setattr(AgentSession, "_build_turn_result", _build_result)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )

    session.run_prompt_events("修复并展示任务进度", event_sink=events.append)

    progress_names = [
        event.event_name
        for event in events
        if hasattr(event, "event_name")
    ]
    assert progress_names == [
        "task_step_started",
        "task_checkpoint_saved",
        "task_step_finished",
    ]


def test_agent_session_emits_task_plan_created_from_runtime(tmp_path: Path) -> None:
    events: list[object] = []
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=_FinalAnswerGateway(),
        memory_extraction_enabled=False,
    )

    session.run_prompt_events("整理一个长任务并记录进度", event_sink=events.append)

    progress_names = [
        event.event_name
        for event in events
        if hasattr(event, "event_name")
    ]
    assert "task_plan_created" in progress_names


def test_agent_session_emits_recovery_for_model_failure(tmp_path: Path) -> None:
    events: list[object] = []
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=_FailingGateway(),
        memory_extraction_enabled=False,
    )

    result = session.run_prompt_events("执行一个会遇到模型错误的长任务", event_sink=events.append)

    assert result.status == "failed"
    recovery_events = [
        event for event in events if getattr(event, "event_name", "") == "task_recovery_suggested"
    ]
    assert recovery_events
    assert recovery_events[-1].category == "model_error"
    assert recovery_events[-1].suggested_action == "retry_or_switch_model"


def test_session_always_permissions_persist_on_resume_not_new_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """edit_diff 与工具权限 always 随 session 恢复；new() 不继承。"""
    from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
    from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver

    def _run(self, request):
        del self
        # 模拟用户在本 turn 选择 always：写回共享 session 状态
        assert request.session_interaction_state is not None
        resolver = HumanInteractionResolver(
            permission_mode=request.path_policy.permission_mode,
            session_interaction_state=request.session_interaction_state,
        )
        resolver.record(
            HumanInteractionRequest(
                interaction_type="edit_diff",
                tool_name="file_write",
                question="Approve?",
                args_summary={"path": "a.txt", "diff_preview": "+a"},
            ),
            HumanInteractionResponse(approved=True, answer="always"),
            turn=1,
        )
        resolver.record(
            HumanInteractionRequest(
                interaction_type="approval",
                tool_name="external_directory",
                question="Allow external directory?",
                args_summary={
                    "permission_patterns": ["C:/Users/test/.haagent/*"],
                    "permission_always": ["C:/Users/test/.haagent/*"],
                },
            ),
            HumanInteractionResponse(approved=True, answer="always"),
            turn=1,
        )
        return RunResult(
            status=RunStatus.COMPLETED,
            state_history=[RunStatus.EXECUTING, RunStatus.COMPLETED],
            episode_path=tmp_path / ".runs" / "episodes" / "episode-always",
        )

    def _build_result(self, prompt, result):
        del prompt, result
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=self.turn_count + 1,
            status="completed",
            episode_path=tmp_path / ".runs" / "episodes" / "episode-always",
            provider="fake",
            final_response="ok",
            verification_status="not_run",
        )

    monkeypatch.setattr(ChatTurnRunner, "run", _run)
    monkeypatch.setattr(AgentSession, "_build_turn_result", _build_result)

    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    session.run_prompt_events("write files")
    assert session._session_interaction_state.edit_diff_session_always is True
    assert session._session_interaction_state.permission_rules == [
        {"permission": "external_directory", "pattern": "C:/Users/test/.haagent/*"},
    ]

    metadata = json.loads((session.session_path / "session.json").read_text(encoding="utf-8"))
    assert metadata["edit_diff_session_always"] is True
    assert metadata["permission_rules"] == session._session_interaction_state.permission_rules

    resumed = AgentSession.resume(session.session_path, model_gateway=None)
    assert resumed._session_interaction_state.edit_diff_session_always is True
    assert resumed._session_interaction_state.permission_rules == session._session_interaction_state.permission_rules

    session.new()
    assert session._session_interaction_state.edit_diff_session_always is False
    assert session._session_interaction_state.permission_rules == []
    new_meta = json.loads((session.session_path / "session.json").read_text(encoding="utf-8"))
    assert new_meta.get("edit_diff_session_always") is False
    assert new_meta.get("permission_rules") == []
def test_switch_model_gateway_failure_restores_all_fields_and_closes_rejected_gateway(tmp_path) -> None:
    class Gateway:
        provider_name = "fake"

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    previous = Gateway()
    rejected = Gateway()
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=previous,  # type: ignore[arg-type]
        model_ref=ModelRef("old-connection", "old-model", "old-variant"),
        memory_extraction_enabled=False,
    )
    session._current_cancellation_token = None

    def fail_metadata_write() -> None:
        raise OSError("disk full")

    session._write_session_metadata = fail_metadata_write  # type: ignore[method-assign]

    with pytest.raises(OSError, match="disk full"):
        session.switch_model_gateway(ModelRef("new-connection", "new-model", "new-variant"), rejected)

    assert session.model_gateway is previous
    assert session.model_ref == ModelRef("old-connection", "old-model", "old-variant")
    assert rejected.closed is True
    assert previous.closed is False


def test_reload_replaces_model_ref_atomically(tmp_path) -> None:
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_ref=ModelRef("main", "model", "deep"),
        memory_extraction_enabled=False,
    )
    target = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_ref=ModelRef("main", "model", "old"),
        memory_extraction_enabled=False,
    )

    session.reload(target.session_path, model_ref=ModelRef("main", "model"))

    assert session.model_ref == ModelRef("main", "model")
