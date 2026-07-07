"""
tests/unit/runtime/test_agent_session.py - AgentSession 生命周期测试

验证会话运行失败时仍会释放运行态资源，避免 TUI 误判任务仍在取消中。
"""

import json
from pathlib import Path

import pytest

from haagent.runtime.session.agent import AgentSession, ChatTurnResult
from haagent.runtime.session.task_ledger import load_task_ledger
from haagent.runtime.session.turn import ChatTurnRunner
from haagent.runtime.orchestration.recorder import RunResult
from haagent.runtime.orchestration.state import RunStatus


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

    [turn] = session.turn_summaries()

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

    assert captured["task_ledger"]["status"] == "planning"
    assert captured["task_ledger"]["goal"] == ""
    persisted = load_task_ledger(session.session_path / "task-ledger.json")
    assert persisted.goal == "完成一个需要多步骤恢复的长任务"
    assert persisted.updated_turn == 1
    assert persisted.checkpoints
