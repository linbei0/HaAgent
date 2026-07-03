"""
tests/unit/runtime/test_agent_session.py - AgentSession 生命周期测试

验证会话运行失败时仍会释放运行态资源，避免 TUI 误判任务仍在取消中。
"""

from pathlib import Path

import pytest

from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.turn import ChatTurnRunner


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
