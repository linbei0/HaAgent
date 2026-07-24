"""
tests/unit/tools/test_session_history_tool.py - 会话历史工具测试

验证工具只读取注入的当前会话，并返回有界对话证据。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.tools.router import ToolRouter
from haagent.tools.session_history import session_history


def _write_turn(session_path: Path) -> None:
    session_path.mkdir(parents=True)
    (session_path / "turns.jsonl").write_text(
        json.dumps(
            {
                "turn_index": 1,
                "request": "检查认证配置",
                "summary": "认证使用 OAuth。",
                "status": "completed",
                "episode_path": "episodes/turn-1",
                "verification_status": "success",
                "assistant_display_text": "认证配置位于 auth/settings.py。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_session_history_returns_current_session_dialogue_evidence(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turn(session_path)

    result = session_history({"query": "认证配置"}, session_path, runs_root=tmp_path)

    assert result["status"] == "success"
    [item] = result["results"]
    assert item["turn_index"] == 1
    assert item["assistant_response"] == "认证配置位于 auth/settings.py。"
    assert "tool_calls" not in item
    assert result["diagnostics"]["scope"] == "current_session"


def test_session_history_requires_current_session_injection(tmp_path: Path) -> None:
    result = session_history({"query": "认证"}, None, runs_root=tmp_path)

    assert result["status"] == "error"
    assert result["error"]["type"] == "session_history_unavailable"


def test_session_history_router_writes_query_and_selection_to_episode_trace(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turn(session_path)
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "goal": "history test",
                "allowed_tools": ["session_history"],
                "acceptance_criteria": [],
                "verification_commands": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)
    router = ToolRouter(
        ["session_history"],
        writer,
        workspace_root=tmp_path,
        session_path=session_path,
        runs_root=tmp_path,
    )

    result = router.dispatch("session_history", {"query": "认证配置", "limit": 1})

    assert result["status"] == "success"
    assert "_trace_result" not in result
    [trace] = [
        json.loads(line)
        for line in (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert trace["tool_name"] == "session_history"
    assert trace["args"] == {"query": "认证配置", "limit": 1}
    assert trace["result"]["diagnostics"]["selected_turns"] == [1]
    assert "results" not in trace["result"]
