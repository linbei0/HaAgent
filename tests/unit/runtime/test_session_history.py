"""
tests/unit/runtime/test_session_history.py - 当前会话历史检索测试

验证 session_history 只读取当前 session 的对话证据，并保持 episode 细节边界。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from haagent.runtime.session.history import SessionHistoryError, SessionHistoryRetriever


def _write_turns(session_path: Path, rows: list[dict[str, object]]) -> None:
    session_path.mkdir(parents=True)
    (session_path / "turns.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _turn(
    index: int,
    *,
    request: str,
    summary: str,
    response: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "turn_index": index,
        "request": request,
        "summary": summary,
        "status": "completed",
        "episode_path": f"episodes/turn-{index}",
        "verification_status": "success",
    }
    if response is not None:
        row["assistant_display_text"] = response
    return row


def test_search_ranks_relevance_then_newer_turn(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(1, request="检查认证模块", summary="认证模块使用 OAuth。", response="已找到配置。"),
            _turn(2, request="继续认证模块", summary="认证模块使用 OAuth，配置在 auth/settings.py。", response="路径是 auth/settings.py。"),
            _turn(3, request="整理文档", summary="无认证内容。", response="完成文档整理。"),
        ],
    )

    result = SessionHistoryRetriever(session_path).search("认证模块 OAuth")

    assert [item.turn_index for item in result.results] == [2, 1]
    assert result.results[0].assistant_response == "路径是 auth/settings.py。"


def test_search_matches_natural_chinese_query_path_and_symbol_tokens(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(
                1,
                request="修复 src/haagent/context/builder.py 的缓存",
                summary="为 ContextBuilder 添加缓存诊断。",
                response="已修改 src/haagent/context/builder.py。",
            ),
        ],
    )

    result = SessionHistoryRetriever(session_path).search("之前 builder.py 的缓存最后是怎么处理的")

    assert [item.turn_index for item in result.results] == [1]


def test_search_returns_only_dialogue_evidence(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(
                1,
                request="运行测试",
                summary="pytest 成功。",
                response="测试已通过，未展示工具参数。",
            ),
        ],
    )

    [item] = SessionHistoryRetriever(session_path).search("pytest").results

    assert set(item.to_dict()) == {
        "turn_index",
        "request",
        "summary",
        "assistant_response",
        "status",
        "verification_status",
        "episode_ref",
    }


def test_search_finds_query_only_present_after_truncated_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(
                1,
                request="旧请求... [truncated]",
                summary="保留了一条旧记录。",
                response="已完成。",
            ),
        ],
    )

    monkeypatch.setattr(
        "haagent.runtime.session.history.load_task",
        lambda _: SimpleNamespace(goal="前置说明" * 200 + "认证模块最终采用 OAuth PKCE。"),
    )

    [item] = SessionHistoryRetriever(session_path, runs_root=tmp_path).search(
        "之前认证模块最后怎么决定的",
    ).results

    assert item.turn_index == 1
    assert "认证模块最终采用 OAuth PKCE" in item.request


def test_search_uses_query_centered_episode_response_when_display_text_is_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(
                1,
                request="确认部署决定",
                summary="保留了一条旧记录。",
                response="旧展示文本... [truncated]",
            ),
        ],
    )

    class _Package:
        def final_response_text(self) -> str:
            return "前置分析" * 700 + "最终部署决定：使用蓝绿发布。" + "后续说明" * 700

    monkeypatch.setattr(
        "haagent.runtime.session.history.load_inspect_episode_package",
        lambda _: _Package(),
    )

    [item] = SessionHistoryRetriever(session_path, runs_root=tmp_path).search("蓝绿发布").results

    assert "最终部署决定：使用蓝绿发布。" in item.assistant_response
    assert len(item.assistant_response) <= 4_000


def test_search_does_not_read_episode_for_complete_turn_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [_turn(1, request="检查缓存", summary="缓存决定已记录。", response="使用本地缓存。")],
    )
    monkeypatch.setattr(
        "haagent.runtime.session.history.load_inspect_episode_package",
        lambda _: pytest.fail("complete turn evidence must not load episode"),
    )

    [item] = SessionHistoryRetriever(session_path, runs_root=tmp_path).search("缓存").results

    assert item.assistant_response == "使用本地缓存。"


def test_search_returns_valid_results_when_one_episode_is_damaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(1, request="发布", summary="发布记录。", response="旧内容... [truncated]"),
            _turn(2, request="发布", summary="发布记录。", response="旧内容... [truncated]"),
        ],
    )

    class _Package:
        def final_response_text(self) -> str:
            return "发布最终使用蓝绿方案。"

    def _load(path: Path):
        assert path.parent == tmp_path / "episodes"
        if path.name == "turn-1":
            raise OSError("damaged")
        return _Package()

    monkeypatch.setattr("haagent.runtime.session.history.load_inspect_episode_package", _load)

    result = SessionHistoryRetriever(session_path, runs_root=tmp_path).search("发布", limit=2)

    assert [item.turn_index for item in result.results] == [2, 1]
    assert result.diagnostics["failed_episode_turns"] == [1]
    assert result.results[1].assistant_response.endswith("... [truncated]")


def test_search_returns_empty_result_for_empty_session(tmp_path: Path) -> None:
    result = SessionHistoryRetriever(tmp_path / "session").search("anything")

    assert result.results == []
    assert result.diagnostics["turn_count"] == 0


def test_search_skips_turn_when_missing_episode_leaves_no_assistant_response(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    _write_turns(
        session_path,
        [
            _turn(
                1,
                request="找回发布结论",
                summary="发布结论已压缩。",
                response=None,
            ),
        ],
    )

    result = SessionHistoryRetriever(session_path, runs_root=tmp_path).search("发布结论")

    assert result.results == []
    assert result.diagnostics["skipped_turns"] == [1]


def test_search_rejects_malformed_turn_record(tmp_path: Path) -> None:
    session_path = tmp_path / "session"
    session_path.mkdir()
    (session_path / "turns.jsonl").write_text("not json\n", encoding="utf-8")

    with pytest.raises(SessionHistoryError, match="invalid turns.jsonl line 1"):
        SessionHistoryRetriever(session_path).search("anything")
