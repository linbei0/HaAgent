"""
tests/unit/models/test_sse_stream.py - SSE 流式解析回归

验证 transport 在 HTTP 响应结束前就会 yield / 回调首个 delta，
防止再次把 stream=true 错误缓冲成整包响应。
"""

from __future__ import annotations

import threading

import pytest

from haagent.models import transport


def test_iter_sse_events_yields_before_response_is_exhausted() -> None:
    """完整 event 一旦就绪就应 yield；不得等整个 response 读完。"""

    release_tail = threading.Event()
    first_yielded = threading.Event()
    errors: list[BaseException] = []
    events: list[dict[str, object]] = []

    def blocking_response():
        yield b'data: {"id": "first"}\n'
        yield b"\n"
        # 首个 event 结束后故意阻塞；若实现先收集 list，next() 会卡在这里。
        if not release_tail.wait(timeout=2.0):
            raise TimeoutError("response tail was never released")
        yield b'data: {"id": "second"}\n'
        yield b"\n"

    def consume() -> None:
        try:
            iterator = transport._iter_sse_events(blocking_response())
            first = next(iterator)
            events.append(first)
            first_yielded.set()
            second = next(iterator)
            events.append(second)
        except BaseException as error:  # pragma: no cover - 测试辅助线程
            errors.append(error)
            first_yielded.set()

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    assert first_yielded.wait(timeout=1.0), "first SSE event was not yielded before response ended"
    assert events == [{"id": "first"}]
    release_tail.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert errors == []
    assert events == [{"id": "first"}, {"id": "second"}]


def test_parse_openai_chat_stream_calls_on_delta_before_response_ends() -> None:
    release_tail = threading.Event()
    first_delta = threading.Event()
    deltas: list[str] = []
    errors: list[BaseException] = []
    result_box: dict[str, object] = {}

    def blocking_response():
        yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        if not release_tail.wait(timeout=2.0):
            raise TimeoutError("response tail was never released")
        yield b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def on_delta(text: str) -> None:
        deltas.append(text)
        if len(deltas) == 1:
            first_delta.set()

    def consume() -> None:
        try:
            result_box["parsed"] = transport._parse_openai_chat_stream(blocking_response(), on_delta)
        except BaseException as error:  # pragma: no cover - 测试辅助线程
            errors.append(error)
            first_delta.set()

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    assert first_delta.wait(timeout=1.0), "on_delta was not called before response ended"
    assert deltas == ["Hi"]
    release_tail.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert errors == []
    assert deltas == ["Hi", " there"]
    parsed = result_box["parsed"]
    assert isinstance(parsed, dict)
    choices = parsed["choices"]
    assert isinstance(choices, list)
    message = choices[0]["message"]
    assert message["content"] == "Hi there"


def test_iter_sse_events_skips_done_and_parses_multi_data_lines() -> None:
    response = [
        b'data: {"a": 1}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
        # 多 data 行按 SSE 规范用 \n 拼接；拼后必须仍是合法 JSON 对象。
        b'data: {"text": "line1",\n',
        b'data: "num": 2}\n',
        b"\n",
    ]
    assert list(transport._iter_sse_events(response)) == [{"a": 1}, {"text": "line1", "num": 2}]


def test_iter_sse_events_handles_chunk_that_includes_event_separator() -> None:
    """单 chunk 内含 data + \\n\\n 时也必须立即形成完整 event。"""

    response = [
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
    ]
    assert list(transport._iter_sse_events(response)) == [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
    ]
