"""
src/haagent/runtime/performance.py - 有界交互性能轨迹

记录 run 各阶段、模型 attempt 与工具耗时；只存数字/枚举/hash，不写敏感正文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelUsage


@dataclass
class _ModelAttemptRecord:
    attempt: int
    request_payload_bytes: int = 0
    request_to_headers_ms: float | None = None
    time_to_first_sse_ms: float | None = None
    time_to_first_text_ms: float | None = None
    stream_duration_ms: float | None = None
    total_model_call_ms: float | None = None
    status: str | None = None
    _first_sse_at_ms: float | None = None


@dataclass
class _ModelTurnRecord:
    turn: int
    message_count: int
    visible_tool_count: int
    tool_schema_bytes: int
    stable_prefix_fingerprint: str
    input_tokens: int | None = None
    attempts: list[_ModelAttemptRecord] = field(default_factory=list)


@dataclass
class _ToolRecord:
    turn: int
    tool_name: str
    duration_ms: float
    execution_effect: str
    status: str


@dataclass
class PerformanceTrace:
    """内存中的有界 performance 轨迹；达到上限时保留旧证据并递增 dropped。"""

    performance_schema_version: str = "1.0"
    max_model_turns: int = 64
    max_model_attempts: int = 64
    max_tools: int = 256
    _clock: Callable[[], float] = field(default=time.perf_counter, repr=False)
    _started_at: float = 0.0
    _run_start_at: float | None = None
    _context_build_start_at: float | None = None
    _context_built_at: float | None = None
    _postprocess_start_at: float | None = None
    _finished_at: float | None = None
    _status: str | None = None
    _model_turns: list[_ModelTurnRecord] = field(default_factory=list)
    _tools: list[_ToolRecord] = field(default_factory=list)
    _dropped_model_attempts: int = 0
    _dropped_model_turns: int = 0
    _dropped_tools: int = 0
    _cache_diagnostics: dict[str, dict[str, object]] = field(default_factory=dict)
    _current_turn: _ModelTurnRecord | None = None
    _total_attempt_slots_used: int = 0

    @classmethod
    def start(
        cls,
        clock: Callable[[], float] = time.perf_counter,
        *,
        max_model_turns: int = 64,
        max_model_attempts: int = 64,
        max_tools: int = 256,
    ) -> PerformanceTrace:
        now = clock()
        return cls(
            max_model_turns=max_model_turns,
            max_model_attempts=max_model_attempts,
            max_tools=max_tools,
            _clock=clock,
            _started_at=now,
        )

    def mark_run_start(self) -> None:
        """标记 orchestrator/run 真正开始，用于 submit_to_run_start 与 run_setup。"""

        self._run_start_at = self._clock()

    def mark_context_built(self) -> None:
        """标记初始 context 构建完成。"""

        self._context_built_at = self._clock()

    def mark_context_build_start(self) -> None:
        """标记 context build 起点，使 run setup 与 context build 可独立解释。"""

        self._context_build_start_at = self._clock()

    def mark_postprocess_start(self) -> None:
        """最后一个模型/工具事件后开始 postprocess 计时。"""

        self._postprocess_start_at = self._clock()

    def begin_model_turn(
        self,
        turn: int,
        message_count: int,
        visible_tool_count: int,
        schema_bytes: int,
        stable_prefix_fingerprint: str,
    ) -> None:
        if len(self._model_turns) >= self.max_model_turns:
            self._dropped_model_turns += 1
            self._current_turn = None
            return
        record = _ModelTurnRecord(
            turn=turn,
            message_count=message_count,
            visible_tool_count=visible_tool_count,
            tool_schema_bytes=schema_bytes,
            stable_prefix_fingerprint=stable_prefix_fingerprint,
        )
        self._model_turns.append(record)
        self._current_turn = record

    def record_transport_event(self, event: ModelTransportEvent) -> None:
        """归并单次 attempt 的 transport 边界；达到上限时丢弃新 attempt 并计数。"""

        turn = self._current_turn
        if turn is None:
            return
        attempt = self._find_or_create_attempt(turn, event.attempt)
        if attempt is None:
            return
        if event.kind == "attempt_started":
            if attempt.status is None:
                attempt.status = "running"
            return
        if event.kind == "request_prepared":
            if event.request_payload_bytes is not None:
                attempt.request_payload_bytes = event.request_payload_bytes
            return
        if event.kind == "headers_received":
            attempt.request_to_headers_ms = _non_negative(event.elapsed_ms)
            return
        if event.kind == "first_sse":
            attempt.time_to_first_sse_ms = _non_negative(event.elapsed_ms)
            attempt._first_sse_at_ms = event.elapsed_ms
            return
        if event.kind == "first_text":
            attempt.time_to_first_text_ms = _non_negative(event.elapsed_ms)
            return
        if event.kind in {"attempt_finished", "attempt_failed"}:
            attempt.total_model_call_ms = _non_negative(event.elapsed_ms)
            if attempt._first_sse_at_ms is not None:
                attempt.stream_duration_ms = _non_negative(
                    event.elapsed_ms - attempt._first_sse_at_ms,
                )
            attempt.status = "completed" if event.kind == "attempt_finished" else "failed"

    def record_model_usage(self, turn: int, usage: ModelUsage | None) -> None:
        if usage is None:
            return
        for record in self._model_turns:
            if record.turn == turn:
                record.input_tokens = usage.input_tokens
                return

    def record_tool(
        self,
        turn: int,
        tool_name: str,
        duration_ms: float,
        execution_effect: str,
        status: str,
    ) -> None:
        if len(self._tools) >= self.max_tools:
            self._dropped_tools += 1
            return
        self._tools.append(
            _ToolRecord(
                turn=turn,
                tool_name=tool_name,
                duration_ms=_non_negative(duration_ms),
                execution_effect=execution_effect,
                status=status,
            ),
        )

    def record_cache_diagnostic(self, component: str, value: dict[str, object]) -> None:
        """记录固定字段的 cache 诊断；不接受正文、路径或任意嵌套 payload。"""

        if component not in {"instructions", "skills", "tool_schema"}:
            return
        allowed = {"status", "count", "chars", "bytes", "fingerprint"}
        self._cache_diagnostics[component] = {
            key: item for key, item in value.items() if key in allowed
        }

    def finish(self, status: str) -> None:
        self._status = status
        self._finished_at = self._clock()

    def to_dict(self) -> dict[str, object]:
        run_setup_ms: float | None = None
        context_build_ms: float | None = None
        submit_to_run_start_ms: float | None = None
        postprocess_ms: float | None = None
        total_turn_ms: float | None = None

        if self._run_start_at is not None:
            submit_to_run_start_ms = _ms_delta(self._started_at, self._run_start_at)
        if self._run_start_at is not None and self._context_build_start_at is not None:
            run_setup_ms = _ms_delta(self._run_start_at, self._context_build_start_at)
        if self._context_build_start_at is not None and self._context_built_at is not None:
            context_build_ms = _ms_delta(self._context_build_start_at, self._context_built_at)
        if self._postprocess_start_at is not None and self._finished_at is not None:
            postprocess_ms = _ms_delta(self._postprocess_start_at, self._finished_at)
        if self._finished_at is not None:
            total_turn_ms = _ms_delta(self._started_at, self._finished_at)

        model_turns: list[dict[str, object]] = []
        for turn in self._model_turns:
            attempts = [
                {
                    "attempt": item.attempt,
                    "request_payload_bytes": item.request_payload_bytes,
                    "request_to_headers_ms": item.request_to_headers_ms,
                    "time_to_first_sse_ms": item.time_to_first_sse_ms,
                    "time_to_first_text_ms": item.time_to_first_text_ms,
                    "stream_duration_ms": item.stream_duration_ms,
                    "total_model_call_ms": item.total_model_call_ms,
                    "status": item.status or "running",
                }
                for item in turn.attempts
            ]
            model_turns.append(
                {
                    "turn": turn.turn,
                    "message_count": turn.message_count,
                    "visible_tool_count": turn.visible_tool_count,
                    "tool_schema_bytes": turn.tool_schema_bytes,
                    "stable_prefix_fingerprint": turn.stable_prefix_fingerprint,
                    "input_tokens": turn.input_tokens,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                },
            )

        tools = [
            {
                "turn": item.turn,
                "tool_name": item.tool_name,
                "duration_ms": item.duration_ms,
                "execution_effect": item.execution_effect,
                "status": item.status,
            }
            for item in self._tools
        ]

        return {
            "performance_schema_version": self.performance_schema_version,
            "submit_to_run_start_ms": submit_to_run_start_ms,
            "run_setup_ms": run_setup_ms,
            "context_build_ms": context_build_ms,
            "model_turns": model_turns,
            "tools": tools,
            "cache_diagnostics": dict(self._cache_diagnostics),
            "postprocess_ms": postprocess_ms,
            "total_turn_ms": total_turn_ms,
            "status": self._status,
            "dropped": {
                "model_turns": self._dropped_model_turns,
                "model_attempts": self._dropped_model_attempts,
                "tools": self._dropped_tools,
            },
        }

    def _find_or_create_attempt(
        self,
        turn: _ModelTurnRecord,
        attempt_number: int,
    ) -> _ModelAttemptRecord | None:
        for item in turn.attempts:
            if item.attempt == attempt_number:
                return item
        # 全局 attempt 上限：保留已有证据，新 attempt 只递增 dropped。
        if self._total_attempt_slots_used >= self.max_model_attempts:
            self._dropped_model_attempts += 1
            return None
        record = _ModelAttemptRecord(attempt=attempt_number)
        turn.attempts.append(record)
        self._total_attempt_slots_used += 1
        return record


def _ms_delta(start: float, end: float) -> float:
    return _non_negative((end - start) * 1000.0)


def _non_negative(value: float) -> float:
    return value if value >= 0.0 else 0.0
