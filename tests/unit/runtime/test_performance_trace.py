"""
tests/unit/runtime/test_performance_trace.py - PerformanceTrace 契约测试

锁定 schema、单调时钟计时语义、有界记录与敏感字段排除。
"""

from __future__ import annotations

import pytest

from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelUsage
from haagent.runtime.orchestration.recorder import RunRecorder
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.performance import PerformanceTrace


def test_performance_trace_uses_monotonic_clock_and_bounds_records() -> None:
    ticks = iter([10.0, 10.010, 10.020, 10.025])
    trace = PerformanceTrace.start(clock=lambda: next(ticks), max_model_attempts=1, max_tools=1)
    trace.mark_run_start()
    trace.mark_context_build_start()
    trace.mark_context_built()
    trace.record_tool(
        turn=1,
        tool_name="file_read",
        duration_ms=4.0,
        execution_effect="read_only",
        status="success",
    )
    trace.record_tool(
        turn=1,
        tool_name="grep",
        duration_ms=5.0,
        execution_effect="read_only",
        status="success",
    )
    value = trace.to_dict()
    assert value["performance_schema_version"] == "1.0"
    assert value["run_setup_ms"] == pytest.approx(10.0)
    assert value["context_build_ms"] == pytest.approx(5.0)
    assert len(value["tools"]) == 1
    assert value["dropped"]["tools"] == 1


def test_performance_trace_records_independent_model_attempts_and_excludes_secrets() -> None:
    ticks = iter([100.0, 100.001, 100.010, 100.020, 100.030, 100.050, 100.060, 100.080, 100.100])
    trace = PerformanceTrace.start(clock=lambda: next(ticks), max_model_attempts=2, max_tools=8)
    trace.mark_run_start()
    trace.mark_context_built()
    trace.begin_model_turn(
        turn=1,
        message_count=3,
        visible_tool_count=2,
        schema_bytes=128,
        stable_prefix_fingerprint="sha256:" + ("a" * 64),
    )
    # attempt 1: started -> prepared -> headers -> first_sse -> first_text -> finished
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_started", attempt=1, elapsed_ms=0.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(
            kind="request_prepared",
            attempt=1,
            elapsed_ms=1.0,
            request_payload_bytes=512,
        ),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="headers_received", attempt=1, elapsed_ms=10.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="first_sse", attempt=1, elapsed_ms=20.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="first_text", attempt=1, elapsed_ms=30.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_failed", attempt=1, elapsed_ms=40.0),
    )
    # attempt 2 completes
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_started", attempt=2, elapsed_ms=0.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(
            kind="request_prepared",
            attempt=2,
            elapsed_ms=2.0,
            request_payload_bytes=520,
        ),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_finished", attempt=2, elapsed_ms=50.0),
    )
    trace.record_model_usage(
        turn=1,
        usage=ModelUsage(input_tokens=120, output_tokens=40, total_tokens=160),
    )
    trace.mark_postprocess_start()
    trace.finish(status="completed")
    value = trace.to_dict()

    assert value["model_turns"][0]["turn"] == 1
    assert value["model_turns"][0]["attempt_count"] == 2
    assert value["model_turns"][0]["input_tokens"] == 120
    assert len(value["model_turns"][0]["attempts"]) == 2
    assert value["model_turns"][0]["attempts"][0]["attempt"] == 1
    assert value["model_turns"][0]["attempts"][0]["status"] == "failed"
    assert value["model_turns"][0]["attempts"][0]["request_to_headers_ms"] == pytest.approx(10.0)
    assert value["model_turns"][0]["attempts"][0]["time_to_first_sse_ms"] == pytest.approx(20.0)
    assert value["model_turns"][0]["attempts"][0]["time_to_first_text_ms"] == pytest.approx(30.0)
    assert value["model_turns"][0]["attempts"][1]["status"] == "completed"
    assert value["status"] == "completed"
    assert value["postprocess_ms"] is not None
    assert value["total_turn_ms"] is not None

    serialized = str(value)
    for forbidden in ("prompt", "arguments", "response_body", "api_key", "Authorization"):
        assert forbidden not in serialized


def test_performance_trace_drops_extra_model_attempts_without_erasing_existing() -> None:
    ticks = iter([1.0 + i * 0.001 for i in range(20)])
    trace = PerformanceTrace.start(clock=lambda: next(ticks), max_model_attempts=1, max_tools=8)
    trace.begin_model_turn(
        turn=1,
        message_count=1,
        visible_tool_count=0,
        schema_bytes=0,
        stable_prefix_fingerprint="sha256:" + ("b" * 64),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_started", attempt=1, elapsed_ms=0.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_finished", attempt=1, elapsed_ms=5.0),
    )
    trace.record_transport_event(
        ModelTransportEvent(kind="attempt_started", attempt=2, elapsed_ms=0.0),
    )
    value = trace.to_dict()
    assert len(value["model_turns"][0]["attempts"]) == 1
    assert value["dropped"]["model_attempts"] == 1
    assert value["model_turns"][0]["attempt_count"] == 1


def test_performance_trace_unmeasured_stages_are_none() -> None:
    ticks = iter([5.0, 5.0])
    trace = PerformanceTrace.start(clock=lambda: next(ticks))
    value = trace.to_dict()
    assert value["submit_to_run_start_ms"] is None
    assert value["run_setup_ms"] is None
    assert value["context_build_ms"] is None
    assert value["postprocess_ms"] is None
    assert value["total_turn_ms"] is None
    assert value["status"] is None
    assert value["model_turns"] == []
    assert value["tools"] == []


def test_performance_trace_keeps_unfinished_attempt_duration_unknown() -> None:
    trace = PerformanceTrace.start()
    trace.begin_model_turn(
        turn=1,
        message_count=1,
        visible_tool_count=0,
        schema_bytes=0,
        stable_prefix_fingerprint="sha256:" + ("c" * 64),
    )
    trace.record_transport_event(ModelTransportEvent(kind="attempt_started", attempt=1, elapsed_ms=0.0))

    attempt = trace.to_dict()["model_turns"][0]["attempts"][0]
    assert attempt["total_model_call_ms"] is None


def test_performance_trace_bounds_model_turns() -> None:
    trace = PerformanceTrace.start(max_model_turns=1)
    for turn in (1, 2):
        trace.begin_model_turn(
            turn=turn,
            message_count=1,
            visible_tool_count=0,
            schema_bytes=0,
            stable_prefix_fingerprint="sha256:" + ("d" * 64),
        )

    value = trace.to_dict()
    assert [item["turn"] for item in value["model_turns"]] == [1]
    assert value["dropped"]["model_turns"] == 1


def test_run_recorder_measures_actual_postprocess_work(tmp_path) -> None:
    now = [10.0]
    captured: dict[str, object] = {}

    class Writer:
        path = tmp_path

        def finalize_cost_metadata(self) -> None:
            now[0] += 0.020

        def write_episode_metadata(self, status: str) -> None:
            now[0] += 0.030

        def write_performance(self, value: dict[str, object]) -> None:
            captured.update(value)

    trace = PerformanceTrace.start(clock=lambda: now[0])
    recorder = RunRecorder(Writer(), performance_trace=trace)

    recorder.finish(RunStatus.COMPLETED)

    assert captured["postprocess_ms"] == pytest.approx(50.0)


def test_performance_trace_records_bounded_non_sensitive_cache_diagnostics() -> None:
    trace = PerformanceTrace.start()
    trace.record_cache_diagnostic(
        "skills",
        {
            "status": "hit",
            "count": 2,
            "chars": 80,
            "fingerprint": "sha256:" + ("e" * 64),
        },
    )

    value = trace.to_dict()
    assert value["cache_diagnostics"]["skills"] == {
        "status": "hit",
        "count": 2,
        "chars": 80,
        "fingerprint": "sha256:" + ("e" * 64),
    }
