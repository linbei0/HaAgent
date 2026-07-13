"""
tests/unit/models/test_retry.py - 统一重试内核测试

验证唯一重试控制器的分类、取消、流式提交与模型失败适配边界。
"""

import httpx
import pytest

from haagent.models import transport
from haagent.models.http_transport import ModelHttpTransport
from haagent.models.transport import _invoke_transport
from haagent.models.types import ModelFailureDetails
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import (
    ReplaySafety,
    RetryableOperationError,
    RetryController,
    RetryFailure,
    RetryOperation,
    RetryPolicy,
    StreamAttemptState,
)


def test_provider_transport_internal_type_error_is_not_reinvoked() -> None:
    calls = 0

    def broken_transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("internal parser bug")

    with pytest.raises(TypeError, match="internal parser bug"):
        _invoke_transport(
            broken_transport,
            broken_transport,
            {},
            "key",
            on_delta=None,
            attempt=1,
            telemetry_sink=None,
        )

    assert calls == 1


def test_retries_rate_limit_with_jittered_delay() -> None:
    calls = 0
    delays: list[float] = []
    events = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableOperationError(
                RetryFailure(category="rate_limited", status_code=429, retryable=True)
            )
        return "ok"

    result = RetryController(
        RetryPolicy(max_attempts=3),
        sleep=delays.append,
        random_value=lambda: 0.5,
    ).execute(
        RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
        operation,
        cancellation_token=None,
        on_event=events.append,
    )

    assert result == "ok"
    assert calls == 2
    assert sum(delays) == pytest.approx(6)
    assert all(0 < delay_seconds <= 0.1 for delay_seconds in delays)
    assert events[0].next_attempt == 2
    assert events[0].source == "backoff"


def test_automatic_backoff_uses_an_increasing_jitter_range() -> None:
    calls = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableOperationError(RetryFailure(category="network", retryable=True))
        return "ok"

    events = []
    RetryController(
        RetryPolicy(base_delay_seconds=2, minimum_delay_seconds=2, max_delay_seconds=30),
        sleep=delays.append,
        random_value=lambda: 0.5,
    ).execute(
        RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
        operation,
        on_event=events.append,
    )

    assert [event.delay_seconds for event in events] == pytest.approx([3, 6])


@pytest.mark.parametrize("category", ["auth", "quota_exhausted"])
def test_does_not_retry_auth_or_quota_failure(category: str) -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RetryableOperationError(RetryFailure(category=category, retryable=False))

    with pytest.raises(RetryableOperationError):
        RetryController(RetryPolicy()).execute(
            RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
            operation,
            cancellation_token=None,
        )

    assert calls == 1


def test_retry_after_within_policy_limit_wins_over_backoff() -> None:
    calls = 0
    delays: list[float] = []
    events = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableOperationError(
                RetryFailure(category="rate_limited", retryable=True, retry_after_seconds=3)
            )
        return "ok"

    result = RetryController(
        RetryPolicy(), sleep=delays.append, random_value=lambda: 0.0
    ).execute(
        RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
        operation,
        cancellation_token=None,
        on_event=events.append,
    )

    assert result == "ok"
    assert sum(delays) == pytest.approx(3)
    assert all(0 < delay_seconds <= 0.1 for delay_seconds in delays)
    assert events[0].source == "retry_after"


@pytest.mark.parametrize("retry_after_seconds", [0, 61])
def test_invalid_or_excessive_retry_after_records_fallback_fact(
    retry_after_seconds: float,
) -> None:
    calls = 0
    delays: list[float] = []
    events = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableOperationError(
                RetryFailure(
                    category="server",
                    retryable=True,
                    retry_after_seconds=retry_after_seconds,
                )
            )
        return "ok"

    result = RetryController(
        RetryPolicy(), sleep=delays.append, random_value=lambda: 0.5
    ).execute(
        RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
        operation,
        cancellation_token=None,
        on_event=events.append,
    )

    assert result == "ok"
    assert sum(delays) == pytest.approx(3)
    assert all(0 < delay_seconds <= 0.1 for delay_seconds in delays)
    assert events[0].source == "backoff"
    assert events[0].retry_after_ignored is True


def test_max_attempts_one_performs_one_call() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RetryableOperationError(RetryFailure(category="server", retryable=True))

    with pytest.raises(RetryableOperationError):
        RetryController(RetryPolicy(max_attempts=1)).execute(
            RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
            operation,
            cancellation_token=None,
        )

    assert calls == 1


def test_cancellation_during_wait_checks_each_delay_slice() -> None:
    calls = 0
    token = CancellationToken()
    delays: list[float] = []

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RetryableOperationError(RetryFailure(category="server", retryable=True))

    def cancel_during_second_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)
        if len(delays) == 2:
            token.cancel()

    with pytest.raises(RunCancelled):
        RetryController(
            RetryPolicy(base_delay_seconds=0.25),
            sleep=cancel_during_second_sleep,
            random_value=lambda: 1.0,
        ).execute(
            RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY),
            operation,
            cancellation_token=token,
        )

    assert calls == 1
    assert delays == [0.1, 0.1]


def test_idempotency_key_operation_without_provider_support_runs_once() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise RetryableOperationError(RetryFailure(category="server", retryable=True))

    with pytest.raises(RetryableOperationError):
        RetryController(RetryPolicy()).execute(
            RetryOperation(
                "model.generate",
                ReplaySafety.IDEMPOTENCY_KEY_REQUIRED,
                idempotency_key="stable-key",
            ),
            operation,
            cancellation_token=None,
        )

    assert calls == 1


def test_stream_can_retry_before_first_delta() -> None:
    calls = 0
    stream_state = StreamAttemptState()

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableOperationError(RetryFailure(category="network", retryable=True))
        return "ok"

    result = RetryController(RetryPolicy(), sleep=lambda _: None).execute(
        RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY, streaming=True),
        operation,
        cancellation_token=None,
        stream_state=stream_state,
    )

    assert result == "ok"
    assert calls == 2


def test_committed_state_does_not_change_non_streaming_operation() -> None:
    calls = 0
    stream_state = StreamAttemptState()
    stream_state.emit("ignored", lambda _: None)

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableOperationError(RetryFailure(category="network", retryable=True))
        return "ok"

    result = RetryController(RetryPolicy(), sleep=lambda _: None).execute(
        RetryOperation("tool.read", ReplaySafety.SAFE_TO_REPLAY, streaming=False),
        operation,
        cancellation_token=None,
        stream_state=stream_state,
    )

    assert result == "ok"
    assert calls == 2


def test_committed_stream_interrupt_is_not_replayed() -> None:
    calls = 0
    deltas: list[str] = []
    stream_state = StreamAttemptState()

    def operation() -> None:
        nonlocal calls
        calls += 1
        stream_state.emit("partial", deltas.append)
        raise RetryableOperationError(RetryFailure(category="network", retryable=True))

    with pytest.raises(RetryableOperationError) as raised:
        RetryController(RetryPolicy()).execute(
            RetryOperation("model.generate", ReplaySafety.SAFE_TO_REPLAY, streaming=True),
            operation,
            cancellation_token=None,
            stream_state=stream_state,
        )

    assert calls == 1
    assert deltas == ["partial"]
    assert raised.value.failure.category == "stream_interrupted"
    assert raised.value.failure.retryable is False


def test_model_failure_details_adapt_without_losing_safe_facts() -> None:
    details = ModelFailureDetails(
        category="server",
        status_code=503,
        provider_code="overloaded",
        retry_after_seconds=2,
        request_id="req_123",
        retryable=True,
    )

    assert details.to_retry_failure() == RetryFailure(
        category="server",
        status_code=503,
        provider_code="overloaded",
        retry_after_seconds=2,
        request_id="req_123",
        retryable=True,
    )


def _failing_http_transport(status_code: int = 503, content: bytes = b"{}") -> ModelHttpTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=content,
            headers={"content-type": "application/json", "Authorization": "Bearer secret"},
            request=request,
        )

    return ModelHttpTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    "call_transport",
    [
        lambda http: transport._responses_transport({}, "secret", http_transport=http),
        lambda http: transport._chat_completions_transport({}, "secret", http_transport=http),
        lambda http: transport._anthropic_transport({}, "secret", http_transport=http),
        lambda http: transport._google_gemini_transport(
            {}, "secret", "https://gemini.example", http_transport=http
        ),
        lambda http: transport._responses_stream_transport(
            {}, "secret", "https://openai.example", lambda _: None, http_transport=http
        ),
        lambda http: transport._chat_completions_stream_transport(
            {}, "secret", "https://openai.example", lambda _: None, http_transport=http
        ),
        lambda http: transport._anthropic_stream_transport(
            {}, "secret", "https://anthropic.example", lambda _: None, http_transport=http
        ),
        lambda http: transport._google_gemini_stream_transport(
            {}, "secret", "https://gemini.example", lambda _: None, http_transport=http
        ),
    ],
)
def test_all_transport_entrypoints_preserve_structured_http_failure(call_transport) -> None:
    http = _failing_http_transport(503)
    try:
        with pytest.raises(transport.ModelCallError) as raised:
            call_transport(http)
    finally:
        http.close()

    assert isinstance(raised.value, transport.ModelCallError)
    assert raised.value.details is not None
    assert raised.value.details.category == "server"
    assert raised.value.details.retryable is True
    assert "secret" not in str(raised.value)


def test_transport_marks_invalid_json_as_non_retryable_response_parse_error() -> None:
    http = _failing_http_transport(200, content=b"not-json")
    try:
        with pytest.raises(transport.ModelCallError) as raised:
            transport._responses_transport({}, "secret", http_transport=http)
    finally:
        http.close()

    assert raised.value.details is not None
    assert raised.value.details.category == "response_parse"
    assert raised.value.details.retryable is False
