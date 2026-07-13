"""
src/haagent/models/gateway_retry.py - 模型网关重试适配

将模型错误和流式提交状态接入统一 RetryController，不在网关层重复实现重试循环。
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelCallError, ModelFailureDetails
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.retry import (
    ReplaySafety,
    RetryController,
    RetryEvent,
    RetryFailure,
    RetryOperation,
    RetryableOperationError,
    StreamAttemptState,
)


ResultT = TypeVar("ResultT")


def default_retry_controller(controller: RetryController | None) -> RetryController:
    """兼容旧 gateway 构造方式，同时允许 session 注入其唯一 controller。"""

    return controller or RetryController()


def execute_model_request(
    controller: RetryController,
    *,
    provider: str,
    invoke: Callable[[Callable[[str], None] | None, int], ResultT],
    event_sink: Callable[[str], None] | None,
    cancellation_token: CancellationToken | None,
    retry_event_sink: Callable[[RetryEvent], None] | None,
    retry_exhausted_sink: Callable[[RetryFailure, int], None] | None,
    telemetry_sink: Callable[[ModelTransportEvent], None] | None = None,
) -> ResultT:
    """执行一次模型请求；流式首个 delta 后将失败标记为不可重放。"""

    # 控制器持有同一对象；attempt 间只重置 committed，避免重放已展示文本。
    stream_state = StreamAttemptState() if event_sink is not None else None
    attempt = 0
    attempt_started_at = 0.0
    first_text_emitted = False

    def publish(kind: str, *, request_payload_bytes: int | None = None) -> None:
        if telemetry_sink is None:
            return
        elapsed_ms = max(0.0, (time.perf_counter() - attempt_started_at) * 1000.0)
        telemetry_sink(
            ModelTransportEvent(
                kind=kind,  # type: ignore[arg-type]
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                request_payload_bytes=request_payload_bytes,
            ),
        )

    def invoke_attempt() -> ResultT:
        nonlocal attempt, attempt_started_at, first_text_emitted
        attempt += 1
        attempt_started_at = time.perf_counter()
        first_text_emitted = False
        if stream_state is not None:
            stream_state.committed = False
        publish("attempt_started")
        try:
            if event_sink is None or stream_state is None:
                result = invoke(None, attempt)
            else:
                def on_delta(delta: str) -> None:
                    nonlocal first_text_emitted
                    if delta and not first_text_emitted:
                        first_text_emitted = True
                        publish("first_text")
                    assert stream_state is not None
                    stream_state.emit(delta, event_sink)

                result = invoke(on_delta, attempt)
        except Exception:
            publish("attempt_failed")
            raise
        publish("attempt_finished")
        return result

    scheduled_attempts = 0

    def on_retry(event: RetryEvent) -> None:
        nonlocal scheduled_attempts
        scheduled_attempts = event.next_attempt - 1
        if retry_event_sink is not None:
            retry_event_sink(event)

    try:
        return controller.execute(
            RetryOperation(
                name=f"{provider}.generate",
                replay_safety=ReplaySafety.SAFE_TO_REPLAY,
                streaming=event_sink is not None,
            ),
            invoke_attempt,
            cancellation_token=cancellation_token,
            on_event=on_retry,
            stream_state=stream_state,
            error_adapter=_model_error_retry_failure,
        )
    except RetryableOperationError as error:
        if error.failure.category != "stream_interrupted":
            raise
        if retry_exhausted_sink is not None:
            # 该回调同时承载最终 attempt 证据；编排层会把流式中断与预算耗尽区分展示。
            retry_exhausted_sink(error.failure, scheduled_attempts + 1)
        raise ModelCallError(
            "model stream interrupted after partial output",
            details=ModelFailureDetails(
                category="stream_interrupted",
                status_code=error.failure.status_code,
                provider_code=error.failure.provider_code,
                request_id=error.failure.request_id,
                retryable=False,
            ),
        ) from error
    except ModelCallError as error:
        if error.details is not None and error.details.retryable and retry_exhausted_sink is not None:
            retry_exhausted_sink(error.details.to_retry_failure(), scheduled_attempts + 1)
        raise


def stream_delta_sink(
    stream_state: StreamAttemptState,
    event_sink: Callable[[str], None],
) -> Callable[[str], None]:
    """把 provider delta 回调接到本 attempt 的提交状态。"""

    return lambda delta: stream_state.emit(delta, event_sink)


def _model_error_retry_failure(error: Exception) -> RetryFailure | None:
    if isinstance(error, ModelCallError) and error.details is not None:
        return error.details.to_retry_failure()
    return None


def unexpected_model_error(error: Exception, *, message: str = "model request failed") -> ModelCallError:
    """未知 provider 异常只暴露固定摘要，避免把凭据写入事件和 episode。"""

    return ModelCallError(message)
