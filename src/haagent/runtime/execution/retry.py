"""
src/haagent/runtime/execution/retry.py - 统一重试执行内核

集中处理外部操作的重放安全性、退避、取消与重试事件，调用方只提供结构化失败事实。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Callable, TypeVar

from haagent.runtime.execution.cancellation import CancellationToken


ResultT = TypeVar("ResultT")


class ReplaySafety(StrEnum):
    """声明一次操作是否可以由控制器自动重放。"""

    SAFE_TO_REPLAY = "safe_to_replay"
    IDEMPOTENCY_KEY_REQUIRED = "idempotency_key_required"
    NEVER_REPLAY = "never_replay"


@dataclass(frozen=True)
class RetryOperation:
    """供重试控制器决策的操作元数据。"""

    name: str
    replay_safety: ReplaySafety = ReplaySafety.NEVER_REPLAY
    streaming: bool = False
    idempotency_key: str | None = None
    idempotency_supported: bool = False


@dataclass
class StreamAttemptState:
    """记录当前流式 attempt 是否已向调用方提交过增量。"""

    committed: bool = False

    def emit(self, delta: str, sink: Callable[[str], None]) -> None:
        """在转发首个增量前提交状态，防止后续失败被自动重放。"""

        self.committed = True
        sink(delta)


@dataclass(frozen=True)
class RetryFailure:
    """调用方已经脱敏并分类的失败事实。"""

    category: str
    retryable: bool
    retry_after_seconds: float | None = None
    status_code: int | None = None
    provider_code: str | None = None
    request_id: str | None = None


class RetryableOperationError(RuntimeError):
    """携带重试内核可直接消费的失败事实。"""

    def __init__(self, failure: RetryFailure) -> None:
        super().__init__(f"retryable operation failed: {failure.category}")
        self.failure = failure


@dataclass(frozen=True)
class RetryPolicy:
    """控制器的有界退避策略。"""

    max_attempts: int = 3
    minimum_delay_seconds: float = 2.0
    base_delay_seconds: float = 2.0
    throttling_base_delay_seconds: float = 4.0
    max_delay_seconds: float = 30.0
    max_server_retry_after_seconds: float = 60.0


@dataclass(frozen=True)
class RetryEvent:
    """一次已调度重试的非敏感事实。"""

    operation_name: str
    attempt: int
    next_attempt: int
    category: str
    delay_seconds: float
    source: str
    retry_after_ignored: bool = False


def _delay_for(
    failure: RetryFailure,
    retry_index: int,
    policy: RetryPolicy,
    random_value: Callable[[], float],
) -> tuple[float, str]:
    """按服务端等待提示或 full jitter 计算下一次尝试的延迟。"""

    retry_after = failure.retry_after_seconds
    if retry_after is not None and 0 < retry_after <= policy.max_server_retry_after_seconds:
        return retry_after, "retry_after"
    base_delay = (
        policy.throttling_base_delay_seconds
        if failure.category == "rate_limited"
        else policy.base_delay_seconds
    )
    lower_bound = max(policy.minimum_delay_seconds, base_delay * (2 ** retry_index))
    upper_bound = max(
        lower_bound,
        min(policy.max_delay_seconds, base_delay * (2 ** (retry_index + 1))),
    )
    # 在每轮递增区间内抖动，既避免近零重试，也保留可见的指数退避。
    return lower_bound + random_value() * (upper_bound - lower_bound), "backoff"


def _retry_after_was_ignored(failure: RetryFailure, policy: RetryPolicy) -> bool:
    retry_after = failure.retry_after_seconds
    return retry_after is not None and not (
        0 < retry_after <= policy.max_server_retry_after_seconds
    )


def _stream_is_committed(stream_state: StreamAttemptState | None) -> bool:
    return bool(stream_state is not None and stream_state.committed)


class RetryController:
    """唯一包含重试循环、延迟和取消检查的执行边界。"""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._random_value = random_value

    def execute(
        self,
        operation: RetryOperation,
        invoke: Callable[[], ResultT],
        *,
        cancellation_token: CancellationToken | None = None,
        on_event: Callable[[RetryEvent], None] | None = None,
        stream_state: StreamAttemptState | None = None,
        error_adapter: Callable[[Exception], RetryFailure | None] | None = None,
    ) -> ResultT:
        """执行操作；仅在结构化、可安全重放的失败后尝试下一次调用。"""

        attempt = 1
        while True:
            self._raise_if_cancelled(cancellation_token)
            try:
                return invoke()
            except Exception as error:
                failure = self._adapt_failure(error, error_adapter)
                if failure is None:
                    raise
                if operation.streaming and _stream_is_committed(stream_state):
                    # 已显示的流式增量不能重放，否则用户会看到重复文本。
                    raise RetryableOperationError(
                        replace(failure, category="stream_interrupted", retryable=False)
                    ) from error
                if not self._may_replay(operation, failure, attempt):
                    raise

                delay_seconds, source = _delay_for(
                    failure,
                    retry_index=attempt - 1,
                    policy=self.policy,
                    random_value=self._random_value,
                )
                if on_event is not None:
                    on_event(RetryEvent(
                        operation_name=operation.name,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        category=failure.category,
                        delay_seconds=delay_seconds,
                        source=source,
                        retry_after_ignored=_retry_after_was_ignored(failure, self.policy),
                    ))
                self._raise_if_cancelled(cancellation_token)
                self._sleep_with_cancellation(delay_seconds, cancellation_token)
                attempt += 1

    def _may_replay(
        self,
        operation: RetryOperation,
        failure: RetryFailure,
        attempt: int,
    ) -> bool:
        if not failure.retryable or attempt >= self.policy.max_attempts:
            return False
        if operation.replay_safety is ReplaySafety.SAFE_TO_REPLAY:
            return True
        return (
            operation.replay_safety is ReplaySafety.IDEMPOTENCY_KEY_REQUIRED
            and operation.idempotency_key is not None
            and operation.idempotency_supported
        )

    def _sleep_with_cancellation(
        self,
        delay_seconds: float,
        cancellation_token: CancellationToken | None,
    ) -> None:
        remaining = delay_seconds
        while remaining > 0:
            sleep_seconds = min(remaining, 0.1)
            self._sleep(sleep_seconds)
            self._raise_if_cancelled(cancellation_token)
            remaining -= sleep_seconds

    @staticmethod
    def _adapt_failure(
        error: Exception,
        error_adapter: Callable[[Exception], RetryFailure | None] | None,
    ) -> RetryFailure | None:
        if isinstance(error, RetryableOperationError):
            return error.failure
        if error_adapter is None:
            return None
        return error_adapter(error)

    @staticmethod
    def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
