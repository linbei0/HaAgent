"""
src/haagent/models/adapters/openai_responses.py - OpenAI Responses API 网关

把 Responses API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.capabilities import ModelCapabilities
from haagent.models.http_transport import ModelHttpTransport
from haagent.models.model_options import merge_provider_payload
from haagent.models.model_ref import ModelInvocation
from haagent.models.model_settings import ModelSettings
from haagent.models.adapters.transport import (
    DEFAULT_RESPONSES_ENDPOINT,
    _endpoint_base_url,
    _image_data_url,
    _invoke_transport,
    _normalize_responses_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _openai_responses_stream_error,
    _responses_retrieve_transport,
    _responses_stream_transport,
    _responses_transport,
    _supports_transport_observability,
    _usage_from_fields,
)
from haagent.models.types import (
    ModelCallError,
    ModelFailureDetails,
    ModelGatewayMetadata,
    ModelResponse,
    ModelUsage,
    ProviderTurnState,
    StreamTransport,
    ToolCall,
    Transport,
)
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import (
    ReplaySafety,
    RetryController,
    RetryEvent,
    RetryFailure,
    RetryOperation,
    StreamResetEvent,
)

RetrieveTransport = Callable[[str], dict[str, object]]


def _parse_tool_calls(response: dict[str, object]) -> tuple[list[ToolCall], list[dict[str, Any]]]:
    output = response.get("output")
    if output is None:
        return [], []
    if not isinstance(output, list):
        raise ModelCallError("OpenAI output must be a list when present")

    tool_calls: list[ToolCall] = []
    # 完整 output 序列不展示给用户，但工具续轮必须按原顺序原样回传。
    continuation_items: list[dict[str, Any]] = []
    has_reasoning = False
    for item in output:
        if not isinstance(item, dict):
            raise ModelCallError("unsupported OpenAI output item")
        output_type = item.get("type")
        continuation_items.append(dict(item))
        if output_type in {"message", "output_text", "text"}:
            continue
        if output_type == "reasoning":
            has_reasoning = True
            continue
        if output_type != "function_call":
            raise ModelCallError(f"unsupported OpenAI output type: {output_type}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        call_id = str(item.get("call_id") or item.get("id") or "")
        tool_calls.append(ToolCall(name=name, args=_parse_tool_arguments(arguments), id=call_id))
    return tool_calls, continuation_items if has_reasoning else []


def _parse_openai_responses_usage(response: dict[str, object]) -> ModelUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return _usage_from_fields(
        usage,
        input_field="input_tokens",
        output_field="output_tokens",
        total_field="total_tokens",
        raw_source="openai.responses.usage",
    )


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions messages to Responses API input format."""
    result = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            result.append({"role": "system", "content": msg.get("content", "")})
        elif role == "user":
            result.append({"role": "user", "content": _responses_user_content(msg.get("content", ""))})
        elif role == "assistant":
            continuation = msg.get("provider_turn_state")
            replayed = False
            if isinstance(continuation, dict) and continuation.get("provider") == "openai":
                payload = continuation.get("payload")
                items = payload.get("items") if isinstance(payload, dict) else None
                if isinstance(items, list):
                    for cont_item in items:
                        if isinstance(cont_item, dict):
                            result.append(dict(cont_item))
                    replayed = True
            if replayed:
                continue
            item: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
            for tc in msg.get("tool_calls", []):
                result.append({
                    "type": "function_call",
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "call_id": tc.get("id", ""),
                })
            if item["content"]:
                result.append(item)
        elif role == "tool":
            result.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })
    return result


def _responses_user_content(content: object) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ModelCallError("Responses user message content must be a string or content parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ModelCallError("Responses content parts must be objects")
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_attachment":
            parts.append({"type": "input_image", "image_url": _image_data_url(part)})
        else:
            raise ModelCallError(f"unsupported Responses content part: {part.get('type')}")
    return parts

class OpenAIResponsesGateway:
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        retry_controller: RetryController | None = None,
        require_api_key: bool = True,
        http_transport: ModelHttpTransport | None = None,
        request_config: ModelSettings | None = None,
        retrieve_transport: RetrieveTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._require_api_key = require_api_key
        self._model = model
        self._request_config = request_config or ModelSettings.empty()
        self.model_settings = self._request_config
        configured_base_url = (
            base_url
            if base_url is not None
            else os.environ.get("OPENAI_BASE_URL")
        )
        self._responses_endpoint = _normalize_responses_endpoint(configured_base_url)
        # 仅在需要默认 HTTP path 时创建/绑定 transport；外部注入不越权关闭。
        self._owns_http_transport = False
        self._http_transport: ModelHttpTransport | None = http_transport
        needs_default_http = (
            transport is None or stream_transport is None or retrieve_transport is None
        )
        if needs_default_http and self._http_transport is None:
            self._http_transport = ModelHttpTransport()
            self._owns_http_transport = True
        bound = self._http_transport
        # 流中断后只保存非敏感 meta，供 background retrieve；不得写入 UI/episode 全文 payload。
        self._last_stream_meta: dict[str, object] = {}
        if transport is None:
            self._transport = (
                lambda payload, api_key, *, attempt=1, telemetry_sink=None: _responses_transport(
                    payload,
                    api_key,
                    self._responses_endpoint,
                    http_transport=bound,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                )
            )
        else:
            self._transport = transport
        if stream_transport is None:
            self._stream_transport = (
                lambda payload, api_key, on_delta, *, attempt=1, telemetry_sink=None: (
                    _responses_stream_transport(
                        payload,
                        api_key,
                        self._responses_endpoint,
                        on_delta,
                        http_transport=bound,
                        attempt=attempt,
                        telemetry_sink=telemetry_sink,
                        meta_out=self._last_stream_meta,
                    )
                )
            )
        else:
            self._stream_transport = stream_transport
        if retrieve_transport is None:
            self._retrieve_transport: RetrieveTransport = (
                lambda response_id, *, attempt=1, telemetry_sink=None: _responses_retrieve_transport(
                    response_id,
                    self._api_key or "",
                    self._responses_endpoint,
                    http_transport=bound,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                )
            )
        else:
            self._retrieve_transport = retrieve_transport
        self._retry_controller = default_retry_controller(retry_controller)

    def close(self) -> None:
        """仅关闭自身创建的共享 transport；注入资源保持外部所有权。"""

        if self._owns_http_transport and self._http_transport is not None:
            self._http_transport.close()
            self._owns_http_transport = False

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._responses_endpoint),
            base_url=_endpoint_base_url(self._responses_endpoint),
            request_config=self._request_config.to_traceable_dict(),
        )

    def capabilities(self) -> ModelCapabilities:
        # 仅官方 endpoint + 显式 background opt-in 才声明可 retrieve；兼容端点不得猜测。
        background_retrieval = (
            "supported"
            if (
                self._responses_endpoint == DEFAULT_RESPONSES_ENDPOINT
                and self._request_config.options.get("background") is True
            )
            else "unknown"
        )
        return ModelCapabilities(
            tools="supported",
            streaming="supported",
            vision="supported",
            reasoning="unknown",
            tools_mode="native",
            protocols=frozenset({"responses"}),
            background_response_retrieval=background_retrieval,
        )

    def generate(
        self,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
        telemetry_sink=None,
        stream_reset_sink=None,
    ) -> ModelResponse:
        """调用 OpenAI Responses API，并把 provider 输出收敛成统一 ModelResponse。"""

        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        if self._require_api_key and not self._api_key:
            raise ModelCallError("OPENAI_API_KEY is required for OpenAIResponsesGateway")

        # Responses API uses "input" — convert messages to input format
        payload: dict[str, object] = {
            "model": self._model,
            "input": _messages_to_responses_input(messages),
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["parallel_tool_calls"] = True
        if event_sink is not None:
            payload["stream"] = True
        # settings 以 gateway 绑定为准；invocation 上的 primary options 不得泄漏到本 adapter。
        payload = merge_provider_payload(payload, self._request_config.options)
        can_retrieve = (
            event_sink is not None
            and stream_reset_sink is not None
            and self.capabilities().background_response_retrieval == "supported"
        )
        try:
            if can_retrieve:
                response = self._generate_with_background_retrieve(
                    payload,
                    event_sink=event_sink,
                    cancellation_token=cancellation_token,
                    retry_event_sink=retry_event_sink,
                    retry_exhausted_sink=retry_exhausted_sink,
                    telemetry_sink=telemetry_sink,
                    stream_reset_sink=stream_reset_sink,
                )
            else:
                response = execute_model_request(
                    self._retry_controller,
                    provider=self.provider_name,
                    invoke=lambda on_delta, attempt: _invoke_transport(
                        self._transport,
                        self._stream_transport,
                        payload,
                        self._api_key or "",
                        on_delta=on_delta,
                        attempt=attempt,
                        telemetry_sink=telemetry_sink,
                    ),
                    event_sink=event_sink,
                    cancellation_token=cancellation_token,
                    retry_event_sink=retry_event_sink,
                    retry_exhausted_sink=retry_exhausted_sink,
                    telemetry_sink=telemetry_sink,
                    stream_reset_sink=stream_reset_sink,
                )
        except ModelCallError:
            raise
        except RunCancelled:
            raise
        except Exception as error:
            raise unexpected_model_error(error) from error

        return self._model_response_from_provider(response)

    def _generate_with_background_retrieve(
        self,
        payload: dict[str, object],
        *,
        event_sink: Callable[[str], None] | None,
        cancellation_token: CancellationToken | None,
        retry_event_sink: Callable[[RetryEvent], None] | None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None,
        telemetry_sink,
        stream_reset_sink: Callable[[StreamResetEvent], None] | None,
    ) -> dict[str, object]:
        """create 一次；已知 response_id 后只在同一总预算内 retrieve。"""

        partial_character_count = 0

        def emit_delta(delta: str) -> None:
            nonlocal partial_character_count
            if delta:
                partial_character_count += len(delta)
            if event_sink is not None:
                event_sink(delta)

        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        self._last_stream_meta.clear()
        try:
            return _invoke_transport(
                self._transport,
                self._stream_transport,
                payload,
                self._api_key or "",
                on_delta=emit_delta,
                attempt=1,
                telemetry_sink=telemetry_sink,
            )
        except ModelCallError as error:
            details = error.details
            if details is None or not details.retryable:
                raise

            failure = details.to_retry_failure()
            response_id = self._known_response_id(error)
            if self._retry_controller.policy.max_attempts <= 1:
                self._raise_background_exhausted(
                    error,
                    partial_character_count=partial_character_count,
                    retry_exhausted_sink=retry_exhausted_sink,
                )

            if response_id is None:
                if partial_character_count:
                    self._emit_stream_reset(
                        stream_reset_sink,
                        attempt=1,
                        next_attempt=2,
                        category=failure.category,
                        partial_character_count=partial_character_count,
                    )
                self._retry_controller.wait_before_retry(
                    failure,
                    operation_name=f"{self.provider_name}.generate",
                    attempt=1,
                    cancellation_token=cancellation_token,
                    on_event=retry_event_sink,
                    recovery_kind="discard_and_replay",
                )
                return execute_model_request(
                    self._retry_controller,
                    provider=self.provider_name,
                    invoke=lambda on_delta, attempt: _invoke_transport(
                        self._transport,
                        self._stream_transport,
                        payload,
                        self._api_key or "",
                        on_delta=on_delta,
                        attempt=attempt,
                        telemetry_sink=telemetry_sink,
                    ),
                    event_sink=event_sink,
                    cancellation_token=cancellation_token,
                    retry_event_sink=retry_event_sink,
                    retry_exhausted_sink=retry_exhausted_sink,
                    telemetry_sink=telemetry_sink,
                    stream_reset_sink=stream_reset_sink,
                    first_attempt=2,
                )

            sequence_number = self._last_sequence_number()
            self._retry_controller.wait_before_retry(
                failure,
                operation_name=f"{self.provider_name}.responses.retrieve",
                attempt=1,
                cancellation_token=cancellation_token,
                on_event=retry_event_sink,
                recovery_kind="background_retrieve",
                response_id=response_id,
                sequence_number=sequence_number,
            )
            retrieve_attempt = 1

            def retrieve() -> tuple[dict[str, object], int]:
                nonlocal retrieve_attempt
                retrieve_attempt += 1
                retrieved = self._call_retrieve(
                    response_id,
                    attempt=retrieve_attempt,
                    telemetry_sink=telemetry_sink,
                )
                status = retrieved.get("status")
                if status in {"queued", "in_progress"}:
                    raise ModelCallError(
                        f"OpenAI background response is {status}",
                        details=ModelFailureDetails(
                            category=details.category,
                            retryable=True,
                            provider_code=str(status),
                        ),
                    )
                if status == "completed" or (
                    status == "incomplete" and _is_length_limited_response(retrieved)
                ):
                    return retrieved, retrieve_attempt
                if status in {"failed", "cancelled", "incomplete"}:
                    raise _background_terminal_error(retrieved, status=str(status))
                raise ModelCallError(
                    f"unknown OpenAI background response status: {status}",
                    details=ModelFailureDetails(category="response_parse", retryable=False),
                )

            try:
                retrieved, completed_attempt = self._retry_controller.execute(
                    RetryOperation(
                        f"{self.provider_name}.responses.retrieve",
                        ReplaySafety.SAFE_TO_REPLAY,
                    ),
                    retrieve,
                    cancellation_token=cancellation_token,
                    on_event=retry_event_sink,
                    error_adapter=lambda exc: (
                        exc.details.to_retry_failure()
                        if isinstance(exc, ModelCallError) and exc.details is not None
                        else None
                    ),
                    first_attempt=2,
                    recovery_kind="background_retrieve",
                    response_id=response_id,
                    sequence_number=sequence_number,
                )
            except ModelCallError as retrieve_error:
                if self._is_retryable_stream_error(retrieve_error):
                    interrupted = _stream_interrupted_failure(retrieve_error)
                    if retry_exhausted_sink is not None:
                        retry_exhausted_sink(interrupted, self._retry_controller.policy.max_attempts)
                    raise _stream_interrupted_error(retrieve_error) from retrieve_error
                raise

            if partial_character_count:
                self._emit_stream_reset(
                    stream_reset_sink,
                    attempt=1,
                    next_attempt=completed_attempt,
                    category=failure.category,
                    partial_character_count=partial_character_count,
                )
            return self._reconcile_authoritative_response(retrieved, event_sink=event_sink)

    def _call_retrieve(
        self,
        response_id: str,
        *,
        attempt: int,
        telemetry_sink,
    ) -> dict[str, object]:
        callback = self._retrieve_transport
        if _supports_transport_observability(callback):
            return callback(response_id, attempt=attempt, telemetry_sink=telemetry_sink)
        return callback(response_id)

    def _reconcile_authoritative_response(
        self,
        payload: dict[str, object],
        *,
        event_sink: Callable[[str], None] | None,
    ) -> dict[str, object]:
        """将 completed Response 归一为 output_text，并一次性推送权威全文。"""

        output_text = payload.get("output_text")
        if not isinstance(output_text, str):
            output_text = _extract_responses_output_text(payload)
            payload = {**payload, "output_text": output_text}
        if event_sink is not None and output_text:
            event_sink(output_text)
        return payload

    def _known_response_id(self, error: ModelCallError) -> str | None:
        meta_id = self._last_stream_meta.get("response_id")
        if isinstance(meta_id, str) and meta_id:
            return meta_id
        if error.details is not None and isinstance(error.details.provider_code, str):
            code = error.details.provider_code
            # 测试/注入路径可能把 response_id 放在 provider_code；正式错误码不走 retrieve。
            if code.startswith("resp_"):
                return code
        return None

    def _last_sequence_number(self) -> int | None:
        sequence = self._last_stream_meta.get("last_sequence_number")
        return sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else None

    @staticmethod
    def _is_retryable_stream_error(error: ModelCallError) -> bool:
        return error.details is not None and error.details.retryable is True

    def _emit_stream_reset(
        self,
        stream_reset_sink: Callable[[StreamResetEvent], None] | None,
        *,
        attempt: int,
        next_attempt: int,
        category: str,
        partial_character_count: int,
    ) -> None:
        if stream_reset_sink is None:
            return
        stream_reset_sink(
            StreamResetEvent(
                operation_name=f"{self.provider_name}.generate",
                attempt=attempt,
                next_attempt=next_attempt,
                category=category,
                partial_character_count=partial_character_count,
            )
        )

    def _raise_background_exhausted(
        self,
        error: ModelCallError,
        *,
        partial_character_count: int,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None,
    ) -> None:
        details = error.details
        assert details is not None
        failure = (
            _stream_interrupted_failure(error)
            if partial_character_count
            else details.to_retry_failure()
        )
        if retry_exhausted_sink is not None:
            retry_exhausted_sink(failure, 1)
        if partial_character_count:
            raise _stream_interrupted_error(error) from error
        raise error

    def _model_response_from_provider(self, response: dict[str, object]) -> ModelResponse:
        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ModelCallError("OpenAI response did not include output_text")
        tool_calls, continuation_items = _parse_tool_calls(response)
        continuation = (
            {"kind": "openai_responses", "items": continuation_items}
            if continuation_items
            else None
        )
        return ModelResponse(
            content=output_text,
            tool_calls=tool_calls,
            usage=_parse_openai_responses_usage(response),
            termination=_openai_responses_termination(response, bool(tool_calls)),
            provider_turn_state=(ProviderTurnState("openai", continuation) if continuation else None),
        )


def _openai_responses_termination(response: dict[str, object], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    if response.get("status") == "completed":
        return "completed"
    incomplete = response.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason") in {"max_output_tokens", "max_tokens"}:
        return "length"
    return "unknown"


def _extract_responses_output_text(response: dict[str, object]) -> str:
    """从 completed Response 的 output 列表提取纯文本；缺省返回空串。"""

    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


def _is_length_limited_response(response: dict[str, object]) -> bool:
    details = response.get("incomplete_details")
    return isinstance(details, dict) and details.get("reason") in {
        "max_output_tokens",
        "max_tokens",
    }


def _stream_interrupted_failure(error: ModelCallError) -> RetryFailure:
    details = error.details
    return RetryFailure(
        category="stream_interrupted",
        retryable=False,
        status_code=details.status_code if details else None,
        provider_code=details.provider_code if details else None,
        request_id=details.request_id if details else None,
    )


def _stream_interrupted_error(error: ModelCallError) -> ModelCallError:
    failure = _stream_interrupted_failure(error)
    return ModelCallError(
        "model stream interrupted after partial output",
        details=ModelFailureDetails(
            category="stream_interrupted",
            status_code=failure.status_code,
            provider_code=failure.provider_code,
            request_id=failure.request_id,
            retryable=False,
        ),
    )


def _background_terminal_error(response: dict[str, object], *, status: str) -> ModelCallError:
    """把 background Response 终态映射为不可自动恢复的结构化错误。"""

    error_payload = response.get("error")
    if status == "failed":
        mapped = _openai_responses_stream_error(error_payload, fallback_code="response_failed")
        mapped_details = mapped.details
        assert mapped_details is not None
        return ModelCallError(
            str(mapped),
            details=ModelFailureDetails(
                category=mapped_details.category,
                status_code=mapped_details.status_code,
                provider_code=mapped_details.provider_code,
                request_id=mapped_details.request_id,
                retryable=False,
            ),
        )
    if status == "cancelled":
        details = ModelFailureDetails(category="client", retryable=False, provider_code=status)
    else:
        details = ModelFailureDetails(category="protocol", retryable=False, provider_code=status)
    return ModelCallError(
        f"OpenAI background response ended with status={status}",
        details=details,
    )
