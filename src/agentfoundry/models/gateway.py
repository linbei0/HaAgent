"""
agentfoundry/models/gateway.py - 统一模型网关接口

上层只依赖 ModelGateway 协议；真实 provider 失败必须显式暴露。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agentfoundry.runtime.task_contract import TaskSpec


class ModelCallError(RuntimeError):
    """Raised when a model provider fails explicitly."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelGateway(Protocol):
    provider_name: str

    def generate(
        self,
        task: TaskSpec,
        model_input: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Generate a model response for a task."""


Transport = Callable[[dict[str, object], str], dict[str, object]]
DEFAULT_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"


class OpenAIResponsesGateway:
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        configured_base_url = (
            base_url
            if base_url is not None
            else os.environ.get("OPENAI_BASE_URL")
        )
        self._responses_endpoint = _normalize_responses_endpoint(configured_base_url)
        self._transport = transport or (
            lambda payload, api_key: _responses_transport(
                payload,
                api_key,
                self._responses_endpoint,
            )
        )

    @property
    def responses_endpoint(self) -> str:
        """返回本次 gateway 会请求的 Responses API endpoint，便于审计和测试。"""
        return self._responses_endpoint

    def generate(
        self,
        task: TaskSpec,
        model_input: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """调用 OpenAI Responses API，并把 provider 输出收敛成统一 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError("OPENAI_API_KEY is required for OpenAIResponsesGateway")

        # provider 失败必须显式暴露给 orchestrator，禁止静默回退到 fake model。
        payload: dict[str, object] = {
            "model": self._model,
            "input": model_input if model_input is not None else _prompt_for_task(task),
        }
        if tool_schemas is not None:
            payload["tools"] = tool_schemas
        try:
            response = self._transport(payload, self._api_key)
        except Exception as error:
            raise ModelCallError(str(error)) from error

        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ModelCallError("OpenAI response did not include output_text")
        return ModelResponse(content=output_text, tool_calls=_parse_tool_calls(response))


def _prompt_for_task(task: TaskSpec) -> str:
    constraints = _format_list(task.constraints)
    criteria = _format_list(task.acceptance_criteria)
    return (
        f"Goal: {task.goal}\n"
        f"Constraints:\n{constraints}\n"
        f"Acceptance criteria:\n{criteria}"
    )


def _format_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def _parse_tool_calls(response: dict[str, object]) -> list[ToolCall]:
    output = response.get("output")
    if output is None:
        return []
    if not isinstance(output, list):
        raise ModelCallError("OpenAI output must be a list when present")

    tool_calls: list[ToolCall] = []
    for item in output:
        # 当前只支持 Responses API 的最小 function_call 结构，避免误吞 provider 新格式。
        if not isinstance(item, dict):
            raise ModelCallError("unsupported OpenAI output item")
        output_type = item.get("type")
        if output_type in {"message", "output_text", "text"}:
            continue
        if output_type != "function_call":
            raise ModelCallError(f"unsupported OpenAI output type: {output_type}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        tool_calls.append(ToolCall(name=name, args=_parse_tool_arguments(arguments)))
    return tool_calls


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ModelCallError("invalid tool arguments JSON") from error
    if not isinstance(parsed, dict):
        raise ModelCallError("tool arguments must be a JSON object")
    return parsed


def _normalize_responses_endpoint(base_url: str | None) -> str:
    """把裸域名或 /v1 base URL 规范化为 Responses API endpoint。"""
    if base_url is None or not base_url.strip():
        return DEFAULT_RESPONSES_ENDPOINT
    endpoint = base_url.strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    if endpoint.endswith("/v1/responses"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/responses"
    return f"{endpoint}/v1/responses"


def _responses_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_RESPONSES_ENDPOINT,
) -> dict[str, object]:
    """执行真实 HTTP 请求；保持为函数便于测试注入替身 transport。"""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(f"OpenAI request failed with HTTP {error.code}: {detail}") from error
    return json.loads(body)
