"""
agent_foundry/model_gateway.py - 统一模型网关接口

上层只依赖 ModelGateway 协议；fake provider 用于测试，OpenAI provider 是首个真实适配。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent_foundry.task import TaskSpec


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

    def generate(self, task: TaskSpec) -> ModelResponse:
        """Generate a model response for a task."""


class FakeModelGateway:
    provider_name = "fake"

    def __init__(self, response: ModelResponse | None = None) -> None:
        self._response = response or ModelResponse(
            content="Use the fake tool for the MVP execution step.",
            tool_calls=[ToolCall(name="fake_tool", args={})],
        )

    def generate(self, task: TaskSpec) -> ModelResponse:
        return self._response


Transport = Callable[[dict[str, object], str], dict[str, object]]


class OpenAIResponsesGateway:
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._transport = transport or _responses_transport

    def generate(self, task: TaskSpec) -> ModelResponse:
        """调用 OpenAI Responses API，并把 provider 输出收敛成统一 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError("OPENAI_API_KEY is required for OpenAIResponsesGateway")

        # provider 失败必须显式暴露给 orchestrator，禁止静默回退到 fake model。
        payload = {"model": self._model, "input": _prompt_for_task(task)}
        try:
            response = self._transport(payload, self._api_key)
        except Exception as error:
            raise ModelCallError(str(error)) from error

        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ModelCallError("OpenAI response did not include output_text")
        return ModelResponse(content=output_text, tool_calls=[])


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


def _responses_transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
    """执行真实 HTTP 请求；保持为函数便于测试注入替身 transport。"""
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
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
