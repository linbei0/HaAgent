"""
src/haagent/runtime/execution/guardrails.py - 最小确定性 Guardrails

在 input、tool input 和 output 边界执行少量明确规则，返回结构化拦截结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SECRET_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
KEY_VALUE_PATTERN = re.compile(
    r"\b(api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*\S{8,}",
    re.IGNORECASE,
)
SECRET_REQUEST_VERBS = ["print", "show", "read", "dump", "leak", "expose", "读取", "打印", "泄露", "导出"]
_INTENT_SEGMENT_BOUNDARY = re.compile(r"[\r\n。！？.!?；;]+")


@dataclass(frozen=True)
class GuardrailResult:
    status: str
    scope: str
    rule_id: str
    message: str
    severity: str

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "scope": self.scope,
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity,
        }


def check_user_input(text: str) -> GuardrailResult | None:
    normalized = text.lower()
    asks_for_secret = any(pattern in normalized for pattern in ["~/.ssh", "id_rsa"])
    if asks_for_secret or _requests_secret_in_same_segment(text):
        return GuardrailResult(
            status="blocked",
            scope="input",
            rule_id="input_secret_request",
            message="user request asks to read or disclose secrets",
            severity="high",
        )
    if any(pattern in normalized for pattern in ["bypass workspace", "outside workspace", "工作区外"]):
        return GuardrailResult(
            status="blocked",
            scope="input",
            rule_id="input_workspace_bypass",
            message="user request asks to bypass workspace boundaries",
            severity="high",
        )
    return None


def _requests_secret_in_same_segment(text: str) -> bool:
    """只拦截同一句或同一行中的索取密钥意图，避免文档上下文误伤。"""

    for segment in _INTENT_SEGMENT_BOUNDARY.split(text):
        normalized = segment.lower()
        mentions_key = any(pattern in normalized for pattern in ["api key", "apikey", "api keys", "secret key"])
        has_secret_verb = any(verb in normalized for verb in SECRET_REQUEST_VERBS)
        if mentions_key and has_secret_verb:
            return True
    return False


def check_tool_input(tool_name: str, args: dict[str, Any]) -> GuardrailResult | None:
    # 静态工具 guardrail 登记在 ToolContribution；此处只做 catalog 分发。
    from haagent.tools.catalog import default_tool_catalog

    return default_tool_catalog().check_guardrail(tool_name, args)


def check_assistant_output(text: str) -> GuardrailResult | None:
    if SECRET_TOKEN_PATTERN.search(text) or KEY_VALUE_PATTERN.search(text):
        return GuardrailResult(
            status="blocked",
            scope="output",
            rule_id="output_secret_pattern",
            message="assistant output contains a secret-like token",
            severity="high",
        )
    return None


def guardrail_evidence(result: GuardrailResult) -> str:
    return f"guardrail {result.rule_id}: {result.message}"
