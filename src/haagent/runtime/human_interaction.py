"""
haagent/runtime/human_interaction.py - 人机交互请求协议

定义 runtime 内部传递澄清问题和高风险审批请求的最小结构。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HumanInteractionRequest:
    interaction_type: str
    tool_name: str
    question: str
    reason: str = ""
    risk_level: str | None = None
    args_summary: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanInteractionResponse:
    approved: bool
    answer: str = ""


HumanInteractionHandler = Callable[[HumanInteractionRequest], HumanInteractionResponse]


def interaction_args_summary(tool_name: str, args: dict[str, Any]) -> dict[str, object]:
    if tool_name == "file_write":
        content = str(args.get("content", ""))
        return {
            "content_chars": len(content),
            "mode": str(args.get("mode", "")),
            "path": _summary_value(str(args.get("path", "")), 160),
        }
    if tool_name == "code_run":
        code = str(args.get("code", ""))
        return {
            "code_chars": len(code),
            "cwd": str(args.get("cwd", ".")),
            "timeout_seconds": args.get("timeout_seconds"),
        }
    if tool_name == "apply_patch":
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        return {
            "new_text_chars": len(new_text),
            "old_text_chars": len(old_text),
            "path": _summary_value(str(args.get("path", "")), 160),
        }
    if tool_name == "shell":
        return {
            "command": _summary_value(str(args.get("command", "")), 160),
            "cwd": str(args.get("cwd", ".")),
            "timeout_seconds": args.get("timeout_seconds"),
        }
    if tool_name == "request_user_input":
        return {
            "question": _summary_value(str(args.get("question", "")), 240),
            "reason": _summary_value(str(args.get("reason", "")), 240),
        }
    return {"args_keys": sorted(str(key) for key in args)}


def _summary_value(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"
