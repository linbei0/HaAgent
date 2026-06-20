"""
agentfoundry/tools/router.py - 工具路由器

校验 allowed_tools，分发本地工具，并为每次调用写入 tool-calls.jsonl。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.policy import PolicyDecision, evaluate_tool_call
from agentfoundry.tools.base import ToolHandler, ToolRoutingError, tool_error
from agentfoundry.tools.file_tools import apply_patch, file_read, file_search
from agentfoundry.tools.registry import TOOL_REGISTRY, validate_tool_registry
from agentfoundry.tools.shell import shell


class ToolRouter:
    def __init__(
        self,
        allowed_tools: list[str],
        episode_writer: EpisodeWriter,
        workspace_root: Path,
    ) -> None:
        self._allowed_tools = set(allowed_tools)
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root.resolve()
        self._handlers: dict[str, ToolHandler] = {
            "fake_tool": self._fake_tool,
            "file_search": lambda args: file_search(args, self._workspace_root),
            "file_read": lambda args: file_read(args, self._workspace_root),
            "apply_patch": lambda args: apply_patch(args, self._workspace_root),
            "shell": lambda args: shell(args, self._workspace_root),
        }
        try:
            validate_tool_registry()
        except ValueError as error:
            raise ToolRoutingError(str(error), error_type="tool_registry_invalid") from error
        self._assert_registry_alignment()

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具并保证每次调用都写入 tool-calls.jsonl。"""
        started = time.perf_counter()
        policy_decision: PolicyDecision | None = None
        try:
            if tool_name not in self._allowed_tools:
                result = tool_error("tool_not_allowed", f"tool is not allowed: {tool_name}")
            elif tool_name not in self._handlers:
                result = tool_error("unknown_tool", f"unknown tool: {tool_name}")
            else:
                policy_decision = evaluate_tool_call(TOOL_REGISTRY[tool_name])
                if policy_decision.action == "deny":
                    result = tool_error(
                        "policy_denied",
                        f"{policy_decision.reason}; {policy_decision.approval.reason}",
                    )
                else:
                    validation_error = _validate_args(tool_name, args)
                    if validation_error:
                        result = validation_error
                    else:
                        result = self._handlers[tool_name](args)
        except Exception as error:
            result = tool_error(type(error).__name__, str(error))

        self._write_trace(tool_name, args, result, started, policy_decision)
        return result

    def raise_for_error(self, result: dict[str, Any]) -> None:
        if result.get("status") == "error":
            error = result.get("error") or {}
            raise ToolRoutingError(
                str(error.get("message", "tool failed")),
                error_type=str(error.get("type", "")),
            )

    def _fake_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "args": args}

    def _assert_registry_alignment(self) -> None:
        """Router 和 Registry 必须同步，否则 allowed_tools 审计会和实际执行脱节。"""
        if set(self._handlers) != set(TOOL_REGISTRY):
            missing = sorted(set(TOOL_REGISTRY) - set(self._handlers))
            extra = sorted(set(self._handlers) - set(TOOL_REGISTRY))
            raise ToolRoutingError(f"tool registry mismatch: missing={missing}, extra={extra}")

    def _write_trace(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        started: float,
        policy_decision: PolicyDecision | None,
    ) -> None:
        self._episode_writer.append_tool_call(
            {
                "tool_name": tool_name,
                "args": args,
                "status": result["status"],
                "result": result if result["status"] == "success" else None,
                "error": result.get("error"),
                "policy": policy_decision.to_dict() if policy_decision else None,
                "duration_seconds": time.perf_counter() - started,
            },
        )


def _validate_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    schema = TOOL_REGISTRY[tool_name].parameters
    if schema.get("type") != "object":
        return tool_error("tool_argument_invalid", "tool arguments schema must be object")

    required = schema.get("required", [])
    for name in required:
        if name not in args:
            return tool_error("tool_argument_invalid", f"missing required argument: {name}")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        for name in args:
            if name not in properties:
                return tool_error("tool_argument_invalid", f"unexpected argument: {name}")

    for name, value in args.items():
        property_schema = properties.get(name)
        if not property_schema:
            continue
        expected_type = property_schema.get("type")
        if expected_type and not _matches_json_type(value, expected_type):
            return tool_error(
                "tool_argument_invalid",
                f"argument {name} must be {expected_type}",
            )
    return None


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True
