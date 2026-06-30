"""
haagent/tools/router.py - 工具路由器

校验 allowed_tools，分发本地工具，并为每次调用写入 tool-calls.jsonl。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.guardrails import GuardrailResult, check_tool_input, guardrail_evidence
from haagent.runtime.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    interaction_args_summary,
)
from haagent.runtime.path_policy import PathPolicy, default_path_policy
from haagent.runtime.policy import (
    PolicyDecision,
    deny_tool_approval,
    evaluate_tool_call,
    grant_tool_approval,
)
from haagent.skills import SkillSettings
from haagent.tools.base import ToolHandler, ToolRoutingError, tool_error
from haagent.tools.code_run import code_run
from haagent.tools.file_tools import apply_patch, apply_patch_set, context_find, file_list, file_read, file_search, file_write
from haagent.tools.registry import TOOL_REGISTRY, validate_tool_registry
from haagent.tools.shell import shell
from haagent.tools.skills import skill_list, skill_read
from haagent.tools.web import web_fetch, web_search


class ToolRouter:
    def __init__(
        self,
        allowed_tools: list[str],
        episode_writer: EpisodeWriter,
        workspace_root: Path,
        path_policy: PathPolicy | None = None,
        approval_allowed_tools: list[str] | None = None,
        approved_tools: list[str] | None = None,
        skill_settings: SkillSettings | None = None,
    ) -> None:
        self._allowed_tools = set(allowed_tools)
        self._approval_allowed_tools = list(approval_allowed_tools or [])
        self._approved_tools = list(approved_tools or [])
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root.resolve()
        self._skill_settings = skill_settings
        self._path_policy = path_policy.resolved() if path_policy is not None else default_path_policy(self._workspace_root)
        self._handlers: dict[str, ToolHandler] = {
            "fake_tool": self._fake_tool,
            "file_list": lambda args: file_list(args, self._workspace_root, self._path_policy),
            "file_search": lambda args: file_search(args, self._workspace_root, self._path_policy),
            "context_find": lambda args: context_find(args, self._workspace_root, self._path_policy),
            "file_read": lambda args: file_read(args, self._workspace_root, self._path_policy),
            "request_user_input": self._request_user_input_without_handler,
            "start_memory_update": self._start_memory_update,
            "skill_list": lambda args: skill_list(args, self._workspace_root, self._skill_settings),
            "skill_read": lambda args: skill_read(args, self._workspace_root, self._skill_settings),
            "web_search": web_search,
            "web_fetch": web_fetch,
            "file_write": lambda args: file_write(args, self._workspace_root, self._path_policy),
            "code_run": lambda args: code_run(args, self._workspace_root, self._path_policy),
            "apply_patch": lambda args: apply_patch(args, self._workspace_root, self._path_policy),
            "apply_patch_set": lambda args: apply_patch_set(args, self._workspace_root, self._path_policy),
            "shell": lambda args: shell(args, self._workspace_root, self._path_policy),
        }
        try:
            validate_tool_registry()
        except ValueError as error:
            raise ToolRoutingError(str(error), error_type="tool_registry_invalid") from error
        self._assert_registry_alignment()

    def dispatch(
        self,
        tool_name: str,
        args: dict[str, Any],
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> dict[str, Any]:
        """执行工具并保证每次调用都写入 tool-calls.jsonl。"""
        started = time.perf_counter()
        policy_decision: PolicyDecision | None = None
        guardrail_result: GuardrailResult | None = None
        try:
            if tool_name not in self._allowed_tools:
                result = tool_error("tool_not_allowed", f"tool is not allowed: {tool_name}")
            elif tool_name not in self._handlers:
                result = tool_error("unknown_tool", f"unknown tool: {tool_name}")
            else:
                policy_decision = evaluate_tool_call(
                    TOOL_REGISTRY[tool_name],
                    approval_allowed_tools=self._approval_allowed_tools,
                    approved_tools=self._approved_tools,
                )
                if policy_decision.action == "deny":
                    result, policy_decision, guardrail_result = self._handle_denied_policy(
                        tool_name,
                        args,
                        policy_decision,
                        interaction_handler,
                    )
                elif validation_error := _validate_args(tool_name, args):
                    result = validation_error
                elif guardrail_result := check_tool_input(tool_name, args):
                    result = tool_error(
                        "guardrail_denied",
                        guardrail_evidence(guardrail_result),
                    )
                elif tool_name == "request_user_input":
                    result = self._request_user_input(args, interaction_handler)
                else:
                    result = self._handlers[tool_name](args)
        except Exception as error:
            result = tool_error(type(error).__name__, str(error))

        self._write_trace(tool_name, args, result, started, policy_decision, guardrail_result)
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

    def _request_user_input_without_handler(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._request_user_input(args, None)

    def _start_memory_update(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "success",
            "memory_update_requested": True,
            "reason": str(args.get("reason", "")),
        }

    def _request_user_input(
        self,
        args: dict[str, Any],
        interaction_handler: HumanInteractionHandler | None,
    ) -> dict[str, Any]:
        if interaction_handler is None:
            return tool_error(
                "user_input_unavailable",
                "user input requested but no interaction handler is available",
            )
        question = str(args.get("question", ""))
        response = interaction_handler(
            HumanInteractionRequest(
                interaction_type="user_input",
                tool_name="request_user_input",
                question=question,
                reason=str(args.get("reason", "")),
                risk_level="low",
                args_summary=interaction_args_summary("request_user_input", args),
            ),
        )
        if not response.approved:
            return tool_error("user_input_unavailable", "user input request was not answered")
        answer = response.answer
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "answer_chars": len(answer),
        }

    def _handle_denied_policy(
        self,
        tool_name: str,
        args: dict[str, Any],
        policy_decision: PolicyDecision,
        interaction_handler: HumanInteractionHandler | None,
    ) -> tuple[dict[str, Any], PolicyDecision, GuardrailResult | None]:
        if tool_name not in self._approval_allowed_tools or interaction_handler is None:
            return (
                tool_error(
                    "policy_denied",
                    f"{policy_decision.reason}; {policy_decision.approval.reason}",
                ),
                policy_decision,
                None,
            )
        validation_error = _validate_args(tool_name, args)
        if validation_error:
            return validation_error, policy_decision, None
        response = interaction_handler(
            HumanInteractionRequest(
                interaction_type="approval",
                tool_name=tool_name,
                question=f"Approve high risk tool {tool_name}?",
                reason=policy_decision.approval.reason,
                risk_level=policy_decision.risk_level,
                args_summary=interaction_args_summary(tool_name, args),
            ),
        )
        if not response.approved:
            denied_policy = deny_tool_approval(policy_decision)
            return (
                tool_error(
                    "approval_denied",
                    f"approval denied for high risk tool {tool_name}",
                ),
                denied_policy,
                None,
            )
        granted_policy = grant_tool_approval(policy_decision)
        guardrail_result = check_tool_input(tool_name, args)
        if guardrail_result is not None:
            return (
                tool_error("guardrail_denied", guardrail_evidence(guardrail_result)),
                granted_policy,
                guardrail_result,
            )
        return self._handlers[tool_name](args), granted_policy, None

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
        guardrail_result: GuardrailResult | None,
    ) -> None:
        self._episode_writer.append_tool_call(
            {
                "tool_name": tool_name,
                "args": args,
                "status": result["status"],
                "result": result if result["status"] == "success" else None,
                "error": result.get("error"),
                "policy": policy_decision.to_dict() if policy_decision else None,
                "path_policy": {
                    "permission_mode": self._path_policy.permission_mode,
                    "project_root": str(self._path_policy.project_root),
                    "external_root_count": len(self._path_policy.external_roots),
                },
                "guardrail": guardrail_result.to_dict() if guardrail_result else None,
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
