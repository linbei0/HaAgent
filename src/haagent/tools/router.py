"""
haagent/tools/router.py - 工具路由器

校验 allowed_tools，分发本地工具，并为每次调用写入 tool-calls.jsonl。
"""

from __future__ import annotations

import time
import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.tool_results import prepare_tool_result_for_model
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import (
    RetryController,
    RetryFailure,
    RetryOperation,
    RetryableOperationError,
)
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.guardrails import GuardrailResult, check_tool_input, guardrail_evidence
from haagent.runtime.execution.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    ToolPermissionRequest,
    interaction_args_summary,
)
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy
from haagent.runtime.execution.policy import (
    PolicyDecision,
    deny_tool_approval,
    evaluate_tool_call,
    grant_tool_approval,
)
from haagent.runtime.sandbox.base import SandboxBackend
from haagent.runtime.session.attachments import ImageAttachment
from haagent.skills import SkillSettings
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.base import (
    RecoveryAction,
    ToolExecutionContext,
    ToolFailureCategory,
    ToolRoutingError,
    tool_error,
)
from haagent.tools.handler_factory import build_static_tool_handlers
from haagent.tools.mcp_tools import run_mcp_tool
from haagent.tools.registry import (
    TOOL_REGISTRY,
    ToolDefinition,
    ToolRuntimeRegistry,
    default_tool_runtime_registry,
    validate_tool_registry,
)
from haagent.tools.schema_validation import validate_json_value
from haagent.tools.user_input import UserQuestionValidationError, parse_user_questions

# turn, tool_name, duration_ms, execution_effect, status — 不进入 model-visible result。
ToolPerformanceSink = Callable[[int, str, float, str, str], None]
TodoStateSink = Callable[[list[dict[str, object]], str, int | None], dict[str, object]]
PlanningStateHandler = Callable[[str, dict[str, object], int], dict[str, object]]


class ToolRouter:
    def __init__(
        self,
        allowed_tools: list[str],
        episode_writer: EpisodeWriter,
        workspace_root: Path,
        session_path: Path | None = None,
        runs_root: Path | None = None,
        path_policy: PathPolicy | None = None,
        approval_allowed_tools: list[str] | None = None,
        approved_tools: list[str] | None = None,
        skill_settings: SkillSettings | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRuntimeRegistry | None = None,
        mcp_runtime: Any | None = None,
        agent_runtime: Any | None = None,
        worker_permission_requester: Callable[[str, dict[str, Any], PolicyDecision], Any] | None = None,
        sandbox_backend: SandboxBackend | None = None,
        image_attachment_history: list[ImageAttachment] | None = None,
        retry_controller: RetryController | None = None,
        performance_sink: ToolPerformanceSink | None = None,
        skill_catalog: SkillCatalogService | None = None,
        planning_status: str = "inactive",
        actor_role: str = "main",
        todo_state_sink: TodoStateSink | None = None,
        planning_state_handler: PlanningStateHandler | None = None,
    ) -> None:
        self._allowed_tools = set(allowed_tools)
        self._approval_allowed_tools = list(approval_allowed_tools or [])
        self._approved_tools = list(approved_tools or [])
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root.resolve()
        self._skill_settings = skill_settings
        self._cancellation_token = cancellation_token
        self._retry_controller = retry_controller or RetryController()
        self._tool_registry = tool_registry or default_tool_runtime_registry()
        self._mcp_runtime = mcp_runtime
        self._agent_runtime = agent_runtime
        self._worker_permission_requester = worker_permission_requester
        self._sandbox_backend = sandbox_backend
        self._performance_sink = performance_sink
        self._planning_status = planning_status
        self._actor_role = actor_role
        self._todo_state_sink = todo_state_sink
        self._planning_state_handler = planning_state_handler
        self._image_attachment_history = {
            attachment.id: attachment
            for attachment in image_attachment_history or []
        }
        self._path_policy = path_policy.resolved() if path_policy is not None else default_path_policy(self._workspace_root)
        self._handlers = build_static_tool_handlers(
            workspace_root=self._workspace_root,
            path_policy=self._path_policy,
            session_path=session_path,
            runs_root=runs_root,
            skill_settings=self._skill_settings,
            cancellation_token=self._cancellation_token,
            mcp_runtime=self._mcp_runtime,
            sandbox_backend=self._sandbox_backend,
            skill_catalog=skill_catalog,
            router_handlers={
                "fake_tool": self._fake_tool,
                "load_image_attachment": self._load_image_attachment,
                "agent": self._agent,
                "send_message": self._send_message,
                "task_stop": self._task_stop,
                "task_get": self._task_get,
                "task_list": self._task_list,
                "task_output": self._task_output,
                "request_user_input": self._request_user_input_handler,
                "todo_update": self._todo_update_handler,
                "submit_plan": self._submit_plan_handler,
                "start_memory_update": self._start_memory_update,
            },
        )
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
        *,
        turn: int | None = None,
    ) -> dict[str, Any]:
        """执行工具并保证每次调用都写入 tool-calls.jsonl。"""
        started = time.perf_counter()
        policy_decision: PolicyDecision | None = None
        guardrail_result: GuardrailResult | None = None
        try:
            if denial := self._mode_or_actor_denial(tool_name):
                result = denial
            elif tool_name not in self._allowed_tools:
                result = tool_error("tool_not_allowed", f"tool is not allowed: {tool_name}")
            elif not self._tool_registry.has(tool_name):
                result = tool_error("unknown_tool", f"unknown tool: {tool_name}")
            else:
                tool_definition = self._tool_registry.get(tool_name)
                policy_decision = evaluate_tool_call(
                    tool_definition,
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
                elif validation_error := _validate_args(tool_name, args, self._tool_registry):
                    result = validation_error
                elif guardrail_result := check_tool_input(tool_name, args):
                    result = tool_error(
                        "guardrail_denied",
                        guardrail_evidence(guardrail_result),
                    )
                elif tool_name.startswith("mcp__"):
                    # 动态 MCP 不进静态 binder；其余静态工具统一走 catalog handler map。
                    result = self._execute_tool_operation(
                        tool_definition,
                        lambda: run_mcp_tool(
                            tool_name,
                            args,
                            self._mcp_runtime,
                            cancellation_token=self._cancellation_token,
                        ),
                    )
                else:
                    result = self._execute_tool_operation(
                        tool_definition,
                        lambda: self._run_handler(tool_name, args, interaction_handler, turn=turn),
                    )
        except RunCancelled as error:
            result = tool_error(type(error).__name__, str(error))
            self._write_trace(
                tool_name,
                args,
                result,
                started,
                policy_decision,
                guardrail_result,
                turn=turn,
            )
            raise
        except Exception as error:
            result = tool_error(type(error).__name__, str(error))

        trace_metadata = result.pop("_trace_metadata", None)
        trace_result = result.pop("_trace_result", None)
        result = self._prepare_model_visible_result(tool_name, result)
        self._write_trace(
            tool_name,
            args,
            result,
            started,
            policy_decision,
            guardrail_result,
            trace_metadata=trace_metadata,
            trace_result=trace_result,
            turn=turn,
        )
        return result

    def raise_for_error(self, result: dict[str, Any]) -> None:
        if result.get("status") == "error":
            error = result.get("error") or {}
            raise ToolRoutingError(
                str(error.get("message", "tool failed")),
                error_type=str(error.get("type", "")),
            )

    def record_skipped(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """记录未启动的 tool call；不跑 policy、guardrail 或 handler。"""
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        # 调度器跳过契约：仅接受结构化 not_started 结果，避免误写成功 trace
        if (
            result.get("status") != "error"
            or error.get("type") != "tool_call_skipped"
            or result.get("execution_state") != "not_started"
        ):
            raise ToolRoutingError(
                "record_skipped only accepts tool_call_skipped not_started errors",
                error_type="tool_call_skipped_invalid",
            )
        self._write_trace(
            tool_name,
            args,
            result,
            started=0.0,
            policy_decision=None,
            guardrail_result=None,
            duration_seconds=0.0,
            turn=None,
        )
        return result

    def record_reused(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        source_result: dict[str, Any],
        reused_from_call_index: int,
        turn: int | None = None,
    ) -> dict[str, Any]:
        """记录同批只读调用复用；handler 未再次执行。"""
        result = {
            "status": str(source_result.get("status", "error")),
            "execution_state": "not_started",
            "duplicate_suppressed": True,
            "reused_from_call_index": reused_from_call_index,
            "model_visible": {
                "same_as_previous": True,
                "tool_name": tool_name,
                "reason": "duplicate_read_call_in_same_batch",
            },
        }
        if result["status"] == "error":
            result["error"] = dict(source_result.get("error") or {})
            if isinstance(source_result.get("recovery"), dict):
                result["recovery"] = dict(source_result["recovery"])
        self._write_trace(
            tool_name,
            args,
            result,
            started=0.0,
            policy_decision=None,
            guardrail_result=None,
            duration_seconds=0.0,
            trace_metadata={
                "duplicate_suppressed": True,
                "reused_from_call_index": reused_from_call_index,
            },
            turn=turn,
        )
        return result

    def _prepare_model_visible_result(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        return prepare_tool_result_for_model(
            tool_name,
            result,
            derive_compression_budget(None),
            self._episode_writer.write_tool_artifact,
        )

    def wait_for_agent_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        if self._agent_runtime is None:
            return {}
        return self._agent_runtime.wait_for_task(task_id, timeout=timeout)

    def _fake_tool(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return {"status": "success", "args": args}

    def _load_image_attachment(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        image_id = str(args["image_id"]).strip()
        attachment = self._image_attachment_history.get(image_id)
        if attachment is None:
            return tool_error(
                "image_attachment_not_found",
                f"image attachment not found in session history: {image_id}",
            )
        root = Path(attachment.base_path).resolve() if attachment.base_path else self._workspace_root
        image_path = (root / attachment.relative_path).resolve()
        if not image_path.is_relative_to(root):
            return tool_error(
                "image_attachment_path_invalid",
                f"image attachment path escapes its session root: {image_id}",
            )
        if not image_path.is_file():
            return tool_error(
                "image_attachment_missing_file",
                f"image attachment file is missing: {image_id}",
            )
        loaded_attachment = attachment.with_absolute_path(root)
        return {
            "status": "success",
            "loaded_image_attachment": loaded_attachment,
            "model_visible": {
                "message": "图片已加载，将在下一次模型调用中作为视觉输入。",
                "image_id": attachment.id,
                "filename": attachment.filename,
                "mime_type": attachment.mime_type,
                "size_bytes": attachment.size_bytes,
                "dimensions": f"{attachment.width}x{attachment.height}",
                "relative_path": attachment.relative_path,
            },
        }

    def _start_memory_update(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "memory_update_requested": True,
            "reason": str(args.get("reason", "")),
        }

    def _agent(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        return _agent_runtime_result(
            self._agent_runtime.spawn_worker(
                description=str(args["description"]),
                prompt=str(args["prompt"]),
                subagent_type=args["subagent_type"],
                team_id=args.get("team"),
                model_profile=args.get("model_profile"),
                profile=args.get("profile"),
            ),
        )

    def _send_message(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        return _agent_runtime_result(self._agent_runtime.send_message(str(args["to"]), str(args["message"])))

    def _task_stop(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        return _agent_runtime_result(
            self._agent_runtime.stop_task(
                str(args["task_id"]),
                force=bool(args.get("force", False)),
            ),
        )

    def _task_get(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        return _agent_runtime_result(self._agent_runtime.task_get(str(args["task_id"])))

    def _task_list(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        status = args.get("status")
        return _agent_runtime_result(
            self._agent_runtime.task_list(status=str(status) if status else None),
        )

    def _task_output(
        self,
        args: dict[str, Any],
        _context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._agent_runtime is None:
            return tool_error("agent_runtime_missing", "agent runtime is not configured")
        return _agent_runtime_result(
            self._agent_runtime.task_output(
                str(args["task_id"]),
                max_chars=int(args.get("max_chars", 12000)),
            ),
        )

    def _request_user_input_handler(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        # 与写工具相同：interaction 经执行上下文注入，不在 dispatch 按名分支。
        return self._request_user_input(args, context.interaction_handler)

    def _todo_update_handler(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._todo_state_sink is None:
            return tool_error("session_state_unavailable", "Todo state sink is unavailable")
        items = args.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return tool_error("tool_argument_invalid", "items must be a list of objects", retryable=False)
        result = self._todo_state_sink(
            [dict(item) for item in items],
            str(args.get("explanation", "")),
            context.turn,
        )
        return {"status": "success", **result}

    def _submit_plan_handler(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        if self._planning_state_handler is None:
            return tool_error("session_state_unavailable", "Planning state handler is unavailable")
        if context.interaction_handler is None:
            return tool_error("plan_confirmation_unavailable", "Plan confirmation requires an interaction handler")
        state = self._planning_state_handler("submit", dict(args), context.turn or 0)
        plan_id = str(state.get("plan_id", ""))
        revision = state.get("revision")
        proposal = state.get("proposal")
        if not plan_id or not isinstance(revision, int) or not isinstance(proposal, dict):
            return tool_error("invalid_planning_state", "Submitted Plan state is incomplete", retryable=False)
        response = context.interaction_handler(
            HumanInteractionRequest(
                interaction_type="plan_confirmation",
                tool_name="submit_plan",
                question="确认实施方案",
                reason="批准后将初始化 Todo 并自动开始执行",
                risk_level="low",
                args_summary={"plan_id": plan_id, "revision": revision, "step_count": len(proposal.get("steps", []))},
                plan_id=plan_id,
                plan_revision=revision,
                plan_proposal=dict(proposal),
            ),
        )
        outcome = response.plan_outcome
        if outcome == "revision_requested":
            feedback = response.answer.strip()
            if not feedback:
                return tool_error("invalid_interaction_response", "Plan feedback must be non-empty", retryable=False)
            self._planning_state_handler(
                "feedback",
                {"plan_id": plan_id, "revision": revision, "feedback": feedback},
                context.turn or 0,
            )
            return {"status": "success", "outcome": "revision_requested", "feedback": feedback}
        if outcome == "approved":
            approved = self._planning_state_handler(
                "approve",
                {"plan_id": plan_id, "revision": revision},
                context.turn or 0,
            )
            return {
                "status": "success",
                "outcome": "approved",
                "execution_id": approved.get("execution_id"),
                "control": "end_turn",
            }
        if outcome == "cancelled":
            return {"status": "success", "outcome": "cancelled", "control": "end_turn"}
        return tool_error("invalid_interaction_response", "Plan confirmation outcome is required", retryable=False)

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
        try:
            questions = parse_user_questions(args)
        except UserQuestionValidationError as error:
            return tool_error("tool_argument_invalid", str(error), retryable=False)
        response = interaction_handler(
            HumanInteractionRequest(
                interaction_type="user_input",
                tool_name="request_user_input",
                question="、".join(question.header for question in questions),
                reason=str(args.get("reason", "")),
                risk_level="low",
                args_summary=interaction_args_summary("request_user_input", args),
                questions=questions,
            ),
        )
        if response.outcome is None:
            return tool_error(
                "invalid_interaction_response",
                "user input response must include outcome",
                retryable=False,
            )
        outcome = response.outcome
        expected = {question.id: question for question in questions}
        unknown_ids = sorted(set(response.answers) - set(expected))
        if unknown_ids:
            return tool_error(
                "invalid_interaction_response",
                f"unknown answer ids: {', '.join(unknown_ids)}",
                retryable=False,
            )
        if outcome == "answered":
            missing_ids = [
                question.id
                for question in questions
                if not response.answers.get(question.id)
                or not all(str(value).strip() for value in response.answers[question.id])
            ]
            if missing_ids:
                return tool_error(
                    "invalid_interaction_response",
                    f"answered response is missing required answers: {', '.join(missing_ids)}",
                    retryable=False,
                )
            invalid_single = [
                question.id
                for question in questions
                if not question.multiple and len(response.answers[question.id]) != 1
            ]
            if invalid_single:
                return tool_error(
                    "invalid_interaction_response",
                    f"single-answer questions returned multiple values: {', '.join(invalid_single)}",
                    retryable=False,
                )
        answers = {
            question_id: [str(value) for value in values if str(value).strip()]
            for question_id, values in response.answers.items()
        }
        return {
            "status": "success",
            "outcome": outcome,
            "answers": answers,
            "question_count": len(questions),
            "answered_count": sum(1 for values in answers.values() if values),
            "answer_chars": sum(len(value) for values in answers.values() for value in values),
        }

    def _handle_denied_policy(
        self,
        tool_name: str,
        args: dict[str, Any],
        policy_decision: PolicyDecision,
        interaction_handler: HumanInteractionHandler | None,
    ) -> tuple[dict[str, Any], PolicyDecision, GuardrailResult | None]:
        if (
            tool_name in self._approval_allowed_tools
            and interaction_handler is None
            and self._worker_permission_requester is not None
        ):
            validation_error = _validate_args(tool_name, args, self._tool_registry)
            if validation_error:
                return validation_error, policy_decision, None
            request = self._worker_permission_requester(tool_name, args, policy_decision)
            return (
                tool_error(
                    "approval_pending",
                    f"worker approval pending: {request.request_id}",
                ),
                policy_decision,
                None,
            )
        if tool_name not in self._approval_allowed_tools or interaction_handler is None:
            return (
                tool_error(
                    "policy_denied",
                    f"{policy_decision.reason}; {policy_decision.approval.reason}",
                ),
                policy_decision,
                None,
            )
        context = ToolExecutionContext(interaction_handler=interaction_handler)
        response = context.ask(_tool_permission_request(tool_name, args, policy_decision))
        if response is None:
            return (
                tool_error(
                    "policy_denied",
                    f"{policy_decision.reason}; interactive approval is unavailable",
                ),
                policy_decision,
                None,
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
        validation_error = _validate_args(tool_name, args, self._tool_registry)
        if validation_error:
            return validation_error, granted_policy, None
        guardrail_result = check_tool_input(tool_name, args)
        if guardrail_result is not None:
            return (
                tool_error("guardrail_denied", guardrail_evidence(guardrail_result)),
                granted_policy,
                guardrail_result,
            )
        if tool_name.startswith("mcp__"):
            return (
                self._execute_tool_operation(
                    self._tool_registry.get(tool_name),
                    lambda: run_mcp_tool(
                        tool_name,
                        args,
                        self._mcp_runtime,
                        cancellation_token=self._cancellation_token,
                    ),
                ),
                granted_policy,
                None,
            )
        return (
            self._execute_tool_operation(
                self._tool_registry.get(tool_name),
                lambda: self._run_handler(tool_name, args, interaction_handler),
            ),
            granted_policy,
            None,
        )

    def _execute_tool_operation(
        self,
        tool_definition: ToolDefinition,
        invoke: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """将通过策略与参数校验的真实工具执行交给统一重试边界。"""
        attempts = 0
        retry_events: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None

        def invoke_checked() -> dict[str, Any]:
            nonlocal attempts, last_result
            attempts += 1
            last_result = _normalize_tool_result(tool_definition.name, invoke())
            failure = _retry_failure(last_result)
            if failure is not None:
                # handler 仍返回 observation；这里只在 Router 内部把临时失败适配给统一重试控制器。
                raise RetryableOperationError(failure)
            return last_result

        def record_retry(event: Any) -> None:
            retry_events.append(
                {
                    "attempt": event.attempt,
                    "next_attempt": event.next_attempt,
                    "category": event.category,
                    "delay_seconds": event.delay_seconds,
                    "source": event.source,
                    "retry_after_ignored": event.retry_after_ignored,
                },
            )

        try:
            result = self._retry_controller.execute(
                RetryOperation(
                    name=f"tool.{tool_definition.name}",
                    replay_safety=tool_definition.replay_safety,
                ),
                invoke_checked,
                cancellation_token=self._cancellation_token,
                on_event=record_retry,
            )
        except RetryableOperationError:
            result = last_result or tool_error(
                "tool_contract_invalid",
                "retryable tool failure did not preserve its observation",
                category=ToolFailureCategory.CONTRACT,
            )
        trace_metadata: dict[str, Any] = {"attempt_count": attempts}
        if retry_events:
            trace_metadata["retry_events"] = retry_events
            trace_metadata["recovered_after_retry"] = result.get("status") == "success"
        result["_trace_metadata"] = trace_metadata
        return result

    def _run_handler(
        self,
        tool_name: str,
        args: dict[str, Any],
        interaction_handler: HumanInteractionHandler | None,
        *,
        turn: int | None = None,
    ) -> dict[str, Any]:
        # 静态工具唯一执行入口：catalog 绑定的 handler + 逐次 ToolExecutionContext。
        context = ToolExecutionContext(interaction_handler=interaction_handler, turn=turn)
        return self._handlers[tool_name](args, context)

    def _mode_or_actor_denial(self, tool_name: str) -> dict[str, Any] | None:
        if self._actor_role != "main" and tool_name in {"todo_update", "submit_plan"}:
            return tool_error("tool_actor_denied", f"{tool_name} is only available to the main Agent")
        if tool_name == "submit_plan" and self._planning_status not in {"planning", "awaiting_confirmation"}:
            return tool_error("plan_mode_required", "submit_plan is only available in Plan Mode")
        if self._planning_status not in {"planning", "awaiting_confirmation"}:
            return None
        if tool_name.startswith("mcp__"):
            return tool_error("plan_mode_tool_denied", "dynamic MCP tools are disabled in Plan Mode")
        from haagent.tools.catalog import default_tool_catalog

        allowed = set(
            default_tool_catalog().plan_mode_tools(
                enable_web=True,
                include_session_history=True,
                include_image_attachment=True,
            ),
        )
        if tool_name not in allowed:
            # 安全边界：即使模型缓存了旧 schema，handler 也绝不执行副作用工具。
            return tool_error("plan_mode_tool_denied", f"tool is disabled in Plan Mode: {tool_name}")
        return None

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
        duration_seconds: float | None = None,
        trace_metadata: dict[str, Any] | None = None,
        trace_result: dict[str, Any] | None = None,
        turn: int | None = None,
    ) -> None:
        effective_trace_result = (
            trace_result if trace_result is not None else _result_for_trace(result)
        )
        # 未启动调用强制 duration=0；真实 dispatch 仍用 wall-clock
        measured = (
            float(duration_seconds)
            if duration_seconds is not None
            else time.perf_counter() - started
        )
        trace = {
                "tool_name": tool_name,
                "args": args,
                "status": result["status"],
                "result": effective_trace_result if result["status"] == "success" else None,
                "error": result.get("error"),
                "policy": policy_decision.to_dict() if policy_decision else None,
                "path_policy": {
                    "permission_mode": self._path_policy.permission_mode,
                    "project_root": str(self._path_policy.project_root),
                    "external_root_count": len(self._path_policy.external_roots),
                },
                "guardrail": guardrail_result.to_dict() if guardrail_result else None,
                "duration_seconds": measured,
            }
        if trace_metadata:
            trace.update(trace_metadata)
        self._episode_writer.append_tool_call(trace)
        # 仅 orchestrator 传入 turn 时记 model-turn 性能；不改 model-visible result。
        if self._performance_sink is not None and turn is not None:
            execution_effect = "unknown"
            if self._tool_registry.has(tool_name):
                execution_effect = self._tool_registry.get(tool_name).execution_effect
            self._performance_sink(
                turn,
                tool_name,
                measured * 1000.0,
                execution_effect,
                str(result.get("status", "unknown")),
            )


def _tool_permission_request(
    tool_name: str,
    args: dict[str, Any],
    decision: PolicyDecision,
) -> ToolPermissionRequest:
    summary = interaction_args_summary(tool_name, args)
    pattern = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    always = pattern
    if tool_name == "shell":
        command = str(summary.get("command", "")).strip()
        prefix = command.split(maxsplit=1)[0] if command else "shell"
        always = f"{prefix} *"
        pattern = command or pattern
    return ToolPermissionRequest(
        permission=tool_name,
        patterns=(pattern,),
        always=(always,),
        metadata=summary,
        question=f"Approve high risk tool {tool_name}?",
        reason=decision.approval.reason,
        risk_level=decision.risk_level,
    )


def _validate_args(
    tool_name: str,
    args: dict[str, Any],
    registry: ToolRuntimeRegistry | None = None,
) -> dict[str, Any] | None:
    runtime_registry = registry or default_tool_runtime_registry()
    schema = runtime_registry.get(tool_name).parameters
    if schema.get("type") != "object":
        return tool_error(
            "tool_registry_invalid",
            "tool arguments schema must be object",
            retryable=False,
        )
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    allowed_arguments = list(properties)
    issue = validate_json_value(args, schema)
    if issue is not None:
        unexpected = issue.path if issue.message.startswith("unexpected argument:") else ""
        suggested_args = (
            _suggested_arguments(args, unexpected=unexpected, allowed=allowed_arguments)
            if unexpected and "." not in unexpected and "[" not in unexpected
            else None
        )
        return _argument_error(
            tool_name,
            issue.message,
            args=args,
            required=required if isinstance(required, list) else [],
            allowed=allowed_arguments,
            field=issue.path,
            expected=issue.expected,
            actual=issue.actual,
            suggested_args=suggested_args,
        )
    return None


def _argument_error(
    tool_name: str,
    message: str,
    *,
    args: dict[str, Any],
    required: list[str],
    allowed: list[str],
    field: str,
    expected: object | None = None,
    actual: object | None = None,
    suggested_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovery_args = suggested_args if suggested_args is not None else None
    return tool_error(
        "tool_argument_invalid",
        message,
        category=ToolFailureCategory.ARGUMENT,
        retryable=False,
        recovery=RecoveryAction(
            "correct_arguments",
            "重写调用，使其满足 expected_schema；不要原样重复失败参数。",
            tool_name=tool_name,
            args=recovery_args,
        ),
        tool_name=tool_name,
        field_path=f"$.{field}",
        required_arguments=list(required),
        allowed_arguments=list(allowed),
        received_arguments=sorted(args),
        expected_schema=expected,
        actual_value=actual,
        suggested_args=suggested_args,
    )


def _suggested_arguments(
    args: dict[str, Any],
    *,
    unexpected: str,
    allowed: list[str],
) -> dict[str, Any] | None:
    matches = get_close_matches(unexpected, allowed, n=2, cutoff=0.74)
    if len(matches) != 1 or matches[0] in args:
        return None
    suggested = dict(args)
    suggested[matches[0]] = suggested.pop(unexpected)
    return suggested


def _normalize_tool_result(tool_name: str, result: object) -> dict[str, Any]:
    if not isinstance(result, dict):
        return tool_error(
            "tool_contract_invalid",
            f"tool {tool_name} returned a non-object result",
            category=ToolFailureCategory.CONTRACT,
            result_type=type(result).__name__,
        )
    status = result.get("status")
    if status in {"success", "running"}:
        return result
    if status != "error":
        return tool_error(
            "tool_contract_invalid",
            f"tool {tool_name} returned invalid status",
            category=ToolFailureCategory.CONTRACT,
            result_keys=sorted(str(key) for key in result),
        )
    error = result.get("error")
    required = {"type", "category", "message", "retryable"}
    if not isinstance(error, dict) or not required <= set(error) or not isinstance(error.get("retryable"), bool):
        return tool_error(
            "tool_contract_invalid",
            f"tool {tool_name} returned an invalid error contract",
            category=ToolFailureCategory.CONTRACT,
            result_keys=sorted(str(key) for key in result),
            error_keys=sorted(str(key) for key in error) if isinstance(error, dict) else [],
        )
    recovery = result.get("recovery")
    if recovery is not None and (
        not isinstance(recovery, dict)
        or recovery.get("action") not in {
            "correct_arguments",
            "retry_same_call",
            "use_tool",
            "use_alternate_source",
            "inspect_state",
            "ask_user",
            "stop",
        }
    ):
        return tool_error(
            "tool_contract_invalid",
            f"tool {tool_name} returned an invalid recovery contract",
            category=ToolFailureCategory.CONTRACT,
        )
    return result


def _retry_failure(result: dict[str, Any]) -> RetryFailure | None:
    if result.get("status") != "error":
        return None
    error = result.get("error")
    if not isinstance(error, dict) or error.get("retryable") is not True:
        return None
    retry_after = error.get("retry_after_seconds")
    status_code = error.get("status_code")
    return RetryFailure(
        category=str(error.get("category", "transient")),
        retryable=True,
        retry_after_seconds=float(retry_after) if isinstance(retry_after, (int, float)) else None,
        status_code=int(status_code) if isinstance(status_code, int) else None,
    )


def _result_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    trace_result = dict(result)
    attachment = trace_result.get("loaded_image_attachment")
    if isinstance(attachment, dict) and "path" in attachment:
        trace_result["loaded_image_attachment"] = {
            key: value
            for key, value in attachment.items()
            if key != "path"
        }
    return trace_result


def _agent_runtime_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("is_error") is True:
        return tool_error("agent_runtime_error", str(result.get("error", "agent runtime failed")))
    return result
