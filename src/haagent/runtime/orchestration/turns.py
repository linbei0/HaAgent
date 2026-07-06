"""
src/haagent/runtime/orchestration/turns.py - Run 单轮执行流程

负责 RunOrchestrator 的模型轮询、工具执行、suggestion、安全处理和完成条件判断。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable

from haagent.context.messages import (
    build_assistant_message,
    build_final_response_request_message,
    build_suggestion_message,
    build_tool_result_message,
    generate_tool_call_id,
)
from haagent.models.gateway import ModelGateway, ModelUsage, ToolCall
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.execution.guardrails import check_assistant_output, guardrail_evidence
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver
from haagent.runtime.orchestration.loop_guidance import (
    safety_violation_observation,
    suggestion_for_observation,
)
from haagent.runtime.orchestration.recorder import RunRecorder, RunResult
from haagent.runtime.orchestration.state import RunStatus
from haagent.tools.base import tool_error
from haagent.tools.registry import ToolRuntimeRegistry, export_tool_schemas
from haagent.tools.router import ToolRouter
from haagent.verification.engine import VerificationEngine


@dataclass
class TurnLoopState:
    messages: list[dict[str, Any]]
    context_id: str
    completion_observations: list[dict[str, object]] = field(default_factory=list)
    final_response_requested: bool = False
    has_file_change: bool = False
    has_shell_verification: bool = False
    passed_verification_commands: set[str] = field(default_factory=set)
    verification_engine: VerificationEngine | None = None
    pending_worker_task_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnLoopDependencies:
    model_gateway: ModelGateway
    writer: EpisodeWriter
    recorder: RunRecorder
    router: ToolRouter
    task_goal: str
    allowed_tools: list[str]
    tool_registry: ToolRuntimeRegistry
    verification_commands: list[str]
    workspace_root: object
    max_turns: int | None
    raise_if_cancelled: Callable[[], None]
    emit_event: Callable[[dict[str, object]], None]
    compress_historical_tool_messages: Callable[[list[dict[str, Any]], EpisodeWriter, int, Callable[[dict[str, object]]]], object]
    interaction_handler: HumanInteractionHandler | None
    interaction_resolver: HumanInteractionResolver
    safety_guard: object
    interaction_bridge_factory: Callable[[int, HumanInteractionResolver], HumanInteractionHandler]
    record_guardrail: Callable[[object, int | None], None]
    record_suggestion: Callable[[int, object], None]
    tool_error_is_terminal: Callable[[dict[str, object]], bool]
    update_in_band_verification_progress: Callable[[str, dict[str, object], dict[str, object], list[str], set[str]], None]
    all_declared_verification_commands_passed: Callable[[list[str], set[str]], bool]
    successful_file_change_without_declared_verification: Callable[[list[dict[str, object]], list[str]], bool]
    verification_observation: Callable[[object], dict[str, object]]
    verification_evidence: Callable[[object], str]
    verification_loop_limit_evidence: Callable[[int, object], str]


def run_turn_loop(
    *,
    state: TurnLoopState,
    deps: TurnLoopDependencies,
) -> RunResult | None:
    turn_numbers = count(1) if deps.max_turns is None else range(1, deps.max_turns + 1)
    for turn in turn_numbers:
        deps.raise_if_cancelled()
        if state.pending_worker_task_ids:
            notifications = _wait_for_pending_worker_tasks(state=state, deps=deps)
            deps.writer.append_transcript(
                {
                    "event": "worker_notifications_collected",
                    "turn": turn,
                    "notifications": notifications,
                },
            )
            state.messages.append(_build_worker_notifications_message(notifications))
        tool_schemas = [] if state.final_response_requested else export_tool_schemas(
            deps.allowed_tools,
            registry=deps.tool_registry,
        )
        deps.compress_historical_tool_messages(state.messages, deps.writer, turn, deps.emit_event)

        deps.writer.append_transcript(
            {
                "event": "model_call",
                "provider": deps.model_gateway.provider_name,
                "context_id": state.context_id,
                "turn": turn,
                "goal": deps.task_goal,
            },
        )
        def emit_assistant_delta(delta: str) -> None:
            deps.raise_if_cancelled()
            deps.emit_event(
                {
                    "event_type": "assistant_delta",
                    "turn": turn,
                    "delta": delta,
                },
            )

        if _supports_event_sink(deps.model_gateway):
            model_response = deps.model_gateway.generate(
                messages=state.messages,
                tool_schemas=tool_schemas,
                event_sink=emit_assistant_delta,
            )
        else:
            model_response = deps.model_gateway.generate(
                messages=state.messages,
                tool_schemas=tool_schemas,
            )
        deps.raise_if_cancelled()
        model_metadata = _gateway_metadata(deps.model_gateway)
        deps.writer.append_model_usage(
            turn=turn,
            provider=deps.model_gateway.provider_name,
            model=model_metadata.get("model"),
            usage=model_response.usage,
        )
        output_guardrail = (
            check_assistant_output(model_response.content)
            if not model_response.tool_calls
            else None
        )
        response_record: dict[str, Any] = {
            "event": "model_response",
            "provider": deps.model_gateway.provider_name,
            "model": model_metadata.get("model"),
            "turn": turn,
            "content": (
                "blocked by output guardrail"
                if output_guardrail is not None
                else model_response.content
            ),
            "tool_calls": [
                {"name": tc.name, "args": tc.args}
                for tc in model_response.tool_calls
            ],
        }
        usage_record = _usage_record(model_response.usage)
        if usage_record is not None:
            response_record["usage"] = usage_record
        deps.writer.append_transcript(response_record)
        if output_guardrail is not None:
            deps.record_guardrail(output_guardrail, turn)
            deps.recorder.transition(RunStatus.FAILED)
            deps.writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": FailureCategory.GUARDRAIL.value,
                    "evidence": guardrail_evidence(output_guardrail),
                },
            )
            return deps.recorder.finish(RunStatus.FAILED)

        if state.final_response_requested and model_response.tool_calls:
            deps.recorder.transition(RunStatus.FAILED)
            deps.writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": FailureCategory.MODEL.value,
                    "evidence": "model returned tool calls during final response turn",
                },
            )
            return deps.recorder.finish(RunStatus.FAILED)

        if not model_response.tool_calls:
            result = _handle_no_tool_response(turn=turn, model_response=model_response, state=state, deps=deps)
            if result is not None:
                return result
            continue

        if deps.recorder.state_history[-1] is not RunStatus.EXECUTING:
            deps.recorder.transition(RunStatus.EXECUTING)

        tool_calls_with_ids = _ensure_tool_call_ids(model_response.tool_calls)
        assistant_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args, ensure_ascii=False),
                },
            }
            for tc in tool_calls_with_ids
        ]
        state.messages.append(build_assistant_message(model_response.content, assistant_tool_calls))

        result = _run_tool_calls(turn=turn, tool_calls_with_ids=tool_calls_with_ids, state=state, deps=deps)
        if result is not None:
            return result
    else:
        deps.recorder.transition(RunStatus.FAILED)
        deps.writer.write_failure_attribution(
            {
                "stage": "executing",
                "category": FailureCategory.LOOP_LIMIT.value,
                "evidence": f"exceeded max_turns={deps.max_turns}",
            },
        )
        return deps.recorder.finish(RunStatus.FAILED)
    return None


def _handle_no_tool_response(
    *,
    turn: int,
    model_response,
    state: TurnLoopState,
    deps: TurnLoopDependencies,
) -> RunResult | None:
    if state.final_response_requested and not model_response.content.strip():
        deps.recorder.transition(RunStatus.FAILED)
        deps.writer.write_failure_attribution(
            {
                "stage": "executing",
                "category": FailureCategory.MODEL.value,
                "evidence": "model returned empty final response during final response turn",
            },
        )
        return deps.recorder.finish(RunStatus.FAILED)
    if state.pending_worker_task_ids:
        notifications = _wait_for_pending_worker_tasks(state=state, deps=deps)
        deps.writer.append_transcript(
            {
                "event": "worker_notifications_collected",
                "turn": turn,
                "notifications": notifications,
            },
        )
        state.messages.append(_build_worker_notifications_message(notifications))
        return None
    deps.writer.append_transcript(
        {
            "event": "no_tool_reviewed",
            "turn": turn,
            "guidance_added": False,
            "trigger": None,
        },
    )
    deps.emit_event(
        {
            "event_type": "assistant_message",
            "turn": turn,
            "content": model_response.content,
        },
    )
    deps.recorder.transition(RunStatus.VERIFYING)
    if state.verification_engine is None:
        state.verification_engine = VerificationEngine(deps.writer, deps.workspace_root)
    deps.raise_if_cancelled()
    verification_result = state.verification_engine.run(deps.verification_commands)
    if verification_result.status == "success":
        deps.recorder.transition(RunStatus.COMPLETED)
        deps.writer.write_failure_attribution(None)
        return deps.recorder.finish(RunStatus.COMPLETED)

    verification_obs = deps.verification_observation(verification_result)
    ver_msg = build_suggestion_message(
        f"Verification failed: {deps.verification_evidence(verification_result)}. "
        "Use the failure details to repair the workspace, then try again."
    )
    state.messages.append(ver_msg)
    state.final_response_requested = False
    if deps.max_turns is not None and turn == deps.max_turns:
        deps.recorder.transition(RunStatus.FAILED)
        deps.writer.write_failure_attribution(
            {
                "stage": "verifying",
                "category": FailureCategory.LOOP_LIMIT.value,
                "evidence": deps.verification_loop_limit_evidence(
                    deps.max_turns,
                    verification_result,
                ),
            },
        )
        return deps.recorder.finish(RunStatus.FAILED)
    return None


def _run_tool_calls(
    *,
    turn: int,
    tool_calls_with_ids: list[ToolCall],
    state: TurnLoopState,
    deps: TurnLoopDependencies,
) -> RunResult | None:
    turn_broke_early = False
    pending_suggestion_messages: list[str] = []
    visible_result_fingerprints: set[str] = set()
    for tool_index, tool_call in enumerate(tool_calls_with_ids):
        deps.raise_if_cancelled()
        deps.emit_event(
            {
                "event_type": "tool_started",
                "turn": turn,
                "tool_name": tool_call.name,
                "args": tool_call.args,
            },
        )
        tool_result = deps.router.dispatch(
            tool_call.name,
            tool_call.args,
            interaction_handler=(
                deps.interaction_bridge_factory(turn, deps.interaction_resolver)
                if deps.interaction_handler is not None
                else None
            ),
        )
        deps.raise_if_cancelled()
        if tool_result.get("status") == "error":
            error = tool_result.get("error") or {}
            failed_event = {
                "event_type": "tool_failed",
                "turn": turn,
                "tool_name": tool_call.name,
                "args": tool_call.args,
                "error": error,
            }
            deps.emit_event(failed_event)
        observation = {
            "tool_name": tool_call.name,
            "args": tool_call.args,
            "result": tool_result,
        }
        deps.writer.append_transcript(
            {
                "event": "tool_observation",
                "turn": turn,
                **observation,
            },
        )
        message_result = tool_result
        if tool_result.get("status") != "error":
            fingerprint = _tool_visible_result_fingerprint(tool_call.name, tool_call.args, tool_result)
            if fingerprint in visible_result_fingerprints:
                message_result = {
                    "status": "success",
                    "model_visible": {
                        "same_as_previous": True,
                        "tool_name": tool_call.name,
                        "reason": "duplicate_tool_result_in_same_turn",
                    },
                }
            else:
                visible_result_fingerprints.add(fingerprint)
        state.messages.append(build_tool_result_message(tool_call.id, tool_call.name, message_result))
        if tool_call.name == "agent" and tool_result.get("status") == "running":
            task_id = str(tool_result.get("task_id") or "").strip()
            if task_id:
                state.pending_worker_task_ids.append(task_id)

        violation = deps.safety_guard.check(tool_call.name, tool_call.args, tool_result)
        if violation is not None and violation.should_abort:
            abort_obs = safety_violation_observation(violation.message, violation.recovery_suggestion)
            deps.writer.append_transcript({"event": "safety_abort", "turn": turn, **abort_obs})
            deps.emit_event(
                {
                    "event_type": "safety_abort",
                    "turn": turn,
                    "violation_type": violation.type,
                    "message": violation.message,
                }
            )
            deps.recorder.transition(RunStatus.FAILED)
            deps.writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": FailureCategory.LOOP_LIMIT.value,
                    "evidence": violation.message,
                }
            )
            return deps.recorder.finish(RunStatus.FAILED)

        if tool_result.get("status") == "error":
            if deps.tool_error_is_terminal(tool_result):
                deps.router.raise_for_error(tool_result)
            suggestion = suggestion_for_observation(observation)
            if violation is not None:
                safety_obs = safety_violation_observation(violation.message, violation.recovery_suggestion)
                deps.writer.append_transcript({"event": "safety_warning", "turn": turn, **safety_obs})
                pending_suggestion_messages.append(str(safety_obs.get("result", {}).get("message", "")))
            elif suggestion is not None:
                deps.record_suggestion(turn, suggestion)
                pending_suggestion_messages.append(suggestion.message)
            turn_broke_early = True
            for skipped_call in tool_calls_with_ids[tool_index + 1 :]:
                skipped_result = tool_error(
                    "tool_call_skipped",
                    "tool call skipped because an earlier tool call in the same assistant message failed.",
                )
                deps.writer.append_tool_call(
                    {
                        "tool_name": skipped_call.name,
                        "args": skipped_call.args,
                        "status": "error",
                        "result": None,
                        "error": skipped_result["error"],
                        "policy": None,
                        "guardrail": None,
                        "duration_seconds": 0.0,
                    },
                )
                skipped_observation = {
                    "tool_name": skipped_call.name,
                    "args": skipped_call.args,
                    "result": skipped_result,
                }
                deps.writer.append_transcript(
                    {
                        "event": "tool_observation",
                        "turn": turn,
                        **skipped_observation,
                    },
                )
                state.messages.append(
                    build_tool_result_message(
                        skipped_call.id,
                        skipped_call.name,
                        skipped_result,
                    ),
                )
            break

        deps.emit_event(
            {
                "event_type": "tool_finished",
                "turn": turn,
                "tool_name": tool_call.name,
                "args": tool_call.args,
                "result": tool_result,
            },
        )
        suggestion = suggestion_for_observation(observation)
        if suggestion is not None:
            deps.record_suggestion(turn, suggestion)
            pending_suggestion_messages.append(suggestion.message)

        if tool_call.name in {"apply_patch", "apply_patch_set", "file_write"}:
            state.completion_observations = [observation]
            state.has_file_change = True
        else:
            state.completion_observations.append(observation)
        if tool_call.name in {"shell", "code_run"} and tool_result.get("exit_code") == 0:
            state.has_shell_verification = True
        deps.update_in_band_verification_progress(
            tool_call.name,
            tool_call.args,
            tool_result,
            deps.verification_commands,
            state.passed_verification_commands,
        )

    for suggestion_message in pending_suggestion_messages:
        if suggestion_message:
            state.messages.append(build_suggestion_message(suggestion_message))

    if not turn_broke_early and (
        deps.all_declared_verification_commands_passed(
            deps.verification_commands,
            state.passed_verification_commands,
        )
        or (state.has_file_change and state.has_shell_verification and not deps.verification_commands)
        or deps.successful_file_change_without_declared_verification(
            state.completion_observations,
            deps.verification_commands,
        )
    ):
        state.final_response_requested = True
        state.messages.append(build_final_response_request_message())
    return None


def _wait_for_pending_worker_tasks(
    *,
    state: TurnLoopState,
    deps: TurnLoopDependencies,
) -> list[dict[str, Any]]:
    task_ids = list(state.pending_worker_task_ids)
    notifications: list[dict[str, Any]] = []
    for task_id in task_ids:
        while True:
            deps.raise_if_cancelled()
            notification = deps.router.wait_for_agent_task(task_id, timeout=0.2)
            deps.raise_if_cancelled()
            if notification:
                notifications.append(notification)
                break
    state.pending_worker_task_ids.clear()
    return notifications


def _build_worker_notifications_message(notifications: list[dict[str, Any]]) -> dict[str, Any]:
    lines = ["Worker notifications:"]
    for notification in notifications:
        status = str(notification.get("status") or "unknown")
        task_id = str(notification.get("task_id") or "unknown-task")
        agent_id = str(notification.get("agent_id") or "unknown-agent")
        summary = str(
            notification.get("summary")
            or notification.get("result_excerpt")
            or notification.get("error")
            or ""
        )
        if summary:
            lines.append(f"- {agent_id} ({task_id}) {status}: {summary[:500]}")
        else:
            lines.append(f"- {agent_id} ({task_id}) {status}")
    return {"role": "user", "content": "\n".join(lines)}


def _ensure_tool_call_ids(tool_calls: list[ToolCall]) -> list[ToolCall]:
    tool_calls_with_ids: list[ToolCall] = []
    for tc in tool_calls:
        if tc.id:
            tool_calls_with_ids.append(tc)
        else:
            tool_calls_with_ids.append(ToolCall(name=tc.name, args=tc.args, id=generate_tool_call_id()))
    return tool_calls_with_ids


def _tool_visible_result_fingerprint(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    visible_result = result.get("model_visible")
    if visible_result is None:
        visible_result = {k: v for k, v in result.items() if k != "status"}
    return json.dumps(
        {
            "tool_name": tool_name,
            "args": args,
            "visible_result": visible_result,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _supports_event_sink(model_gateway: ModelGateway) -> bool:
    try:
        signature = inspect.signature(model_gateway.generate)
    except (TypeError, ValueError):
        return False
    return "event_sink" in signature.parameters


def _usage_record(usage: ModelUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "raw_usage_source": usage.raw_source,
    }


def _gateway_metadata(model_gateway: ModelGateway) -> dict[str, object]:
    metadata_getter = getattr(model_gateway, "metadata", None)
    if not callable(metadata_getter):
        return {"model": None}
    metadata = metadata_getter()
    return {"model": metadata.model}
