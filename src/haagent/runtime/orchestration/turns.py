"""
src/haagent/runtime/orchestration/turns.py - Run 单轮执行流程

负责 RunOrchestrator 的模型轮询、工具执行、suggestion、安全处理和完成条件判断。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from typing import Any, Callable

from haagent.context.messages import (
    build_assistant_message,
    build_final_response_request_message,
    build_suggestion_message,
    build_tool_result_message,
    generate_tool_call_id,
)
from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelGateway, ModelUsage, ToolCall
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.cancellation import RunCancelled
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.execution.guardrails import check_assistant_output, guardrail_evidence
from haagent.runtime.execution.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    HumanInteractionResponse,
)
from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver
from haagent.runtime.execution.progress_guard import ProgressDecision, ProgressFrame, ProgressGuard
from haagent.runtime.orchestration.loop_guidance import suggestion_for_observation
from haagent.runtime.orchestration.recorder import RunRecorder, RunResult
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.events.bus import (
    AssistantDeltaBusEvent,
    AssistantIntermediateBusEvent,
    AssistantMessageBusEvent,
    RuntimeBusEvent,
    ToolFailedBusEvent,
    ToolFinishedBusEvent,
    ToolStartedBusEvent,
    bus_event_to_dict,
    coerce_bus_event,
)
from haagent.runtime.orchestration.task_progress import (
    map_failure_to_recovery,
    task_budget_warning_event,
    task_checkpoint_saved_event,
    task_recovery_suggested_event,
    task_step_blocked_event,
    task_step_progress_event,
)
from haagent.runtime.performance import PerformanceTrace
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
    changed_files: list[dict[str, object]] = field(default_factory=list)
    verification_engine: VerificationEngine | None = None
    pending_worker_task_ids: list[str] = field(default_factory=list)
    progress_warn_emitted: bool = False


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
    emit_event: Callable[[RuntimeBusEvent], None]
    compress_historical_tool_messages: Callable[
        [list[dict[str, Any]], EpisodeWriter, int, Callable[[RuntimeBusEvent | dict[str, object]], None]],
        object,
    ]
    interaction_handler: HumanInteractionHandler | None
    interaction_resolver: HumanInteractionResolver
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
    task_step_id: str = "step-001"
    task_step_title: str = ""
    cancellation_token: CancellationToken | None = None
    max_parallel_read_tools: int = 4
    performance_trace: PerformanceTrace | None = None
    persist_performance: Callable[[], None] | None = None
    tool_schema_cache: object | None = None
    progress_guard: ProgressGuard | None = None
    progress_guard_mode: str = "warn"
    on_progress_blocked: Callable[[int, ProgressDecision], None] | None = None


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
        def record_schema_cache(value: dict[str, object]) -> None:
            if deps.performance_trace is not None:
                deps.performance_trace.record_cache_diagnostic("tool_schema", value)

        tool_schemas = [] if state.final_response_requested else export_tool_schemas(
            deps.allowed_tools,
            registry=deps.tool_registry,
            cache=deps.tool_schema_cache,
            diagnostics_sink=record_schema_cache,
        )
        deps.compress_historical_tool_messages(state.messages, deps.writer, turn, deps.emit_event)

        schema_bytes = len(
            json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        if deps.performance_trace is not None:
            deps.performance_trace.begin_model_turn(
                turn=turn,
                message_count=len(state.messages),
                visible_tool_count=len(tool_schemas),
                schema_bytes=schema_bytes,
                stable_prefix_fingerprint=_stable_prefix_fingerprint(
                    state.messages,
                    tool_schemas,
                ),
            )

        max_attempts = _retry_max_attempts(deps.model_gateway)
        model_attempt = 1
        deps.writer.append_transcript(
            {
                "event": "model_call",
                "provider": deps.model_gateway.provider_name,
                "context_id": state.context_id,
                "turn": turn,
                "attempt": model_attempt,
                "max_attempts": max_attempts,
                "goal": deps.task_goal,
            },
        )
        deps.emit_event(
            coerce_bus_event(
                task_step_progress_event(
                    step_id=deps.task_step_id,
                    title=_task_step_title(deps),
                    phase="model_turn_started",
                    summary=f"model turn {turn} started",
                ),
            ),
        )
        def emit_assistant_delta(delta: str) -> None:
            deps.raise_if_cancelled()
            deps.emit_event(AssistantDeltaBusEvent(turn=turn, delta=delta))

        set_route_event_sink = getattr(deps.model_gateway, "set_route_event_sink", None)
        if callable(set_route_event_sink):
            def emit_model_route_event(route_event) -> None:
                from haagent.runtime.events.bus import ModelRouteFallbackBusEvent

                bus_event = ModelRouteFallbackBusEvent(
                    turn=turn,
                    kind=route_event.kind,
                    reason=route_event.reason,
                    from_connection=route_event.from_connection,
                    from_model=route_event.from_model,
                    from_protocol=route_event.from_protocol,
                    to_connection=route_event.to_connection,
                    to_model=route_event.to_model,
                    to_protocol=route_event.to_protocol,
                    required_capabilities=route_event.required_capabilities,
                    missing_capabilities=route_event.missing_capabilities,
                )
                deps.emit_event(bus_event)
                deps.writer.append_transcript(bus_event_to_dict(bus_event))

            set_route_event_sink(emit_model_route_event)

        generate_kwargs: dict[str, object] = {"messages": state.messages, "tool_schemas": tool_schemas}
        if _supports_generate_parameter(deps.model_gateway, "event_sink"):
            generate_kwargs["event_sink"] = emit_assistant_delta
        if _supports_generate_parameter(deps.model_gateway, "retry_event_sink"):
            def emit_retry(retry_event) -> None:
                nonlocal model_attempt
                deps.writer.append_transcript(
                    {
                        "event": "model_attempt_failed",
                        "turn": turn,
                        "attempt": retry_event.attempt,
                        "category": retry_event.category,
                    },
                )
                model_attempt = retry_event.next_attempt
                retry_record = {
                    "event": "model_retry_scheduled",
                    "turn": turn,
                    "attempt": retry_event.attempt,
                    "next_attempt": retry_event.next_attempt,
                    "category": retry_event.category,
                    "delay_seconds": retry_event.delay_seconds,
                    "source": retry_event.source,
                    "retry_after_ignored": retry_event.retry_after_ignored,
                }
                deps.writer.append_transcript(retry_record)
                deps.emit_event({"event_type": "model_retry_scheduled", **{key: value for key, value in retry_record.items() if key != "event"}})
                deps.writer.append_transcript(
                    {
                        "event": "model_call",
                        "provider": deps.model_gateway.provider_name,
                        "context_id": state.context_id,
                        "turn": turn,
                        "attempt": model_attempt,
                        "max_attempts": max_attempts,
                        "goal": deps.task_goal,
                    },
                )
                if deps.persist_performance is not None:
                    deps.persist_performance()
            generate_kwargs["retry_event_sink"] = emit_retry
        if _supports_generate_parameter(deps.model_gateway, "retry_exhausted_sink"):
            def emit_retry_exhausted(failure, attempt: int) -> None:
                deps.writer.append_transcript(
                    {
                        "event": "model_attempt_failed",
                        "turn": turn,
                        "attempt": attempt,
                        "category": failure.category,
                        "status_code": failure.status_code,
                        "request_id": failure.request_id,
                    },
                )
                if failure.category == "stream_interrupted":
                    return
                retry_record = {
                    "event": "model_retry_exhausted",
                    "turn": turn,
                    "attempt": attempt,
                    "category": failure.category,
                    "status_code": failure.status_code,
                    "request_id": failure.request_id,
                }
                deps.writer.append_transcript(retry_record)
                deps.emit_event({"event_type": "model_retry_exhausted", **{key: value for key, value in retry_record.items() if key != "event"}})
                if deps.persist_performance is not None:
                    deps.persist_performance()
            generate_kwargs["retry_exhausted_sink"] = emit_retry_exhausted
        if _supports_generate_parameter(deps.model_gateway, "cancellation_token"):
            generate_kwargs["cancellation_token"] = deps.cancellation_token
        if deps.performance_trace is not None and _supports_generate_parameter(
            deps.model_gateway,
            "telemetry_sink",
        ):
            def emit_transport(event: ModelTransportEvent) -> None:
                assert deps.performance_trace is not None
                deps.performance_trace.record_transport_event(event)
                # attempt 边界写盘；首个 SSE/text 只更新内存，避免逐 token I/O。
                if event.kind in {
                    "attempt_started",
                    "attempt_finished",
                    "attempt_failed",
                } and deps.persist_performance is not None:
                    deps.persist_performance()

            generate_kwargs["telemetry_sink"] = emit_transport
        model_response = deps.model_gateway.generate(**generate_kwargs)
        deps.raise_if_cancelled()
        if deps.performance_trace is not None:
            deps.performance_trace.record_model_usage(turn, model_response.usage)
            if deps.persist_performance is not None:
                deps.persist_performance()
        model_metadata = _gateway_metadata(deps.model_gateway)
        deps.writer.append_model_usage(
            turn=turn,
            attempt=model_attempt,
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

        terminal_memory_response = _is_terminal_memory_settlement_response(model_response)
        # 普通工具轮次的 assistant 文本作为过程展示；记忆结算是回答后的收尾动作，
        # 不能把已经完成的用户答案降级成过程并强制模型再生成一次短总结。
        if model_response.content.strip() and not terminal_memory_response:
            deps.emit_event(AssistantIntermediateBusEvent(turn=turn, content=model_response.content))

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
        if terminal_memory_response:
            result = _handle_no_tool_response(turn=turn, model_response=model_response, state=state, deps=deps)
            if result is not None:
                return result
            continue
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
    deps.emit_event(AssistantMessageBusEvent(turn=turn, content=model_response.content))
    deps.recorder.transition(RunStatus.VERIFYING)
    if state.verification_engine is None:
        state.verification_engine = VerificationEngine(deps.writer, deps.workspace_root)
    deps.raise_if_cancelled()
    if state.changed_files:
        verification_result = state.verification_engine.run(
            deps.verification_commands,
            changed_files=state.changed_files,
        )
    else:
        verification_result = state.verification_engine.run(deps.verification_commands)
    if verification_result.status == "success":
        deps.recorder.transition(RunStatus.COMPLETED)
        deps.writer.write_failure_attribution(None)
        return deps.recorder.finish(RunStatus.COMPLETED)

    verification_obs = deps.verification_observation(verification_result)
    deps.emit_event(
        coerce_bus_event(
            task_checkpoint_saved_event(
                step_id=deps.task_step_id,
                title=_task_step_title(deps),
                status="failed",
                evidence_count=0,
                checkpoint_count=0,
            ),
        ),
    )
    recovery = map_failure_to_recovery(
        {
            "event_type": "verification_failed",
            "reason": deps.verification_evidence(verification_result),
        },
    )
    if recovery is not None:
        deps.emit_event(
            coerce_bus_event(
                task_recovery_suggested_event(
                    step_id=deps.task_step_id,
                    title=_task_step_title(deps),
                    category=recovery.category,
                    reason=recovery.reason,
                    suggested_action=recovery.suggested_action,
                ),
            ),
        )
    ver_msg = build_suggestion_message(
        f"Verification failed: {deps.verification_evidence(verification_result)}. "
        "Use the failure details to repair the workspace, then try again."
    )
    state.messages.append(ver_msg)
    state.final_response_requested = False
    if deps.max_turns is not None and turn == deps.max_turns:
        deps.emit_event(
            coerce_bus_event(
                task_budget_warning_event(
                    step_id=deps.task_step_id,
                    title=_task_step_title(deps),
                    category="turn_budget",
                    reason=f"verification failed on final turn {turn}/{deps.max_turns}",
                    suggested_action="checkpoint_and_resume",
                ),
            ),
        )
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
    loaded_image_attached_this_turn = False
    pending_suggestion_messages: list[str] = []
    visible_result_fingerprints: set[str] = set()
    terminal_error: dict[str, object] | None = None
    verification_count_before = len(state.passed_verification_commands)
    tool_results = _dispatch_tool_calls(turn=turn, tool_calls=tool_calls_with_ids, deps=deps)
    for tool_call, tool_result in zip(tool_calls_with_ids, tool_results):
        deps.raise_if_cancelled()
        if tool_result.get("status") == "error":
            error = tool_result.get("error") or {}
            if not isinstance(error, dict):
                error = {"type": "unknown", "message": str(error)}
            failed_event = ToolFailedBusEvent(
                turn=turn,
                tool_name=tool_call.name,
                args=dict(tool_call.args),
                error=dict(error),
                execution_state=str(tool_result.get("execution_state", "")),
            )
            deps.emit_event(failed_event)
            recovery = map_failure_to_recovery(bus_event_to_dict(failed_event))
            if recovery is not None:
                deps.emit_event(
                    coerce_bus_event(
                        task_recovery_suggested_event(
                            step_id=deps.task_step_id,
                            title=_task_step_title(deps),
                            category=recovery.category,
                            reason=recovery.reason,
                            suggested_action=recovery.suggested_action,
                        ),
                    ),
                )
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
        loaded_image_message = _build_loaded_image_attachment_message(tool_result)
        if loaded_image_message is not None:
            state.messages.append(loaded_image_message)
            loaded_image_attached_this_turn = True
        if tool_call.name == "agent" and tool_result.get("status") == "running":
            task_id = str(tool_result.get("task_id") or "").strip()
            if task_id:
                state.pending_worker_task_ids.append(task_id)

        if tool_result.get("status") == "error":
            if deps.tool_error_is_terminal(tool_result):
                terminal_error = terminal_error or tool_result
            suggestion = suggestion_for_observation(observation)
            if suggestion is not None:
                deps.record_suggestion(turn, suggestion)
                pending_suggestion_messages.append(suggestion.message)
            turn_broke_early = True
            continue

        deps.emit_event(
            ToolFinishedBusEvent(
                turn=turn,
                tool_name=tool_call.name,
                args=dict(tool_call.args),
                result=dict(tool_result),
            ),
        )
        suggestion = suggestion_for_observation(observation)
        if suggestion is not None:
            deps.record_suggestion(turn, suggestion)
            pending_suggestion_messages.append(suggestion.message)

        if tool_call.name in {"apply_patch", "apply_patch_set", "file_write"}:
            state.completion_observations = [observation]
            state.has_file_change = True
            _record_changed_files(state.changed_files, tool_result)
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

    success_count = sum(1 for result in tool_results if result.get("status") != "error")
    if success_count:
        deps.emit_event(
            coerce_bus_event(
                task_step_progress_event(
                    step_id=deps.task_step_id,
                    title=_task_step_title(deps),
                    phase="tool_batch_finished",
                    summary=f"completed {success_count} tool call(s)",
                    evidence_count=success_count,
                ),
            ),
        )

    progress_result = _apply_progress_guard(
        turn=turn,
        tool_calls=tool_calls_with_ids,
        tool_results=tool_results,
        state=state,
        deps=deps,
        verification_count_before=verification_count_before,
    )
    if progress_result is not None:
        return progress_result

    if terminal_error is not None:
        deps.router.raise_for_error(terminal_error)

    if not turn_broke_early and not loaded_image_attached_this_turn and (
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


def _apply_progress_guard(
    *,
    turn: int,
    tool_calls: list[ToolCall],
    tool_results: list[dict[str, Any]],
    state: TurnLoopState,
    deps: TurnLoopDependencies,
    verification_count_before: int,
) -> RunResult | None:
    """完整工具 batch 后观察 ProgressGuard；mode 控制 warn/block 行为。"""
    guard = deps.progress_guard
    if guard is None:
        return None
    mode = deps.progress_guard_mode or "warn"
    frame = _build_progress_frame(
        tool_calls=tool_calls,
        tool_results=tool_results,
        state=state,
        verification_count_before=verification_count_before,
    )
    decision = guard.observe(frame)
    if frame.workspace_changed or frame.verification_progressed:
        state.progress_warn_emitted = False

    effective = _effective_progress_level(decision.level, mode=mode)
    if effective == "none":
        return None
    if effective == "warn":
        if not state.progress_warn_emitted:
            suggestion = (
                f"ProgressGuard ({decision.pattern or 'unknown'}): {decision.reason} "
                "请更换策略，避免重复相同 action/observation。"
            )
            state.messages.append(build_suggestion_message(suggestion))
            state.progress_warn_emitted = True
            deps.writer.append_transcript(
                {
                    "event": "progress_guard_warning",
                    "turn": turn,
                    "pattern": decision.pattern,
                    "reason": decision.reason,
                },
            )
        return None
    # block：可恢复交互
    if deps.on_progress_blocked is not None:
        deps.on_progress_blocked(turn, decision)
    deps.emit_event(
        coerce_bus_event(
            task_step_blocked_event(
                step_id=deps.task_step_id,
                title=_task_step_title(deps),
                category=decision.pattern or "progress_guard",
                reason=decision.reason,
                suggested_action="continue, replan, or stop",
            ),
        ),
    )
    deps.writer.append_transcript(
        {
            "event": "progress_guard_blocked",
            "turn": turn,
            "pattern": decision.pattern,
            "reason": decision.reason,
        },
    )
    return _resolve_progress_block(turn=turn, decision=decision, state=state, deps=deps)


def _effective_progress_level(level: str, *, mode: str) -> str:
    if mode == "off":
        return "none"
    if mode == "warn" and level == "block":
        # warn 模式不进入可恢复 block，只提示
        return "warn"
    return level


def _resolve_progress_block(
    *,
    turn: int,
    decision: ProgressDecision,
    state: TurnLoopState,
    deps: TurnLoopDependencies,
) -> RunResult | None:
    handler = _interaction_handler_for_turn(turn, deps)
    if handler is None:
        return _fail_progress_block(
            turn=turn,
            deps=deps,
            evidence=f"progress_guard blocked ({decision.pattern}): no interaction handler; {decision.reason}",
        )
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="progress_guard",
        question="检测到任务可能陷入重复。请输入 continue、replan 或 stop。",
        reason=decision.reason,
        args_summary={"choices": ["continue", "replan", "stop"], "pattern": decision.pattern},
    )
    try:
        response = handler(request)
    except Exception as error:
        return _fail_progress_block(
            turn=turn,
            deps=deps,
            evidence=f"progress_guard blocked interaction failed: {error}",
        )
    if not isinstance(response, HumanInteractionResponse):
        return _fail_progress_block(
            turn=turn,
            deps=deps,
            evidence="progress_guard blocked: invalid interaction response",
        )
    choice = _normalize_progress_choice(response.answer)
    if choice in {"continue", "replan"} and deps.progress_guard is not None:
        deps.progress_guard.reset()
        state.progress_warn_emitted = False
        deps.writer.append_transcript(
            {
                "event": "progress_guard_recovered",
                "turn": turn,
                "choice": choice,
                "pattern": decision.pattern,
            },
        )
        if choice == "replan":
            state.messages.append(
                build_suggestion_message(
                    "用户要求 replan：请更换策略与工具路径，不要重复相同 action/observation。",
                ),
            )
        return None
    return _fail_progress_block(
        turn=turn,
        deps=deps,
        evidence=(
            f"progress_guard blocked ({decision.pattern}): user chose stop or empty; "
            f"{decision.reason}"
        ),
    )


def _fail_progress_block(*, turn: int, deps: TurnLoopDependencies, evidence: str) -> RunResult:
    deps.recorder.transition(RunStatus.FAILED)
    deps.writer.write_failure_attribution(
        {
            "stage": "executing",
            "category": FailureCategory.LOOP_LIMIT.value,
            "evidence": evidence,
        },
    )
    return deps.recorder.finish(RunStatus.FAILED)


def _normalize_progress_choice(answer: str) -> str:
    text = " ".join(str(answer or "").strip().lower().split())
    if text in {"continue", "replan", "stop"}:
        return text
    if text.startswith("continue"):
        return "continue"
    if text.startswith("replan"):
        return "replan"
    if text.startswith("stop"):
        return "stop"
    return ""


def _build_progress_frame(
    *,
    tool_calls: list[ToolCall],
    tool_results: list[dict[str, Any]],
    state: TurnLoopState,
    verification_count_before: int,
) -> ProgressFrame:
    pairs: list[tuple[str, dict[str, Any], str, str]] = []
    has_running_tool = False
    waiting_approval = False
    waiting_user_input = False
    workspace_changed = False
    for tool_call, tool_result in zip(tool_calls, tool_results):
        status = str(tool_result.get("status") or "success")
        execution_state = str(tool_result.get("execution_state") or "")
        if status == "running" or execution_state in {"running", "pending", "not_started"}:
            has_running_tool = True
        if execution_state in {"awaiting_approval", "approval_required"} or tool_result.get("approval_required"):
            waiting_approval = True
        if tool_call.name == "request_user_input":
            waiting_user_input = True
        pair_status = "error" if status == "error" else "success"
        pairs.append(
            (
                tool_call.name,
                dict(tool_call.args),
                _model_visible_observation(tool_result),
                pair_status,
            ),
        )
        if status != "error" and tool_call.name in {"apply_patch", "apply_patch_set", "file_write"}:
            workspace_changed = True
        if status != "error" and tool_result.get("changed_files"):
            workspace_changed = True
    verification_progressed = len(state.passed_verification_commands) > verification_count_before
    return ProgressFrame(
        pairs=tuple(pairs),
        workspace_changed=workspace_changed,
        verification_progressed=verification_progressed,
        context_chars=_estimate_context_chars(state.messages),
        has_running_tool=has_running_tool,
        has_running_worker=bool(state.pending_worker_task_ids),
        waiting_approval=waiting_approval,
        waiting_user_input=waiting_user_input,
    )


def _model_visible_observation(result: dict[str, Any]) -> str:
    visible = result.get("model_visible")
    if visible is not None:
        return json.dumps(visible, ensure_ascii=False, sort_keys=True, default=str)
    skip = {
        "duration_ms",
        "duration",
        "episode_path",
        "tool_call_id",
        "response_id",
        "timestamp",
    }
    cleaned = {key: value for key, value in result.items() if key not in skip}
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=str)


def _estimate_context_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(json.dumps(content, ensure_ascii=False, default=str))
    return total


def _dispatch_tool_calls(
    *,
    turn: int,
    tool_calls: list[ToolCall],
    deps: TurnLoopDependencies,
) -> list[dict[str, Any]]:
    """按 effect 划分只读批次与串行屏障，并保持原始 tool-call 顺序。"""
    if not tool_calls:
        return []
    if deps.max_parallel_read_tools <= 0:
        raise ValueError("max_parallel_read_tools must be > 0")

    interaction_handler = _interaction_handler_for_turn(turn, deps)
    results: list[dict[str, Any] | None] = [None] * len(tool_calls)
    index = 0
    stop_remaining = False

    while index < len(tool_calls):
        if stop_remaining:
            _skip_remaining_tool_calls(
                tool_calls=tool_calls,
                results=results,
                start_index=index,
                deps=deps,
            )
            break

        deps.raise_if_cancelled()
        tool_call = tool_calls[index]
        if _tool_call_is_parallel_safe(tool_call, deps=deps):
            batch_end = index + 1
            while batch_end < len(tool_calls) and _tool_call_is_parallel_safe(
                tool_calls[batch_end],
                deps=deps,
            ):
                batch_end += 1
            failed = _run_read_batch(
                turn=turn,
                tool_calls=tool_calls,
                start_index=index,
                end_index=batch_end,
                results=results,
                deps=deps,
                interaction_handler=interaction_handler,
            )
            if failed:
                stop_remaining = True
                if batch_end < len(tool_calls):
                    _skip_remaining_tool_calls(
                        tool_calls=tool_calls,
                        results=results,
                        start_index=batch_end,
                        deps=deps,
                    )
            index = batch_end
            continue

        # 串行屏障：写入、外部副作用、交互、高风险与未知工具独占执行
        deps.raise_if_cancelled()
        deps.emit_event(_tool_started_event(turn, tool_call))
        result = _dispatch_tool_call(tool_call, deps, interaction_handler, turn=turn)
        results[index] = result
        if result.get("status") == "error":
            stop_remaining = True
            _skip_remaining_tool_calls(
                tool_calls=tool_calls,
                results=results,
                start_index=index + 1,
                deps=deps,
            )
            break
        index += 1

    return [item if item is not None else _not_started_tool_result() for item in results]


def _tool_call_is_parallel_safe(tool_call: ToolCall, *, deps: TurnLoopDependencies) -> bool:
    if tool_call.name not in deps.allowed_tools:
        return False
    if not deps.tool_registry.has(tool_call.name):
        return False
    definition = deps.tool_registry.get(tool_call.name)
    return (
        definition.execution_effect == "read_only"
        and definition.risk_level != "high"
    )


def _not_started_tool_result() -> dict[str, Any]:
    return {
        "status": "error",
        "error": {
            "type": "tool_call_skipped",
            "message": (
                "tool call was not started because an earlier call "
                "in the same model response failed"
            ),
        },
        "execution_state": "not_started",
    }


def _skip_remaining_tool_calls(
    *,
    tool_calls: list[ToolCall],
    results: list[dict[str, Any] | None],
    start_index: int,
    deps: TurnLoopDependencies,
) -> None:
    for index in range(start_index, len(tool_calls)):
        if results[index] is not None:
            continue
        tool_call = tool_calls[index]
        skipped = _not_started_tool_result()
        # 未启动调用只写 trace，不进入 dispatch/policy/handler
        results[index] = deps.router.record_skipped(tool_call.name, tool_call.args, skipped)


def _run_read_batch(
    *,
    turn: int,
    tool_calls: list[ToolCall],
    start_index: int,
    end_index: int,
    results: list[dict[str, Any] | None],
    deps: TurnLoopDependencies,
    interaction_handler: HumanInteractionHandler | None,
) -> bool:
    """滑动窗口执行连续只读批次；返回是否出现 error。"""
    batch_indices = list(range(start_index, end_index))
    if not batch_indices:
        return False
    if len(batch_indices) == 1:
        index = batch_indices[0]
        deps.raise_if_cancelled()
        deps.emit_event(_tool_started_event(turn, tool_calls[index]))
        results[index] = _dispatch_tool_call(tool_calls[index], deps, interaction_handler, turn=turn)
        return results[index].get("status") == "error"

    max_workers = min(deps.max_parallel_read_tools, len(batch_indices))
    next_pos = 0
    in_flight: dict[Future[dict[str, Any]], int] = {}
    failed = False
    cancelled_error: RunCancelled | None = None

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="haagent-tool") as executor:
        def submit_available() -> None:
            nonlocal next_pos
            # 只维持最多 max_workers 个 in-flight，失败后不再领取
            while (
                not failed
                and cancelled_error is None
                and next_pos < len(batch_indices)
                and len(in_flight) < max_workers
            ):
                deps.raise_if_cancelled()
                index = batch_indices[next_pos]
                next_pos += 1
                # tool_started 仅在取得 active slot 后、提交 worker 前发出
                deps.emit_event(_tool_started_event(turn, tool_calls[index]))
                future = executor.submit(
                    _dispatch_tool_call,
                    tool_calls[index],
                    deps,
                    interaction_handler,
                    turn=turn,
                )
                in_flight[future] = index

        try:
            submit_available()
        except RunCancelled as error:
            cancelled_error = error

        while in_flight and cancelled_error is None:
            done, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                try:
                    result = future.result()
                except RunCancelled as error:
                    cancelled_error = error
                    continue
                results[index] = result
                if result.get("status") == "error":
                    failed = True
            if cancelled_error is not None:
                break
            if not failed:
                try:
                    submit_available()
                except RunCancelled as error:
                    cancelled_error = error
                    break

        while in_flight:
            done, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                index = in_flight.pop(future)
                try:
                    result = future.result()
                except RunCancelled as error:
                    cancelled_error = error
                    continue
                results[index] = result
                if result.get("status") == "error":
                    failed = True

    if cancelled_error is not None:
        raise cancelled_error

    if failed:
        for index in batch_indices:
            if results[index] is None:
                tool_call = tool_calls[index]
                skipped = _not_started_tool_result()
                results[index] = deps.router.record_skipped(
                    tool_call.name,
                    tool_call.args,
                    skipped,
                )
    return failed


def _dispatch_tool_call(
    tool_call: ToolCall,
    deps: TurnLoopDependencies,
    interaction_handler: HumanInteractionHandler | None,
    *,
    turn: int,
) -> dict[str, Any]:
    try:
        return deps.router.dispatch(
            tool_call.name,
            tool_call.args,
            interaction_handler=interaction_handler,
            turn=turn,
        )
    except RunCancelled:
        raise
    except Exception as error:
        return tool_error(type(error).__name__, str(error))


def _interaction_handler_for_turn(
    turn: int,
    deps: TurnLoopDependencies,
) -> HumanInteractionHandler | None:
    if deps.interaction_handler is None:
        return None
    handler = deps.interaction_bridge_factory(turn, deps.interaction_resolver)
    interaction_lock = Lock()

    def locked_handler(request):
        with interaction_lock:
            return handler(request)

    return locked_handler


def _tool_started_event(turn: int, tool_call: ToolCall) -> ToolStartedBusEvent:
    return ToolStartedBusEvent(
        turn=turn,
        tool_name=tool_call.name,
        args=dict(tool_call.args),
    )


def _record_changed_files(changed_files: list[dict[str, object]], tool_result: dict[str, object]) -> None:
    raw_changes = tool_result.get("changed_files")
    if not isinstance(raw_changes, list):
        return
    for change in raw_changes:
        if isinstance(change, dict):
            changed_files.append(dict(change))


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


def _build_loaded_image_attachment_message(tool_result: dict[str, Any]) -> dict[str, Any] | None:
    if tool_result.get("status") != "success":
        return None
    attachment = tool_result.get("loaded_image_attachment")
    if not isinstance(attachment, dict):
        return None
    image_id = str(attachment.get("id") or "").strip()
    if not image_id:
        return None
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Loaded historical image: {image_id}"},
            {
                "type": "image_attachment",
                "id": image_id,
                "filename": str(attachment.get("filename") or ""),
                "mime_type": str(attachment.get("mime_type") or ""),
                "size_bytes": int(attachment.get("size_bytes") or 0),
                "width": int(attachment.get("width") or 0),
                "height": int(attachment.get("height") or 0),
                "sha256": str(attachment.get("sha256") or ""),
                "relative_path": str(attachment.get("relative_path") or ""),
                "path": str(attachment.get("path") or ""),
            },
        ],
    }


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


def _is_terminal_memory_settlement_response(model_response) -> bool:
    """仅记忆结算工具伴随非空回答时，该回答就是本轮最终用户输出。"""
    return bool(model_response.content.strip()) and bool(model_response.tool_calls) and all(
        tool_call.name == "start_memory_update" for tool_call in model_response.tool_calls
    )


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
    return _supports_generate_parameter(model_gateway, "event_sink")


def _supports_generate_parameter(model_gateway: ModelGateway, parameter: str) -> bool:
    try:
        signature = inspect.signature(model_gateway.generate)
    except (TypeError, ValueError):
        return False
    return parameter in signature.parameters


def _retry_max_attempts(model_gateway: ModelGateway) -> int | None:
    """从 session 注入的 controller 读取审计上限；旧测试替身不强行声明该字段。"""

    controller = getattr(model_gateway, "_retry_controller", None)
    policy = getattr(controller, "policy", None)
    value = getattr(policy, "max_attempts", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _task_step_title(deps: TurnLoopDependencies) -> str:
    return deps.task_step_title or deps.task_goal


def _usage_record(usage: ModelUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "raw_usage_source": usage.raw_source,
    }


def _stable_prefix_fingerprint(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
) -> str:
    """哈希请求的稳定 system 前缀与工具 schema，不纳入动态历史和运行 ID。"""

    system_prefix: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            break
        system_prefix.append(message)
    payload = json.dumps(
        {"system_messages": system_prefix, "tool_schemas": tool_schemas},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _gateway_metadata(model_gateway: ModelGateway) -> dict[str, object]:
    metadata_getter = getattr(model_gateway, "metadata", None)
    if not callable(metadata_getter):
        return {"model": None}
    metadata = metadata_getter()
    return {"model": metadata.model}
