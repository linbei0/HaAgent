"""
haagent/runtime/orchestrator.py - Run Orchestrator 状态机

串联 task 加载、模型调用、工具执行和 episode trace 写入。
使用累积对话历史（messages list）替代每轮重建 context 块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from haagent.context.builder import ContextBuildError, ContextBuilder
from haagent.context.messages import (
    build_assistant_message,
    build_final_response_request_message,
    build_suggestion_message,
    build_tool_result_message,
    generate_tool_call_id,
)
from haagent.context.observation_compaction import (
    OBSERVATION_MICROCOMPACT_CHAR_LIMIT,
    OBSERVATION_MICROCOMPACT_HEAD_CHARS,
    OBSERVATION_MICROCOMPACT_TAIL_CHARS,
)
from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import ModelCallError, ModelGateway, ToolCall
from haagent.runtime.cancellation import CancellationToken, RunCancelled
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.failure import FailureCategory
from haagent.runtime.full_compact import FullCompactResult, maybe_full_compact_messages
from haagent.runtime.full_compact_contract import FullCompactEligibility
from haagent.runtime.guardrails import (
    GuardrailResult,
    check_assistant_output,
    check_user_input,
    guardrail_evidence,
)
from haagent.runtime.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    HumanInteractionResponse,
)
from haagent.runtime.human_interaction_resolver import (
    HumanInteractionResolution,
    HumanInteractionResolver,
)
from haagent.runtime.loop_guidance import (
    ToolSuggestion,
    safety_violation_observation,
    suggestion_for_observation,
    suggestion_observation,
)
from haagent.runtime.safety_guard import SafetyGuard
from haagent.runtime.plan import build_plan
from haagent.runtime.path_policy import default_path_policy, load_path_policy
from haagent.runtime.run_recorder import RunRecorder, RunResult
from haagent.runtime.state import RunStatus
from haagent.runtime.task_contract import TaskLoadError, load_task, resolve_workspace_root
from haagent.runtime.workspace_preflight import build_workspace_preflight
from haagent.tools.base import ToolRoutingError, tool_error
from haagent.tools.registry import export_tool_schemas
from haagent.tools.router import ToolRouter
from haagent.verification.engine import DEFAULT_COMMAND_TIMEOUT_SECONDS, VerificationEngine


class RunOrchestrator:
    def __init__(
        self,
        runs_root: Path,
        model_gateway: ModelGateway | None = None,
        max_turns: int = 3,
        session_summary: str | None = None,
        session_compaction: dict[str, object] | None = None,
        tool_result_microcompact_count: int = 0,
        working_state: dict[str, object] | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        interaction_handler: HumanInteractionHandler | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._model_gateway = model_gateway or FakeModelGateway()
        self._max_turns = max_turns
        self._session_summary = session_summary
        self._session_compaction = session_compaction
        self._tool_result_microcompact_count = max(0, tool_result_microcompact_count)
        self._working_state = working_state
        self._event_sink = event_sink
        self._interaction_handler = interaction_handler
        self._cancellation_token = cancellation_token

    def _emit_event(self, event: dict[str, object]) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def _raise_if_cancelled(self) -> None:
        if self._cancellation_token is not None:
            self._cancellation_token.raise_if_cancelled()

    def run(self, task_path: Path) -> RunResult:
        """执行一次 run，并把所有阶段变化写入 transcript.jsonl。"""
        writer = EpisodeWriter.create(self._runs_root, task_path)
        recorder = RunRecorder(writer)
        transition = recorder.transition

        transition(RunStatus.CREATED)

        try:
            self._raise_if_cancelled()
            task = load_task(task_path)
            workspace_candidate = _workspace_root_candidate(task.workspace_root, task_path)
            writer.write_workspace_preflight(build_workspace_preflight(workspace_candidate))
            workspace_root = resolve_workspace_root(task, task_path)
            path_policy = load_path_policy(task.path_policy) if task.path_policy is not None else default_path_policy(workspace_root)
            writer.write_episode_metadata(
                status=RunStatus.CREATED.value,
                provider=self._model_gateway.provider_name,
                workspace_root=workspace_root,
            )
            transition(RunStatus.PLANNING)
            writer.write_environment(workspace_root)
            writer.write_sandbox_metadata(
                workspace_root,
                command_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            )
            self._raise_if_cancelled()
            plan = build_plan(task)
            writer.write_plan(plan)
            writer.append_transcript(
                {
                    "event": "planning",
                    "plan_path": "plan.json",
                    "planned_step_count": len(plan["planned_steps"]),
                },
            )
            if isinstance(self._session_compaction, dict) and self._session_compaction.get("decision") == "compacted":
                writer.append_transcript(
                    {
                        "event": "session_memory_compaction",
                        "trigger_kind": "session_memory",
                        "decision": self._session_compaction.get("decision"),
                        "original_turn_count": self._session_compaction.get("original_turn_count"),
                        "compacted_turn_count": self._session_compaction.get("compacted_turn_count"),
                        "preserved_recent_count": self._session_compaction.get("preserved_recent_count"),
                        "original_chars": self._session_compaction.get("original_chars"),
                        "final_chars": self._session_compaction.get("final_chars"),
                        "saved_chars": self._session_compaction.get("saved_chars"),
                        "reason": self._session_compaction.get("reason"),
                    },
                )
            input_guardrail = check_user_input(task.goal)
            if input_guardrail is not None:
                _record_guardrail(writer, self._emit_event, input_guardrail)
                transition(RunStatus.FAILED)
                writer.write_failure_attribution(
                    {
                        "stage": "planning",
                        "category": FailureCategory.GUARDRAIL.value,
                        "evidence": guardrail_evidence(input_guardrail),
                    },
                )
                return recorder.finish(RunStatus.FAILED)

            router = ToolRouter(
                task.allowed_tools,
                writer,
                workspace_root=workspace_root,
                path_policy=path_policy,
                approval_allowed_tools=task.policy["approval_allowed_tools"],
                approved_tools=task.policy["approved_tools"],
            )
            verification_engine: VerificationEngine | None = None
            passed_verification_commands: set[str] = set()
            has_file_change = False
            has_shell_verification = False
            safety_guard = SafetyGuard()
            interaction_resolver = HumanInteractionResolver()

            # 构建初始消息（一次，不再每轮重建）
            context = ContextBuilder(
                task=task,
                workspace_root=workspace_root,
                provider_name=self._model_gateway.provider_name,
                episode_writer=writer,
                session_summary=self._session_summary,
                session_compaction=self._session_compaction,
                tool_result_microcompact_count=self._tool_result_microcompact_count,
                working_state=self._working_state,
                interaction_state=interaction_resolver.state_records(),
            ).build()
            if context.manifest.full_compact_contract is not None:
                writer.append_transcript(
                    {"event": "full_compact_contract", **context.manifest.full_compact_contract},
                )
            messages: list[dict[str, Any]] = list(context.messages)
            if context.manifest.full_compact_contract is not None:
                if context.manifest.full_compact_contract.get("eligible") is True:
                    writer.append_transcript(
                        {
                            "event": "full_compact_start",
                            "context_id": context.context_id,
                            "reason": context.manifest.full_compact_contract.get("reason"),
                            "preserve_recent": context.manifest.full_compact_contract.get("required_preserve_recent", 6),
                            "pre_message_count": len(messages),
                        },
                    )
                full_compact_result = maybe_full_compact_messages(
                    messages=messages,
                    eligibility=_full_compact_eligibility_from_manifest(context.manifest.full_compact_contract),
                    gateway=self._model_gateway,
                    preserve_recent=_full_compact_preserve_recent(context.manifest.full_compact_contract),
                )
                if context.manifest.full_compact_contract.get("eligible") is True:
                    writer.append_transcript(
                        {
                            "event": "full_compact_success" if full_compact_result.applied else "full_compact_failed",
                            "context_id": context.context_id,
                            **_full_compact_event_fields(full_compact_result),
                        },
                    )
                _write_full_compact_manifest_result(writer, context.context_id, full_compact_result.manifest)
                messages = full_compact_result.messages

            # 追踪 completion 用的最后一次文件变更 observation（供 _successful_file_change_without_declared_verification）
            completion_observations: list[dict[str, object]] = []
            final_response_requested = False

            for turn in range(1, self._max_turns + 1):
                self._raise_if_cancelled()
                tool_schemas = [] if final_response_requested else export_tool_schemas(task.allowed_tools)
                _microcompact_old_tool_messages(messages, writer, turn, emit_event=self._emit_event)

                writer.append_transcript(
                    {
                        "event": "model_call",
                        "provider": self._model_gateway.provider_name,
                        "context_id": context.context_id,
                        "turn": turn,
                        "goal": task.goal,
                    },
                )
                model_response = self._model_gateway.generate(
                    messages=messages,
                    tool_schemas=tool_schemas,
                )
                self._raise_if_cancelled()
                output_guardrail = (
                    check_assistant_output(model_response.content)
                    if not model_response.tool_calls
                    else None
                )
                writer.append_transcript(
                    {
                        "event": "model_response",
                        "provider": self._model_gateway.provider_name,
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
                    },
                )
                if output_guardrail is not None:
                    _record_guardrail(writer, self._emit_event, output_guardrail, turn=turn)
                    transition(RunStatus.FAILED)
                    writer.write_failure_attribution(
                        {
                            "stage": "executing",
                            "category": FailureCategory.GUARDRAIL.value,
                            "evidence": guardrail_evidence(output_guardrail),
                        },
                    )
                    return recorder.finish(RunStatus.FAILED)

                if final_response_requested and model_response.tool_calls:
                    transition(RunStatus.FAILED)
                    writer.write_failure_attribution(
                        {
                            "stage": "executing",
                            "category": FailureCategory.MODEL.value,
                            "evidence": "model returned tool calls during final response turn",
                        },
                    )
                    return recorder.finish(RunStatus.FAILED)

                if not model_response.tool_calls:
                    if final_response_requested and not model_response.content.strip():
                        transition(RunStatus.FAILED)
                        writer.write_failure_attribution(
                            {
                                "stage": "executing",
                                "category": FailureCategory.MODEL.value,
                                "evidence": "model returned empty final response during final response turn",
                            },
                        )
                        return recorder.finish(RunStatus.FAILED)
                    writer.append_transcript(
                        {
                            "event": "no_tool_reviewed",
                            "turn": turn,
                            "guidance_added": False,
                            "trigger": None,
                        },
                    )
                    self._emit_event(
                        {
                            "event_type": "assistant_message",
                            "turn": turn,
                            "content": model_response.content,
                        },
                    )
                    transition(RunStatus.VERIFYING)
                    if verification_engine is None:
                        verification_engine = VerificationEngine(writer, workspace_root)
                    self._raise_if_cancelled()
                    verification_result = verification_engine.run(task.verification_commands)
                    if verification_result.status == "success":
                        transition(RunStatus.COMPLETED)
                        writer.write_failure_attribution(None)
                        return recorder.finish(RunStatus.COMPLETED)

                    # Verification failed — inject result as user message and continue
                    verification_obs = _verification_observation(verification_result)
                    ver_msg = build_suggestion_message(
                        f"Verification failed: {_verification_evidence(verification_result)}. "
                        "Use the failure details to repair the workspace, then try again."
                    )
                    messages.append(ver_msg)
                    final_response_requested = False
                    if turn == self._max_turns:
                        transition(RunStatus.FAILED)
                        writer.write_failure_attribution(
                            {
                                "stage": "verifying",
                                "category": FailureCategory.LOOP_LIMIT.value,
                                "evidence": _verification_loop_limit_evidence(
                                    self._max_turns,
                                    verification_result,
                                ),
                            },
                        )
                        return recorder.finish(RunStatus.FAILED)
                    continue

                if recorder.state_history[-1] is not RunStatus.EXECUTING:
                    transition(RunStatus.EXECUTING)

                # Assign IDs to tool calls if missing (fake/test gateways may omit them)
                tool_calls_with_ids: list[ToolCall] = []
                for tc in model_response.tool_calls:
                    if tc.id:
                        tool_calls_with_ids.append(tc)
                    else:
                        tool_calls_with_ids.append(ToolCall(name=tc.name, args=tc.args, id=generate_tool_call_id()))

                # Append assistant message with tool_calls
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
                messages.append(build_assistant_message(model_response.content, assistant_tool_calls))

                turn_broke_early = False
                pending_suggestion_messages: list[str] = []
                for tool_index, tool_call in enumerate(tool_calls_with_ids):
                    self._raise_if_cancelled()
                    self._emit_event(
                        {
                            "event_type": "tool_started",
                            "turn": turn,
                            "tool_name": tool_call.name,
                            "args": tool_call.args,
                        },
                    )
                    tool_result = router.dispatch(
                        tool_call.name,
                        tool_call.args,
                        interaction_handler=(
                            _interaction_bridge(self, writer, turn, interaction_resolver)
                            if self._interaction_handler is not None
                            else None
                        ),
                    )
                    self._raise_if_cancelled()
                    if tool_result.get("status") == "error":
                        error = tool_result.get("error") or {}
                        failed_event = {
                            "event_type": "tool_failed",
                            "turn": turn,
                            "tool_name": tool_call.name,
                            "args": tool_call.args,
                            "error": error,
                        }
                        self._emit_event(failed_event)
                    observation = {
                        "tool_name": tool_call.name,
                        "args": tool_call.args,
                        "result": tool_result,
                    }
                    writer.append_transcript(
                        {
                            "event": "tool_observation",
                            "turn": turn,
                            **observation,
                        },
                    )

                    # Append tool result to messages
                    messages.append(build_tool_result_message(tool_call.id, tool_call.name, tool_result))

                    violation = safety_guard.check(tool_call.name, tool_call.args, tool_result)
                    if violation is not None and violation.should_abort:
                        abort_obs = safety_violation_observation(violation.message, violation.recovery_suggestion)
                        writer.append_transcript({"event": "safety_abort", "turn": turn, **abort_obs})
                        self._emit_event(
                            {
                                "event_type": "safety_abort",
                                "turn": turn,
                                "violation_type": violation.type,
                                "message": violation.message,
                            }
                        )
                        transition(RunStatus.FAILED)
                        writer.write_failure_attribution(
                            {
                                "stage": "executing",
                                "category": FailureCategory.LOOP_LIMIT.value,
                                "evidence": violation.message,
                            }
                        )
                        return recorder.finish(RunStatus.FAILED)

                    if tool_result.get("status") == "error":
                        if _tool_error_is_terminal(tool_result):
                            router.raise_for_error(tool_result)
                        suggestion = suggestion_for_observation(observation)
                        if violation is not None:
                            safety_obs = safety_violation_observation(violation.message, violation.recovery_suggestion)
                            writer.append_transcript({"event": "safety_warning", "turn": turn, **safety_obs})
                            pending_suggestion_messages.append(str(safety_obs.get("result", {}).get("message", "")))
                        elif suggestion is not None:
                            _record_suggestion(writer, self._emit_event, turn, suggestion)
                            pending_suggestion_messages.append(suggestion.message)
                        turn_broke_early = True
                        for skipped_call in tool_calls_with_ids[tool_index + 1 :]:
                            skipped_result = tool_error(
                                "tool_call_skipped",
                                "tool call skipped because an earlier tool call in the same assistant message failed.",
                            )
                            writer.append_tool_call(
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
                            writer.append_transcript(
                                {
                                    "event": "tool_observation",
                                    "turn": turn,
                                    **skipped_observation,
                                },
                            )
                            messages.append(
                                build_tool_result_message(
                                    skipped_call.id,
                                    skipped_call.name,
                                    skipped_result,
                                ),
                            )
                        break

                    self._emit_event(
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
                        _record_suggestion(writer, self._emit_event, turn, suggestion)
                        pending_suggestion_messages.append(suggestion.message)

                    if tool_call.name in {"apply_patch", "apply_patch_set", "file_write"}:
                        completion_observations = [observation]
                        has_file_change = True
                    else:
                        completion_observations.append(observation)
                    if tool_call.name in {"shell", "code_run"} and tool_result.get("exit_code") == 0:
                        has_shell_verification = True
                    _update_in_band_verification_progress(
                        tool_call.name,
                        tool_call.args,
                        tool_result,
                        task.verification_commands,
                        passed_verification_commands,
                    )

                for suggestion_message in pending_suggestion_messages:
                    if suggestion_message:
                        messages.append(build_suggestion_message(suggestion_message))

                if not turn_broke_early and (
                    _all_declared_verification_commands_passed(
                        task.verification_commands,
                        passed_verification_commands,
                    )
                    or (has_file_change and has_shell_verification and not task.verification_commands)
                    or _successful_file_change_without_declared_verification(
                        completion_observations,
                        task.verification_commands,
                    )
                ):
                    final_response_requested = True
                    messages.append(build_final_response_request_message())
            else:
                transition(RunStatus.FAILED)
                writer.write_failure_attribution(
                    {
                        "stage": "executing",
                        "category": FailureCategory.LOOP_LIMIT.value,
                        "evidence": f"exceeded max_turns={self._max_turns}",
                    },
                )
                return recorder.finish(RunStatus.FAILED)
        except RunCancelled as error:
            transition(RunStatus.CANCELLED)
            writer.write_failure_attribution(
                {
                    "stage": recorder.state_history[-2].value if len(recorder.state_history) > 1 else "created",
                    "category": FailureCategory.RUNTIME.value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.CANCELLED)
        except ToolRoutingError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": _tool_failure_category(error).value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)
        except ModelCallError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "planning",
                    "category": FailureCategory.MODEL.value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)
        except ContextBuildError as error:
            transition(RunStatus.FAILED)
            category = (
                FailureCategory.TASK_SPEC
                if "unknown allowed_tools" in str(error)
                else FailureCategory.CONTEXT
            )
            writer.write_failure_attribution(
                {
                    "stage": "planning",
                    "category": category.value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)
        except TaskLoadError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "created",
                    "category": FailureCategory.TASK_SPEC.value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)
        except Exception as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": recorder.state_history[-2].value if len(recorder.state_history) > 1 else "created",
                    "category": _unexpected_failure_category(error, recorder.state_history).value,
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)


def _verification_evidence(verification_result) -> str:
    lines = [f"command: {verification_result.failed_command}"]
    if verification_result.timeout or verification_result.failure_reason == "timeout":
        lines.append("timeout: true")
    else:
        lines.append(f"exit_code={verification_result.exit_code}")
    if verification_result.stdout_excerpt:
        lines.append(f"stdout: {verification_result.stdout_excerpt}")
    if verification_result.stderr_excerpt:
        lines.append(f"stderr: {verification_result.stderr_excerpt}")
    return "\n".join(lines)


def _full_compact_eligibility_from_manifest(contract: dict[str, Any]) -> FullCompactEligibility:
    return FullCompactEligibility(
        eligible=contract.get("eligible") is True,
        reason=str(contract.get("reason", "unknown")),
        trigger_kind=contract.get("trigger_kind") if isinstance(contract.get("trigger_kind"), str) else None,
        required_preserve_recent=_full_compact_preserve_recent(contract),
    )


def _full_compact_preserve_recent(contract: dict[str, Any]) -> int:
    preserve_recent = contract.get("required_preserve_recent", 6)
    return preserve_recent if isinstance(preserve_recent, int) else 6


def _full_compact_event_fields(result: FullCompactResult) -> dict[str, object]:
    return {
        "applied": result.applied,
        "reason": result.reason,
        "pre_message_count": result.pre_message_count,
        "post_message_count": result.post_message_count,
        "older_message_count": result.older_message_count,
        "preserved_recent_count": result.preserved_recent_count,
        "summary_chars": result.summary_chars,
    }


def _write_full_compact_manifest_result(
    writer: EpisodeWriter,
    context_id: str,
    full_compact: dict[str, Any],
) -> None:
    manifest_path = writer.path / "contexts" / f"{context_id}-manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_compact"] = full_compact
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _microcompact_old_tool_messages(
    messages: list[dict[str, Any]],
    writer: EpisodeWriter,
    turn: int,
    emit_event: Callable[[dict[str, object]], None] | None = None,
) -> None:
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= OBSERVATION_MICROCOMPACT_CHAR_LIMIT:
            continue
        compacted = _collapse_text_head_tail(
            content,
            head_chars=OBSERVATION_MICROCOMPACT_HEAD_CHARS,
            tail_chars=OBSERVATION_MICROCOMPACT_TAIL_CHARS,
        )
        if len(compacted) >= len(content):
            continue
        message["content"] = compacted
        event = {
            "event": "tool_result_microcompact",
            "turn": turn,
            "message_index": index,
            "tool_name": str(message.get("name", "unknown_tool")),
            "original_chars": len(content),
            "final_chars": len(compacted),
            "decision": "collapsed",
            "reason": "old_tool_result_over_budget",
        }
        writer.append_transcript(event)
        if emit_event is not None:
            emit_event(event)


def _collapse_text_head_tail(text: str, *, head_chars: int, tail_chars: int) -> str:
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    collapsed_chars = len(text) - head_chars - tail_chars
    marker = f"...[collapsed {collapsed_chars} chars]..."
    if tail:
        return f"{head}\n{marker}\n{tail}"
    return f"{head}\n{marker}"


def _verification_loop_limit_evidence(max_turns: int, verification_result) -> str:
    return (
        f"verification did not pass before max_turns={max_turns}\n"
        f"{_verification_evidence(verification_result)}"
    )


def _successful_file_change_without_declared_verification(
    completion_observations: list[dict[str, object]],
    verification_commands: list[str],
) -> bool:
    if verification_commands or not completion_observations:
        return False
    latest = completion_observations[-1]
    tool_name = latest.get("tool_name")
    result = latest.get("result")
    return (
        tool_name == "file_write"
        and isinstance(result, dict)
        and result.get("status") == "success"
    )


def _record_suggestion(
    writer: EpisodeWriter,
    emit_event: Callable[[dict[str, object]], None],
    turn: int,
    suggestion: ToolSuggestion,
) -> None:
    event = {
        "event_type": "loop_suggestion_added",
        "turn": turn,
        "trigger": suggestion.trigger,
        "tool_name": suggestion.tool_name,
        "message": suggestion.message,
    }
    writer.append_transcript({"event": "loop_suggestion_added", **_transcript_event(event)})
    emit_event(event)


def _record_guardrail(
    writer: EpisodeWriter,
    emit_event: Callable[[dict[str, object]], None],
    guardrail: GuardrailResult,
    turn: int | None = None,
) -> None:
    event: dict[str, object] = {
        "event_type": "guardrail_triggered",
        "status": guardrail.status,
        "scope": guardrail.scope,
        "rule_id": guardrail.rule_id,
        "severity": guardrail.severity,
        "message": guardrail.message,
    }
    if turn is not None:
        event["turn"] = turn
    writer.append_transcript({"event": "guardrail_triggered", **_transcript_event(event)})
    emit_event(event)


def _tool_error_is_terminal(tool_result: dict[str, object]) -> bool:
    error = tool_result.get("error") if isinstance(tool_result.get("error"), dict) else {}
    error_type = str(error.get("type", ""))
    if error_type in {"approval_denied", "policy_denied", "guardrail_denied", "tool_not_allowed", "unknown_tool"}:
        return True
    if error_type in {"patch_text_not_found", "patch_text_not_unique", "code_run_failed", "timeout"}:
        return False
    if tool_result.get("suggestions"):
        return False
    if tool_result.get("suggested_tool"):
        return False
    return error_type == "tool_argument_invalid"


def _interaction_bridge(
    orchestrator: RunOrchestrator,
    writer: EpisodeWriter,
    turn: int,
    interaction_resolver: HumanInteractionResolver,
) -> HumanInteractionHandler:
    def handle(request: HumanInteractionRequest) -> HumanInteractionResponse:
        if resolution := interaction_resolver.resolve(request):
            reused_event = _interaction_reused_event(turn, resolution)
            writer.append_interaction_event("interaction_reused", _transcript_event(reused_event))
            orchestrator._emit_event(reused_event)
            return resolution.to_response()
        requested_event = _interaction_requested_event(turn, request)
        writer.append_interaction_event(str(requested_event["event_type"]), _transcript_event(requested_event))
        orchestrator._emit_event(requested_event)
        if orchestrator._interaction_handler is None:
            response = HumanInteractionResponse(approved=False, answer="")
        else:
            response = orchestrator._interaction_handler(request)
        response_event = _interaction_response_event(turn, request, response)
        writer.append_interaction_event(str(response_event["event_type"]), _transcript_event(response_event))
        orchestrator._emit_event(response_event)
        interaction_resolver.record(request, response, turn=turn)
        return response

    return handle


def _interaction_requested_event(turn: int, request: HumanInteractionRequest) -> dict[str, object]:
    event_type = "approval_requested" if request.interaction_type == "approval" else "user_input_requested"
    return {
        "event_type": event_type,
        "turn": turn,
        "tool_name": request.tool_name,
        "question": request.question,
        "reason": request.reason,
        "risk_level": request.risk_level,
        "args_summary": request.args_summary,
        "approved": None,
    }


def _interaction_reused_event(turn: int, resolution: HumanInteractionResolution) -> dict[str, object]:
    return {
        "event_type": "interaction_reused",
        "turn": turn,
        "interaction_type": resolution.interaction_type,
        "tool_name": resolution.tool_name,
        "question": resolution.question,
        "status": resolution.status,
        "approved": resolution.approved,
        "resolved_turn": resolution.turn,
        "signature": resolution.signature,
    }


def _interaction_response_event(
    turn: int,
    request: HumanInteractionRequest,
    response: HumanInteractionResponse,
) -> dict[str, object]:
    if request.interaction_type == "approval":
        event_type = "approval_granted" if response.approved else "approval_denied"
        return {
            "event_type": event_type,
            "turn": turn,
            "tool_name": request.tool_name,
            "question": request.question,
            "approved": response.approved,
        }
    return {
        "event_type": "user_input_received",
        "turn": turn,
        "tool_name": request.tool_name,
        "question": request.question,
        "answer": response.answer,
        "answer_chars": len(response.answer),
        "approved": response.approved,
    }


def _transcript_event(event: dict[str, object]) -> dict[str, object]:
    record = dict(event)
    record.pop("event_type", None)
    return record


def _verification_observation(verification_result) -> dict[str, object]:
    return {
        "tool_name": "verification",
        "args": {"command": verification_result.failed_command},
        "result": {
            "status": "error",
            "command": verification_result.failed_command,
            "exit_code": verification_result.exit_code,
            "failure_reason": verification_result.failure_reason,
            "timeout": verification_result.timeout,
            "stdout": verification_result.stdout_excerpt,
            "stderr": verification_result.stderr_excerpt,
        },
    }


def _update_in_band_verification_progress(
    tool_name: str,
    tool_args: dict[str, object],
    tool_result: dict[str, object],
    verification_commands: list[str],
    passed_verification_commands: set[str],
) -> None:
    if tool_name in {"apply_patch", "apply_patch_set"}:
        passed_verification_commands.clear()
        return
    if tool_name != "shell":
        return
    command = tool_args.get("command")
    if not isinstance(command, str) or command not in verification_commands:
        return
    if tool_result.get("status") == "success" and tool_result.get("exit_code") == 0:
        passed_verification_commands.add(command)


def _all_declared_verification_commands_passed(
    verification_commands: list[str],
    passed_verification_commands: set[str],
) -> bool:
    expected_commands = set(verification_commands)
    return bool(expected_commands) and expected_commands.issubset(passed_verification_commands)


def _unexpected_failure_category(error: Exception, state_history: list[RunStatus]) -> FailureCategory:
    previous_status = state_history[-2] if len(state_history) > 1 else state_history[-1]
    if isinstance(error, TypeError) and previous_status is RunStatus.PLANNING:
        return FailureCategory.MODEL_CALL
    return FailureCategory.RUNTIME


def _tool_failure_category(error: ToolRoutingError) -> FailureCategory:
    if error.error_type == "approval_denied":
        return FailureCategory.USER_DENIED
    if error.error_type == "guardrail_denied":
        return FailureCategory.GUARDRAIL
    if error.error_type in {"invalid_tool_arguments", "tool_argument_invalid"}:
        return FailureCategory.TOOL_ARGUMENT
    return FailureCategory.TOOL_INTERFACE


def _workspace_root_candidate(raw_root: str | None, task_path: Path) -> Path:
    candidate = task_path.parent if raw_root is None else Path(raw_root)
    if raw_root is not None and not candidate.is_absolute():
        candidate = task_path.parent / candidate
    return candidate.resolve(strict=False)
