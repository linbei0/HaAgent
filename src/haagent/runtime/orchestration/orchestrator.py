"""
src/haagent/runtime/orchestration/orchestrator.py - Run Orchestrator 状态机

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
from haagent.models.provider_profile import user_config_dir
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.execution.guardrails import (
    GuardrailResult,
    check_assistant_output,
    check_user_input,
    guardrail_evidence,
)
from haagent.runtime.execution.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    HumanInteractionResponse,
)
from haagent.runtime.execution.human_interaction_resolver import (
    HumanInteractionResolution,
    HumanInteractionResolver,
)
from haagent.runtime.orchestration.loop_guidance import (
    ToolSuggestion,
    safety_violation_observation,
    suggestion_for_observation,
    suggestion_observation,
)
from haagent.runtime.execution.safety_guard import SafetyGuard
from haagent.runtime.orchestration.preparation import prepare_initial_messages, prepare_run_setup
from haagent.runtime.orchestration.recorder import RunRecorder, RunResult
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.orchestration.turns import TurnLoopDependencies, TurnLoopState, run_turn_loop
from haagent.runtime.sandbox import SandboxBackend, create_sandbox_backend
from haagent.runtime.settings import DEFAULT_RUN_MAX_TURNS, load_runtime_settings
from haagent.runtime.contracts.task import TaskLoadError
from haagent.tools.base import ToolRoutingError, tool_error
from haagent.tools.registry import ToolRuntimeRegistry, default_tool_runtime_registry
from haagent.tools.router import ToolRouter
from haagent.verification.engine import DEFAULT_COMMAND_TIMEOUT_SECONDS, VerificationEngine


class RunOrchestrator:
    def __init__(
        self,
        runs_root: Path,
        model_gateway: ModelGateway | None = None,
        max_turns: int | None = DEFAULT_RUN_MAX_TURNS,
        session_summary: str | None = None,
        session_compaction: dict[str, object] | None = None,
        tool_result_microcompact_count: int = 0,
        working_state: dict[str, object] | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        interaction_handler: HumanInteractionHandler | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRuntimeRegistry | None = None,
        mcp_runtime: Any | None = None,
        leader_session_id: str | None = None,
        worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None,
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
        self._tool_registry = tool_registry or default_tool_runtime_registry()
        self._mcp_runtime = mcp_runtime
        self._leader_session_id = leader_session_id or "leader"
        self._worker_permission_requester = worker_permission_requester

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
        sandbox_backend: SandboxBackend | None = None

        try:
            setup = prepare_run_setup(
                task_path=task_path,
                writer=writer,
                provider_name=self._model_gateway.provider_name,
                transition=transition,
                raise_if_cancelled=self._raise_if_cancelled,
                session_compaction=self._session_compaction,
            )
            task = setup.task
            workspace_root = setup.workspace_root
            path_policy = setup.path_policy
            runtime_settings = load_runtime_settings()
            sandbox_backend = create_sandbox_backend(
                settings=runtime_settings.sandbox,
                workspace_root=workspace_root,
                session_id=writer.path.name,
                command_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            )
            writer.write_sandbox_metadata(sandbox_backend.metadata())
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
                cancellation_token=self._cancellation_token,
                tool_registry=self._tool_registry,
                mcp_runtime=self._mcp_runtime,
                agent_runtime=MultiAgentRuntime(
                    runs_root=self._runs_root,
                    workspace_root=workspace_root,
                    leader_session_id=self._leader_session_id,
                    model_gateway=self._model_gateway,
                    path_policy=path_policy,
                    inherited_allowed_tools=task.allowed_tools,
                    inherited_approval_allowed_tools=task.policy["approval_allowed_tools"],
                    inherited_approved_tools=task.policy["approved_tools"],
                    event_sink=self._emit_event,
                    interaction_handler=self._interaction_handler,
                    enable_web=bool({"web_search", "web_fetch"} & set(task.allowed_tools)),
                    mcp_tool_names=[
                        tool for tool in task.allowed_tools if tool.startswith("mcp__")
                    ],
                    tool_registry=self._tool_registry,
                    mcp_runtime=self._mcp_runtime,
                    worker_max_turns=self._max_turns,
                ),
                worker_permission_requester=self._worker_permission_requester,
                sandbox_backend=sandbox_backend,
            )
            verification_engine: VerificationEngine | None = None
            passed_verification_commands: set[str] = set()
            has_file_change = False
            has_shell_verification = False
            safety_guard = SafetyGuard()
            interaction_resolver = HumanInteractionResolver()

            prepared_messages = prepare_initial_messages(
                context_builder_cls=ContextBuilder,
                task=task,
                workspace_root=workspace_root,
                provider_name=self._model_gateway.provider_name,
                writer=writer,
                model_gateway=self._model_gateway,
                session_summary=self._session_summary,
                session_compaction=self._session_compaction,
                tool_result_microcompact_count=self._tool_result_microcompact_count,
                working_state=self._working_state,
                interaction_resolver=interaction_resolver,
                tool_registry=self._tool_registry,
            )
            worker_notifications = _worker_notification_context(self._leader_session_id)
            if worker_notifications:
                prepared_messages.messages.append(
                    {"role": "user", "content": worker_notifications},
                )
            context_id = prepared_messages.context_id
            messages = prepared_messages.messages
            turn_result = run_turn_loop(
                state=TurnLoopState(
                    messages=messages,
                    context_id=context_id,
                    verification_engine=verification_engine,
                ),
                deps=TurnLoopDependencies(
                    model_gateway=self._model_gateway,
                    writer=writer,
                    recorder=recorder,
                    router=router,
                    task_goal=task.goal,
                    allowed_tools=task.allowed_tools,
                    tool_registry=self._tool_registry,
                    verification_commands=task.verification_commands,
                    workspace_root=workspace_root,
                    max_turns=self._max_turns,
                    raise_if_cancelled=self._raise_if_cancelled,
                    emit_event=self._emit_event,
                    microcompact_old_tool_messages=_microcompact_old_tool_messages,
                    interaction_handler=self._interaction_handler,
                    interaction_resolver=interaction_resolver,
                    safety_guard=safety_guard,
                    interaction_bridge_factory=lambda turn, resolver: _interaction_bridge(
                        self,
                        writer,
                        turn,
                        resolver,
                    ),
                    record_guardrail=lambda guardrail, turn=None: _record_guardrail(
                        writer,
                        self._emit_event,
                        guardrail,
                        turn=turn,
                    ),
                    record_suggestion=lambda turn, suggestion: _record_suggestion(
                        writer,
                        self._emit_event,
                        turn,
                        suggestion,
                    ),
                    tool_error_is_terminal=_tool_error_is_terminal,
                    update_in_band_verification_progress=_update_in_band_verification_progress,
                    all_declared_verification_commands_passed=_all_declared_verification_commands_passed,
                    successful_file_change_without_declared_verification=_successful_file_change_without_declared_verification,
                    verification_observation=_verification_observation,
                    verification_evidence=_verification_evidence,
                    verification_loop_limit_evidence=_verification_loop_limit_evidence,
                ),
            )
            if turn_result is not None:
                return turn_result
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
        finally:
            if sandbox_backend is not None:
                sandbox_backend.close()


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
        if _is_artifact_backed_tool_message(content):
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
            "event_type": "tool_result_microcompact",
            "turn": turn,
            "message_index": index,
            "tool_name": str(message.get("name", "unknown_tool")),
            "original_chars": len(content),
            "final_chars": len(compacted),
            "decision": "collapsed",
            "reason": "old_tool_result_over_budget",
        }
        writer.append_transcript({"event": "tool_result_microcompact", **_transcript_event(event)})
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


def _is_artifact_backed_tool_message(content: str) -> bool:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("artifact_path"), str) and payload.get("truncated") is True


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
    if error_type in {"approval_denied", "approval_pending", "policy_denied", "guardrail_denied", "tool_not_allowed", "unknown_tool"}:
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
    if request.interaction_type == "approval":
        event_type = "approval_requested"
    elif request.interaction_type == "edit_diff":
        event_type = "edit_diff_requested"
    else:
        event_type = "user_input_requested"
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
    if request.interaction_type == "edit_diff":
        event_type = "edit_diff_granted" if response.approved else "edit_diff_denied"
        return {
            "event_type": event_type,
            "turn": turn,
            "tool_name": request.tool_name,
            "question": request.question,
            "answer": response.answer,
            "approved": response.approved,
            "args_summary": request.args_summary,
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


def _worker_notification_context(leader_session_id: str) -> str | None:
    store = TeamStore(user_config_dir() / "teams")
    lines: list[str] = []
    for team in store.list_teams_for_leader(leader_session_id):
        for item in store.read_notifications(team.team_id, limit=5):
            details: list[str] = []
            task_id = str(item.get("task_id", "")).strip()
            if task_id:
                details.append(f"task={task_id}")
            request_id = str(item.get("request_id", "")).strip()
            if request_id:
                details.append(f"request={request_id}")
            if item.get("needs_attention") is True:
                details.append("attention=yes")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(
                "- "
                f"{item.get('agent_id', '')} "
                f"{item.get('status', '')}: "
                f"{item.get('summary', '')}"
                f"{suffix}"
            )
    if not lines:
        return None
    return "Worker Notifications:\n" + "\n".join(lines[-10:])
