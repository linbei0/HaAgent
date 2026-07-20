"""
src/haagent/runtime/orchestration/orchestrator.py - Run Orchestrator 状态机

串联 task 加载、模型调用、工具执行和 episode trace 写入。
使用累积对话历史（messages list）替代每轮重建 context 块。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.messages import compress_historical_tool_messages
from haagent.context.builder import ContextBuildError, ContextBuilder
from haagent.models.fake import FakeModelGateway
from haagent.models.types import ModelCallError, ModelGateway
from haagent.models.config.connections import user_config_dir
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.command import describe_shell_contract
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.execution.guardrails import (
    GuardrailResult,
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
    SessionInteractionState,
)
from haagent.runtime.orchestration.loop_guidance import ToolSuggestion
from haagent.runtime.execution.progress_guard import ProgressDecision, ProgressGuard
from haagent.runtime.orchestration.preparation import prepare_initial_messages, prepare_run_setup
from haagent.runtime.orchestration.recorder import RunRecorder, RunResult
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.orchestration.task_progress import map_failure_to_recovery
from haagent.runtime.orchestration.task_progress import task_plan_created_event
from haagent.runtime.orchestration.task_progress import task_recovery_suggested_event
from haagent.runtime.events.bus import RuntimeBusEvent, coerce_bus_event
from haagent.runtime.orchestration.turns import TurnLoopDependencies, TurnLoopState, run_turn_loop
from haagent.runtime.performance import PerformanceTrace
from haagent.runtime.sandbox import SandboxBackend, create_sandbox_backend
from haagent.runtime.settings import DEFAULT_RUN_MAX_TURNS, load_runtime_settings
from haagent.runtime.contracts.task import TaskLoadError
from haagent.tools.access import ToolAccessManager
from haagent.tools.base import ToolRoutingError
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
        historical_tool_compression_count: int = 0,
        working_state: dict[str, object] | None = None,
        task_ledger: dict[str, object] | None = None,
        event_sink: Callable[[RuntimeBusEvent], None] | None = None,
        interaction_handler: HumanInteractionHandler | None = None,
        cancellation_token: CancellationToken | None = None,
        tool_registry: ToolRuntimeRegistry | None = None,
        mcp_runtime: Any | None = None,
        leader_session_id: str | None = None,
        worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None,
        session_interaction_state: SessionInteractionState | None = None,
        performance_trace: PerformanceTrace | None = None,
        skill_catalog: object | None = None,
        instruction_cache: object | None = None,
        tool_schema_cache: object | None = None,
        working_state_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._model_gateway = model_gateway or FakeModelGateway()
        self._max_turns = max_turns
        self._session_summary = session_summary
        self._session_compaction = session_compaction
        self._historical_tool_compression_count = max(0, historical_tool_compression_count)
        self._working_state = working_state
        self._task_ledger = task_ledger
        self._event_sink = event_sink
        self._interaction_handler = interaction_handler
        self._cancellation_token = cancellation_token
        self._tool_registry = tool_registry or default_tool_runtime_registry()
        self._mcp_runtime = mcp_runtime
        self._episode_session_id = leader_session_id
        self._leader_session_id = leader_session_id or "leader"
        self._worker_permission_requester = worker_permission_requester
        # 跨 turn 共享；always 写回此对象，由 AgentSession 持久化
        self._session_interaction_state = session_interaction_state or SessionInteractionState()
        # 普通交互由 session 注入；advanced/CI 直接 run 时在入口补建。
        self._performance_trace = performance_trace
        self._skill_catalog = skill_catalog
        self._instruction_cache = instruction_cache
        self._tool_schema_cache = tool_schema_cache
        self._working_state_sink = working_state_sink

    def _emit_event(self, event: RuntimeBusEvent | dict[str, object]) -> None:
        # 兼容仍发 raw dict 的 multi_agent / compression / task_progress 路径。
        if self._event_sink is not None:
            self._event_sink(coerce_bus_event(event))

    def _raise_if_cancelled(self) -> None:
        if self._cancellation_token is not None:
            self._cancellation_token.raise_if_cancelled()

    def run(self, task_path: Path) -> RunResult:
        """执行一次 run，并把所有阶段变化写入 transcript.jsonl。"""
        performance_trace = self._performance_trace or PerformanceTrace.start()
        performance_trace.mark_run_start()
        writer = EpisodeWriter.create(
            self._runs_root,
            task_path,
            session_id=self._episode_session_id,
        )
        recorder = RunRecorder(writer, performance_trace=performance_trace)
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
            # 执行器先确定实际解释器，再把同一事实放进模型可见 schema 与任务上下文。
            runtime_tool_registry = self._tool_registry.with_description_overrides(
                {
                    "shell": "\n\n".join(
                        (
                            self._tool_registry.get("shell").description,
                            describe_shell_contract(sandbox_backend.shell_contract()),
                        ),
                    ),
                },
            )
            model_capabilities = None
            capabilities = getattr(self._model_gateway, "capabilities", None)
            if callable(capabilities):
                model_capabilities = capabilities()
            access_snapshot = ToolAccessManager.resolve(
                task.allowed_tools,
                registry=runtime_tool_registry,
                mcp_runtime=self._mcp_runtime,
                model_capabilities=model_capabilities,
                image_attachment_history=bool(task.image_attachment_history),
            )
            allowed_set = set(access_snapshot.allowed_tools)
            task = replace(
                task,
                allowed_tools=list(access_snapshot.allowed_tools),
                policy={
                    key: [name for name in values if name in allowed_set]
                    for key, values in task.policy.items()
                },
            )
            writer.append_transcript(
                {
                    "event": "tool_access_snapshot",
                    "allowed_tools": list(access_snapshot.allowed_tools),
                    "denied_tools": dict(access_snapshot.denied_tools),
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

            def tool_performance_sink(
                turn: int,
                tool_name: str,
                duration_ms: float,
                execution_effect: str,
                status: str,
            ) -> None:
                performance_trace.record_tool(
                    turn,
                    tool_name,
                    duration_ms,
                    execution_effect,
                    status,
                )
                recorder.persist_performance()

            router = ToolRouter(
                task.allowed_tools,
                writer,
                workspace_root=workspace_root,
                path_policy=path_policy,
                approval_allowed_tools=task.policy["approval_allowed_tools"],
                approved_tools=task.policy["approved_tools"],
                cancellation_token=self._cancellation_token,
                tool_registry=runtime_tool_registry,
                mcp_runtime=self._mcp_runtime,
                # 仅当本 run 允许 agent/task 工具时才装配 MultiAgentRuntime，避免普通对话热路径耦合。
                agent_runtime=_maybe_multi_agent_runtime(
                    allowed_tools=task.allowed_tools,
                    runs_root=self._runs_root,
                    workspace_root=workspace_root,
                    leader_session_id=self._leader_session_id,
                    model_gateway=self._model_gateway,
                    path_policy=path_policy,
                    approval_allowed_tools=task.policy["approval_allowed_tools"],
                    approved_tools=task.policy["approved_tools"],
                    event_sink=self._emit_event,
                    interaction_handler=self._interaction_handler,
                    tool_registry=runtime_tool_registry,
                    mcp_runtime=self._mcp_runtime,
                    worker_max_turns=self._max_turns,
                    parent_task_step_id=_current_task_step_id(self._task_ledger),
                ),
                worker_permission_requester=self._worker_permission_requester,
                sandbox_backend=sandbox_backend,
                image_attachment_history=task.image_attachment_history,
                performance_sink=tool_performance_sink,
                skill_catalog=self._skill_catalog,
            )
            verification_engine: VerificationEngine | None = None
            progress_guard = ProgressGuard()
            # permission_mode 与 always 权限规则都经结构化 session 状态跨 turn 复用。
            interaction_resolver = HumanInteractionResolver(
                permission_mode=path_policy.permission_mode,
                session_interaction_state=self._session_interaction_state,
            )

            performance_trace.mark_context_build_start()
            prepared_messages = prepare_initial_messages(
                context_builder_cls=ContextBuilder,
                task=task,
                workspace_root=workspace_root,
                provider_name=self._model_gateway.provider_name,
                writer=writer,
                model_gateway=self._model_gateway,
                session_summary=self._session_summary,
                session_compaction=self._session_compaction,
                historical_tool_compression_count=self._historical_tool_compression_count,
                working_state=self._working_state,
                task_ledger=self._task_ledger,
                interaction_resolver=interaction_resolver,
                tool_registry=runtime_tool_registry,
                instruction_cache=self._instruction_cache,
                skill_catalog=self._skill_catalog,
            )
            performance_trace.mark_context_built()
            for component, diagnostic in prepared_messages.cache_diagnostics.items():
                if isinstance(diagnostic, dict):
                    performance_trace.record_cache_diagnostic(component, diagnostic)
            worker_notifications = _worker_notification_context(self._leader_session_id)
            if worker_notifications:
                prepared_messages.messages.append(
                    {"role": "user", "content": worker_notifications},
                )
            context_id = prepared_messages.context_id
            messages = prepared_messages.messages
            task_step_id = _current_task_step_id(self._task_ledger) or "step-001"
            self._emit_event(
                task_plan_created_event(
                    step_id=task_step_id,
                    title=task.goal,
                    owner="main",
                    status="running",
                    summary="task plan ready",
                ),
            )
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
                    tool_registry=runtime_tool_registry,
                    verification_commands=task.verification_commands,
                    workspace_root=workspace_root,
                    max_turns=self._max_turns,
                    raise_if_cancelled=self._raise_if_cancelled,
                    emit_event=self._emit_event,
                    compress_historical_tool_messages=lambda messages, writer, turn, emit_event: compress_historical_tool_messages(
                        messages,
                        _compression_budget_for_gateway(self._model_gateway),
                        writer=writer,
                        turn=turn,
                        emit_event=emit_event,
                    ),
                    interaction_handler=self._interaction_handler,
                    interaction_resolver=interaction_resolver,
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
                    verification_evidence=_verification_evidence,
                    verification_loop_limit_evidence=_verification_loop_limit_evidence,
                    task_step_id=task_step_id,
                    task_step_title=task.goal,
                    cancellation_token=self._cancellation_token,
                    performance_trace=performance_trace,
                    persist_performance=recorder.persist_performance,
                    tool_schema_cache=self._tool_schema_cache,
                    progress_guard=progress_guard,
                    progress_guard_mode=runtime_settings.progress_guard_mode,
                    on_progress_blocked=lambda turn, decision: _record_progress_block_working_state(
                        self,
                        turn=turn,
                        decision=decision,
                    ),
                ),
            )
            if turn_result is not None:
                return turn_result
        except RunCancelled as error:
            _emit_task_recovery(
                self._emit_event,
                self._task_ledger,
                event_type="run_cancelled",
                reason=str(error),
                title=_task_title_from_locals(locals()),
            )
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
            _emit_task_recovery(
                self._emit_event,
                self._task_ledger,
                event_type="tool_failed",
                reason=str(error),
                error_type=error.error_type,
                title=_task_title_from_locals(locals()),
            )
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
            failed_stage = recorder.state_history[-1].value if recorder.state_history else "planning"
            _emit_task_recovery(
                self._emit_event,
                self._task_ledger,
                event_type="model_failed",
                reason=str(error),
                title=_task_title_from_locals(locals()),
            )
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": failed_stage,
                    "category": (
                        FailureCategory.MODEL_PROTOCOL.value
                        if error.details is not None and error.details.category == "protocol"
                        else FailureCategory.MODEL.value
                    ),
                    "evidence": str(error),
                },
            )
            return recorder.finish(RunStatus.FAILED)
        except ContextBuildError as error:
            _emit_task_recovery(
                self._emit_event,
                self._task_ledger,
                event_type="context_build_failed",
                reason=str(error),
                title=_task_title_from_locals(locals()),
            )
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
            _emit_task_recovery(
                self._emit_event,
                self._task_ledger,
                event_type="model_failed",
                reason=str(error),
                title=_task_title_from_locals(locals()),
            )
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


def _compression_budget_for_gateway(model_gateway: ModelGateway):
    metadata = None
    metadata_fn = getattr(model_gateway, "metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
    return derive_compression_budget(metadata)


def _verification_loop_limit_evidence(max_turns: int, verification_result) -> str:
    return (
        f"verification did not pass before max_turns={max_turns}\n"
        f"{_verification_evidence(verification_result)}"
    )


def _record_suggestion(
    writer: EpisodeWriter,
    emit_event: Callable[[RuntimeBusEvent | dict[str, object]], None],
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
    emit_event: Callable[[RuntimeBusEvent | dict[str, object]], None],
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


def _record_progress_block_working_state(
    orchestrator: RunOrchestrator,
    *,
    turn: int,
    decision: ProgressDecision,
) -> None:
    """block 时写入可恢复 working state 摘要，不复制 episode 证据。"""
    current = orchestrator._working_state
    if not isinstance(current, dict):
        current = {
            "current_goal": "",
            "key_findings": [],
            "completed_actions": [],
            "next_steps": [],
            "last_updated_turn": 0,
        }
        orchestrator._working_state = current
    next_steps = list(current.get("next_steps") or [])
    recovery = f"progress_guard blocked ({decision.pattern}): continue/replan/stop"
    if recovery not in next_steps:
        next_steps.insert(0, recovery)
    current["next_steps"] = next_steps[:5]
    current["last_updated_turn"] = int(turn)
    findings = list(current.get("key_findings") or [])
    reason = f"progress_guard:{decision.pattern}:{decision.reason}"[:240]
    if reason and reason not in findings:
        findings.append(reason)
    current["key_findings"] = findings[-5:]
    sink = getattr(orchestrator, "_working_state_sink", None)
    if sink is not None:
        # 必须在进入 user-input 等待前持久化，不能只依赖 run 结束后的 session flush。
        sink(dict(current))


def _tool_error_is_terminal(tool_result: dict[str, object]) -> bool:
    """普通工具失败交回模型；审批与策略边界必须暂停或结束当前执行。"""
    error = tool_result.get("error") if isinstance(tool_result.get("error"), dict) else {}
    return str(error.get("type", "")) in {
        "approval_denied",
        "approval_pending",
        "policy_denied",
        "guardrail_denied",
    }


def _interaction_bridge(
    orchestrator: RunOrchestrator,
    writer: EpisodeWriter,
    turn: int,
    interaction_resolver: HumanInteractionResolver,
) -> HumanInteractionHandler:
    def handle(request: HumanInteractionRequest) -> HumanInteractionResponse:
        if resolution := interaction_resolver.resolve(request):
            # 自动模式、session always 或用户输入复用：只写 reused，不伪造用户点击。
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
        "answer": resolution.answer,
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


def _current_task_step_id(task_ledger: dict[str, object] | None) -> str:
    if not isinstance(task_ledger, dict):
        return ""
    value = task_ledger.get("current_step_id")
    return value if isinstance(value, str) else ""


def _emit_task_recovery(
    emit_event: Callable[[RuntimeBusEvent | dict[str, object]], None],
    task_ledger: dict[str, object] | None,
    *,
    event_type: str,
    reason: str,
    title: str,
    error_type: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "event_type": event_type,
        "reason": reason,
    }
    if error_type:
        payload["error"] = {
            "type": error_type,
            "message": reason,
        }
    suggestion = map_failure_to_recovery(payload)
    if suggestion is None:
        return
    emit_event(
        task_recovery_suggested_event(
            step_id=_current_task_step_id(task_ledger) or suggestion.step_id or "step-001",
            title=title or "Current task",
            category=suggestion.category,
            reason=suggestion.reason,
            suggested_action=suggestion.suggested_action,
        ),
    )


def _task_title_from_locals(values: dict[str, object]) -> str:
    task = values.get("task")
    goal = getattr(task, "goal", None)
    if isinstance(goal, str) and goal.strip():
        return goal
    return "Current task"


# leader 侧多 agent 控制面工具；任一出现在 allowed_tools 才需要 MultiAgentRuntime。
_MULTI_AGENT_CONTROL_TOOLS = frozenset(
    {
        "agent",
        "send_message",
        "task_stop",
        "task_get",
        "task_list",
        "task_output",
    },
)


def _needs_multi_agent_runtime(allowed_tools: list[str]) -> bool:
    return bool(_MULTI_AGENT_CONTROL_TOOLS.intersection(allowed_tools))


def _maybe_multi_agent_runtime(
    *,
    allowed_tools: list[str],
    runs_root: Path,
    workspace_root: Path,
    leader_session_id: str,
    model_gateway: ModelGateway,
    path_policy: Any,
    approval_allowed_tools: list[str],
    approved_tools: list[str],
    event_sink: Callable[[RuntimeBusEvent], None] | None,
    interaction_handler: HumanInteractionHandler | None,
    tool_registry: ToolRuntimeRegistry,
    mcp_runtime: Any,
    worker_max_turns: int | None,
    parent_task_step_id: str,
) -> Any | None:
    if not _needs_multi_agent_runtime(allowed_tools):
        return None
    # 局部 import：普通 run 不加载 multi_agent.runtime 重模块。
    from haagent.multi_agent.runtime import MultiAgentRuntime

    return MultiAgentRuntime(
        runs_root=runs_root,
        workspace_root=workspace_root,
        leader_session_id=leader_session_id,
        model_gateway=model_gateway,
        path_policy=path_policy,
        inherited_allowed_tools=allowed_tools,
        inherited_approval_allowed_tools=approval_allowed_tools,
        inherited_approved_tools=approved_tools,
        event_sink=event_sink,
        interaction_handler=interaction_handler,
        enable_web=bool({"web_search", "web_fetch"} & set(allowed_tools)),
        mcp_tool_names=[tool for tool in allowed_tools if tool.startswith("mcp__")],
        tool_registry=tool_registry,
        mcp_runtime=mcp_runtime,
        worker_max_turns=worker_max_turns,
        parent_task_step_id=parent_task_step_id,
    )


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
            parent_step_id = str(item.get("parent_step_id", "")).strip()
            if parent_step_id:
                details.append(f"parent_step={parent_step_id}")
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
