"""
haagent/runtime/orchestrator.py - Run Orchestrator 状态机

串联 task 加载、模型调用、工具执行和 episode trace 写入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from haagent.context.builder import ContextBuildError, ContextBuilder
from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import ModelCallError, ModelGateway
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.failure import FailureCategory
from haagent.runtime.plan import build_plan
from haagent.runtime.state import RunStatus
from haagent.runtime.task_contract import TaskLoadError, load_task, resolve_workspace_root
from haagent.runtime.workspace_preflight import build_workspace_preflight
from haagent.tools.base import ToolRoutingError
from haagent.tools.registry import export_tool_schemas
from haagent.tools.router import ToolRouter
from haagent.verification.engine import DEFAULT_COMMAND_TIMEOUT_SECONDS, VerificationEngine


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state_history: list[RunStatus]
    episode_path: Path


class RunOrchestrator:
    def __init__(
        self,
        runs_root: Path,
        model_gateway: ModelGateway | None = None,
        max_turns: int = 3,
        session_summary: str | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._model_gateway = model_gateway or FakeModelGateway()
        self._max_turns = max_turns
        self._session_summary = session_summary

    def run(self, task_path: Path) -> RunResult:
        """执行一次 run，并把所有阶段变化写入 transcript.jsonl。"""
        state_history: list[RunStatus] = []
        writer = EpisodeWriter.create(self._runs_root, task_path)

        def transition(status: RunStatus) -> None:
            # 状态流转是 episode 的关键事实来源，必须先落 trace 再继续执行下一步。
            state_history.append(status)
            writer.append_transcript({"event": "state_transition", "status": status.value})

        transition(RunStatus.CREATED)

        try:
            task = load_task(task_path)
            workspace_candidate = _workspace_root_candidate(task.workspace_root, task_path)
            writer.write_workspace_preflight(build_workspace_preflight(workspace_candidate))
            workspace_root = resolve_workspace_root(task, task_path)
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
            plan = build_plan(task)
            writer.write_plan(plan)
            writer.append_transcript(
                {
                    "event": "planning",
                    "plan_path": "plan.json",
                    "planned_step_count": len(plan["planned_steps"]),
                },
            )

            router = ToolRouter(
                task.allowed_tools,
                writer,
                workspace_root=workspace_root,
                approval_allowed_tools=task.policy["approval_allowed_tools"],
                approved_tools=task.policy["approved_tools"],
            )
            verification_engine: VerificationEngine | None = None
            observations: list[dict[str, object]] = []
            completion_observations: list[dict[str, object]] = []
            passed_verification_commands: set[str] = set()
            final_response_requested = False
            for turn in range(1, self._max_turns + 1):
                context = ContextBuilder(
                    task=task,
                    workspace_root=workspace_root,
                    provider_name=self._model_gateway.provider_name,
                    episode_writer=writer,
                    observations=observations,
                    final_response_requested=final_response_requested,
                    session_summary=self._session_summary,
                ).build()
                tool_schemas = [] if final_response_requested else export_tool_schemas(task.allowed_tools)
                # 每一轮模型调用都绑定独立 context_id，便于复盘工具观察如何进入下一轮。
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
                    task,
                    model_input=context.model_input,
                    tool_schemas=tool_schemas,
                    observations=observations,
                )
                writer.append_transcript(
                    {
                        "event": "model_response",
                        "provider": self._model_gateway.provider_name,
                        "turn": turn,
                        "content": model_response.content,
                        "tool_calls": [
                            {"name": tool_call.name, "args": tool_call.args}
                            for tool_call in model_response.tool_calls
                        ],
                    },
                )

                if final_response_requested and model_response.tool_calls:
                    transition(RunStatus.FAILED)
                    writer.write_failure_attribution(
                        {
                            "stage": "executing",
                            "category": FailureCategory.MODEL.value,
                            "evidence": "model returned tool calls during final response turn",
                        },
                    )
                    return _finish_run(writer, RunStatus.FAILED, state_history)

                if not model_response.tool_calls:
                    transition(RunStatus.VERIFYING)
                    if verification_engine is None:
                        verification_engine = VerificationEngine(writer, workspace_root)
                    verification_result = verification_engine.run(task.verification_commands)
                    if verification_result.status == "success":
                        transition(RunStatus.COMPLETED)
                        writer.write_failure_attribution(None)
                        return _finish_run(writer, RunStatus.COMPLETED, state_history)

                    observations = [_verification_observation(verification_result)]
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
                        return _finish_run(writer, RunStatus.FAILED, state_history)
                    continue

                if state_history[-1] is not RunStatus.EXECUTING:
                    transition(RunStatus.EXECUTING)

                observations = []
                # 工具失败以结构化结果返回；orchestrator 在这里显式转换成 failed run。
                for tool_call in model_response.tool_calls:
                    tool_result = router.dispatch(tool_call.name, tool_call.args)
                    router.raise_for_error(tool_result)
                    observation = {
                        "tool_name": tool_call.name,
                        "args": tool_call.args,
                        "result": tool_result,
                    }
                    observations.append(observation)
                    if tool_call.name == "apply_patch":
                        completion_observations = [observation]
                    else:
                        completion_observations.append(observation)
                    writer.append_transcript(
                        {
                            "event": "tool_observation",
                            "turn": turn,
                            **observation,
                        },
                    )
                    _update_in_band_verification_progress(
                        tool_call.name,
                        tool_call.args,
                        tool_result,
                        task.verification_commands,
                        passed_verification_commands,
                    )
                if _all_declared_verification_commands_passed(
                    task.verification_commands,
                    passed_verification_commands,
                ):
                    observations = list(completion_observations)
                    final_response_requested = True
            else:
                transition(RunStatus.FAILED)
                writer.write_failure_attribution(
                    {
                        "stage": "executing",
                        "category": FailureCategory.LOOP_LIMIT.value,
                        "evidence": f"exceeded max_turns={self._max_turns}",
                    },
                )
                return _finish_run(writer, RunStatus.FAILED, state_history)
        except ToolRoutingError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": _tool_failure_category(error).value,
                    "evidence": str(error),
                },
            )
            return _finish_run(writer, RunStatus.FAILED, state_history)
        except ModelCallError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "planning",
                    "category": FailureCategory.MODEL.value,
                    "evidence": str(error),
                },
            )
            return _finish_run(writer, RunStatus.FAILED, state_history)
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
            return _finish_run(writer, RunStatus.FAILED, state_history)
        except TaskLoadError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "created",
                    "category": FailureCategory.TASK_SPEC.value,
                    "evidence": str(error),
                },
            )
            return _finish_run(writer, RunStatus.FAILED, state_history)
        except Exception as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": state_history[-2].value if len(state_history) > 1 else "created",
                    "category": _unexpected_failure_category(error, state_history).value,
                    "evidence": str(error),
                },
            )
            return _finish_run(writer, RunStatus.FAILED, state_history)


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


def _verification_loop_limit_evidence(max_turns: int, verification_result) -> str:
    return (
        f"verification did not pass before max_turns={max_turns}\n"
        f"{_verification_evidence(verification_result)}"
    )


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
    # 修改文件后，之前通过的验证不再证明当前工作区状态。
    if tool_name == "apply_patch":
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


def _finish_run(
    writer: EpisodeWriter,
    status: RunStatus,
    state_history: list[RunStatus],
) -> RunResult:
    writer.write_episode_metadata(status=status.value)
    return RunResult(status, state_history, writer.path)


def _unexpected_failure_category(error: Exception, state_history: list[RunStatus]) -> FailureCategory:
    previous_status = state_history[-2] if len(state_history) > 1 else state_history[-1]
    if isinstance(error, TypeError) and previous_status is RunStatus.PLANNING:
        return FailureCategory.MODEL_CALL
    return FailureCategory.RUNTIME


def _tool_failure_category(error: ToolRoutingError) -> FailureCategory:
    if error.error_type in {"invalid_tool_arguments", "tool_argument_invalid"}:
        return FailureCategory.TOOL_ARGUMENT
    return FailureCategory.TOOL_INTERFACE


def _workspace_root_candidate(raw_root: str | None, task_path: Path) -> Path:
    candidate = task_path.parent if raw_root is None else Path(raw_root)
    if raw_root is not None and not candidate.is_absolute():
        candidate = task_path.parent / candidate
    return candidate.resolve(strict=False)
