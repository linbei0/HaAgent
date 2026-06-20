"""
agentfoundry/runtime/orchestrator.py - Run Orchestrator 状态机

串联 task 加载、模型调用、工具执行和 episode trace 写入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentfoundry.context.builder import ContextBuildError, ContextBuilder
from agentfoundry.models.fake import FakeModelGateway
from agentfoundry.models.gateway import ModelCallError, ModelGateway
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.failure import FailureCategory
from agentfoundry.runtime.plan import build_plan
from agentfoundry.runtime.state import RunStatus
from agentfoundry.runtime.task_contract import TaskLoadError, load_task, resolve_workspace_root
from agentfoundry.tools.base import ToolRoutingError
from agentfoundry.tools.registry import export_tool_schemas
from agentfoundry.tools.router import ToolRouter
from agentfoundry.verification.engine import VerificationEngine


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
    ) -> None:
        self._runs_root = runs_root
        self._model_gateway = model_gateway or FakeModelGateway()
        self._max_turns = max_turns

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
            workspace_root = resolve_workspace_root(task, task_path)
            writer.write_episode_metadata(
                status=RunStatus.CREATED.value,
                provider=self._model_gateway.provider_name,
                workspace_root=workspace_root,
            )
            transition(RunStatus.PLANNING)
            writer.write_environment(workspace_root)
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
            )
            observations: list[dict[str, object]] = []
            has_entered_executing = False
            for turn in range(1, self._max_turns + 1):
                context = ContextBuilder(
                    task=task,
                    workspace_root=workspace_root,
                    provider_name=self._model_gateway.provider_name,
                    episode_writer=writer,
                    observations=observations,
                ).build()
                tool_schemas = export_tool_schemas(task.allowed_tools)
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

                if not model_response.tool_calls:
                    break

                if not has_entered_executing:
                    transition(RunStatus.EXECUTING)
                    has_entered_executing = True

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
                    writer.append_transcript(
                        {
                            "event": "tool_observation",
                            "turn": turn,
                            **observation,
                        },
                    )
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

            transition(RunStatus.VERIFYING)
            verification_result = VerificationEngine(writer, workspace_root).run(task.verification_commands)
            if verification_result.status == "failed":
                transition(RunStatus.FAILED)
                writer.write_failure_attribution(
                    {
                        "stage": "verifying",
                        "category": FailureCategory.VERIFICATION.value,
                        "evidence": _verification_evidence(verification_result),
                    },
                )
                return _finish_run(writer, RunStatus.FAILED, state_history)

            transition(RunStatus.COMPLETED)
            writer.write_failure_attribution(None)
            return _finish_run(writer, RunStatus.COMPLETED, state_history)
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
