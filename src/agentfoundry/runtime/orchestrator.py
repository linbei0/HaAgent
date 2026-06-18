"""
agentfoundry/runtime/orchestrator.py - Run Orchestrator 状态机

串联 task 加载、模型调用、工具执行和 episode trace 写入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentfoundry.models.fake import FakeModelGateway
from agentfoundry.models.gateway import ModelCallError, ModelGateway
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.state import RunStatus
from agentfoundry.runtime.task_contract import TaskSpec, load_task
from agentfoundry.tools.base import ToolRoutingError
from agentfoundry.tools.router import ToolRouter
from agentfoundry.verification.engine import VerificationEngine


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state_history: list[RunStatus]
    episode_path: Path


class RunOrchestrator:
    def __init__(self, runs_root: Path, model_gateway: ModelGateway | None = None) -> None:
        self._runs_root = runs_root
        self._model_gateway = model_gateway or FakeModelGateway()

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
            transition(RunStatus.PLANNING)
            writer.write_context_manifest(self._context_manifest(task))
            writer.write_environment()
            # 模型调用和响应分开记录，方便后续区分 provider 故障与工具故障。
            writer.append_transcript(
                {
                    "event": "model_call",
                    "provider": self._model_gateway.provider_name,
                    "goal": task.goal,
                },
            )
            model_response = self._model_gateway.generate(task)
            writer.append_transcript(
                {
                    "event": "model_response",
                    "provider": self._model_gateway.provider_name,
                    "content": model_response.content,
                    "tool_calls": [
                        {"name": tool_call.name, "args": tool_call.args}
                        for tool_call in model_response.tool_calls
                    ],
                },
            )

            transition(RunStatus.EXECUTING)
            router = ToolRouter(task.allowed_tools, writer, workspace_root=task_path.parent)
            # 工具失败以结构化结果返回；orchestrator 在这里显式转换成 failed run。
            for tool_call in model_response.tool_calls:
                tool_result = router.dispatch(tool_call.name, tool_call.args)
                router.raise_for_error(tool_result)

            transition(RunStatus.VERIFYING)
            verification_result = VerificationEngine(writer, task_path.parent).run(task.verification_commands)
            if verification_result.status == "failed":
                transition(RunStatus.FAILED)
                writer.write_failure_attribution(
                    {
                        "stage": "verifying",
                        "category": "Verification Failure",
                        "evidence": (
                            f"{verification_result.failed_command} "
                            f"exited with {verification_result.exit_code}"
                        ),
                    },
                )
                return RunResult(RunStatus.FAILED, state_history, writer.path)

            transition(RunStatus.COMPLETED)
            writer.write_failure_attribution(None)
            return RunResult(RunStatus.COMPLETED, state_history, writer.path)
        except ToolRoutingError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "executing",
                    "category": "Tool Interface Failure",
                    "evidence": str(error),
                },
            )
            return RunResult(RunStatus.FAILED, state_history, writer.path)
        except ModelCallError as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": "planning",
                    "category": "Model Failure",
                    "evidence": str(error),
                },
            )
            return RunResult(RunStatus.FAILED, state_history, writer.path)
        except Exception as error:
            transition(RunStatus.FAILED)
            writer.write_failure_attribution(
                {
                    "stage": state_history[-2].value if len(state_history) > 1 else "created",
                    "category": "Runtime Failure",
                    "evidence": str(error),
                },
            )
            return RunResult(RunStatus.FAILED, state_history, writer.path)

    def _context_manifest(self, task: TaskSpec) -> dict[str, object]:
        return {
            "goal": task.goal,
            "allowed_tools": task.allowed_tools,
            "gateway": self._model_gateway.provider_name,
        }
