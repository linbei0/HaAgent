from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_foundry.episode import EpisodeWriter
from agent_foundry.model_gateway import FakeModelGateway, ModelCallError, ModelGateway
from agent_foundry.states import RunStatus
from agent_foundry.task import TaskSpec, load_task
from agent_foundry.tools import ToolRouter, ToolRoutingError


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
        state_history: list[RunStatus] = []
        writer = EpisodeWriter.create(self._runs_root, task_path)

        def transition(status: RunStatus) -> None:
            state_history.append(status)
            writer.append_transcript({"event": "state_transition", "status": status.value})

        transition(RunStatus.CREATED)

        try:
            task = load_task(task_path)
            transition(RunStatus.PLANNING)
            writer.write_context_manifest(self._context_manifest(task))
            writer.write_environment()
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
            for tool_call in model_response.tool_calls:
                tool_result = router.dispatch(tool_call.name, tool_call.args)
                router.raise_for_error(tool_result)

            transition(RunStatus.VERIFYING)
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
