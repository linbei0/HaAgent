from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_foundry.episode import EpisodeWriter
from agent_foundry.states import RunStatus
from agent_foundry.task import TaskSpec, load_task
from agent_foundry.tools import ToolRouter, ToolRoutingError


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state_history: list[RunStatus]
    episode_path: Path


class RunOrchestrator:
    def __init__(self, runs_root: Path) -> None:
        self._runs_root = runs_root

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

            transition(RunStatus.EXECUTING)
            ToolRouter(task.allowed_tools, writer).dispatch("fake_tool", {"goal": task.goal})

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
            "gateway": "fake",
        }
