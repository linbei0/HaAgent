"""
src/haagent/runtime/orchestration/recorder.py - Run 记录器

集中记录 runtime 状态转换，并在 run 结束时写入 episode 终态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.state import RunStatus


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state_history: list[RunStatus]
    episode_path: Path


@dataclass
class RunRecorder:
    writer: EpisodeWriter
    state_history: list[RunStatus] = field(default_factory=list)

    def transition(self, status: RunStatus) -> None:
        self.state_history.append(status)
        self.writer.append_transcript({"event": "state_transition", "status": status.value})

    def finish(self, status: RunStatus) -> RunResult:
        self.writer.finalize_cost_metadata()
        self.writer.write_episode_metadata(status=status.value)
        return RunResult(status, list(self.state_history), self.writer.path)
