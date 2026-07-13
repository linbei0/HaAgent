"""
src/haagent/runtime/orchestration/recorder.py - Run 记录器

集中记录 runtime 状态转换，并在 run 结束时写入 episode 终态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.performance import PerformanceTrace


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state_history: list[RunStatus]
    episode_path: Path


@dataclass
class RunRecorder:
    writer: EpisodeWriter
    state_history: list[RunStatus] = field(default_factory=list)
    performance_trace: PerformanceTrace | None = None

    def transition(self, status: RunStatus) -> None:
        self.state_history.append(status)
        self.writer.append_transcript({"event": "state_transition", "status": status.value})

    def finish(self, status: RunStatus) -> RunResult:
        # terminal 路径统一落 performance；写失败归 recording，不伪装成模型/工具失败。
        if self.performance_trace is not None:
            self.performance_trace.mark_postprocess_start()
        self.writer.finalize_cost_metadata()
        self.writer.write_episode_metadata(status=status.value)
        if self.performance_trace is not None:
            try:
                # postprocess 包含 cost/episode metadata 写入；performance 自身是最终快照。
                self.performance_trace.finish(status.value)
                self.writer.write_performance(self.performance_trace.to_dict())
            except Exception as error:
                raise RuntimeError(f"failed to write performance.json: {error}") from error
        return RunResult(status, list(self.state_history), self.writer.path)

    def persist_performance(self) -> None:
        """模型 attempt / 工具完成后刷新 performance artifact（非逐 token）。"""

        if self.performance_trace is None:
            return
        self.writer.write_performance(self.performance_trace.to_dict())
