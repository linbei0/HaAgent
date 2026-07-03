from __future__ import annotations

import json
from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.recorder import RunRecorder
from haagent.runtime.orchestration.state import RunStatus


def test_run_recorder_records_state_transitions_and_finish(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)

    recorder = RunRecorder(writer)
    recorder.transition(RunStatus.CREATED)
    recorder.transition(RunStatus.COMPLETED)
    result = recorder.finish(RunStatus.COMPLETED)

    transcript = [
        json.loads(line)
        for line in (writer.path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode = json.loads((writer.path / "episode.json").read_text(encoding="utf-8"))

    assert [event["status"] for event in transcript] == ["created", "completed"]
    assert result.status == RunStatus.COMPLETED
    assert result.state_history == [RunStatus.CREATED, RunStatus.COMPLETED]
    assert result.episode_path == writer.path
    assert episode["status"] == "completed"
