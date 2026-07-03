"""
tests/unit/runtime/test_episode_writer.py - EpisodeWriter 文件产物测试

验证 episode package 的核心文件会被稳定创建。
"""

from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter


def test_episode_writer_creates_required_files(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Create episode package
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    writer = EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
    writer.write_context_manifest({"allowed_tools": ["fake_tool"]})
    writer.write_environment()
    writer.write_sandbox_metadata(tmp_path, command_timeout_seconds=60)
    writer.write_failure_attribution(None)

    assert (writer.path / "task.yaml").exists()
    assert (writer.path / "context-manifest.json").exists()
    assert (writer.path / "transcript.jsonl").exists()
    assert (writer.path / "tool-calls.jsonl").exists()
    assert (writer.path / "failure-attribution.md").exists()
    assert (writer.path / "environment.json").exists()
    assert (writer.path / "sandbox.json").exists()
