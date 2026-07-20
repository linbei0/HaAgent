"""
tests/unit/runtime/test_episode_writer.py - EpisodeWriter 文件产物测试

验证 episode package 的核心文件会被稳定创建。
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from haagent.models.types import ModelGatewayMetadata, ModelUsage
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend


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
    writer.write_sandbox_metadata(
        LocalSubprocessSandboxBackend(
            workspace_root=tmp_path,
            command_timeout_seconds=60,
        ).metadata(),
    )
    writer.write_failure_attribution(None)

    assert (writer.path / "task.yaml").exists()
    assert (writer.path / "context-manifest.json").exists()
    assert (writer.path / "transcript.jsonl").exists()
    assert (writer.path / "tool-calls.jsonl").exists()
    assert (writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8") == ""
    assert (writer.path / "verification" / "files.jsonl").read_text(encoding="utf-8") == ""
    assert (writer.path / "failure-attribution.md").exists()
    assert (writer.path / "environment.json").exists()
    assert (writer.path / "sandbox.json").exists()
    assert (writer.path / "cost.json").exists()


def test_episode_writer_groups_packages_by_day_and_session(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: Create episode package\n", encoding="utf-8")
    runs_root = tmp_path / ".runs"

    session_writer = EpisodeWriter.create(
        runs_root=runs_root,
        task_path=task_path,
        session_id="session-123",
    )
    standalone_writer = EpisodeWriter.create(runs_root=runs_root, task_path=task_path)

    session_parts = session_writer.path.relative_to(runs_root).parts
    assert session_parts[0] == "episodes"
    assert session_parts[4] == "session-123"
    assert len(session_parts) == 6
    standalone_parts = standalone_writer.path.relative_to(runs_root).parts
    assert standalone_parts[0] == "episodes"
    assert standalone_parts[4] == "runs"
    assert len(standalone_parts) == 6


def test_episode_writer_initializes_cost_metadata(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)

    cost = json.loads((writer.path / "cost.json").read_text(encoding="utf-8"))

    assert cost == {
        "cost_schema_version": "1.0",
        "usage_available": False,
        "pricing_available": False,
        "currency": None,
        "estimated_cost": None,
        "pricing_source": None,
        "reason": "model gateway did not provide usage metadata",
        "model_calls": [],
        "totals": {
            "model_call_count": 0,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }


def test_episode_writer_writes_extended_environment_schema(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)

    writer.write_environment(
        workspace_root=tmp_path,
        model_metadata=ModelGatewayMetadata(
            provider="openai-chat",
            model="gpt-test",
            endpoint="https://user:pass@example.test/v1/chat/completions?key=secret",
            base_url="https://token@example.test/v1?api_key=secret",
            profile_name="main",
        ),
        allowed_tools=["fake_tool"],
        registry_tool_count=3,
        entrypoint="run",
    )

    environment = json.loads((writer.path / "environment.json").read_text(encoding="utf-8"))
    assert environment["environment_schema_version"] == "1.0"
    assert environment["workspace_root"] == str(tmp_path)
    assert isinstance(environment["process"]["executable"], str)
    assert isinstance(environment["process"]["cwd"], str)
    assert environment["haagent"]["entrypoint"] == "run"
    assert environment["haagent"]["package_version"]
    assert environment["model"] == {
        "provider": "openai-chat",
        "model": "gpt-test",
        "endpoint": "https://example.test/v1/chat/completions",
        "base_url": "https://example.test/v1",
        "profile_name": "main",
        "request_config": None,
    }
    assert environment["tools"] == {
        "allowed_tool_count": 1,
        "registry_tool_count": 3,
        "allowed_tools": ["fake_tool"],
    }


def test_episode_writer_appends_usage_and_accumulates_totals(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)

    writer.append_model_usage(
        turn=1,
        provider="openai",
        model="gpt-test",
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            raw_source="openai.responses.usage",
        ),
    )
    writer.append_model_usage(
        turn=2,
        provider="openai",
        model="gpt-test",
        usage=ModelUsage(
            input_tokens=7,
            output_tokens=3,
            total_tokens=10,
            raw_source="openai.responses.usage",
        ),
    )

    cost = json.loads((writer.path / "cost.json").read_text(encoding="utf-8"))
    assert cost["usage_available"] is True
    assert cost["pricing_available"] is False
    assert cost["estimated_cost"] is None
    assert cost["reason"] == "pricing unavailable: no reliable catalog match"
    assert cost["model_calls"] == [
        {
            "turn": 1,
            "provider": "openai",
            "model": "gpt-test",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "raw_usage_source": "openai.responses.usage",
        },
        {
            "turn": 2,
            "provider": "openai",
            "model": "gpt-test",
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "raw_usage_source": "openai.responses.usage",
        },
    ]
    assert cost["totals"] == {
        "model_call_count": 2,
        "input_tokens": 17,
        "output_tokens": 8,
        "total_tokens": 25,
    }


def test_episode_writer_keeps_usage_unavailable_when_usage_is_none(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)

    writer.append_model_usage(turn=1, provider="fake", model="fake-model", usage=None)

    cost = json.loads((writer.path / "cost.json").read_text(encoding="utf-8"))
    assert cost["usage_available"] is False
    assert cost["model_calls"] == []
    assert cost["totals"] == {
        "model_call_count": 0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_episode_writer_records_optional_model_attempt_without_changing_token_totals(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)

    writer.append_model_usage(
        turn=1,
        attempt=2,
        provider="openai",
        model="gpt-test",
        usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )

    cost = json.loads((writer.path / "cost.json").read_text(encoding="utf-8"))
    assert cost["model_calls"][0]["attempt"] == 2
    assert cost["totals"]["total_tokens"] == 5


def test_episode_writer_appends_tool_calls_with_instance_write_lock(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode"
    episode_path.mkdir()
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
    writer = EpisodeWriter(path=episode_path, task_path=task_path)

    assert hasattr(writer, "_write_lock")

    def append(index: int) -> None:
        writer.append_tool_call(
            {
                "tool_name": "fake_tool",
                "args": {"index": index},
                "status": "success",
                "result": {"index": index},
                "error": None,
                "policy": None,
                "guardrail": None,
                "duration_seconds": 0.0,
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(80)))

    records = [
        json.loads(line)
        for line in (episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(record["args"]["index"] for record in records) == list(range(80))


def test_episode_writer_writes_performance_artifact(tmp_path: Path) -> None:
    writer = create_writer(tmp_path)
    payload = {
        "performance_schema_version": "1.0",
        "submit_to_run_start_ms": None,
        "run_setup_ms": 1.5,
        "context_build_ms": 1.5,
        "model_turns": [],
        "tools": [],
        "postprocess_ms": None,
        "total_turn_ms": None,
        "status": "completed",
        "dropped": {"model_attempts": 0, "tools": 0},
    }

    writer.write_performance(payload)

    written = json.loads((writer.path / "performance.json").read_text(encoding="utf-8"))
    assert written == payload


def test_episode_writer_write_performance_failure_exposes_artifact_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = create_writer(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    # EpisodeWriter 为 frozen dataclass，不能 setattr 实例方法；改写 Path.write_text。
    monkeypatch.setattr(Path, "write_text", boom)

    with pytest.raises(RuntimeError, match="performance.json"):
        writer.write_performance({"performance_schema_version": "1.0"})


def create_writer(tmp_path: Path) -> EpisodeWriter:
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
    return EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
