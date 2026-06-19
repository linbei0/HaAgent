"""
tests/test_episode_validator.py - Episode Validator 测试

验证 episode.json 和 failure.json 的 inspect 侧 schema 校验行为。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.runtime.episode_validator import (
    EpisodeValidationError,
    read_episode_metadata,
    read_failure_record,
)


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validator_accepts_valid_episode_metadata(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", valid_episode_json(tmp_path))

    metadata, warnings = read_episode_metadata(episode_path)

    assert metadata is not None
    assert metadata["status"] == "completed"
    assert warnings == []


def test_validator_rejects_unknown_episode_version(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", {"episode_version": "9.9"})

    with pytest.raises(EpisodeValidationError, match="unsupported episode_version: 9.9"):
        read_episode_metadata(episode_path)


def test_validator_rejects_invalid_episode_status(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", valid_episode_json(tmp_path, status="done-ish"))

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: episode.json status is invalid: done-ish",
    ):
        read_episode_metadata(episode_path)


def test_validator_accepts_legacy_missing_episode_and_failure_json(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    metadata, warnings = read_episode_metadata(episode_path)
    failure_record = read_failure_record(episode_path)

    assert metadata is None
    assert warnings == ["warning: episode.json missing; inspecting legacy episode", ""]
    assert failure_record is None


def test_validator_rejects_failure_unknown_category(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(
        episode_path / "failure.json",
        {
            "status": "failed",
            "failure": {
                "category": "Surprise Failure",
                "stage": "verifying",
                "evidence": "bad",
            },
        },
    )

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: failure.json category is invalid: Surprise Failure",
    ):
        read_failure_record(episode_path)
