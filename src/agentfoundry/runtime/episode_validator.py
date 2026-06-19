"""
agentfoundry/runtime/episode_validator.py - Episode schema 校验模块

集中提供 inspect 读取 episode.json 和 failure.json 时使用的兼容校验逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode import EPISODE_VERSION
from agentfoundry.runtime.failure import FailureCategory
from agentfoundry.runtime.state import RunStatus


REQUIRED_PACKAGE_FILES = [
    "episode.json",
    "task.yaml",
    "context-manifest.json",
    "transcript.jsonl",
    "tool-calls.jsonl",
    "verification/commands.jsonl",
    "failure-attribution.md",
    "failure.json",
    "environment.json",
]


class EpisodeValidationError(RuntimeError):
    """Episode package 存在但 schema 损坏或版本不兼容时抛出。"""


def validate_episode_package(episode_path: Path) -> None:
    """校验 v1 episode package 的完整性和关键 trace 文件结构。"""
    for relative_path in REQUIRED_PACKAGE_FILES:
        path = episode_path / relative_path
        if not path.exists():
            raise EpisodeValidationError(f"episode package missing required file: {relative_path}")

    read_episode_metadata(episode_path)
    read_failure_record(episode_path)
    _read_json(episode_path / "context-manifest.json")
    _read_json(episode_path / "environment.json")

    _validate_jsonl_fields(episode_path / "transcript.jsonl", "transcript.jsonl", ["event"])
    _validate_jsonl_fields(episode_path / "tool-calls.jsonl", "tool-calls.jsonl", ["tool_name", "status"])
    _validate_jsonl_fields(
        episode_path / "verification" / "commands.jsonl",
        "verification/commands.jsonl",
        ["command", "status"],
    )


def read_episode_metadata(episode_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """读取并校验 episode.json；缺失时按 legacy episode 兼容。"""
    metadata_path = episode_path / "episode.json"
    if not metadata_path.exists():
        return None, ["warning: episode.json missing; inspecting legacy episode", ""]

    metadata = _read_json(metadata_path)
    version = metadata.get("episode_version")
    if version != EPISODE_VERSION:
        raise EpisodeValidationError(f"unsupported episode_version: {version}")

    _validate_episode_metadata(metadata)
    return metadata, []


def read_failure_record(episode_path: Path) -> dict[str, Any] | None:
    """读取并校验 failure.json；缺失时按 legacy episode 兼容。"""
    path = episode_path / "failure.json"
    if not path.exists():
        return None

    record = _read_json(path)
    _validate_failure_record(record)
    return record


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EpisodeValidationError(f"{path.name} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise EpisodeValidationError(f"{path.name} must contain a JSON object")
    return value


def _validate_jsonl_fields(path: Path, label: str, required_fields: list[str]) -> None:
    """逐行校验 JSONL 可解析，并检查该 trace 类型的最小字段。"""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EpisodeValidationError(f"{label} line {line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise EpisodeValidationError(f"{label} line {line_number} must contain a JSON object")
        for field_name in required_fields:
            if field_name not in record:
                raise EpisodeValidationError(
                    f"{label} line {line_number} missing required field: {field_name}",
                )


def _validate_episode_metadata(metadata: dict[str, Any]) -> None:
    """校验 episode.json 的 inspect 契约，不扩大现有严格度。"""
    required_fields = [
        "episode_version",
        "created_at",
        "task_path",
        "status",
        "provider",
        "workspace_root",
    ]
    for field_name in required_fields:
        if field_name not in metadata:
            raise EpisodeValidationError(
                f"corrupt episode: episode.json missing required field: {field_name}",
            )

    status = metadata["status"]
    if status not in {run_status.value for run_status in RunStatus}:
        raise EpisodeValidationError(f"corrupt episode: episode.json status is invalid: {status}")


def _validate_failure_record(record: dict[str, Any]) -> None:
    """校验 failure.json，区分 legacy 缺失和存在但损坏的结构。"""
    status = record.get("status")
    if status not in {"success", "failed"}:
        raise EpisodeValidationError(f"corrupt episode: failure.json status is invalid: {status}")

    failure = record.get("failure")
    if status == "success":
        if failure is not None:
            raise EpisodeValidationError(
                "corrupt episode: failure.json success record must have failure=null",
            )
        return

    if not isinstance(failure, dict):
        raise EpisodeValidationError("corrupt episode: failure.json failed record must have failure object")

    for field_name in ["category", "stage", "evidence"]:
        if field_name not in failure:
            raise EpisodeValidationError(
                f"corrupt episode: failure.json missing failure field: {field_name}",
            )

    category = failure["category"]
    if category not in {failure_category.value for failure_category in FailureCategory}:
        raise EpisodeValidationError(f"corrupt episode: failure.json category is invalid: {category}")
