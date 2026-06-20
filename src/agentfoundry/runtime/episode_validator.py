"""
agentfoundry/runtime/episode_validator.py - Episode schema 校验模块

集中提供 inspect 读取 episode.json 和 failure.json 时使用的兼容校验逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode import EPISODE_VERSION
from agentfoundry.runtime.failure import FailureCategory
from agentfoundry.runtime.state import RunStatus


REQUIRED_PACKAGE_FILES = [
    "episode.json",
    "task.yaml",
    "plan.json",
    "context-manifest.json",
    "transcript.jsonl",
    "tool-calls.jsonl",
    "verification/commands.jsonl",
    "failure-attribution.md",
    "failure.json",
    "environment.json",
    "sandbox.json",
]


class EpisodeValidationError(RuntimeError):
    """Episode package 存在但 schema 损坏或版本不兼容时抛出。"""


@dataclass(frozen=True)
class EpisodePackageView:
    episode_metadata: dict[str, Any]
    failure_record: dict[str, Any]
    context_manifest: dict[str, Any]
    transcript: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    verification_commands: list[dict[str, Any]]
    plan: dict[str, Any] = field(default_factory=dict)
    sandbox: dict[str, Any] = field(default_factory=dict)


def validate_episode_package(episode_path: Path) -> None:
    """校验 v1 episode package 的完整性和关键 trace 文件结构。"""
    _read_validated_episode_package(episode_path)


def load_validated_episode_package(episode_path: Path) -> EpisodePackageView:
    """校验并返回已解析的 episode package view，供后续审计/导出复用。"""
    return _read_validated_episode_package(episode_path)


def _read_validated_episode_package(episode_path: Path) -> EpisodePackageView:
    """读取、校验并组装 package view；保持 validate 入口只负责触发该流程。"""
    for relative_path in REQUIRED_PACKAGE_FILES:
        path = episode_path / relative_path
        if not path.exists():
            raise EpisodeValidationError(f"episode package missing required file: {relative_path}")

    episode_metadata, _warnings = read_episode_metadata(episode_path)
    failure_record = read_failure_record(episode_path)
    plan = _read_json(episode_path / "plan.json")
    context_manifest = _read_json(episode_path / "context-manifest.json")
    environment = _read_json(episode_path / "environment.json")
    sandbox = _read_json(episode_path / "sandbox.json")

    transcript = _validate_jsonl_fields(episode_path / "transcript.jsonl", "transcript.jsonl", ["event"])
    tool_calls = _validate_jsonl_fields(episode_path / "tool-calls.jsonl", "tool-calls.jsonl", ["tool_name", "status"])
    verification_commands = _validate_jsonl_fields(
        episode_path / "verification" / "commands.jsonl",
        "verification/commands.jsonl",
        ["command", "status"],
    )
    if episode_metadata is None or failure_record is None:
        raise EpisodeValidationError("episode package must contain v1 episode.json and failure.json")
    _validate_environment(environment)
    _validate_sandbox(sandbox)
    _validate_plan(plan)
    _validate_tool_calls(tool_calls)
    _validate_verification_commands(verification_commands)
    _validate_cross_file_consistency(
        episode_path,
        episode_metadata,
        failure_record,
        context_manifest,
        environment,
        sandbox,
        transcript,
    )
    return EpisodePackageView(
        episode_metadata=episode_metadata,
        failure_record=failure_record,
        plan=plan,
        context_manifest=context_manifest,
        transcript=transcript,
        tool_calls=tool_calls,
        verification_commands=verification_commands,
        sandbox=sandbox,
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


def _validate_jsonl_fields(path: Path, label: str, required_fields: list[str]) -> list[dict[str, Any]]:
    """逐行校验 JSONL 可解析，并检查该 trace 类型的最小字段。"""
    records = []
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
        records.append(record)
    return records


def _validate_tool_calls(records: list[dict[str, Any]]) -> None:
    """校验 tool-calls.jsonl 的最小字段类型和值域。

    ToolRouter 当前写入 success/error；failed 保留为旧 episode 兼容状态。
    """
    for index, record in enumerate(records, start=1):
        if not isinstance(record["tool_name"], str):
            raise EpisodeValidationError(f"tool-calls.jsonl line {index} tool_name must be a string")
        status = record["status"]
        if status not in {"success", "error", "failed"}:
            raise EpisodeValidationError(f"tool-calls.jsonl line {index} status is invalid: {status}")


def _validate_verification_commands(records: list[dict[str, Any]]) -> None:
    """校验 verification commands trace 的最小字段类型和值域。"""
    for index, record in enumerate(records, start=1):
        if not isinstance(record["command"], str):
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} command must be a string",
            )
        status = record["status"]
        if status not in {"success", "failed", "timeout"}:
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} status is invalid: {status}",
            )
        if "exit_code" in record and not (
            isinstance(record["exit_code"], int) or record["exit_code"] is None
        ):
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} exit_code must be an integer or null",
            )
        if "timeout" in record and not isinstance(record["timeout"], bool):
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} timeout must be a bool",
            )


def _validate_environment(environment: dict[str, Any]) -> None:
    """校验 environment.json 的最小审计字段类型。"""
    for field_name in ["python", "platform"]:
        if not isinstance(environment.get(field_name), str):
            raise EpisodeValidationError(f"environment.json {field_name} must be a string")
    _validate_iso_datetime_field(
        environment.get("created_at"),
        label="environment.json created_at",
    )
    if "workspace_root" in environment and not isinstance(environment["workspace_root"], str):
        raise EpisodeValidationError("environment.json workspace_root must be a string")


def _validate_sandbox(sandbox: dict[str, Any]) -> None:
    """校验 sandbox.json 的最小 sandbox 元数据字段类型。"""
    for field_name in [
        "workspace_root",
        "filesystem_boundary",
        "network_policy",
        "process_policy",
        "credential_policy",
    ]:
        if not isinstance(sandbox.get(field_name), str):
            raise EpisodeValidationError(f"sandbox.json {field_name} must be a string")
    resource_limits = sandbox.get("resource_limits")
    if not isinstance(resource_limits, dict):
        raise EpisodeValidationError("sandbox.json resource_limits must be an object")
    command_timeout = resource_limits.get("command_timeout_seconds")
    if not (
        isinstance(command_timeout, int | float)
        and not isinstance(command_timeout, bool)
    ):
        raise EpisodeValidationError(
            "sandbox.json resource_limits.command_timeout_seconds must be a number",
        )


def _validate_plan(plan: dict[str, Any]) -> None:
    """校验 Agent Plan Trace v0 的最小字段类型。"""
    if not isinstance(plan.get("goal"), str):
        raise EpisodeValidationError("plan.json goal must be a string")
    for field_name in [
        "allowed_tools",
        "acceptance_criteria",
        "verification_commands",
        "planned_steps",
    ]:
        value = plan.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise EpisodeValidationError(f"plan.json {field_name} must be a list of strings")


def _validate_cross_file_consistency(
    episode_path: Path,
    episode_metadata: dict[str, Any],
    failure_record: dict[str, Any],
    context_manifest: dict[str, Any],
    environment: dict[str, Any],
    sandbox: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> None:
    """校验 episode 根状态、失败记录、context 索引与 transcript 末态一致。"""
    state_transitions = [
        record
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    if not state_transitions:
        raise EpisodeValidationError("transcript.jsonl missing state_transition")

    episode_status = str(episode_metadata["status"])
    final_status = str(state_transitions[-1].get("status"))
    if episode_status != final_status:
        raise EpisodeValidationError(
            f"episode status {episode_status} does not match transcript final status {final_status}",
        )

    failure_status = failure_record["status"]
    if failure_status == "success" and episode_status != RunStatus.COMPLETED.value:
        raise EpisodeValidationError("failure.json status success requires episode status completed")
    if failure_status == "failed" and episode_status != RunStatus.FAILED.value:
        raise EpisodeValidationError("failure.json status failed requires episode status failed")

    environment_workspace_root = environment.get("workspace_root")
    if (
        environment_workspace_root is not None
        and environment_workspace_root != episode_metadata["workspace_root"]
    ):
        raise EpisodeValidationError(
            "environment.json workspace_root does not match episode.json workspace_root",
        )
    if sandbox["workspace_root"] != episode_metadata["workspace_root"]:
        raise EpisodeValidationError(
            "sandbox.json workspace_root does not match episode.json workspace_root",
        )

    contexts = context_manifest.get("contexts")
    context_count = context_manifest.get("context_count")
    if not isinstance(contexts, list):
        raise EpisodeValidationError("context-manifest.json contexts must be a list")
    if not isinstance(context_count, int):
        raise EpisodeValidationError("context-manifest.json context_count must be an integer")
    if context_count != len(contexts):
        raise EpisodeValidationError(
            f"context-manifest.json context_count {context_count} does not match contexts length {len(contexts)}",
        )
    _validate_context_items(episode_path, contexts)


def _validate_context_items(episode_path: Path, contexts: list[Any]) -> None:
    """校验 context-manifest contexts 索引项和对应文件存在性。"""
    required_fields = ["context_id", "model_input_path", "manifest_path"]
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise EpisodeValidationError(f"context-manifest.json contexts[{index}] must be an object")
        for field_name in required_fields:
            if field_name not in context:
                raise EpisodeValidationError(
                    f"context-manifest.json contexts[{index}] missing required field: {field_name}",
                )
            if not isinstance(context[field_name], str):
                raise EpisodeValidationError(
                    f"context-manifest.json contexts[{index}].{field_name} must be a string",
                )
        for field_name in ["model_input_path", "manifest_path"]:
            if not _is_episode_internal_file(episode_path, context[field_name]):
                raise EpisodeValidationError(
                    f"context-manifest.json contexts[{index}].{field_name} file missing: {context[field_name]}",
                )
        context_json = _read_json(episode_path / context["manifest_path"])
        _validate_context_json(index, context, context_json)


def _validate_context_json(index: int, context_index: dict[str, Any], context_json: dict[str, Any]) -> None:
    label = str(context_index["manifest_path"])
    if not isinstance(context_json.get("context_id"), str):
        raise EpisodeValidationError(f"{label} context_id must be a string")
    budget = context_json.get("budget")
    if not isinstance(budget, dict):
        raise EpisodeValidationError(f"{label} budget must be an object")
    _validate_context_budget_object(
        budget,
        label=f"{label} budget",
        count_fields={
            "character_count": "int",
            "character_limit": "int",
        },
        status_field="status",
    )
    sources = context_json.get("sources")
    if not isinstance(sources, list):
        raise EpisodeValidationError(f"{label} sources must be a list")
    for source_index, source in enumerate(sources):
        _validate_context_source(label, source_index, source)
    _validate_context_index_budget(index, context_index, sources)


def _validate_context_budget_object(
    budget: dict[str, Any],
    label: str,
    count_fields: dict[str, str],
    status_field: str,
) -> None:
    for field_name in count_fields:
        if not isinstance(budget.get(field_name), int):
            raise EpisodeValidationError(f"{label}.{field_name} must be an int")
    status = budget.get(status_field)
    if status not in {"within_limit", "over_limit"}:
        raise EpisodeValidationError(f"{label}.{status_field} is invalid: {status}")


def _validate_context_source(label: str, source_index: int, source: Any) -> None:
    if not isinstance(source, dict):
        raise EpisodeValidationError(f"{label} sources[{source_index}] must be an object")
    for field_name in ["source_type", "name", "description"]:
        if not isinstance(source.get(field_name), str):
            raise EpisodeValidationError(f"{label} sources[{source_index}].{field_name} must be a string")
    if not _non_empty_string(source.get("inclusion_reason")):
        raise EpisodeValidationError(f"{label} sources[{source_index}].inclusion_reason must be a non-empty string")
    budget = source.get("budget")
    if not isinstance(budget, dict):
        raise EpisodeValidationError(f"{label} sources[{source_index}].budget must be an object")
    if not isinstance(budget.get("char_count"), int):
        raise EpisodeValidationError(f"{label} sources[{source_index}].budget.char_count must be an int")
    if not isinstance(budget.get("included_in_model_input"), bool):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.included_in_model_input must be a bool",
        )
    if not _non_empty_string(budget.get("inclusion_reason")):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.inclusion_reason must be a non-empty string",
        )


def _validate_context_index_budget(index: int, context_index: dict[str, Any], sources: list[Any]) -> None:
    label = f"context-manifest.json contexts[{index}].budget"
    budget = context_index.get("budget")
    if not isinstance(budget, dict):
        raise EpisodeValidationError(f"{label} must be an object")
    if budget.get("context_id") != context_index["context_id"]:
        raise EpisodeValidationError(f"{label}.context_id must match context_id")
    for field_name in ["total_chars", "max_chars", "source_count", "included_source_count"]:
        if not isinstance(budget.get(field_name), int):
            raise EpisodeValidationError(f"{label}.{field_name} must be an int")
    status = budget.get("status")
    if status not in {"within_limit", "over_limit"}:
        raise EpisodeValidationError(f"{label}.status is invalid: {status}")
    if budget["source_count"] != len(sources):
        raise EpisodeValidationError(f"{label}.source_count does not match sources length")
    included_count = sum(
        1
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("budget"), dict)
        and source["budget"].get("included_in_model_input") is True
    )
    if budget["included_source_count"] != included_count:
        raise EpisodeValidationError(f"{label}.included_source_count does not match included sources")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_episode_internal_file(episode_path: Path, relative_path: str) -> bool:
    candidate = (episode_path / relative_path).resolve()
    episode_root = episode_path.resolve()
    try:
        candidate.relative_to(episode_root)
    except ValueError:
        return False
    return candidate.is_file()


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

    _validate_iso_datetime(metadata["created_at"])
    for field_name in ["task_path", "provider", "workspace_root"]:
        if not isinstance(metadata[field_name], str):
            raise EpisodeValidationError(
                f"corrupt episode: episode.json {field_name} must be a string",
            )


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

    stage = failure["stage"]
    if stage not in {run_status.value for run_status in RunStatus}:
        raise EpisodeValidationError(f"corrupt episode: failure.json stage is invalid: {stage}")

    if not isinstance(failure["evidence"], str):
        raise EpisodeValidationError("corrupt episode: failure.json evidence must be a string")


def _validate_iso_datetime(value: Any) -> None:
    """校验 created_at 是 ISO 时间字符串；Z 后缀转为 Python 可解析的 +00:00。"""
    _validate_iso_datetime_field(value, label="corrupt episode: episode.json created_at")


def _validate_iso_datetime_field(value: Any, label: str) -> None:
    """校验指定字段是 ISO 时间字符串；Z 后缀转为 Python 可解析的 +00:00。"""
    if not isinstance(value, str):
        raise EpisodeValidationError(f"{label} must be an ISO string")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EpisodeValidationError(
            f"{label} is invalid: {value}",
        ) from error
