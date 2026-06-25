"""
haagent/runtime/episode_validator.py - Episode schema 校验模块

集中提供 inspect、eval export 读取 episode package 时使用的严格 schema 校验逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from haagent.runtime.episode import EPISODE_VERSION
from haagent.runtime.failure import FailureCategory
from haagent.runtime.state import RunStatus


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
VERIFICATION_COMMANDS_FILE = "verification/commands.jsonl"
INSPECT_PRE_VERIFICATION_CORE_FILES = [
    "episode.json",
    "task.yaml",
    "transcript.jsonl",
    "tool-calls.jsonl",
    "failure-attribution.md",
    "failure.json",
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
    workspace_preflight: dict[str, Any] = field(default_factory=dict)
    verification_reached: bool = True


def validate_episode_package(episode_path: Path) -> None:
    """校验 v1 episode package 的完整性和关键 trace 文件结构。"""
    _read_validated_episode_package(episode_path)


def load_validated_episode_package(episode_path: Path) -> EpisodePackageView:
    """校验并返回已解析的 episode package view，供后续审计/导出复用。"""
    return _read_validated_episode_package(episode_path)


def load_inspect_episode_package(episode_path: Path) -> EpisodePackageView:
    """读取 inspect 用 package view；允许 verifying 前失败的 episode 缺少 verification trace。"""
    return _read_inspect_episode_package(episode_path)


def _read_validated_episode_package(
    episode_path: Path,
    allow_missing_pre_verification: bool = False,
) -> EpisodePackageView:
    """读取、校验并组装 package view；保持 validate 入口只负责触发该流程。"""
    required_files = (
        [path for path in REQUIRED_PACKAGE_FILES if path != VERIFICATION_COMMANDS_FILE]
        if allow_missing_pre_verification
        else REQUIRED_PACKAGE_FILES
    )
    for relative_path in required_files:
        path = episode_path / relative_path
        if not path.exists():
            raise EpisodeValidationError(f"episode package missing required file: {relative_path}")

    episode_metadata, _warnings = read_episode_metadata(episode_path)
    failure_record = read_failure_record(episode_path)
    plan = _read_json(episode_path / "plan.json")
    context_manifest = _read_json(episode_path / "context-manifest.json")
    environment = _read_json(episode_path / "environment.json")
    sandbox = _read_json(episode_path / "sandbox.json")
    workspace_preflight = _read_workspace_preflight(episode_path)

    transcript = _validate_jsonl_fields(episode_path / "transcript.jsonl", "transcript.jsonl", ["event"])
    tool_calls = _validate_jsonl_fields(episode_path / "tool-calls.jsonl", "tool-calls.jsonl", ["tool_name", "status"])
    verification_path = episode_path / VERIFICATION_COMMANDS_FILE
    verification_reached = True
    if verification_path.exists():
        verification_commands = _validate_jsonl_fields(
            verification_path,
            VERIFICATION_COMMANDS_FILE,
            ["command", "status"],
        )
    elif (
        allow_missing_pre_verification
        and _can_omit_verification_commands(episode_metadata, transcript)
    ):
        verification_commands = []
        verification_reached = False
    else:
        raise EpisodeValidationError(
            f"episode package missing required file: {VERIFICATION_COMMANDS_FILE}",
        )
    _validate_environment(environment)
    _validate_sandbox(sandbox)
    if workspace_preflight:
        _validate_workspace_preflight(workspace_preflight)
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
        workspace_preflight=workspace_preflight,
        verification_reached=verification_reached,
    )


def _read_inspect_episode_package(episode_path: Path) -> EpisodePackageView:
    """读取 inspect 视图；仅对当前 run 的 verifying 前失败放宽后续阶段文件要求。"""
    for relative_path in INSPECT_PRE_VERIFICATION_CORE_FILES:
        path = episode_path / relative_path
        if not path.exists():
            raise EpisodeValidationError(f"episode package missing required file: {relative_path}")

    episode_metadata, _warnings = read_episode_metadata(
        episode_path,
        allow_nullable_runtime_fields=True,
    )
    failure_record = read_failure_record(episode_path)
    transcript = _validate_jsonl_fields(episode_path / "transcript.jsonl", "transcript.jsonl", ["event"])
    tool_calls = _validate_jsonl_fields(episode_path / "tool-calls.jsonl", "tool-calls.jsonl", ["tool_name", "status"])
    workspace_preflight = _read_workspace_preflight(episode_path)
    if workspace_preflight:
        _validate_workspace_preflight(workspace_preflight)

    if not _can_omit_verification_commands(episode_metadata, transcript):
        _validate_episode_metadata(episode_metadata)
        return _read_validated_episode_package(
            episode_path,
            allow_missing_pre_verification=True,
        )

    _validate_tool_calls(tool_calls)
    verification_path = episode_path / VERIFICATION_COMMANDS_FILE
    if verification_path.exists():
        verification_commands = _validate_jsonl_fields(
            verification_path,
            VERIFICATION_COMMANDS_FILE,
            ["command", "status"],
        )
        verification_reached = True
    else:
        verification_commands = []
        verification_reached = False
    _validate_verification_commands(verification_commands)

    plan = _read_optional_json(episode_path / "plan.json")
    if plan is not None:
        _validate_plan(plan)
    context_manifest = _read_optional_json(episode_path / "context-manifest.json")
    if context_manifest is None:
        context_manifest = {"context_count": 0, "contexts": []}
    else:
        _validate_context_manifest_for_inspect(episode_path, context_manifest)
    environment = _read_optional_json(episode_path / "environment.json")
    if environment is not None:
        _validate_environment(environment)
    sandbox = _read_optional_json(episode_path / "sandbox.json")
    if sandbox is not None:
        _validate_sandbox(sandbox)

    _validate_pre_verification_failure_consistency(
        episode_metadata,
        failure_record,
        transcript,
        environment,
        sandbox,
    )
    inspect_metadata = dict(episode_metadata)
    if inspect_metadata.get("provider") is None:
        inspect_metadata["provider"] = "unknown"
    if inspect_metadata.get("workspace_root") is None:
        inspect_metadata["workspace_root"] = "unknown"

    return EpisodePackageView(
        episode_metadata=inspect_metadata,
        failure_record=failure_record,
        plan=plan or {},
        context_manifest=context_manifest,
        transcript=transcript,
        tool_calls=tool_calls,
        verification_commands=verification_commands,
        sandbox=sandbox or {},
        workspace_preflight=workspace_preflight,
        verification_reached=verification_reached,
    )


def _can_omit_verification_commands(
    episode_metadata: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> bool:
    if episode_metadata.get("status") != RunStatus.FAILED.value:
        return False
    states = [
        record.get("status")
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    return RunStatus.VERIFYING.value not in states


def read_episode_metadata(
    episode_path: Path,
    *,
    allow_nullable_runtime_fields: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """读取并校验 episode.json；缺失即视为损坏 package。"""
    metadata_path = episode_path / "episode.json"
    if not metadata_path.exists():
        raise EpisodeValidationError("episode package missing required file: episode.json")

    metadata = _read_json(metadata_path)
    version = metadata.get("episode_version")
    if version != EPISODE_VERSION:
        raise EpisodeValidationError(f"unsupported episode_version: {version}")

    _validate_episode_metadata(
        metadata,
        allow_nullable_runtime_fields=allow_nullable_runtime_fields,
    )
    return metadata, []


def read_failure_record(episode_path: Path) -> dict[str, Any]:
    """读取并校验 failure.json；缺失即视为损坏 package。"""
    path = episode_path / "failure.json"
    if not path.exists():
        raise EpisodeValidationError("episode package missing required file: failure.json")

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


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_workspace_preflight(episode_path: Path) -> dict[str, Any]:
    preflight = _read_optional_json(episode_path / "workspace" / "preflight.json")
    return preflight or {}


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
    """校验 tool-calls.jsonl 的当前字段类型和值域。"""
    for index, record in enumerate(records, start=1):
        if not isinstance(record["tool_name"], str):
            raise EpisodeValidationError(f"tool-calls.jsonl line {index} tool_name must be a string")
        status = record["status"]
        if status not in {"success", "error"}:
            raise EpisodeValidationError(f"tool-calls.jsonl line {index} status is invalid: {status}")
        if "policy" not in record:
            raise EpisodeValidationError(
                f"tool-calls.jsonl line {index} missing required field: policy",
            )
        _validate_tool_call_policy(index, record)


def _validate_tool_call_policy(index: int, record: dict[str, Any]) -> None:
    policy = record["policy"]
    if policy is None and _tool_policy_not_evaluated(record):
        return
    if not isinstance(policy, dict):
        raise EpisodeValidationError(f"tool-calls.jsonl line {index} policy must be an object")
    for field_name in ["tool_name", "risk_level", "action", "reason"]:
        if not isinstance(policy.get(field_name), str):
            raise EpisodeValidationError(
                f"tool-calls.jsonl line {index} policy.{field_name} must be a string",
            )
    approval = policy.get("approval")
    if not isinstance(approval, dict):
        raise EpisodeValidationError(
            f"tool-calls.jsonl line {index} policy.approval must be an object",
        )
    if not isinstance(approval.get("required"), bool):
        raise EpisodeValidationError(
            f"tool-calls.jsonl line {index} policy.approval.required must be a bool",
        )
    for field_name in ["status", "reason"]:
        if not isinstance(approval.get(field_name), str):
            raise EpisodeValidationError(
                f"tool-calls.jsonl line {index} policy.approval.{field_name} must be a string",
            )


def _tool_policy_not_evaluated(record: dict[str, Any]) -> bool:
    error = record.get("error")
    return (
        record.get("status") == "error"
        and isinstance(error, dict)
        and error.get("type") in {"tool_not_allowed", "unknown_tool"}
    )


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
        for field_name in [
            "exit_code",
            "timeout",
            "stdout_excerpt",
            "stderr_excerpt",
            "stdout_truncated",
            "stderr_truncated",
            "stdout_original_length",
            "stderr_original_length",
            "redacted",
        ]:
            if field_name not in record:
                raise EpisodeValidationError(
                    f"verification/commands.jsonl line {index} missing required field: {field_name}",
                )
        if not (isinstance(record["exit_code"], int) or record["exit_code"] is None):
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} exit_code must be an integer or null",
            )
        if not isinstance(record["timeout"], bool):
            raise EpisodeValidationError(
                f"verification/commands.jsonl line {index} timeout must be a bool",
            )
        for field_name in ["stdout_excerpt", "stderr_excerpt"]:
            if not isinstance(record[field_name], str):
                raise EpisodeValidationError(
                    f"verification/commands.jsonl line {index} {field_name} must be a string",
                )
        for field_name in ["stdout_truncated", "stderr_truncated", "redacted"]:
            if not isinstance(record[field_name], bool):
                raise EpisodeValidationError(
                    f"verification/commands.jsonl line {index} {field_name} must be a bool",
                )
        for field_name in ["stdout_original_length", "stderr_original_length"]:
            if not isinstance(record[field_name], int):
                raise EpisodeValidationError(
                    f"verification/commands.jsonl line {index} {field_name} must be an int",
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


def _validate_workspace_preflight(preflight: dict[str, Any]) -> None:
    """校验 workspace preflight 审计记录的轻量结构。"""
    if not isinstance(preflight.get("workspace_root"), str):
        raise EpisodeValidationError("workspace/preflight.json workspace_root must be a string")
    for field_name in ["exists", "is_git_repo", "modifies_original_workspace"]:
        if not isinstance(preflight.get(field_name), bool):
            raise EpisodeValidationError(f"workspace/preflight.json {field_name} must be a bool")
    git_branch = preflight.get("git_branch")
    if git_branch is not None and not isinstance(git_branch, str):
        raise EpisodeValidationError("workspace/preflight.json git_branch must be a string or null")
    git_dirty = preflight.get("git_dirty")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise EpisodeValidationError("workspace/preflight.json git_dirty must be a bool or null")
    git_status = preflight.get("git_status")
    if git_status not in {"missing", "not_git_repo", "unknown", "clean", "dirty"}:
        raise EpisodeValidationError(
            f"workspace/preflight.json git_status is invalid: {git_status}",
        )
    summary = preflight.get("git_dirty_summary")
    if not isinstance(summary, dict):
        raise EpisodeValidationError("workspace/preflight.json git_dirty_summary must be an object")
    for field_name in ["total", "modified", "untracked", "deleted", "renamed", "other"]:
        value = summary.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise EpisodeValidationError(
                f"workspace/preflight.json git_dirty_summary.{field_name} must be an int",
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


def _validate_context_manifest_for_inspect(
    episode_path: Path,
    context_manifest: dict[str, Any],
) -> None:
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


def _validate_pre_verification_failure_consistency(
    episode_metadata: dict[str, Any],
    failure_record: dict[str, Any],
    transcript: list[dict[str, Any]],
    environment: dict[str, Any] | None,
    sandbox: dict[str, Any] | None,
) -> None:
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
    if episode_status != RunStatus.FAILED.value:
        raise EpisodeValidationError("pre-verification inspect view requires failed episode status")
    if failure_record["status"] != "failed":
        raise EpisodeValidationError("failure.json status failed requires episode status failed")

    workspace_root = episode_metadata.get("workspace_root")
    if (
        environment is not None
        and environment.get("workspace_root") is not None
        and environment.get("workspace_root") != workspace_root
    ):
        raise EpisodeValidationError(
            "environment.json workspace_root does not match episode.json workspace_root",
        )
    if (
        sandbox is not None
        and workspace_root is not None
        and sandbox["workspace_root"] != workspace_root
    ):
        raise EpisodeValidationError(
            "sandbox.json workspace_root does not match episode.json workspace_root",
        )


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
    if not isinstance(context_json.get("message_count"), int):
        raise EpisodeValidationError(f"{label} message_count must be an int")


def _validate_next_action(label: str, next_action: Any) -> None:
    if not isinstance(next_action, dict):
        raise EpisodeValidationError(f"{label} next_action must be an object")
    status = next_action.get("status")
    if status not in {"none", "continue", "handle_error", "decide"}:
        raise EpisodeValidationError(f"{label} next_action.status is invalid: {status}")
    if not isinstance(next_action.get("reason"), str):
        raise EpisodeValidationError(f"{label} next_action.reason must be a string")
    observation_index = next_action.get("based_on_observation_index")
    if observation_index is not None and not isinstance(observation_index, int):
        raise EpisodeValidationError(
            f"{label} next_action.based_on_observation_index must be an integer or null",
        )
    tool_name = next_action.get("based_on_tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise EpisodeValidationError(
            f"{label} next_action.based_on_tool_name must be a string or null",
        )


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
    for field_name in ["raw_char_count", "model_input_char_count"]:
        if not isinstance(budget.get(field_name), int):
            raise EpisodeValidationError(
                f"{label} sources[{source_index}].budget.{field_name} must be an int",
            )
    if not isinstance(budget.get("included_in_model_input"), bool):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.included_in_model_input must be a bool",
        )
    if not isinstance(budget.get("truncated"), bool):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.truncated must be a bool",
        )
    if not _non_empty_string(budget.get("inclusion_reason")):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.inclusion_reason must be a non-empty string",
        )
    if not budget["included_in_model_input"] and not _non_empty_string(budget.get("exclusion_reason")):
        raise EpisodeValidationError(
            f"{label} sources[{source_index}].budget.exclusion_reason must be a non-empty string when excluded",
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


def _validate_episode_metadata(
    metadata: dict[str, Any],
    *,
    allow_nullable_runtime_fields: bool = False,
) -> None:
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
    if not isinstance(metadata["task_path"], str):
        raise EpisodeValidationError(
            "corrupt episode: episode.json task_path must be a string",
        )
    for field_name in ["provider", "workspace_root"]:
        if allow_nullable_runtime_fields and metadata[field_name] is None:
            continue
        if not isinstance(metadata[field_name], str):
            raise EpisodeValidationError(
                f"corrupt episode: episode.json {field_name} must be a string",
            )


def _validate_failure_record(record: dict[str, Any]) -> None:
    """校验 failure.json 的当前失败归因结构。"""
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
