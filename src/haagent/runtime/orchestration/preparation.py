"""
src/haagent/runtime/orchestration/preparation.py - Run 前置准备流程

负责 RunOrchestrator 的 task 加载、workspace 初始化、plan 记录、context 构建和 full compact。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.sections import context_budget_from_compression_budget
from haagent.models.types import ModelGateway
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.context.compression.full import FullCompactEligibility, FullCompactResult, maybe_full_compact_messages
from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver
from haagent.runtime.execution.path_policy import default_path_policy, load_path_policy
from haagent.runtime.contracts.plan import build_plan
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.contracts.task import TaskSpec, load_task, resolve_workspace_root
from haagent.tools.registry import ToolRuntimeRegistry
from haagent.runtime.contracts.workspace_preflight import build_workspace_preflight


@dataclass(frozen=True)
class RunSetup:
    task: TaskSpec
    workspace_root: Path
    path_policy: dict[str, object]


@dataclass(frozen=True)
class PreparedMessages:
    context_id: str
    messages: list[dict[str, Any]]
    cache_diagnostics: dict[str, dict[str, object]]


def prepare_run_setup(
    *,
    task_path: Path,
    writer: EpisodeWriter,
    provider_name: str,
    transition: Callable[[RunStatus], None],
    raise_if_cancelled: Callable[[], None],
    session_compaction: dict[str, object] | None,
) -> RunSetup:
    raise_if_cancelled()
    task = load_task(task_path)
    workspace_candidate = workspace_root_candidate(task.workspace_root, task_path)
    writer.write_workspace_preflight(build_workspace_preflight(workspace_candidate))
    workspace_root = resolve_workspace_root(task, task_path)
    path_policy = load_path_policy(task.path_policy) if task.path_policy is not None else default_path_policy(workspace_root)
    writer.write_episode_metadata(
        status=RunStatus.CREATED.value,
        provider=provider_name,
        workspace_root=workspace_root,
    )
    transition(RunStatus.PLANNING)
    writer.write_environment(workspace_root, entrypoint="run")
    raise_if_cancelled()
    plan = build_plan(task)
    writer.write_plan(plan)
    writer.append_transcript(
        {
            "event": "planning",
            "plan_path": "plan.json",
            "planned_step_count": len(plan["planned_steps"]),
        },
    )
    if isinstance(session_compaction, dict) and session_compaction.get("decision") == "compacted":
        writer.append_transcript(
            {
                "event": "session_memory_compaction",
                "trigger_kind": "session_memory",
                "decision": session_compaction.get("decision"),
                "original_turn_count": session_compaction.get("original_turn_count"),
                "compacted_turn_count": session_compaction.get("compacted_turn_count"),
                "preserved_recent_count": session_compaction.get("preserved_recent_count"),
                "original_chars": session_compaction.get("original_chars"),
                "final_chars": session_compaction.get("final_chars"),
                "saved_chars": session_compaction.get("saved_chars"),
                "reason": session_compaction.get("reason"),
            },
        )
    return RunSetup(task=task, workspace_root=workspace_root, path_policy=path_policy)


def prepare_initial_messages(
    *,
    context_builder_cls,
    task: TaskSpec,
    workspace_root: Path,
    provider_name: str,
    writer: EpisodeWriter,
    model_gateway: ModelGateway,
    session_summary: str | None,
    session_compaction: dict[str, object] | None,
    historical_tool_compression_count: int,
    working_state: dict[str, object] | None,
    interaction_resolver: HumanInteractionResolver,
    task_ledger: dict[str, object] | None = None,
    tool_registry: ToolRuntimeRegistry | None = None,
    instruction_cache: object | None = None,
    skill_catalog: object | None = None,
) -> PreparedMessages:
    writer.write_environment(
        workspace_root,
        model_metadata=_gateway_metadata(model_gateway, provider_name),
        allowed_tools=task.allowed_tools,
        registry_tool_count=_registry_tool_count(tool_registry),
        entrypoint="run",
    )
    context = context_builder_cls(
        task=task,
        workspace_root=workspace_root,
        provider_name=provider_name,
        episode_writer=writer,
        session_summary=session_summary,
        session_compaction=session_compaction,
        historical_tool_compression_count=historical_tool_compression_count,
        working_state=working_state,
        task_ledger=task_ledger,
        interaction_state=interaction_resolver.state_records(),
        compaction_budget=context_budget_from_compression_budget(
            derive_compression_budget(_gateway_metadata(model_gateway, provider_name)),
        ),
        tool_registry=tool_registry,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()
    if context.manifest.full_compact_contract is not None:
        writer.append_transcript(
            {"event": "full_compact_contract", **context.manifest.full_compact_contract},
        )
    messages: list[dict[str, Any]] = _with_worker_context_messages(
        list(context.messages),
        task.worker_context,
    )
    if context.manifest.full_compact_contract is not None:
        if context.manifest.full_compact_contract.get("eligible") is True:
            writer.append_transcript(
                {
                    "event": "full_compact_start",
                    "context_id": context.context_id,
                    "reason": context.manifest.full_compact_contract.get("reason"),
                    "preserve_recent": context.manifest.full_compact_contract.get("required_preserve_recent", 6),
                    "pre_message_count": len(messages),
                },
            )
        full_compact_result = maybe_full_compact_messages(
            messages=messages,
            eligibility=full_compact_eligibility_from_manifest(context.manifest.full_compact_contract),
            gateway=model_gateway,
            preserve_recent=full_compact_preserve_recent(context.manifest.full_compact_contract),
        )
        if context.manifest.full_compact_contract.get("eligible") is True:
            writer.append_transcript(
                {
                    "event": "full_compact_success" if full_compact_result.applied else "full_compact_failed",
                    "context_id": context.context_id,
                    **full_compact_event_fields(full_compact_result),
                },
            )
        write_full_compact_manifest_result(writer, context.context_id, full_compact_result.manifest)
        messages = full_compact_result.messages
    source_diagnostics = getattr(context.manifest, "source_diagnostics", {})
    raw_cache = source_diagnostics.get("cache", {}) if isinstance(source_diagnostics, dict) else {}
    cache_diagnostics = raw_cache if isinstance(raw_cache, dict) else {}
    return PreparedMessages(
        context_id=context.context_id,
        messages=messages,
        cache_diagnostics=cache_diagnostics,
    )


def _gateway_metadata(model_gateway: ModelGateway, provider_name: str):
    metadata_getter = getattr(model_gateway, "metadata", None)
    if callable(metadata_getter):
        return metadata_getter()
    from haagent.models.types import ModelGatewayMetadata

    return ModelGatewayMetadata(
        provider=provider_name,
        model=None,
        endpoint=None,
        base_url=None,
        profile_name=None,
    )


def _registry_tool_count(tool_registry: ToolRuntimeRegistry | None) -> int:
    if tool_registry is None:
        return 0
    return len(tool_registry.static_tools) + len(tool_registry.dynamic_tools)


def full_compact_eligibility_from_manifest(contract: dict[str, Any]) -> FullCompactEligibility:
    return FullCompactEligibility(
        eligible=contract.get("eligible") is True,
        reason=str(contract.get("reason", "unknown")),
        trigger_kind=contract.get("trigger_kind") if isinstance(contract.get("trigger_kind"), str) else None,
        required_preserve_recent=full_compact_preserve_recent(contract),
    )


def _with_worker_context_messages(
    messages: list[dict[str, Any]],
    worker_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not worker_context:
        return messages
    system_prompt = worker_context.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return messages
    lines = [
        "Worker profile context:",
        f"- agent_id: {worker_context.get('agent_id', '')}",
        f"- agent_profile: {worker_context.get('agent_profile', '')}",
        f"- leader_session_id: {worker_context.get('leader_session_id', '')}",
        f"- team_id: {worker_context.get('team_id', '')}",
        f"- inbox_enabled: {bool(worker_context.get('inbox_enabled'))}",
        f"- system_prompt: {system_prompt}",
    ]
    return [{"role": "system", "content": "\n".join(lines)}, *messages]


def full_compact_preserve_recent(contract: dict[str, Any]) -> int:
    preserve_recent = contract.get("required_preserve_recent", 6)
    return preserve_recent if isinstance(preserve_recent, int) else 6


def full_compact_event_fields(result: FullCompactResult) -> dict[str, object]:
    return {
        "applied": result.applied,
        "reason": result.reason,
        "pre_message_count": result.pre_message_count,
        "post_message_count": result.post_message_count,
        "older_message_count": result.older_message_count,
        "preserved_recent_count": result.preserved_recent_count,
        "summary_chars": result.summary_chars,
    }


def write_full_compact_manifest_result(
    writer: EpisodeWriter,
    context_id: str,
    full_compact: dict[str, Any],
) -> None:
    manifest_path = writer.path / "contexts" / f"{context_id}-manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_compact"] = full_compact
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def workspace_root_candidate(raw_root: str | None, task_path: Path) -> Path:
    candidate = task_path.parent if raw_root is None else Path(raw_root)
    if raw_root is not None and not candidate.is_absolute():
        candidate = task_path.parent / candidate
    return candidate.resolve(strict=False)
