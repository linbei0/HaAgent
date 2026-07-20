"""
src/haagent/runtime/session/turn_completion.py - 单轮结束后的结果与进度处理

从 episode / runtime bus 构建 ChatTurnResult，合成 task 进度事件，
并处理 in-band verification 与压缩诊断计数。不负责模型调用本身。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from haagent.runtime.episodes.package_types import VerificationCommandRecord
from haagent.runtime.episodes.validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.session.turn import summary_value


@dataclass(frozen=True)
class ChatTurnResult:
    session_id: str
    turn_index: int
    status: str
    episode_path: Path
    provider: str
    final_response: str
    verification_status: str
    failed_stage: str = "none"
    failure_category: str = "none"
    reason: str = "none"
    summary_error: str | None = None
    memory_candidates_created: int = 0
    memory_extraction_status: str = "skipped"
    memory_extraction_reason: str = ""

def build_turn_result(
    *,
    session_id: str,
    turn_index: int,
    provider_name: str,
    result: Any,
) -> ChatTurnResult:
    try:
        package = load_inspect_episode_package(result.episode_path)
    except EpisodeValidationError as error:
        return ChatTurnResult(
            session_id=session_id,
            turn_index=turn_index,
            status=result.status.value,
            episode_path=result.episode_path,
            provider=provider_name,
            final_response="none",
            verification_status="not_run",
            summary_error=str(error),
        )

    failure = package.failure.failure
    return ChatTurnResult(
        session_id=session_id,
        turn_index=turn_index,
        status=result.status.value,
        episode_path=result.episode_path,
        provider=str(package.metadata.provider or provider_name),
        final_response=package.final_response_text(),
        verification_status=verification_status(
            package.verification_commands,
            package.verification_reached,
        ),
        failed_stage="none" if failure is None else failure.stage,
        failure_category="none" if failure is None else failure.category,
        reason="none" if failure is None else failure.evidence,
    )


def turn_summary(prompt: str, result: ChatTurnResult) -> str:
    return "\n".join(
        [
            f"- user_request: {summary_value(prompt, 160)}",
            f"  status: {result.status}",
            f"  episode_id: {result.episode_path.name}",
            f"  assistant_final_response: {summary_value(result.final_response, 220)}",
            f"  verification: {result.verification_status}",
        ],
    )


def run_final_response(transcript: list[dict[str, Any]]) -> str:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return str(record.get("content", ""))
    return "none"


def verification_status(
    commands: list[VerificationCommandRecord],
    verification_reached: bool,
) -> str:
    if not verification_reached or not commands:
        return "not_run"
    for command in commands:
        if command.status != "success":
            return "failed"
    return "success"


def with_in_band_verification(
    result: ChatTurnResult,
    runtime_events: list[object],
) -> ChatTurnResult:
    if result.verification_status != "not_run":
        return result
    status = in_band_verification_status(runtime_events)
    if status == "not_run":
        return result
    return replace(result, verification_status=status)


def task_step_started_event(ledger) -> dict[str, object] | None:
    step = ledger.active_step()
    if step is None:
        return None
    return {
        "event_type": "task_step_started",
        "step_id": step.id,
        "title": step.title,
        "owner": step.owner,
        "status": step.status,
        "summary": f"started task step {step.id}: {summary_value(step.title, 120)}",
        "evidence_count": len(step.evidence_refs),
        "checkpoint_count": len(ledger.checkpoints),
    }


def task_turn_closed_events(ledger, result: ChatTurnResult) -> list[dict[str, object]]:
    step = ledger.active_step()
    if step is None:
        return []
    events: list[dict[str, object]] = []
    if result.verification_status in {"success", "failed"}:
        events.append(
            {
                "event_type": "task_checkpoint_saved",
                "step_id": step.id,
                "title": step.title,
                "owner": step.owner,
                "status": result.verification_status,
                "summary": f"verification {result.verification_status}",
                "evidence_count": len(step.evidence_refs),
                "checkpoint_count": len(ledger.checkpoints),
            },
        )
    if ledger.status == "completed" and step.status == "completed":
        events.append(
            {
                "event_type": "task_step_finished",
                "step_id": step.id,
                "title": step.title,
                "owner": step.owner,
                "status": "completed",
                "summary": f"completed task step {step.id}: {summary_value(step.title, 120)}",
                "evidence_count": len(step.evidence_refs),
                "checkpoint_count": len(ledger.checkpoints),
            },
        )
    elif ledger.status in {"blocked", "failed", "cancelled"}:
        events.append(
            {
                "event_type": "task_step_blocked",
                "step_id": step.id,
                "title": step.title,
                "owner": step.owner,
                "status": ledger.status,
                "category": ledger.status,
                "summary": f"task step {step.id} ended as {ledger.status}",
                "suggested_action": "resume_or_replan",
                "evidence_count": len(step.evidence_refs),
                "checkpoint_count": len(ledger.checkpoints),
            },
        )
    return events


def in_band_verification_status(runtime_events: list[object]) -> str:
    from haagent.runtime.events.bus import ToolFailedBusEvent, ToolFinishedBusEvent, bus_event_to_dict, coerce_bus_event

    saw_failed_verification = False
    for raw_event in runtime_events:
        event = coerce_bus_event(raw_event)
        if isinstance(event, (ToolFinishedBusEvent, ToolFailedBusEvent)):
            tool_name = event.tool_name
            args = event.args
            result = event.result if isinstance(event, ToolFinishedBusEvent) else {}
            event_type = event.event_type
        else:
            payload = bus_event_to_dict(event)
            event_type = str(payload.get("event_type", ""))
            if event_type not in {"tool_finished", "tool_failed"}:
                continue
            tool_name = str(payload.get("tool_name", ""))
            args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if tool_name not in {"shell", "code_run"}:
            continue
        if not looks_like_verification_command(args):
            continue
        if event_type == "tool_finished" and result.get("status") == "success" and result.get("exit_code") == 0:
            return "success"
        saw_failed_verification = True
    return "failed" if saw_failed_verification else "not_run"


def looks_like_verification_command(args: dict[str, object]) -> bool:
    text_parts = [
        str(args.get("command", "")),
        str(args.get("code", "")),
    ]
    command_text = " ".join(text_parts).lower()
    markers = (
        "pytest",
        "haagent check",
        "ruff",
        "mypy",
        "npm test",
        "pnpm test",
        "yarn test",
        "vitest",
        "tox",
        "cargo test",
        "go test",
    )
    return any(marker in command_text for marker in markers)


def memory_update_requested(runtime_events: list[object]) -> bool:
    from haagent.runtime.events.bus import ToolFinishedBusEvent, bus_event_to_dict, coerce_bus_event

    for raw_event in runtime_events:
        event = coerce_bus_event(raw_event)
        if isinstance(event, ToolFinishedBusEvent):
            if event.tool_name != "start_memory_update":
                continue
            result = event.result
        else:
            payload = bus_event_to_dict(event)
            if payload.get("event_type") != "tool_finished" or payload.get("tool_name") != "start_memory_update":
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if result.get("status") == "success" and result.get("memory_update_requested") is True:
            return True
    return False


def count_historical_tool_compression_events(runtime_events: list[object]) -> int:
    from haagent.runtime.events.bus import bus_event_to_dict, coerce_bus_event

    total = 0
    for raw_event in runtime_events:
        payload = bus_event_to_dict(coerce_bus_event(raw_event))
        if (
            payload.get("event_type") == "compression_diagnostic" or payload.get("event") == "compression_diagnostic"
        ) and payload.get("stage") == "historical_tool_message":
            total += 1
    return total
