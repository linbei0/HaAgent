"""
src/haagent/runtime/session/planning_state.py - Plan Mode 结构化状态机

保存最新 Plan revision，并提供进入、修改、批准、执行与取消的严格转换。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4


PLANNING_STATE_SCHEMA_VERSION = 1
PLANNING_STATUSES = {
    "inactive",
    "planning",
    "awaiting_confirmation",
    "approved_pending_execution",
    "execution_started",
    "cancelled",
}
PLAN_MODE_STATUSES = {"planning", "awaiting_confirmation"}


class PlanningStateError(RuntimeError):
    """planning-state 文件损坏或状态转换不合法时抛出。"""


@dataclass(frozen=True)
class PlanVerification:
    required: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return {"required": self.required, "description": self.description}


@dataclass(frozen=True)
class PlanStep:
    id: str
    content: str
    completion_condition: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "completion_condition": self.completion_condition,
        }


@dataclass(frozen=True)
class PlanProposal:
    goal: str
    summary: str
    steps: list[PlanStep]
    verification: PlanVerification
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "verification": self.verification.to_dict(),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class PlanningState:
    status: str
    plan_id: str | None = None
    revision: int = 0
    proposal: PlanProposal | None = None
    approved_revision: int | None = None
    execution_id: str | None = None
    execution_started_revision: int | None = None
    updated_turn: int = 0
    schema_version: int = PLANNING_STATE_SCHEMA_VERSION

    @property
    def is_plan_mode(self) -> bool:
        return self.status in PLAN_MODE_STATUSES

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "plan_id": self.plan_id,
            "revision": self.revision,
            "proposal": self.proposal.to_dict() if self.proposal is not None else None,
            "approved_revision": self.approved_revision,
            "execution_id": self.execution_id,
            "execution_started_revision": self.execution_started_revision,
            "updated_turn": self.updated_turn,
        }


def empty_planning_state() -> PlanningState:
    return PlanningState(status="inactive")


def enter_plan_mode(*, updated_turn: int) -> PlanningState:
    return PlanningState(
        status="planning",
        plan_id=f"plan-{uuid4().hex}",
        updated_turn=updated_turn,
    )


def submit_plan_revision(state: PlanningState, raw: object, *, updated_turn: int) -> PlanningState:
    if state.status != "planning" or state.plan_id is None:
        raise PlanningStateError("plan revision can only be submitted while planning")
    revision = state.revision + 1
    proposal = proposal_from_tool_args(raw, revision=revision)
    return replace(
        state,
        status="awaiting_confirmation",
        revision=revision,
        proposal=proposal,
        approved_revision=None,
        execution_id=None,
        execution_started_revision=None,
        updated_turn=updated_turn,
    )


def request_plan_revision(
    state: PlanningState,
    *,
    plan_id: str,
    revision: int,
    updated_turn: int,
) -> PlanningState:
    _require_latest(state, plan_id=plan_id, revision=revision)
    return replace(state, status="planning", updated_turn=updated_turn)


def approve_plan(
    state: PlanningState,
    *,
    plan_id: str,
    revision: int,
    updated_turn: int,
) -> PlanningState:
    _require_latest(state, plan_id=plan_id, revision=revision)
    return replace(
        state,
        status="approved_pending_execution",
        approved_revision=revision,
        execution_id=f"execution-{uuid4().hex}",
        updated_turn=updated_turn,
    )


def mark_plan_execution_started(
    state: PlanningState,
    *,
    execution_id: str,
    updated_turn: int,
) -> PlanningState:
    if state.status != "approved_pending_execution" or state.execution_id != execution_id:
        raise PlanningStateError("plan execution does not match the approved revision")
    return replace(
        state,
        status="execution_started",
        execution_started_revision=state.approved_revision,
        updated_turn=updated_turn,
    )


def cancel_plan(state: PlanningState, *, updated_turn: int) -> PlanningState:
    if state.status in {"inactive", "execution_started", "cancelled"}:
        raise PlanningStateError("there is no cancellable plan")
    return replace(state, status="cancelled", updated_turn=updated_turn)


def load_planning_state(path: Path) -> PlanningState:
    if not path.exists():
        raise PlanningStateError(f"session package missing required file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningStateError(f"cannot load planning state: {path}") from error
    return planning_state_from_dict(raw)


def write_planning_state(path: Path, state: PlanningState) -> None:
    validated = planning_state_from_dict(state.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def planning_state_from_dict(raw: object) -> PlanningState:
    if not isinstance(raw, dict):
        raise PlanningStateError("planning state must be an object")
    required = {
        "schema_version",
        "status",
        "plan_id",
        "revision",
        "proposal",
        "approved_revision",
        "execution_id",
        "execution_started_revision",
        "updated_turn",
    }
    if set(raw) != required or raw.get("schema_version") != PLANNING_STATE_SCHEMA_VERSION:
        raise PlanningStateError("unsupported planning state schema")
    status = _required_string(raw, "status")
    if status not in PLANNING_STATUSES:
        raise PlanningStateError("planning status is invalid")
    proposal_raw = raw.get("proposal")
    proposal = proposal_from_dict(proposal_raw) if proposal_raw is not None else None
    state = PlanningState(
        status=status,
        plan_id=_optional_string(raw.get("plan_id"), "plan_id"),
        revision=_required_int(raw, "revision"),
        proposal=proposal,
        approved_revision=_optional_int(raw.get("approved_revision"), "approved_revision"),
        execution_id=_optional_string(raw.get("execution_id"), "execution_id"),
        execution_started_revision=_optional_int(raw.get("execution_started_revision"), "execution_started_revision"),
        updated_turn=_required_int(raw, "updated_turn"),
    )
    _validate_state_consistency(state)
    return state


def proposal_from_tool_args(raw: object, *, revision: int) -> PlanProposal:
    if not isinstance(raw, dict):
        raise PlanningStateError("plan proposal must be an object")
    required = {"goal", "summary", "steps", "verification", "assumptions"}
    if set(raw) != required:
        raise PlanningStateError("plan proposal fields are invalid")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PlanningStateError("plan must contain at least one step")
    verification_raw = raw.get("verification")
    if not isinstance(verification_raw, dict):
        raise PlanningStateError("verification must be an object")
    verification_required = verification_raw.get("required")
    verification_description = verification_raw.get("description")
    if not isinstance(verification_required, bool) or not isinstance(verification_description, str):
        raise PlanningStateError("verification fields are invalid")
    _validate_text(verification_description, "verification.description", 480)
    if len(steps_raw) + int(verification_required) > 20:
        raise PlanningStateError("plan steps plus verification cannot exceed 20 Todo items")
    steps: list[PlanStep] = []
    for index, item in enumerate(steps_raw):
        if not isinstance(item, dict) or set(item) != {"content", "completion_condition"}:
            raise PlanningStateError("plan step fields are invalid")
        content = item.get("content")
        condition = item.get("completion_condition")
        if not isinstance(content, str) or not isinstance(condition, str):
            raise PlanningStateError("plan step content must be strings")
        _validate_text(content, "step.content", 240)
        _validate_text(condition, "step.completion_condition", 240)
        steps.append(
            PlanStep(
                id=f"step-{revision:03d}-{index + 1:03d}",
                content=content.strip(),
                completion_condition=condition.strip(),
            ),
        )
    assumptions = raw.get("assumptions")
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        raise PlanningStateError("assumptions must be a list of strings")
    if len(assumptions) > 10:
        raise PlanningStateError("assumptions cannot exceed 10 items")
    for assumption in assumptions:
        _validate_text(assumption, "assumption", 240)
    goal = _required_string(raw, "goal")
    summary = _required_string(raw, "summary")
    _validate_text(goal, "goal", 240)
    _validate_text(summary, "summary", 1200)
    return PlanProposal(
        goal=goal.strip(),
        summary=summary.strip(),
        steps=steps,
        verification=PlanVerification(required=verification_required, description=verification_description.strip()),
        assumptions=[item.strip() for item in assumptions],
    )


def proposal_from_dict(raw: object) -> PlanProposal:
    if not isinstance(raw, dict):
        raise PlanningStateError("proposal must be an object")
    if set(raw) != {"goal", "summary", "steps", "verification", "assumptions"}:
        raise PlanningStateError("proposal fields are invalid")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PlanningStateError("proposal steps are invalid")
    steps: list[PlanStep] = []
    for item in steps_raw:
        if not isinstance(item, dict) or set(item) != {"id", "content", "completion_condition"}:
            raise PlanningStateError("proposal step fields are invalid")
        step = PlanStep(
            id=_required_string(item, "id"),
            content=_required_string(item, "content"),
            completion_condition=_required_string(item, "completion_condition"),
        )
        _validate_text(step.id, "step.id", 80)
        _validate_text(step.content, "step.content", 240)
        _validate_text(step.completion_condition, "step.completion_condition", 240)
        steps.append(step)
    verification_raw = raw.get("verification")
    if not isinstance(verification_raw, dict) or set(verification_raw) != {"required", "description"}:
        raise PlanningStateError("verification fields are invalid")
    required = verification_raw.get("required")
    description = verification_raw.get("description")
    if not isinstance(required, bool) or not isinstance(description, str):
        raise PlanningStateError("verification fields are invalid")
    assumptions = raw.get("assumptions")
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        raise PlanningStateError("assumptions are invalid")
    proposal = PlanProposal(
        goal=_required_string(raw, "goal"),
        summary=_required_string(raw, "summary"),
        steps=steps,
        verification=PlanVerification(required=required, description=description),
        assumptions=list(assumptions),
    )
    # 复用 tool 校验边界，但保留磁盘中 runtime step id。
    _validate_text(proposal.goal, "goal", 240)
    _validate_text(proposal.summary, "summary", 1200)
    _validate_text(proposal.verification.description, "verification.description", 480)
    if len(proposal.steps) + int(required) > 20 or len(proposal.assumptions) > 10:
        raise PlanningStateError("proposal exceeds item limits")
    return proposal


def format_planning_state_for_model(state: PlanningState) -> str:
    lines = [
        "Plan Mode：先只读调查，再按需澄清，最后必须调用 submit_plan 提交完整结构化方案。",
        "禁止写文件、执行命令、启动后台任务、worker、Todo、memory update 和动态 MCP。",
    ]
    if state.proposal is not None:
        lines.extend(
            [
                f"latest_plan_revision: {state.revision}",
                f"goal: {state.proposal.goal}",
                "steps:",
                *[f"- {step.id}: {step.content}；完成条件：{step.completion_condition}" for step in state.proposal.steps],
            ],
        )
    return "\n".join(lines)


def _require_latest(state: PlanningState, *, plan_id: str, revision: int) -> None:
    if (
        state.status != "awaiting_confirmation"
        or state.plan_id != plan_id
        or state.revision != revision
        or state.proposal is None
    ):
        raise PlanningStateError("plan id or revision is stale")


def _validate_state_consistency(state: PlanningState) -> None:
    if state.status == "inactive":
        if any((state.plan_id, state.proposal, state.approved_revision, state.execution_id, state.execution_started_revision)) or state.revision:
            raise PlanningStateError("inactive planning state contains active data")
        return
    if state.plan_id is None:
        raise PlanningStateError("active planning state requires plan_id")
    if state.status in {"awaiting_confirmation", "approved_pending_execution", "execution_started"} and state.proposal is None:
        raise PlanningStateError("planning state requires proposal")
    if state.status in {"approved_pending_execution", "execution_started"}:
        if state.approved_revision != state.revision or state.execution_id is None:
            raise PlanningStateError("approved planning state is incomplete")
    if state.status == "execution_started" and state.execution_started_revision != state.approved_revision:
        raise PlanningStateError("execution_started revision mismatch")


def _validate_text(value: str, field_name: str, limit: int) -> None:
    if not value.strip():
        raise PlanningStateError(f"{field_name} must be non-empty")
    if len(value.strip()) > limit:
        raise PlanningStateError(f"{field_name} cannot exceed {limit} characters")


def _required_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise PlanningStateError(f"{key} must be a string")
    return value


def _optional_string(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PlanningStateError(f"{key} must be a string or null")
    return value


def _required_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlanningStateError(f"{key} must be a non-negative integer")
    return value


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlanningStateError(f"{key} must be a non-negative integer or null")
    return value
