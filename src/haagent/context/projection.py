"""
src/haagent/context/projection.py - 动态模型上下文投影

集中把 session 事实确定性地投影为版本化上下文 sections，供首次构建和运行中同步复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from haagent.context.versioned_state import content_digest
from haagent.runtime.session.planning_state import (
    PlanningState,
    PlanningStateError,
    format_planning_state_for_model,
    planning_state_from_dict,
)
from haagent.runtime.session.task_ledger import TaskLedgerError, format_task_ledger_for_model
from haagent.runtime.session.working_state import WorkingStateError, format_working_state_for_model


CONTEXT_STATE_SECTION_TITLES = {
    "working_state": "Working State",
    "task_ledger": "Task Ledger",
    "planning_state": "Planning State",
    "memory_index": "Memory/SOP Navigation Index",
    "memory": "Relevant Memory",
    "interaction_history": "Interaction History",
}
SESSION_FACT_SECTION_KEYS = frozenset(
    {"session_summary", "working_state", "task_ledger", "planning_state", "interaction_history"},
)


class ModelContextProjectionError(ValueError):
    """动态 session 事实无法确定性投影时抛出。"""


@dataclass(frozen=True)
class ModelContextFacts:
    session_summary: str | None = None
    working_state: object | None = None
    task_ledger: object | None = None
    planning_state: object | None = None
    interaction_records: tuple[Mapping[str, object], ...] = ()


class ContextFactsSource(Protocol):
    def __call__(self) -> ModelContextFacts: ...


class MutableContextFactsSource:
    """Session lifecycle 先创建、AgentSession 再按领域状态刷新的一处事实 seam。"""

    def __init__(self, facts: ModelContextFacts | None = None) -> None:
        self._facts = facts or ModelContextFacts()

    def update(self, facts: ModelContextFacts) -> None:
        self._facts = facts

    def __call__(self) -> ModelContextFacts:
        return self._facts


@dataclass(frozen=True)
class ProjectedModelContext:
    sections: Mapping[str, object]
    source_digests: Mapping[str, str]
    requires_rebuild: bool = False

    @classmethod
    def create(
        cls,
        sections: Mapping[str, object],
        *,
        requires_rebuild: bool = False,
    ) -> "ProjectedModelContext":
        normalized = {str(key): value for key, value in sorted(sections.items())}
        return cls(
            sections=normalized,
            source_digests={key: content_digest(value) for key, value in normalized.items()},
            requires_rebuild=requires_rebuild,
        )


def project_model_context(facts: ModelContextFacts) -> ProjectedModelContext:
    """把原始 session 事实转成稳定、可散列的模型投影。"""

    sections: dict[str, object] = {}
    try:
        session_summary = (facts.session_summary or "").strip()
        if session_summary:
            sections["session_summary"] = session_summary
        if facts.working_state is not None:
            content = format_working_state_for_model(facts.working_state)
            if content:
                sections["working_state"] = format_context_state_section("working_state", content)
        if facts.task_ledger is not None:
            content = format_task_ledger_for_model(facts.task_ledger)
            if content:
                sections["task_ledger"] = format_context_state_section("task_ledger", content)
        if facts.planning_state is not None:
            planning = (
                facts.planning_state
                if isinstance(facts.planning_state, PlanningState)
                else planning_state_from_dict(facts.planning_state)
            )
            if planning.status not in {"inactive", "cancelled"}:
                sections["planning_state"] = format_context_state_section(
                    "planning_state",
                    format_planning_state_for_model(planning),
                )
        if facts.interaction_records:
            sections["interaction_history"] = format_interaction_state_for_model(
                facts.interaction_records,
            )
    except (WorkingStateError, TaskLedgerError, PlanningStateError, TypeError, ValueError) as error:
        raise ModelContextProjectionError(f"invalid model context facts: {error}") from error
    return ProjectedModelContext.create(sections)


def merge_session_projection(
    base_sections: Mapping[str, object],
    projection: ProjectedModelContext,
) -> ProjectedModelContext:
    """只替换 projection 拥有的 session sections，保留 memory 等首次选择结果。"""

    merged = {key: value for key, value in base_sections.items() if key not in SESSION_FACT_SECTION_KEYS}
    merged.update(projection.sections)
    return ProjectedModelContext.create(merged)


def format_context_state_section(key: str, content: object) -> object:
    if not isinstance(content, str):
        return content
    title = CONTEXT_STATE_SECTION_TITLES.get(key)
    if title is None or content.startswith(f"{title}:"):
        return content
    return f"{title}:\n{content}"


def format_interaction_state_for_model(records: Sequence[Mapping[str, object]]) -> str:
    lines = [f"- {_interaction_state_summary(record)}" for record in records[-8:]]
    return str(format_context_state_section("interaction_history", "\n".join(lines)))


def _interaction_state_summary(record: Mapping[str, object]) -> str:
    parts = [
        f"type={_safe_state_value(record.get('type'), 'interaction')}",
        f"tool={_safe_state_value(record.get('tool'), 'unknown')}",
        f"status={_safe_state_value(record.get('status'), 'unknown')}",
    ]
    question = str(record.get("question") or "")
    if question:
        parts.append(f"question={json.dumps(question, ensure_ascii=False)}")
    if record.get("type") == "user_input":
        for key in ("outcome", "question_count", "answered_count", "answer_chars"):
            if key in record:
                parts.append(f"{key}={_safe_state_value(record[key], 'unknown')}")
    if "approved" in record:
        parts.append(f"approved={str(bool(record['approved'])).lower()}")
    if "turn" in record:
        parts.append(f"turn={record['turn']}")
    return " ".join(parts)


def _safe_state_value(value: object, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback
