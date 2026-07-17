"""
src/haagent/context/source_definitions.py - 内部 ContextSourceDefinition 注册表

固定、确定性、有序的 source 元数据；选择/压缩/标题同源。
不是用户插件，不引入 DI 或 embedding。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ContextSourceDefinition:
    id: str
    placement: Literal["system", "task"]
    title: str
    reason: str
    selection_priority: int
    compaction_priority: int
    hard_required: bool = False
    include_empty_on_skip: bool = False
    compaction_source: str | None = None
    compaction_kind: str | None = None
    # ContextCandidateInputs 字段名 → 内容
    content_field: str = ""
    skip_reason_field: str | None = None
    metadata_field: str | None = None

    @property
    def source_type(self) -> str:
        return self.id

    def resolved_compaction_source(self) -> str:
        return self.compaction_source or self.id

    def resolved_compaction_kind(self) -> str:
        return self.compaction_kind or self.id


# 固定顺序：collect 按此遍历，保证可复现。
CONTEXT_SOURCE_DEFINITIONS: tuple[ContextSourceDefinition, ...] = (
    ContextSourceDefinition(
        id="soul",
        placement="system",
        title="Agent Soul",
        reason="deterministic_soul_context",
        selection_priority=25,
        compaction_priority=68,
        include_empty_on_skip=True,
        content_field="soul",
        skip_reason_field="soul_skip_reason",
        metadata_field="soul_metadata",
    ),
    ContextSourceDefinition(
        id="project_instructions",
        placement="system",
        title="Project Instructions",
        reason="workspace_agents_md_found",
        selection_priority=10,
        compaction_priority=80,
        compaction_source="project",
        content_field="project_instructions",
    ),
    ContextSourceDefinition(
        id="prompt_pack",
        placement="system",
        title="Prompt Packs",
        reason="explicit_prompt_command",
        selection_priority=15,
        compaction_priority=85,
        hard_required=True,
        content_field="prompt_packs",
        metadata_field="prompt_pack_metadata",
    ),
    ContextSourceDefinition(
        id="session_summary",
        placement="system",
        title="Session Summary",
        reason="resumed_session",
        selection_priority=30,
        compaction_priority=70,
        compaction_source="session",
        content_field="session_summary",
    ),
    ContextSourceDefinition(
        id="working_state",
        placement="task",
        title="Working State",
        reason="working_state_present",
        selection_priority=20,
        compaction_priority=75,
        content_field="working_state",
    ),
    ContextSourceDefinition(
        id="task_ledger",
        placement="task",
        title="Task Ledger",
        reason="task_ledger_present",
        selection_priority=18,
        compaction_priority=78,
        content_field="task_ledger",
    ),
    ContextSourceDefinition(
        id="memory_index",
        placement="task",
        title="Memory/SOP Navigation Index",
        reason="confirmed_memory_index_available",
        selection_priority=45,
        compaction_priority=65,
        include_empty_on_skip=True,
        compaction_source="memory",
        compaction_kind="memory_index",
        content_field="memory_index",
        skip_reason_field="memory_index_skip_reason",
        metadata_field="memory_index_metadata",
    ),
    ContextSourceDefinition(
        id="memory",
        placement="task",
        title="Relevant Memory",
        reason="memory_retrieval_match",
        selection_priority=50,
        compaction_priority=60,
        content_field="memory_block",
    ),
    ContextSourceDefinition(
        id="interaction_history",
        placement="task",
        title="Interaction History",
        reason="recent_interaction_state",
        selection_priority=40,
        compaction_priority=55,
        compaction_source="interaction_state",
        compaction_kind="interaction",
        content_field="interaction_state",
    ),
    ContextSourceDefinition(
        id="skills",
        placement="system",
        title="Available Skills",
        reason="allowed_skill_tools_present",
        selection_priority=60,
        compaction_priority=50,
        content_field="skills_block",
    ),
)


_DEFINITIONS_BY_ID = {item.id: item for item in CONTEXT_SOURCE_DEFINITIONS}


def get_source_definition(source_id: str) -> ContextSourceDefinition | None:
    return _DEFINITIONS_BY_ID.get(source_id)
