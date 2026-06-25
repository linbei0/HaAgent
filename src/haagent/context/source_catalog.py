"""
haagent/context/source_catalog.py - Context source catalog

集中生成 context manifest sources，并为每个 source 计算 raw/model 输入预算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from haagent.context.manifest import ContextSource, ContextSourceBudget
from haagent.context.observations import (
    observation_summary,
    observation_tool_name,
    raw_observation_summary,
)
from haagent.runtime.task_contract import TaskSpec
from haagent.tools.registry import TOOL_REGISTRY


AUDIT_SOURCE_EXCLUSION_REASON = "Audit evidence is stored in the episode and is not sent to the model by default."


@dataclass(frozen=True)
class ContextSourceCatalog:
    task: TaskSpec
    tool_workflow_hints: list[str]
    project_instructions: str | None
    project_instruction_lines: list[str]
    plan_lines: list[str]
    pending_next_step_lines: list[str]
    observations: list[dict[str, object]]
    session_summary: str | None
    session_summary_lines: list[str]
    working_state: object | None
    working_state_model_content: str
    working_state_raw_content: str
    episode_path: Path
    interaction_state: list[dict[str, object]] = field(default_factory=list)
    interaction_state_lines: list[str] = field(default_factory=list)

    def sources_with_budget(self) -> list[ContextSource]:
        return [self._source_with_budget(source) for source in self._sources()]

    def _sources(self) -> list[ContextSource]:
        sources = [
            self._project_instructions_source(),
            ContextSource("task", "goal", "Goal from task.yaml", "The model needs the task goal."),
            ContextSource("task", "constraints", "Constraints from task.yaml", "The model must obey task constraints."),
            ContextSource("task", "allowed_tools", "Allowed tools from task.yaml", "The model needs the allowed tool list."),
            ContextSource(
                "tool_workflow",
                "editing_workflow",
                "Concise tool sequencing guidance",
                "The model needs stable guidance for context discovery, editing, and verification.",
            ),
            ContextSource(
                "task",
                "acceptance_criteria",
                "Acceptance criteria from task.yaml",
                "The model needs completion criteria.",
            ),
            ContextSource(
                "task",
                "verification_commands",
                "Verification commands from task.yaml",
                "The model needs to know how the run will be verified.",
            ),
            ContextSource(
                "plan",
                "plan.json",
                "Agent plan trace for this episode",
                "The model needs the planned steps produced during planning.",
            ),
            ContextSource(
                "pending_next_step",
                "pending_next_step",
                "Deterministic next-step guidance from the latest tool result",
                "The model needs explicit continuation guidance after tool observations.",
            ),
        ]
        if self.session_summary is not None:
            sources.append(
                ContextSource(
                    "session_summary",
                    "session_summary",
                    "Bounded summary of previous chat turns",
                    "The model needs concise prior chat context without full episode traces.",
                ),
            )
        if self.working_state is not None:
            sources.append(
                ContextSource(
                    "working_state",
                    "working_state",
                    "Bounded short-term working state for the current chat session",
                    "The model needs concise current goal, findings, progress, and next steps without full history.",
                ),
            )
        if self.interaction_state:
            sources.append(
                ContextSource(
                    "interaction_state",
                    "human_interaction_state",
                    "Human interaction state recorded during this run",
                    "Structured human interaction facts are included in the current context.",
                ),
            )
        sources.extend(
            ContextSource(
                "tool_catalog",
                tool,
                TOOL_REGISTRY[tool].description,
                "Allowed tool description is included for model tool selection.",
            )
            for tool in self.task.allowed_tools
        )
        sources.extend(
            ContextSource(
                "observation",
                observation_tool_name(observation),
                "Tool observation from previous turn",
                "Previous tool result is needed for the next model turn.",
            )
            for observation in self.observations
        )
        sources.extend(self._audit_sources())
        return sources

    def _source_with_budget(self, source: ContextSource) -> ContextSource:
        raw_content = self._source_raw_content(source)
        model_input_content = self._source_content(source)
        exclusion_reason = _source_exclusion_reason(source)
        included = exclusion_reason is None
        model_input_char_count = len(model_input_content) if included else 0
        return ContextSource(
            source_type=source.source_type,
            name=source.name,
            description=source.description,
            inclusion_reason=source.inclusion_reason,
            status=source.status,
            budget=ContextSourceBudget(
                raw_char_count=len(raw_content),
                model_input_char_count=model_input_char_count,
                included_in_model_input=included,
                truncated=included and len(raw_content) > model_input_char_count,
                inclusion_reason=source.inclusion_reason,
                exclusion_reason=exclusion_reason,
            ),
        )

    def _source_content(self, source: ContextSource) -> str:
        if source.source_type == "project_instructions":
            return "\n".join(self.project_instruction_lines)
        if source.source_type == "session_summary":
            return "\n".join(["Session Summary:", *self.session_summary_lines])
        if source.source_type == "working_state":
            return self.working_state_model_content
        if source.source_type == "interaction_state":
            return "\n".join(
                [
                    "Human Interaction State:",
                    *self.interaction_state_lines,
                ],
            )
        if source.source_type == "task":
            return _task_source_content(source.name, self.task)
        if source.source_type == "tool_workflow":
            return "\n".join(["tool_workflow:", *_format_list(self.tool_workflow_hints)])
        if source.source_type == "tool_catalog":
            return f"- {source.name}: {TOOL_REGISTRY[source.name].description}"
        if source.source_type == "plan":
            return "\n".join(["Plan:", *self.plan_lines])
        if source.source_type == "pending_next_step":
            return "\n".join(["Pending next step:", *self.pending_next_step_lines])
        if source.source_type == "observation":
            for observation in self.observations:
                if observation_tool_name(observation) == source.name:
                    return (
                        f"- {observation_tool_name(observation)}: "
                        f"{json.dumps(observation_summary(observation), ensure_ascii=False, sort_keys=True)}"
                    )
        return ""

    def _source_raw_content(self, source: ContextSource) -> str:
        if source.source_type == "project_instructions":
            return self.project_instructions or ""
        if source.source_type == "session_summary":
            return self.session_summary or ""
        if source.source_type == "working_state":
            return self.working_state_raw_content
        if source.source_type == "interaction_state":
            return json.dumps(self.interaction_state, ensure_ascii=False, sort_keys=True)
        if source.source_type == "observation":
            for observation in self.observations:
                if observation_tool_name(observation) == source.name:
                    return (
                        f"- {observation_tool_name(observation)}: "
                        f"{json.dumps(raw_observation_summary(observation), ensure_ascii=False, sort_keys=True)}"
                    )
        if source.source_type.startswith("audit_"):
            return self._audit_source_raw_content(source.name)
        return self._source_content(source)

    def _project_instructions_source(self) -> ContextSource:
        if self.project_instructions is None:
            return ContextSource(
                "project_instructions",
                "AGENTS.md",
                "workspace AGENTS.md not found",
                "Absence is recorded so audits can see no project instructions were loaded.",
                status="absent",
            )
        return ContextSource(
            "project_instructions",
            "AGENTS.md",
            "Project instructions from workspace AGENTS.md",
            "Workspace AGENTS.md is the project instruction source for this run.",
            status="present",
        )

    def _audit_sources(self) -> list[ContextSource]:
        return [
            ContextSource(
                "audit_trace",
                "transcript.jsonl",
                "Episode transcript trace",
                "Audit trace is tracked for exclusion from model input.",
                status="excluded",
            ),
            ContextSource(
                "audit_tool_calls",
                "tool-calls.jsonl",
                "Full tool call trace, including policy records",
                "Tool and policy evidence is tracked for exclusion from model input.",
                status="excluded",
            ),
            ContextSource(
                "audit_verification",
                "verification/commands.jsonl",
                "Verification command evidence",
                "Verification evidence is tracked for exclusion from model input.",
                status="excluded",
            ),
            ContextSource(
                "audit_failure",
                "failure.json",
                "Failure attribution evidence",
                "Failure evidence is tracked for exclusion from model input.",
                status="excluded",
            ),
            ContextSource(
                "audit_eval_export",
                "eval export",
                "Eval export artifacts",
                "Eval export evidence is tracked for exclusion from model input.",
                status="excluded",
            ),
        ]

    def _audit_source_raw_content(self, name: str) -> str:
        paths = {
            "transcript.jsonl": self.episode_path / "transcript.jsonl",
            "tool-calls.jsonl": self.episode_path / "tool-calls.jsonl",
            "verification/commands.jsonl": self.episode_path / "verification" / "commands.jsonl",
            "failure.json": self.episode_path / "failure.json",
            "eval export": self.episode_path / "eval-case.json",
        }
        path = paths[name]
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


def _source_exclusion_reason(source: ContextSource) -> str | None:
    if source.source_type.startswith("audit_"):
        return AUDIT_SOURCE_EXCLUSION_REASON
    return None


def _task_source_content(name: str, task: TaskSpec) -> str:
    if name == "goal":
        return f"goal: {task.goal}"
    if name == "constraints":
        return "\n".join(["constraints:", *_format_list(task.constraints)])
    if name == "allowed_tools":
        return "\n".join(
            [
                "allowed_tools:",
                *[f"- {tool}: {TOOL_REGISTRY[tool].description}" for tool in task.allowed_tools],
            ],
        )
    if name == "acceptance_criteria":
        return "\n".join(["acceptance_criteria:", *_format_list(task.acceptance_criteria)])
    if name == "verification_commands":
        return "\n".join(["verification_commands:", *_format_list(task.verification_commands)])
    return ""


def _format_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]
