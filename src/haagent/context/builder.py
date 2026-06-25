"""
haagent/context/builder.py - Context Builder v1

生成可审计的标准化模型输入与上下文 manifest。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from haagent.context.manifest import (
    ContextBudget,
    ContextIndex,
    ContextManifest,
    ContextSource,
)
from haagent.context.observations import observation_summary, observation_tool_name
from haagent.context.source_catalog import ContextSourceCatalog
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.task_contract import TaskSpec
from haagent.runtime.working_state import (
    WorkingStateError,
    format_working_state_for_model,
    raw_working_state_text,
)
from haagent.tools.registry import TOOL_REGISTRY


CONTEXT_MANIFEST_VERSION = "1.2"
CONTEXT_CHARACTER_LIMIT = 12000
PROJECT_INSTRUCTIONS_CHAR_LIMIT = 2000
SESSION_SUMMARY_CHAR_LIMIT = 1000
TOOL_WORKFLOW_HINTS = [
    "Prefer context_find before file_search when the user describes functionality without paths.",
    "After context_find, use file_read on the most relevant candidate before editing.",
    "Use apply_patch_set for related edits across multiple files or multiple replacements.",
    "Use apply_patch only for a single isolated replacement.",
    'Use workspace-relative paths in tool arguments; use cwd=\'.\' or omit cwd for the workspace root.',
    "After file changes, read changed files or run verification before claiming completion.",
]


class ContextBuildError(RuntimeError):
    """Raised when context cannot be built from the task contract."""


@dataclass(frozen=True)
class BuiltContext:
    context_id: str
    model_input: str
    manifest: ContextManifest


class ContextBuilder:
    def __init__(
        self,
        task: TaskSpec,
        workspace_root: Path,
        provider_name: str,
        episode_writer: EpisodeWriter,
        observations: list[dict[str, object]] | None = None,
        final_response_requested: bool = False,
        session_summary: str | None = None,
        working_state: dict[str, object] | None = None,
        interaction_state: list[dict[str, object]] | None = None,
    ) -> None:
        self._task = task
        self._workspace_root = workspace_root
        self._provider_name = provider_name
        self._episode_writer = episode_writer
        self._observations = list(observations or [])
        self._final_response_requested = final_response_requested
        self._session_summary = session_summary
        self._working_state = working_state
        self._interaction_state = list(interaction_state or [])
        self._project_instructions: str | None = None
        self._plan: dict[str, object] | None = None

    def build(self) -> BuiltContext:
        """构建第一版上下文：不检索文件，只写任务事实和工具目录。"""
        self._validate_tools()
        self._project_instructions = self._read_project_instructions()
        self._plan = self._read_plan()
        context_id = self._next_context_id()
        contexts_dir = self._episode_writer.path / "contexts"
        contexts_dir.mkdir(parents=True, exist_ok=True)

        model_input = self._render_model_input()
        budget = _context_budget(model_input)
        if budget.status != "within_limit":
            raise ContextBuildError(
                "context character budget exceeded: "
                f"{budget.character_count} > {budget.character_limit}",
            )
        model_input_path = contexts_dir / f"{context_id}.txt"
        manifest_path = contexts_dir / f"{context_id}.json"
        model_input_path.write_text(model_input, encoding="utf-8")

        manifest = self._context_manifest(context_id, budget)
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_run_manifest(
            ContextIndex(
                context_id=context_id,
                model_input_path=f"contexts/{context_id}.txt",
                manifest_path=f"contexts/{context_id}.json",
                budget=_context_budget_summary(context_id, budget, manifest.sources),
            ),
        )
        return BuiltContext(context_id=context_id, model_input=model_input, manifest=manifest)

    def _validate_tools(self) -> None:
        unknown_tools = [tool for tool in self._task.allowed_tools if tool not in TOOL_REGISTRY]
        if unknown_tools:
            raise ContextBuildError(f"unknown allowed_tools: {', '.join(unknown_tools)}")

    def _next_context_id(self) -> str:
        contexts_dir = self._episode_writer.path / "contexts"
        if not contexts_dir.exists():
            return "0001"
        existing = sorted(path.stem for path in contexts_dir.glob("*.txt") if path.stem.isdigit())
        if not existing:
            return "0001"
        return f"{int(existing[-1]) + 1:04d}"

    def _render_model_input(self) -> str:
        return "\n".join(
            [
                "HaAgent Context v1",
                "",
                "Instructions:",
                "- Use only the task facts and allowed tools listed below.",
                "- Report failures explicitly; do not invent successful outcomes.",
                "",
                "Project Instructions:",
                *self._format_project_instructions(),
                *self._format_session_summary_block(),
                *self._format_working_state_block(),
                *self._format_interaction_state_block(),
                "",
                "Facts:",
                f"goal: {self._task.goal}",
                "constraints:",
                *_format_list(self._task.constraints),
                "allowed_tools:",
                *[
                    f"- {tool}: {TOOL_REGISTRY[tool].description}"
                    for tool in self._task.allowed_tools
                ],
                "tool_workflow:",
                *_format_list(self._tool_workflow_hints()),
                "acceptance_criteria:",
                *_format_list(self._task.acceptance_criteria),
                "verification_commands:",
                *_format_list(self._task.verification_commands),
                "",
                "Plan:",
                *self._format_plan(),
                "",
                "Observations:",
                *self._format_observations(),
                "",
                "Pending next step:",
                *self._format_pending_next_step(),
                "",
            ],
        )

    def _context_manifest(self, context_id: str, budget: ContextBudget) -> ContextManifest:
        sources = self._source_catalog().sources_with_budget()
        return ContextManifest(
            context_id=context_id,
            provider=self._provider_name,
            workspace_root=str(self._workspace_root),
            generated_at=_now_iso(),
            budget=budget,
            sources=sources,
            next_action=self._next_action(),
        )

    def _source_catalog(self) -> ContextSourceCatalog:
        working_state_model_content = ""
        working_state_raw_content = ""
        if self._working_state is not None:
            working_state_model_content = self._working_state_model_content()
            working_state_raw_content = raw_working_state_text(self._working_state)
        return ContextSourceCatalog(
            task=self._task,
            tool_workflow_hints=self._tool_workflow_hints(),
            project_instructions=self._project_instructions,
            project_instruction_lines=self._format_project_instructions(),
            plan_lines=self._format_plan(),
            pending_next_step_lines=self._format_pending_next_step(),
            observations=self._observations,
            session_summary=self._session_summary,
            session_summary_lines=self._format_session_summary(),
            working_state=self._working_state,
            working_state_model_content=working_state_model_content,
            working_state_raw_content=working_state_raw_content,
            interaction_state=self._interaction_state,
            interaction_state_lines=self._format_interaction_state(),
            episode_path=self._episode_writer.path,
        )

    def _format_observations(self) -> list[str]:
        """把上一轮工具观察压成稳定 JSON 行，方便人工审计和测试复现。"""
        if not self._observations:
            return ["- none"]
        return [
            (
                f"- {observation_tool_name(observation)}: "
                f"{json.dumps(observation_summary(observation), ensure_ascii=False, sort_keys=True)}"
            )
            for observation in self._observations
        ]

    def _format_pending_next_step(self) -> list[str]:
        return [f"- {self._next_action()['reason']}"]

    def _next_action(self) -> dict[str, object]:
        if not self._observations:
            return {
                "status": "none",
                "reason": "none",
                "based_on_observation_index": None,
                "based_on_tool_name": None,
            }
        observation_index = len(self._observations) - 1
        latest_observation = self._observations[observation_index]
        latest_tool_name = observation_tool_name(latest_observation)
        if self._final_response_requested:
            return {
                "status": "continue",
                "reason": (
                    "The runtime has enough successful evidence to request the final answer. "
                    "Produce a concise final answer now; do not call tools again."
                ),
                "based_on_observation_index": observation_index,
                "based_on_tool_name": latest_tool_name,
            }
        latest_result = self._observations[-1].get("result", {})
        status = latest_result.get("status") if isinstance(latest_result, dict) else None
        if latest_tool_name in {"loop_suggestion", "safety_warning"} and isinstance(latest_result, dict):
            next_action_status = "continue"
            reason = str(latest_result.get("message") or "Continue with the next step.")
        elif latest_tool_name == "verification" and status == "error":
            next_action_status = "handle_error"
            reason = "Use the verification failure summary to repair the workspace, then stop for verification again."
        elif status == "suggestion":
            next_action_status = "continue"
            reason = str(latest_result.get("message") or "Continue with the next step.")
        elif status == "warning":
            next_action_status = "continue"
            reason = str(latest_result.get("recovery_suggestion") or "Change strategy and continue.")
        elif status == "success":
            next_action_status = "continue"
            reason = (
                "Continue from the latest successful tool observation. "
                "If the acceptance criteria are satisfied, produce the final answer."
            )
        elif status == "error":
            next_action_status = "handle_error"
            reason = "Use the latest tool error to adjust parameters, or stop and explain the failure explicitly."
        else:
            next_action_status = "decide"
            reason = "Use the latest tool observation to decide the next action explicitly."
        return {
            "status": next_action_status,
            "reason": reason,
            "based_on_observation_index": observation_index,
            "based_on_tool_name": latest_tool_name,
        }

    def _read_plan(self) -> dict[str, object]:
        path = self._episode_writer.path / "plan.json"
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ContextBuildError(f"plan.json missing: {path}") from error
        except json.JSONDecodeError as error:
            raise ContextBuildError(f"plan.json is not valid JSON: {error.msg}") from error
        if not isinstance(plan, dict):
            raise ContextBuildError("plan.json must contain a JSON object")
        planned_steps = plan.get("planned_steps")
        if not isinstance(planned_steps, list) or not all(isinstance(step, str) for step in planned_steps):
            raise ContextBuildError("plan.json planned_steps must be a list of strings")
        return plan

    def _format_plan(self) -> list[str]:
        if self._plan is None:
            return ["- none"]
        return _format_list(self._plan["planned_steps"])

    def _tool_workflow_hints(self) -> list[str]:
        allowed_tools = set(self._task.allowed_tools)
        hints: list[str] = []
        if {"context_find", "file_read"} <= allowed_tools:
            hints.extend(TOOL_WORKFLOW_HINTS[:2])
        if "apply_patch_set" in allowed_tools:
            hints.append(TOOL_WORKFLOW_HINTS[2])
        if "apply_patch" in allowed_tools:
            hints.append(TOOL_WORKFLOW_HINTS[3])
        if allowed_tools & {"shell", "code_run"}:
            hints.append(TOOL_WORKFLOW_HINTS[4])
        if allowed_tools & {"file_write", "apply_patch", "apply_patch_set"}:
            hints.append(TOOL_WORKFLOW_HINTS[5])
        return hints or ["Use the allowed tools only as needed for the task."]

    def _read_project_instructions(self) -> str | None:
        path = self._workspace_root / "AGENTS.md"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise ContextBuildError(f"failed to read AGENTS.md: {error}") from error

    def _format_project_instructions(self) -> list[str]:
        if self._project_instructions is None:
            return ["- none"]
        if not self._project_instructions.strip():
            return ["- empty"]
        return self._project_instructions[:PROJECT_INSTRUCTIONS_CHAR_LIMIT].splitlines()

    def _format_session_summary_block(self) -> list[str]:
        if self._session_summary is None:
            return []
        return ["", "Session Summary:", *self._format_session_summary()]

    def _format_session_summary(self) -> list[str]:
        if self._session_summary is None or not self._session_summary.strip():
            return ["- none"]
        return self._session_summary[:SESSION_SUMMARY_CHAR_LIMIT].splitlines()

    def _format_working_state_block(self) -> list[str]:
        if self._working_state is None:
            return []
        content = self._working_state_model_content()
        if not content:
            return []
        return ["", *content.splitlines()]

    def _format_interaction_state_block(self) -> list[str]:
        if not self._interaction_state:
            return []
        return ["", "Human Interaction State:", *self._format_interaction_state()]

    def _format_interaction_state(self) -> list[str]:
        return [f"- {_interaction_state_summary(record)}" for record in self._interaction_state[-8:]]

    def _working_state_model_content(self) -> str:
        try:
            return format_working_state_for_model(self._working_state)
        except WorkingStateError as error:
            raise ContextBuildError(f"invalid working_state: {error}") from error

    def _write_run_manifest(self, index: ContextIndex) -> None:
        manifest_path = self._episode_writer.path / "context-manifest.json"
        contexts = []
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            contexts = list(existing.get("contexts", []))
        contexts.append(index.to_dict())
        run_manifest = {
            "version": CONTEXT_MANIFEST_VERSION,
            "generated_at": _now_iso(),
            "context_count": len(contexts),
            "summary": {
                "provider": self._provider_name,
                "workspace_root": str(self._workspace_root),
                "goal": self._task.goal,
                "allowed_tools": self._task.allowed_tools,
            },
            "contexts": contexts,
        }
        self._episode_writer.write_context_manifest(run_manifest)


def _format_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _interaction_state_summary(record: dict[str, object]) -> str:
    parts = [
        f"type={_safe_state_value(record.get('type'), 'interaction')}",
        f"tool={_safe_state_value(record.get('tool'), 'unknown')}",
        f"status={_safe_state_value(record.get('status'), 'unknown')}",
    ]
    question = str(record.get("question") or "")
    if question:
        parts.append(f"question={json.dumps(question, ensure_ascii=False)}")
    if "answer_excerpt" in record:
        parts.append(f"answer_excerpt={json.dumps(str(record['answer_excerpt']), ensure_ascii=False)}")
        parts.append(f"answer_chars={record.get('answer_chars', 0)}")
    if "approved" in record:
        parts.append(f"approved={str(bool(record['approved'])).lower()}")
    if "turn" in record:
        parts.append(f"turn={record['turn']}")
    return " ".join(parts)


def _safe_state_value(value: object, fallback: str) -> str:
    text = str(value or fallback)
    return " ".join(text.split())


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _context_budget_summary(
    context_id: str,
    budget: ContextBudget,
    sources: list[ContextSource],
) -> dict[str, object]:
    included_source_count = sum(
        1
        for source in sources
        if source.budget is not None and source.budget.included_in_model_input
    )
    return {
        "context_id": context_id,
        "total_chars": budget.character_count,
        "max_chars": budget.character_limit,
        "status": budget.status,
        "source_count": len(sources),
        "included_source_count": included_source_count,
    }


def _context_budget(model_input: str) -> ContextBudget:
    character_count = len(model_input)
    status = "within_limit" if character_count <= CONTEXT_CHARACTER_LIMIT else "over_limit"
    return ContextBudget(
        character_count=character_count,
        character_limit=CONTEXT_CHARACTER_LIMIT,
        status=status,
    )
