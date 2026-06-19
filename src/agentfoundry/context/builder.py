"""
agentfoundry/context/builder.py - Context Builder v1

生成可审计的标准化模型输入与上下文 manifest。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentfoundry.context.manifest import (
    ContextBudget,
    ContextIndex,
    ContextManifest,
    ContextSource,
    ContextSourceBudget,
)
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.task_contract import TaskSpec
from agentfoundry.tools.registry import TOOL_REGISTRY


CONTEXT_MANIFEST_VERSION = "1.2"
CONTEXT_CHARACTER_LIMIT = 12000


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
    ) -> None:
        self._task = task
        self._workspace_root = workspace_root
        self._provider_name = provider_name
        self._episode_writer = episode_writer
        self._observations = list(observations or [])
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
                "AgentFoundry Context v1",
                "",
                "Instructions:",
                "- Use only the task facts and allowed tools listed below.",
                "- Report failures explicitly; do not invent successful outcomes.",
                "",
                "Project Instructions:",
                *self._format_project_instructions(),
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
        sources = [
            self._project_instructions_source(),
            ContextSource("task", "goal", "Goal from task.yaml", "The model needs the task goal."),
            ContextSource("task", "constraints", "Constraints from task.yaml", "The model must obey task constraints."),
            ContextSource("task", "allowed_tools", "Allowed tools from task.yaml", "The model needs the allowed tool list."),
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
        sources.extend(
            ContextSource(
                "tool_catalog",
                tool,
                TOOL_REGISTRY[tool].description,
                "Allowed tool description is included for model tool selection.",
            )
            for tool in self._task.allowed_tools
        )
        sources.extend(
            ContextSource(
                "observation",
                _observation_tool_name(observation),
                "Tool observation from previous turn",
                "Previous tool result is needed for the next model turn.",
            )
            for observation in self._observations
        )
        sources = [self._source_with_budget(source) for source in sources]
        return ContextManifest(
            context_id=context_id,
            provider=self._provider_name,
            workspace_root=str(self._workspace_root),
            generated_at=_now_iso(),
            budget=budget,
            sources=sources,
            next_action=self._next_action(),
        )

    def _source_with_budget(self, source: ContextSource) -> ContextSource:
        content = self._source_content(source)
        return ContextSource(
            source_type=source.source_type,
            name=source.name,
            description=source.description,
            inclusion_reason=source.inclusion_reason,
            status=source.status,
            budget=ContextSourceBudget(
                char_count=len(content),
                included_in_model_input=True,
                inclusion_reason=source.inclusion_reason,
            ),
        )

    def _source_content(self, source: ContextSource) -> str:
        if source.source_type == "project_instructions":
            return "\n".join(self._format_project_instructions())
        if source.source_type == "task":
            return _task_source_content(source.name, self._task)
        if source.source_type == "tool_catalog":
            return f"- {source.name}: {TOOL_REGISTRY[source.name].description}"
        if source.source_type == "plan":
            return "\n".join(["Plan:", *self._format_plan()])
        if source.source_type == "pending_next_step":
            return "\n".join(["Pending next step:", *self._format_pending_next_step()])
        if source.source_type == "observation":
            for observation in self._observations:
                if _observation_tool_name(observation) == source.name:
                    return (
                        f"- {_observation_tool_name(observation)}: "
                        f"{json.dumps(_observation_summary(observation), ensure_ascii=False, sort_keys=True)}"
                    )
        return ""

    def _format_observations(self) -> list[str]:
        """把上一轮工具观察压成稳定 JSON 行，方便人工审计和测试复现。"""
        if not self._observations:
            return ["- none"]
        return [
            (
                f"- {_observation_tool_name(observation)}: "
                f"{json.dumps(_observation_summary(observation), ensure_ascii=False, sort_keys=True)}"
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
        latest_result = self._observations[-1].get("result", {})
        status = latest_result.get("status") if isinstance(latest_result, dict) else None
        if status == "success":
            next_action_status = "continue"
            reason = "Continue from the latest successful tool observation and judge whether the acceptance criteria are satisfied."
        elif status in {"error", "failed"}:
            next_action_status = "handle_error"
            reason = "Use the latest tool error to adjust parameters, or stop and explain the failure explicitly."
        else:
            next_action_status = "decide"
            reason = "Use the latest tool observation to decide the next action explicitly."
        return {
            "status": next_action_status,
            "reason": reason,
            "based_on_observation_index": observation_index,
            "based_on_tool_name": _observation_tool_name(latest_observation),
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
        return self._project_instructions.splitlines()

    def _project_instructions_source(self) -> ContextSource:
        if self._project_instructions is None:
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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _observation_tool_name(observation: dict[str, object]) -> str:
    tool_name = observation.get("tool_name", "unknown_tool")
    return str(tool_name)


def _observation_summary(observation: dict[str, object]) -> dict[str, object]:
    return {
        "args": observation.get("args", {}),
        "result": observation.get("result", {}),
    }


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
