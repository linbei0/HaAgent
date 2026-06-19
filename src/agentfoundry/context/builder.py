"""
agentfoundry/context/builder.py - Context Builder v1

生成可审计的标准化模型输入与上下文 manifest。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentfoundry.context.manifest import ContextIndex, ContextManifest, ContextSource
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.task_contract import TaskSpec
from agentfoundry.tools.registry import TOOL_REGISTRY


CONTEXT_MANIFEST_VERSION = "1.2"


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

    def build(self) -> BuiltContext:
        """构建第一版上下文：不检索文件，只写任务事实和工具目录。"""
        self._validate_tools()
        context_id = self._next_context_id()
        contexts_dir = self._episode_writer.path / "contexts"
        contexts_dir.mkdir(parents=True, exist_ok=True)

        model_input = self._render_model_input()
        model_input_path = contexts_dir / f"{context_id}.txt"
        manifest_path = contexts_dir / f"{context_id}.json"
        model_input_path.write_text(model_input, encoding="utf-8")

        manifest = self._context_manifest(context_id)
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_run_manifest(
            ContextIndex(
                context_id=context_id,
                model_input_path=f"contexts/{context_id}.txt",
                manifest_path=f"contexts/{context_id}.json",
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
                "Observations:",
                *self._format_observations(),
                "",
            ],
        )

    def _context_manifest(self, context_id: str) -> ContextManifest:
        sources = [
            ContextSource("task", "goal", "Goal from task.yaml"),
            ContextSource("task", "constraints", "Constraints from task.yaml"),
            ContextSource("task", "allowed_tools", "Allowed tools from task.yaml"),
            ContextSource("task", "acceptance_criteria", "Acceptance criteria from task.yaml"),
            ContextSource("task", "verification_commands", "Verification commands from task.yaml"),
        ]
        sources.extend(
            ContextSource("tool_catalog", tool, TOOL_REGISTRY[tool].description)
            for tool in self._task.allowed_tools
        )
        sources.extend(
            ContextSource(
                "observation",
                _observation_tool_name(observation),
                "Tool observation from previous turn",
            )
            for observation in self._observations
        )
        return ContextManifest(
            context_id=context_id,
            provider=self._provider_name,
            workspace_root=str(self._workspace_root),
            generated_at=_now_iso(),
            sources=sources,
        )

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
