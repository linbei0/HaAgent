"""
haagent/context/builder.py - Context Builder v2

构建对话历史初始消息（system + task），不再每轮重建 context 块。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from haagent.context.manifest import (
    ContextIndex,
    ContextManifest,
)
from haagent.context.messages import (
    build_system_message,
    build_task_message,
)
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.task_contract import TaskSpec
from haagent.runtime.working_state import (
    WorkingStateError,
    format_working_state_for_model,
    raw_working_state_text,
)
from haagent.tools.registry import TOOL_REGISTRY


CONTEXT_MANIFEST_VERSION = "2.0"
PROJECT_INSTRUCTIONS_CHAR_LIMIT = 4000
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
    messages: list[dict]
    manifest: ContextManifest

    @property
    def model_input(self) -> str:
        """Backward-compat: return concatenated message content as a single string."""
        parts: list[str] = []
        for msg in self.messages:
            content = msg.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        return "\n".join(parts)


class ContextBuilder:
    def __init__(
        self,
        task: TaskSpec,
        workspace_root: Path,
        provider_name: str,
        episode_writer: EpisodeWriter,
        observations: list[dict] | None = None,
        final_response_requested: bool = False,
        session_summary: str | None = None,
        working_state: dict | None = None,
        interaction_state: list[dict] | None = None,
    ) -> None:
        self._task = task
        self._workspace_root = workspace_root
        self._provider_name = provider_name
        self._episode_writer = episode_writer
        self._session_summary = session_summary
        self._working_state = working_state
        self._interaction_state = list(interaction_state or [])

    def build(self) -> BuiltContext:
        """构建初始对话消息（system + task），写入 contexts/ 快照。"""
        self._validate_tools()
        project_instructions = self._read_project_instructions()
        plan = self._read_plan()
        context_id = self._next_context_id()
        contexts_dir = self._episode_writer.path / "contexts"
        contexts_dir.mkdir(parents=True, exist_ok=True)

        system_msg = build_system_message(
            project_instructions=project_instructions[:PROJECT_INSTRUCTIONS_CHAR_LIMIT] if project_instructions else None,
            tool_workflow_hints=self._tool_workflow_hints(),
            session_summary=(self._session_summary or "")[:SESSION_SUMMARY_CHAR_LIMIT] or None,
        )
        task_msg = build_task_message(
            task=self._task,
            plan_steps=list(plan.get("planned_steps", [])),
            working_state_content=self._working_state_content(),
            interaction_state_lines=self._format_interaction_state(),
        )
        messages = [system_msg, task_msg]

        system_chars = len(system_msg["content"])
        task_chars = len(task_msg["content"])

        snapshot_path = contexts_dir / f"{context_id}.json"
        snapshot_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = ContextManifest(
            context_id=context_id,
            provider=self._provider_name,
            workspace_root=str(self._workspace_root),
            generated_at=_now_iso(),
            message_count=len(messages),
            system_chars=system_chars,
            task_chars=task_chars,
        )
        manifest_path = contexts_dir / f"{context_id}-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_run_manifest(
            ContextIndex(
                context_id=context_id,
                model_input_path=f"contexts/{context_id}.json",
                manifest_path=f"contexts/{context_id}-manifest.json",
            ),
        )
        return BuiltContext(context_id=context_id, messages=messages, manifest=manifest)

    def _validate_tools(self) -> None:
        unknown_tools = [tool for tool in self._task.allowed_tools if tool not in TOOL_REGISTRY]
        if unknown_tools:
            raise ContextBuildError(f"unknown allowed_tools: {', '.join(unknown_tools)}")

    def _next_context_id(self) -> str:
        contexts_dir = self._episode_writer.path / "contexts"
        if not contexts_dir.exists():
            return "0001"
        existing = sorted(
            path.stem for path in contexts_dir.glob("*.json")
            if path.stem.isdigit()
        )
        if not existing:
            return "0001"
        return f"{int(existing[-1]) + 1:04d}"

    def _read_project_instructions(self) -> str | None:
        path = self._workspace_root / "AGENTS.md"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise ContextBuildError(f"failed to read AGENTS.md: {error}") from error

    def _read_plan(self) -> dict:
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
        if not isinstance(planned_steps, list) or not all(isinstance(s, str) for s in planned_steps):
            raise ContextBuildError("plan.json planned_steps must be a list of strings")
        return plan

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

    def _working_state_content(self) -> str | None:
        if self._working_state is None:
            return None
        try:
            content = format_working_state_for_model(self._working_state)
            return content or None
        except WorkingStateError as error:
            raise ContextBuildError(f"invalid working_state: {error}") from error

    def _format_interaction_state(self) -> list[str]:
        return [f"- {_interaction_state_summary(r)}" for r in self._interaction_state[-8:]]

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


def _interaction_state_summary(record: dict) -> str:
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
