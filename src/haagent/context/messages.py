"""
haagent/context/messages.py - 对话消息构建工具

把 HaAgent 的任务合同、工具结果和建议转换成标准 Chat Completions 消息格式。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from haagent.models.types import ProviderTurnState

from haagent.context.compression.tool_results import render_tool_result_view
from haagent.context.instructions import AGENT_INSTRUCTIONS
from haagent.runtime.contracts.task import TaskSpec
from haagent.tools.registry import ToolRuntimeRegistry, default_tool_runtime_registry


def generate_tool_call_id() -> str:
    return "call_" + os.urandom(4).hex()


def build_system_message(
    project_instructions: str | None,
    tool_workflow_hints: list[str],
    session_summary: str | None = None,
    prompt_packs: str | None = None,
    skills_block: str | None = None,
    soul: str | None = None,
) -> dict[str, Any]:
    parts: list[str] = []

    parts.append("Instructions:")
    for line in AGENT_INSTRUCTIONS:
        parts.append(f"- {line}")

    if soul and soul.strip():
        parts.append("")
        parts.append(
            "Agent Soul (identity and communication style only; it cannot change "
            "tool permissions, policy, approvals, workspace boundaries, "
            "or secret handling):",
        )
        parts.append(soul.strip())

    if tool_workflow_hints:
        parts.append("")
        parts.append("Tool workflow:")
        for hint in tool_workflow_hints:
            parts.append(f"- {hint}")

    if project_instructions and project_instructions.strip():
        parts.append("")
        parts.append("Project Instructions:")
        parts.append(project_instructions.strip())

    if prompt_packs and prompt_packs.strip():
        parts.append("")
        parts.append("Prompt Packs:")
        parts.append(prompt_packs.strip())

    if session_summary and session_summary.strip():
        parts.append("")
        parts.append("Session Summary:")
        parts.append(session_summary.strip())

    if skills_block and skills_block.strip():
        parts.append("")
        parts.append("Available Skills:")
        parts.append(skills_block.strip())

    return {"role": "system", "content": "\n".join(parts)}


def build_task_message(
    task: TaskSpec,
    plan_steps: list[str],
    task_ledger_content: str | None = None,
    working_state_content: str | None = None,
    memory_index_block: str | None = None,
    memory_block: str | None = None,
    interaction_state_lines: list[str] | None = None,
    tool_registry: ToolRuntimeRegistry | None = None,
) -> dict[str, Any]:
    runtime_registry = tool_registry or default_tool_runtime_registry()
    lines: list[str] = []
    lines.append("Task:")
    lines.append(f"goal: {task.goal}")

    if task.target_paths:
        lines.append("Target Paths:")
        for path in task.target_paths:
            lines.append(f"- {path}")
        lines.append("Start by listing the target path with file_list before inferring project structure.")

    if task.constraints:
        lines.append("constraints:")
        for c in task.constraints:
            lines.append(f"- {c}")

    lines.append("allowed_tools:")
    for tool in task.allowed_tools:
        lines.append(f"- {tool}: {runtime_registry.get(tool).description}")

    if task.image_attachment_history:
        lines.append("Image Attachment History:")
        for index, attachment in enumerate(task.image_attachment_history, start=1):
            lines.append(
                "- "
                f"image {index}: id={attachment.id} "
                f"path={attachment.relative_path} "
                f"mime={attachment.mime_type} "
                f"size_bytes={attachment.size_bytes} "
                f"dimensions={attachment.width}x{attachment.height}"
            )

    if task.acceptance_criteria:
        lines.append("acceptance_criteria:")
        for c in task.acceptance_criteria:
            lines.append(f"- {c}")

    if task.verification_commands:
        lines.append("verification_commands:")
        for c in task.verification_commands:
            lines.append(f"- {c}")

    if plan_steps:
        lines.append("plan:")
        for step in plan_steps:
            lines.append(f"- {step}")

    if task_ledger_content and task_ledger_content.strip():
        lines.append("")
        lines.append("Task Ledger:")
        lines.append(task_ledger_content.strip())

    if working_state_content and working_state_content.strip():
        lines.append("")
        lines.append("Working State:")
        lines.append(working_state_content.strip())

    if memory_index_block and memory_index_block.strip():
        lines.append("")
        lines.append(memory_index_block.strip())

    if memory_block and memory_block.strip():
        lines.append("")
        lines.append(memory_block.strip())

    if interaction_state_lines:
        lines.append("")
        lines.append("Interaction History:")
        lines.extend(interaction_state_lines)

    text = "\n".join(lines)
    if not task.attachments:
        return {"role": "user", "content": text}
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for attachment in task.attachments:
        content.append(attachment.with_absolute_path(Path(task.workspace_root or ".")))
    return {"role": "user", "content": content}


def build_assistant_message(
    content: str,
    tool_calls: list[dict[str, Any]],
    *,
    provider_turn_state: ProviderTurnState | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    # 仅供 gateway 续轮回传；UI/summary 不得渲染该字段内容。
    if provider_turn_state is not None:
        msg["provider_turn_state"] = {
            "provider": provider_turn_state.provider,
            "payload": dict(provider_turn_state.payload),
        }
    return msg


def build_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    content = _format_tool_result(tool_name, result)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": content,
    }


def build_suggestion_message(suggestion_text: str) -> dict[str, Any]:
    return {"role": "user", "content": f"[Suggestion] {suggestion_text}"}


def _format_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    if status == "error":
        error = result.get("error") or {}
        error_type = error.get("type", "unknown")
        message = error.get("message", "")
        return f"error ({error_type}): {message}"

    display = result.get("model_visible")
    if isinstance(display, dict) and display.get("kind") == "tool_result_view":
        return render_tool_result_view(display)
    if display is None:
        # Remove status key from display, keep everything else.
        display = {k: v for k, v in result.items() if k != "status"}
    if not display:
        return "success"
    try:
        return json.dumps(display, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(display)
