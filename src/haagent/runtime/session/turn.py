"""
src/haagent/runtime/session/turn.py - Chat 单轮运行适配

把自然语言 prompt 转成临时 task contract，并保留 runtime 原始事件的紧凑展示 helper。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from haagent.models.gateway import ModelGateway
from haagent.prompts.commands import parse_prompt_command
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, serialize_path_policy
from haagent.runtime.orchestration.recorder import RunResult
from haagent.runtime.session.attachments import ImageAttachment
from haagent.skills import load_skill_registry
from haagent.tools.registry import ToolRuntimeRegistry
from haagent.tools.presentation import summarize_tool_args, summarize_tool_result


CHAT_ALLOWED_TOOLS = [
    "file_list",
    "file_search",
    "file_read",
    "request_user_input",
    "start_memory_update",
    "file_write",
    "code_run",
    "apply_patch",
    "apply_patch_set",
    "shell",
    "agent",
    "send_message",
    "task_stop",
    "task_get",
    "task_list",
    "task_output",
]
CHAT_WEB_TOOLS = ["web_search", "web_fetch", "skill_market_search"]
CHAT_SKILL_TOOLS = ["skill_list", "skill_read"]
CHAT_APPROVED_TOOLS = ["file_write", "code_run", "apply_patch", "apply_patch_set", "shell"]


class OrchestratorFactory(Protocol):
    def __call__(
        self,
        *,
        runs_root: Path,
        model_gateway: ModelGateway,
        max_turns: int | None,
        session_summary: str | None,
        session_compaction: dict[str, object] | None,
        historical_tool_compression_count: int,
        working_state: dict[str, object] | None,
        event_sink,
        interaction_handler: HumanInteractionHandler | None,
        cancellation_token: CancellationToken,
        tool_registry: ToolRuntimeRegistry | None = None,
        mcp_runtime: object | None = None,
        leader_session_id: str | None = None,
        worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None,
    ):
        ...


@dataclass(frozen=True)
class ChatTurnRequest:
    prompt: str
    workspace_root: Path
    runs_root: Path
    model_gateway: ModelGateway
    max_turns: int | None
    session_summary: str | None
    session_compaction: dict[str, object] | None
    historical_tool_compression_count: int
    working_state: dict[str, object] | None
    path_policy: PathPolicy
    enable_web: bool
    target_paths: list[str]
    event_sink: object
    interaction_handler: HumanInteractionHandler | None
    cancellation_token: CancellationToken
    orchestrator_factory: OrchestratorFactory
    leader_session_id: str | None = None
    tool_registry: ToolRuntimeRegistry | None = None
    mcp_runtime: object | None = None
    mcp_tool_names: list[str] = field(default_factory=list)
    prompt_pack_ids: list[str] = field(default_factory=list)
    allowed_tools_override: list[str] | None = None
    approval_allowed_tools_override: list[str] | None = None
    approved_tools_override: list[str] | None = None
    worker_context: dict[str, object] | None = None
    worker_permission_requester: Callable[[str, dict[str, Any], Any], Any] | None = None
    attachments: list[ImageAttachment] = field(default_factory=list)
    image_attachment_history: list[ImageAttachment] = field(default_factory=list)


class ChatTurnRunner:
    def run(self, request: ChatTurnRequest) -> RunResult:
        clean_prompt = request.prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be non-empty")
        parsed_prompt = parse_prompt_command(clean_prompt)
        prompt_pack_ids = [*request.prompt_pack_ids, *parsed_prompt.prompt_pack_ids]
        with tempfile.TemporaryDirectory(prefix="haagent-chat-") as task_dir:
            task_path = Path(task_dir) / "task.yaml"
            write_chat_task_yaml(
                task_path,
                parsed_prompt.normalized_prompt,
                request.workspace_root,
                path_policy=request.path_policy,
                enable_web=request.enable_web,
                target_paths=request.target_paths,
                mcp_tool_names=request.mcp_tool_names,
                prompt_pack_ids=prompt_pack_ids,
                allowed_tools_override=request.allowed_tools_override,
                approval_allowed_tools_override=request.approval_allowed_tools_override,
                approved_tools_override=request.approved_tools_override,
                worker_context=request.worker_context,
                attachments=request.attachments,
                image_attachment_history=request.image_attachment_history,
            )
            orchestrator = request.orchestrator_factory(
                runs_root=request.runs_root,
                model_gateway=request.model_gateway,
                max_turns=request.max_turns,
                session_summary=request.session_summary,
                session_compaction=request.session_compaction,
                historical_tool_compression_count=request.historical_tool_compression_count,
                working_state=request.working_state,
                event_sink=request.event_sink,
                interaction_handler=request.interaction_handler,
                cancellation_token=request.cancellation_token,
                tool_registry=request.tool_registry,
                mcp_runtime=request.mcp_runtime,
                leader_session_id=request.leader_session_id,
                worker_permission_requester=request.worker_permission_requester,
            )
            return orchestrator.run(task_path)


def write_chat_task_yaml(
    path: Path,
    request: str,
    workspace_root: Path,
    *,
    path_policy: PathPolicy | None = None,
    enable_web: bool = False,
    target_paths: list[str] | None = None,
    mcp_tool_names: list[str] | None = None,
    prompt_pack_ids: list[str] | None = None,
    allowed_tools_override: list[str] | None = None,
    approval_allowed_tools_override: list[str] | None = None,
    approved_tools_override: list[str] | None = None,
    worker_context: dict[str, object] | None = None,
    attachments: list[ImageAttachment] | None = None,
    image_attachment_history: list[ImageAttachment] | None = None,
) -> None:
    mcp_tools = list(mcp_tool_names or [])
    if allowed_tools_override is None:
        allowed_tools = list(CHAT_ALLOWED_TOOLS)
        if image_attachment_history:
            allowed_tools.append("load_image_attachment")
        if enable_web:
            allowed_tools.extend(CHAT_WEB_TOOLS)
        if load_skill_registry(workspace_root=workspace_root).list_skills():
            allowed_tools.extend(CHAT_SKILL_TOOLS)
        if mcp_tools:
            allowed_tools.extend(mcp_tools)
            allowed_tools.extend(["list_mcp_resources", "read_mcp_resource"])
    else:
        allowed_tools = list(allowed_tools_override)
    policy = path_policy or default_path_policy(workspace_root)
    approval_allowed_tools = (
        list(approval_allowed_tools_override)
        if approval_allowed_tools_override is not None
        else [*CHAT_APPROVED_TOOLS, *mcp_tools]
    )
    approved_tools = (
        list(approved_tools_override)
        if approved_tools_override is not None
        else approval_allowed_tools
        if policy.permission_mode in {"auto_approve", "full_access"}
        else []
    )
    task = {
        "goal": request,
        "workspace_root": str(workspace_root.resolve()),
        "path_policy": serialize_path_policy(policy),
        "target_paths": list(target_paths or []),
        "prompt_pack_ids": list(prompt_pack_ids or []),
        "constraints": [],
        "allowed_tools": allowed_tools,
        "acceptance_criteria": ["Complete the requested chat task."],
        "verification_commands": [],
        "policy": {
            "approval_allowed_tools": approval_allowed_tools,
            "approved_tools": approved_tools,
        },
    }
    if attachments:
        task["attachments"] = [_attachment_dict(attachment) for attachment in attachments]
    if image_attachment_history:
        task["image_attachment_history"] = [
            _history_attachment_dict(attachment)
            for attachment in image_attachment_history
        ]
    if worker_context is not None:
        task["worker_context"] = dict(worker_context)
    path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _history_attachment_dict(attachment: ImageAttachment) -> dict[str, object]:
    return _attachment_dict(attachment)


def _attachment_dict(attachment: ImageAttachment) -> dict[str, object]:
    data = attachment.to_dict()
    if attachment.base_path:
        data["base_path"] = str(Path(attachment.base_path).resolve())
    return data


def runtime_event_message(event_type: str, payload: dict[str, object]) -> str:
    if event_type == "tool_started":
        return f"starting tool {payload.get('tool_name', 'unknown')}"
    if event_type == "tool_finished":
        return f"finished tool {payload.get('tool_name', 'unknown')}"
    if event_type == "tool_failed":
        return f"failed tool {payload.get('tool_name', 'unknown')}"
    if event_type == "approval_requested":
        return f"approval requested for {payload.get('tool_name', 'unknown')}"
    if event_type == "approval_granted":
        return f"approval granted for {payload.get('tool_name', 'unknown')}"
    if event_type == "approval_denied":
        return f"approval denied for {payload.get('tool_name', 'unknown')}"
    if event_type == "edit_diff_requested":
        return f"edit diff requested for {payload.get('tool_name', 'unknown')}"
    if event_type == "edit_diff_granted":
        return f"edit diff granted for {payload.get('tool_name', 'unknown')}"
    if event_type == "edit_diff_denied":
        return f"edit diff denied for {payload.get('tool_name', 'unknown')}"
    if event_type == "user_input_requested":
        return summary_value(str(payload.get("question", "")))
    if event_type == "user_input_received":
        return "user input received"
    if event_type == "assistant_delta":
        return summary_value(str(payload.get("delta", "")))
    if event_type == "assistant_message":
        return summary_value(str(payload.get("content", "")))
    if event_type == "guardrail_triggered":
        return summary_value(str(payload.get("message", "guardrail triggered")))
    if event_type == "failure":
        return summary_value(str(payload.get("reason", "chat turn failed")))
    return event_type


def runtime_event_payload(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    if event_type == "tool_started":
        tool_name = str(payload.get("tool_name", "unknown"))
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": tool_name,
            "args_summary": summarize_tool_args(tool_name, args),
        }
    if event_type == "tool_finished":
        tool_name = str(payload.get("tool_name", "unknown"))
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if tool_name == "start_memory_update":
            return {
                "model_turn": payload.get("turn"),
                "tool_name": tool_name,
                "memory_update_requested": bool(result.get("memory_update_requested")),
                "reason": summary_value(str(result.get("reason", "")), 240),
            }
        return {
            "model_turn": payload.get("turn"),
            "tool_name": tool_name,
            "status": str(result.get("status", "unknown")),
            "result_summary": summarize_tool_result(tool_name, result),
        }
    if event_type == "tool_failed":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "error_type": str(error.get("type", "unknown")),
            "message": summary_value(str(error.get("message", ""))),
            "error": {
                "type": str(error.get("type", "unknown")),
                "message": summary_value(str(error.get("message", ""))),
            },
        }
    if event_type in {"approval_requested", "approval_granted", "approval_denied"}:
        args_summary = payload.get("args_summary") if isinstance(payload.get("args_summary"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": summary_value(str(payload.get("question", "")), 240),
            "approved": payload.get("approved"),
            "args_summary": args_summary,
        }
    if event_type in {"edit_diff_requested", "edit_diff_granted", "edit_diff_denied"}:
        args_summary = payload.get("args_summary") if isinstance(payload.get("args_summary"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": summary_value(str(payload.get("question", "")), 240),
            "approved": payload.get("approved"),
            "answer": summary_value(str(payload.get("answer", "")), 80),
            "args_summary": args_summary,
        }
    if event_type == "user_input_requested":
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": summary_value(str(payload.get("question", "")), 240),
            "reason": summary_value(str(payload.get("reason", "")), 240),
        }
    if event_type == "user_input_received":
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": summary_value(str(payload.get("question", "")), 240),
            "answer_chars": payload.get("answer_chars"),
            "approved": payload.get("approved"),
        }
    if event_type == "assistant_delta":
        return {
            "model_turn": payload.get("turn"),
            "delta": str(payload.get("delta", "")),
        }
    if event_type == "assistant_message":
        return {
            "model_turn": payload.get("turn"),
            "content": str(payload.get("content", "")),
        }
    if event_type == "guardrail_triggered":
        return {
            "status": str(payload.get("status", "blocked")),
            "scope": str(payload.get("scope", "unknown")),
            "rule_id": str(payload.get("rule_id", "unknown")),
            "severity": str(payload.get("severity", "unknown")),
            "message": summary_value(str(payload.get("message", ""))),
        }
    if event_type == "failure":
        return {
            "status": str(payload.get("status", "failed")),
            "failed_stage": summary_value(str(payload.get("failed_stage", "unknown"))),
            "failure_category": summary_value(str(payload.get("failure_category", "unknown"))),
            "reason": summary_value(str(payload.get("reason", ""))),
            "episode_path": summary_value(str(payload.get("episode_path", "")), 300),
        }
    return payload


def summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"
