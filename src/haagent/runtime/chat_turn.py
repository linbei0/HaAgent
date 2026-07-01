"""
haagent/runtime/chat_turn.py - Chat 单轮事件映射

把 runtime 事件转换为 ChatEvent 的 message/payload，避免 AgentSession 承载展示细节。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from haagent.models.gateway import ModelGateway
from haagent.runtime.cancellation import CancellationToken
from haagent.runtime.human_interaction import HumanInteractionHandler
from haagent.runtime.path_policy import PathPolicy, default_path_policy, serialize_path_policy
from haagent.runtime.run_recorder import RunResult
from haagent.skills import load_skill_registry
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
]
CHAT_WEB_TOOLS = ["web_search", "web_fetch", "skill_market_search"]
CHAT_SKILL_TOOLS = ["skill_list", "skill_read"]
CHAT_APPROVED_TOOLS = ["file_write", "code_run", "apply_patch", "apply_patch_set", "shell"]


@dataclass(frozen=True)
class ChatEventView:
    event_type: str
    message: str
    payload: dict[str, object]


class ChatEventMapper:
    @staticmethod
    def to_chat_event(event: dict[str, object], turn_index: int | None = None) -> ChatEventView:
        event_type = str(event.get("event_type", "unknown"))
        payload = dict(event)
        payload.pop("event_type", None)
        return ChatEventView(
            event_type=event_type,
            message=runtime_event_message(event_type, payload),
            payload=runtime_event_payload(event_type, payload),
        )


class OrchestratorFactory(Protocol):
    def __call__(
        self,
        *,
        runs_root: Path,
        model_gateway: ModelGateway,
        max_turns: int,
        session_summary: str | None,
        session_compaction: dict[str, object] | None,
        tool_result_microcompact_count: int,
        working_state: dict[str, object] | None,
        event_sink,
        interaction_handler: HumanInteractionHandler | None,
        cancellation_token: CancellationToken,
    ):
        ...


@dataclass(frozen=True)
class ChatTurnRequest:
    prompt: str
    workspace_root: Path
    runs_root: Path
    model_gateway: ModelGateway
    max_turns: int
    session_summary: str | None
    session_compaction: dict[str, object] | None
    tool_result_microcompact_count: int
    working_state: dict[str, object] | None
    path_policy: PathPolicy
    enable_web: bool
    target_paths: list[str]
    event_sink: object
    interaction_handler: HumanInteractionHandler | None
    cancellation_token: CancellationToken
    orchestrator_factory: OrchestratorFactory


class ChatTurnRunner:
    def run(self, request: ChatTurnRequest) -> RunResult:
        clean_prompt = request.prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be non-empty")
        with tempfile.TemporaryDirectory(prefix="haagent-chat-") as task_dir:
            task_path = Path(task_dir) / "task.yaml"
            write_chat_task_yaml(
                task_path,
                clean_prompt,
                request.workspace_root,
                path_policy=request.path_policy,
                enable_web=request.enable_web,
                target_paths=request.target_paths,
            )
            orchestrator = request.orchestrator_factory(
                runs_root=request.runs_root,
                model_gateway=request.model_gateway,
                max_turns=request.max_turns,
                session_summary=request.session_summary,
                session_compaction=request.session_compaction,
                tool_result_microcompact_count=request.tool_result_microcompact_count,
                working_state=request.working_state,
                event_sink=request.event_sink,
                interaction_handler=request.interaction_handler,
                cancellation_token=request.cancellation_token,
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
) -> None:
    allowed_tools = list(CHAT_ALLOWED_TOOLS)
    if enable_web:
        allowed_tools.extend(CHAT_WEB_TOOLS)
    if load_skill_registry(workspace_root=workspace_root).list_skills():
        allowed_tools.extend(CHAT_SKILL_TOOLS)
    policy = path_policy or default_path_policy(workspace_root)
    task = {
        "goal": request,
        "workspace_root": str(workspace_root.resolve()),
        "path_policy": serialize_path_policy(policy),
        "target_paths": list(target_paths or []),
        "constraints": [],
        "allowed_tools": allowed_tools,
        "acceptance_criteria": ["Complete the requested chat task."],
        "verification_commands": [],
        "policy": {
            "approval_allowed_tools": list(CHAT_APPROVED_TOOLS),
            "approved_tools": (
                list(CHAT_APPROVED_TOOLS)
                if policy.permission_mode in {"auto_approve", "full_access"}
                else []
            ),
        },
    }
    path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")


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
