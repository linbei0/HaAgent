"""
haagent/runtime/chat_session.py - 自然语言 Agent 会话

管理 chat 会话状态，并把每条用户请求转成可审计的临时 task contract。
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from haagent.models.gateway import ModelGateway
from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.human_interaction import (
    HumanInteractionHandler,
    interaction_args_summary,
)
from haagent.runtime.orchestrator import RunOrchestrator
from haagent.runtime.working_state import (
    WorkingStateError,
    empty_working_state,
    load_working_state,
    update_working_state,
    write_working_state,
)


CHAT_ALLOWED_TOOLS = [
    "file_list",
    "file_search",
    "context_find",
    "file_read",
    "request_user_input",
    "file_write",
    "code_run",
    "apply_patch",
    "apply_patch_set",
    "shell",
]
CHAT_APPROVED_TOOLS = ["file_write", "code_run", "apply_patch", "apply_patch_set", "shell"]
CHAT_MAX_TURNS = 20
SESSION_SUMMARY_CHAR_LIMIT = 1000


class ChatSessionError(RuntimeError):
    """Chat session package 损坏或无法恢复时抛出。"""


@dataclass(frozen=True)
class ChatEvent:
    event_type: str
    session_id: str
    turn_index: int
    message: str
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "message": self.message,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class ChatTurnResult:
    session_id: str
    turn_index: int
    status: str
    episode_path: Path
    provider: str
    final_response: str
    verification_status: str
    failed_stage: str = "none"
    failure_category: str = "none"
    reason: str = "none"
    summary_error: str | None = None

    def output_lines(self) -> list[str]:
        lines = [
            f"status={self.status}",
            f"episode_path={self.episode_path}",
            f"provider={self.provider}",
            f"final_response={_summary_value(self.final_response)}",
            f"verification={self.verification_status}",
        ]
        if self.summary_error is not None:
            lines.append(f"summary_error={_summary_value(self.summary_error)}")
        if self.status != "completed":
            lines.extend(
                [
                    f"failed_stage={_summary_value(self.failed_stage)}",
                    f"failure_category={_summary_value(self.failure_category)}",
                    f"reason={_summary_value(self.reason)}",
                ],
            )
        return lines


class AgentSession:
    def __init__(
        self,
        *,
        workspace_root: Path,
        runs_root: Path,
        model_gateway: ModelGateway | None = None,
        max_turns: int = CHAT_MAX_TURNS,
        session_id: str | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.runs_root = runs_root
        self.model_gateway = model_gateway
        self.max_turns = max_turns
        self.session_id = session_id or _new_session_id()
        self.turn_count = 0
        self._summaries: list[str] = []
        self._working_state = empty_working_state()
        self.session_path = self.runs_root / "sessions" / self.session_id
        self._created_at = datetime.now(UTC).isoformat()
        self._write_session_metadata()
        self._write_working_state()

    @classmethod
    def resume(
        cls,
        session: str | Path,
        *,
        runs_root: Path | None = None,
        model_gateway: ModelGateway | None = None,
        max_turns: int = CHAT_MAX_TURNS,
    ) -> "AgentSession":
        session_path = _resolve_session_path(session, runs_root or Path(".runs"))
        metadata = _read_session_metadata(session_path)
        turns = _read_session_turns(session_path)

        instance = cls.__new__(cls)
        instance.workspace_root = Path(str(metadata["workspace_root"])).resolve()
        instance.runs_root = session_path.parent.parent
        instance.model_gateway = model_gateway
        instance.max_turns = max_turns
        instance.session_id = str(metadata["session_id"])
        instance.turn_count = int(metadata["turn_count"])
        instance._summaries = _bounded_summaries([str(turn["summary"]) for turn in turns])
        try:
            instance._working_state = load_working_state(session_path / "working_state.json")
        except WorkingStateError as error:
            raise ChatSessionError(str(error)) from error
        instance.session_path = session_path
        instance._created_at = str(metadata["created_at"])
        return instance

    @property
    def provider_name(self) -> str:
        if self.model_gateway is None:
            return "fake"
        return self.model_gateway.provider_name

    def run_prompt(
        self,
        prompt: str,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        return self.run_prompt_events(prompt, interaction_handler=interaction_handler)

    def run_prompt_events(
        self,
        prompt: str,
        event_sink: Callable[[ChatEvent], None] | None = None,
        include_session_events: bool = False,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be non-empty")

        turn_index = self.turn_count + 1
        if include_session_events:
            self._emit_chat_event(
                event_sink,
                event_type="session_started",
                turn_index=turn_index,
                message="chat session started",
                payload=self.status(),
            )
        self._emit_chat_event(
            event_sink,
            event_type="turn_started",
            turn_index=turn_index,
            message="chat turn started",
            payload={"prompt": _summary_value(clean_prompt, 160)},
        )
        runtime_events: list[dict[str, object]] = []

        def on_runtime_event(event: dict[str, object]) -> None:
            runtime_events.append(event)
            self._emit_runtime_event(event_sink, turn_index, event)

        with tempfile.TemporaryDirectory(prefix="haagent-chat-") as task_dir:
            task_path = Path(task_dir) / "task.yaml"
            _write_chat_task_yaml(task_path, clean_prompt, self.workspace_root)
            result = RunOrchestrator(
                runs_root=self.runs_root,
                model_gateway=self.model_gateway,
                max_turns=self.max_turns,
                session_summary=self.summary_text(),
                working_state=self._working_state.to_dict() if not self._working_state.is_empty() else None,
                event_sink=on_runtime_event,
                interaction_handler=interaction_handler,
            ).run(task_path)

        turn_result = self._build_turn_result(clean_prompt, result)
        self.turn_count += 1
        self._working_state = update_working_state(
            self._working_state,
            prompt=clean_prompt,
            result=turn_result,
            runtime_events=runtime_events,
        )
        self._write_working_state()
        turn_summary = _turn_summary(clean_prompt, turn_result)
        self._summaries.append(turn_summary)
        self._summaries = _bounded_summaries(self._summaries)
        self._record_turn(clean_prompt, turn_result, turn_summary)
        if turn_result.status != "completed":
            self._emit_chat_event(
                event_sink,
                event_type="failure",
                turn_index=turn_index,
                message="chat turn failed",
                payload={
                    "status": turn_result.status,
                    "failed_stage": _summary_value(turn_result.failed_stage),
                    "failure_category": _summary_value(turn_result.failure_category),
                    "reason": _summary_value(turn_result.reason),
                },
            )
        self._emit_chat_event(
            event_sink,
            event_type="turn_finished",
            turn_index=turn_index,
            message="chat turn finished",
            payload={
                "status": turn_result.status,
                "episode_path": str(turn_result.episode_path),
                "runtime_event_count": len(runtime_events),
            },
        )
        if include_session_events:
            self._emit_chat_event(
                event_sink,
                event_type="session_finished",
                turn_index=turn_index,
                message="chat session finished",
                payload={"status": turn_result.status},
            )
        return turn_result

    def status(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_path": str(self.session_path.resolve()),
            "workspace_root": str(self.workspace_root),
            "provider": self.provider_name,
            "turn_count": self.turn_count,
            "working_state": self._working_state.status_summary(),
        }

    def new(self) -> None:
        self.session_id = _new_session_id()
        self.turn_count = 0
        self._summaries = []
        self._working_state = empty_working_state()
        self.session_path = self.runs_root / "sessions" / self.session_id
        self._created_at = datetime.now(UTC).isoformat()
        self._write_session_metadata()
        self._write_working_state()

    def summary_text(self) -> str | None:
        if not self._summaries:
            return None
        return "\n".join(_bounded_summaries(self._summaries))

    def session_started_event(self) -> ChatEvent:
        return ChatEvent(
            event_type="session_started",
            session_id=self.session_id,
            turn_index=self.turn_count,
            message="chat session started",
            payload=self.status(),
        )

    def session_finished_event(self) -> ChatEvent:
        return ChatEvent(
            event_type="session_finished",
            session_id=self.session_id,
            turn_index=self.turn_count,
            message="chat session finished",
            payload={"turn_count": self.turn_count},
        )

    def _emit_chat_event(
        self,
        event_sink: Callable[[ChatEvent], None] | None,
        *,
        event_type: str,
        turn_index: int,
        message: str,
        payload: dict[str, object],
    ) -> None:
        if event_sink is None:
            return
        event_sink(
            ChatEvent(
                event_type=event_type,
                session_id=self.session_id,
                turn_index=turn_index,
                message=message,
                payload=payload,
            ),
        )

    def _emit_runtime_event(
        self,
        event_sink: Callable[[ChatEvent], None] | None,
        turn_index: int,
        event: dict[str, object],
    ) -> None:
        event_type = str(event.get("event_type", "unknown"))
        payload = dict(event)
        payload.pop("event_type", None)
        self._emit_chat_event(
            event_sink,
            event_type=event_type,
            turn_index=turn_index,
            message=_runtime_event_message(event_type, payload),
            payload=_runtime_event_payload(event_type, payload),
        )

    def _build_turn_result(self, prompt: str, result) -> ChatTurnResult:
        try:
            package_view = load_inspect_episode_package(result.episode_path)
        except EpisodeValidationError as error:
            return ChatTurnResult(
                session_id=self.session_id,
                turn_index=self.turn_count + 1,
                status=result.status.value,
                episode_path=result.episode_path,
                provider=self.provider_name,
                final_response="none",
                verification_status="not_run",
                summary_error=str(error),
            )

        failure = package_view.failure_record.get("failure")
        if not isinstance(failure, dict):
            failure = {}
        return ChatTurnResult(
            session_id=self.session_id,
            turn_index=self.turn_count + 1,
            status=result.status.value,
            episode_path=result.episode_path,
            provider=str(package_view.episode_metadata.get("provider", self.provider_name)),
            final_response=_run_final_response(package_view.transcript),
            verification_status=_verification_status(
                package_view.verification_commands,
                package_view.verification_reached,
            ),
            failed_stage=str(failure.get("stage", "none")),
            failure_category=str(failure.get("category", "none")),
            reason=str(failure.get("evidence", "none")),
        )

    def _record_turn(self, prompt: str, result: ChatTurnResult, summary: str) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        record = {
            "turn_index": result.turn_index,
            "request": _summary_value(prompt, 300),
            "summary": summary,
            "status": result.status,
            "episode_path": str(result.episode_path),
            "verification_status": result.verification_status,
        }
        with (self.session_path / "turns.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_session_metadata()

    def _write_working_state(self) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        write_working_state(self.session_path / "working_state.json", self._working_state)

    def _write_session_metadata(self) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.session_path / "session.json"
        created_at = self._created_at
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
                created_at = str(existing["created_at"])
        metadata = {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "provider": self.provider_name,
            "created_at": created_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "turn_count": self.turn_count,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_chat_task_yaml(path: Path, request: str, workspace_root: Path) -> None:
    task = {
        "goal": request,
        "workspace_root": str(workspace_root.resolve()),
        "constraints": [],
        "allowed_tools": list(CHAT_ALLOWED_TOOLS),
        "acceptance_criteria": ["Complete the requested chat task."],
        "verification_commands": [],
        "policy": {
            "approval_allowed_tools": list(CHAT_APPROVED_TOOLS),
            "approved_tools": [],
        },
    }
    path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _turn_summary(prompt: str, result: ChatTurnResult) -> str:
    return "\n".join(
        [
            f"- request: {_summary_value(prompt, 160)}",
            f"  status: {result.status}",
            f"  episode_path: {result.episode_path}",
            f"  final_response: {_summary_value(result.final_response, 220)}",
            f"  verification: {result.verification_status}",
        ],
    )


def _bounded_summaries(summaries: list[str]) -> list[str]:
    selected: list[str] = []
    total = 0
    for summary in reversed(summaries):
        extra = len(summary) + (1 if selected else 0)
        if selected and total + extra > SESSION_SUMMARY_CHAR_LIMIT:
            break
        if not selected and extra > SESSION_SUMMARY_CHAR_LIMIT:
            selected.append(summary[:SESSION_SUMMARY_CHAR_LIMIT])
            break
        selected.append(summary)
        total += extra
    return list(reversed(selected))


def _run_final_response(transcript: list[dict[str, Any]]) -> str:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return str(record.get("content", ""))
    return "none"


def _verification_status(commands: list[dict[str, Any]], verification_reached: bool) -> str:
    if not verification_reached or not commands:
        return "not_run"
    if any(command.get("status") != "success" for command in commands):
        return "failed"
    return "success"


def _runtime_event_message(event_type: str, payload: dict[str, object]) -> str:
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
    if event_type == "user_input_requested":
        return _summary_value(str(payload.get("question", "")))
    if event_type == "user_input_received":
        return "user input received"
    if event_type == "assistant_message":
        return _summary_value(str(payload.get("content", "")))
    if event_type == "guardrail_triggered":
        return _summary_value(str(payload.get("message", "guardrail triggered")))
    if event_type == "failure":
        return _summary_value(str(payload.get("reason", "chat turn failed")))
    return event_type


def _runtime_event_payload(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    if event_type == "tool_started":
        tool_name = str(payload.get("tool_name", "unknown"))
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": tool_name,
            "args_summary": _tool_args_summary(tool_name, args),
        }
    if event_type == "tool_finished":
        tool_name = str(payload.get("tool_name", "unknown"))
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": tool_name,
            "status": str(result.get("status", "unknown")),
            "result_summary": _tool_result_summary(tool_name, result),
        }
    if event_type == "tool_failed":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "error_type": str(error.get("type", "unknown")),
            "message": _summary_value(str(error.get("message", ""))),
        }
    if event_type in {"approval_requested", "approval_granted", "approval_denied"}:
        args_summary = payload.get("args_summary") if isinstance(payload.get("args_summary"), dict) else {}
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": _summary_value(str(payload.get("question", "")), 240),
            "approved": payload.get("approved"),
            "args_summary": args_summary,
        }
    if event_type == "user_input_requested":
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": _summary_value(str(payload.get("question", "")), 240),
            "reason": _summary_value(str(payload.get("reason", "")), 240),
        }
    if event_type == "user_input_received":
        return {
            "model_turn": payload.get("turn"),
            "tool_name": str(payload.get("tool_name", "unknown")),
            "question": _summary_value(str(payload.get("question", "")), 240),
            "answer_chars": payload.get("answer_chars"),
            "approved": payload.get("approved"),
        }
    if event_type == "assistant_message":
        return {
            "model_turn": payload.get("turn"),
            "content": _summary_value(str(payload.get("content", ""))),
        }
    if event_type == "guardrail_triggered":
        return {
            "status": str(payload.get("status", "blocked")),
            "scope": str(payload.get("scope", "unknown")),
            "rule_id": str(payload.get("rule_id", "unknown")),
            "severity": str(payload.get("severity", "unknown")),
            "message": _summary_value(str(payload.get("message", ""))),
        }
    if event_type == "failure":
        return {
            "status": str(payload.get("status", "failed")),
            "failed_stage": _summary_value(str(payload.get("failed_stage", "unknown"))),
            "failure_category": _summary_value(str(payload.get("failure_category", "unknown"))),
            "reason": _summary_value(str(payload.get("reason", ""))),
        }
    return payload


def _tool_args_summary(tool_name: str, args: dict[str, object]) -> dict[str, object]:
    if tool_name in {"file_write", "code_run", "apply_patch", "apply_patch_set", "shell", "request_user_input"}:
        return interaction_args_summary(tool_name, args)
    if tool_name == "file_read":
        return {
            "path": _summary_value(str(args.get("path", "")), 160),
            "offset": args.get("offset"),
            "limit": args.get("limit"),
            "keyword": _summary_value(str(args.get("keyword", "")), 80),
        }
    return {"args_keys": sorted(str(key) for key in args)}


def _tool_result_summary(tool_name: str, result: dict[str, object]) -> dict[str, object]:
    if tool_name == "file_read":
        return {
            "path": _summary_value(str(result.get("path", "")), 160),
            "start_line": result.get("start_line"),
            "end_line": result.get("end_line"),
            "line_count": result.get("line_count"),
            "truncated": bool(result.get("truncated")),
        }
    if tool_name == "file_write":
        return {
            "path": _summary_value(str(result.get("path", "")), 160),
            "mode": result.get("mode"),
            "bytes_written": result.get("bytes_written"),
            "created": result.get("created"),
        }
    if tool_name == "code_run":
        return {
            "exit_code": result.get("exit_code"),
            "stdout_chars": len(str(result.get("stdout_excerpt", ""))),
            "stderr_chars": len(str(result.get("stderr_excerpt", ""))),
            "truncated": bool(result.get("truncated")),
        }
    if tool_name == "shell":
        return {
            "exit_code": result.get("exit_code"),
            "stdout_chars": len(str(result.get("stdout", ""))),
            "stderr_chars": len(str(result.get("stderr", ""))),
        }
    return {
        "status": str(result.get("status", "unknown")),
        "result_keys": sorted(str(key) for key in result),
    }


def _summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _resolve_session_path(session: str | Path, runs_root: Path) -> Path:
    raw = Path(session)
    if raw.is_absolute() or raw.exists() or raw.name != str(session):
        return raw.resolve()
    return (runs_root / "sessions" / str(session)).resolve()


def _read_session_metadata(session_path: Path) -> dict[str, object]:
    metadata_path = session_path / "session.json"
    if not metadata_path.exists():
        raise ChatSessionError(f"session package missing required file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ChatSessionError(f"invalid session.json: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ChatSessionError(f"invalid session.json: {metadata_path} must contain an object")
    required_fields = ["session_id", "workspace_root", "provider", "created_at", "updated_at", "turn_count"]
    for field_name in required_fields:
        if field_name not in metadata:
            raise ChatSessionError(f"invalid session.json: missing {field_name}")
    for field_name in ["session_id", "workspace_root", "provider", "created_at", "updated_at"]:
        if not isinstance(metadata[field_name], str):
            raise ChatSessionError(f"invalid session.json: {field_name} must be a string")
    if not isinstance(metadata["turn_count"], int) or isinstance(metadata["turn_count"], bool):
        raise ChatSessionError("invalid session.json: turn_count must be an integer")
    if str(metadata["session_id"]) != session_path.name:
        raise ChatSessionError("invalid session.json: session_id does not match session path")
    return metadata


def _read_session_turns(session_path: Path) -> list[dict[str, object]]:
    turns_path = session_path / "turns.jsonl"
    if not turns_path.exists():
        return []
    turns: list[dict[str, object]] = []
    for index, line in enumerate(turns_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ChatSessionError(f"invalid turns.jsonl line {index}") from error
        if not isinstance(record, dict):
            raise ChatSessionError(f"invalid turns.jsonl line {index}: must contain an object")
        for field_name in ["turn_index", "request", "summary", "status", "episode_path", "verification_status"]:
            if field_name not in record:
                raise ChatSessionError(f"invalid turns.jsonl line {index}: missing {field_name}")
        if not isinstance(record["turn_index"], int) or isinstance(record["turn_index"], bool):
            raise ChatSessionError(f"invalid turns.jsonl line {index}: turn_index must be an integer")
        for field_name in ["request", "summary", "status", "episode_path", "verification_status"]:
            if not isinstance(record[field_name], str):
                raise ChatSessionError(f"invalid turns.jsonl line {index}: {field_name} must be a string")
        turns.append(record)
    return turns


def _new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:8]
