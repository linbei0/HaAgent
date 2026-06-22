"""
haagent/runtime/chat_session.py - 自然语言 Agent 会话

管理 chat 会话状态，并把每条用户请求转成可审计的临时 task contract。
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from haagent.models.gateway import ModelGateway
from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.orchestrator import RunOrchestrator


CHAT_ALLOWED_TOOLS = ["file_list", "file_search", "file_read", "apply_patch", "shell"]
CHAT_APPROVED_TOOLS = ["apply_patch", "shell"]
CHAT_MAX_TURNS = 20
SESSION_SUMMARY_CHAR_LIMIT = 1000


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

    @property
    def provider_name(self) -> str:
        if self.model_gateway is None:
            return "fake"
        return self.model_gateway.provider_name

    def run_prompt(self, prompt: str) -> ChatTurnResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must be non-empty")

        with tempfile.TemporaryDirectory(prefix="haagent-chat-") as task_dir:
            task_path = Path(task_dir) / "task.yaml"
            _write_chat_task_yaml(task_path, clean_prompt, self.workspace_root)
            result = RunOrchestrator(
                runs_root=self.runs_root,
                model_gateway=self.model_gateway,
                max_turns=self.max_turns,
                session_summary=self.summary_text(),
            ).run(task_path)

        turn_result = self._build_turn_result(clean_prompt, result)
        self.turn_count += 1
        self._summaries.append(_turn_summary(clean_prompt, turn_result))
        self._summaries = _bounded_summaries(self._summaries)
        return turn_result

    def status(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "provider": self.provider_name,
            "turn_count": self.turn_count,
        }

    def new(self) -> None:
        self.session_id = _new_session_id()
        self.turn_count = 0
        self._summaries = []

    def summary_text(self) -> str | None:
        if not self._summaries:
            return None
        return "\n".join(_bounded_summaries(self._summaries))

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
            verification_status="not_run",
            failed_stage=str(failure.get("stage", "none")),
            failure_category=str(failure.get("category", "none")),
            reason=str(failure.get("evidence", "none")),
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
            "approved_tools": list(CHAT_APPROVED_TOOLS),
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


def _summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:8]
