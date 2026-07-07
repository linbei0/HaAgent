"""
src/haagent/multi_agent/runtime.py - 进程内 worker 调度运行时

由 coordinator 的 ToolRouter 调用，负责启动独立 AgentSession、记录 team 状态与完成通知。
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haagent.models.gateway import ModelGateway
from haagent.models.gateway_registry import gateway_from_profile
from haagent.models.model_connections import (
    ModelSelection,
    ProviderProfile,
    ProviderProfileError,
    load_active_model_selection,
    load_model_selection_profile,
    user_config_dir,
)
from haagent.multi_agent.backends import BACKEND_REGISTRY, WorkerBackend
from haagent.multi_agent.messages import WorkerMessage, WorkerNotification, WorkerPermissionRequest
from haagent.multi_agent.permissions import WorkerType, worker_tool_policy
from haagent.multi_agent.profiles import WorkerProfileRuntime, resolve_worker_profile
from haagent.multi_agent.team_store import MailboxMessage, TeamStore, WorkerRecord
from haagent.multi_agent.worktree import WorktreeLease, create_worktree
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.execution.human_interaction import interaction_args_summary
from haagent.runtime.execution.policy import PolicyDecision
from haagent.runtime.execution.path_policy import PathPolicy
from haagent.runtime.settings import DEFAULT_INTERACTIVE_MAX_TURNS


RESTART_STATUS_NOTE = "Task restarted; prior interactive context was not preserved."
GatewayFactory = Callable[[ProviderProfile], ModelGateway]


@dataclass
class _WorkerTask:
    agent_id: str
    task_id: str
    team_id: str
    session: Any
    backend: str = "in_process"
    process: Any | None = None
    worktree_lease: WorktreeLease | None = None
    thread: threading.Thread | None = None
    done: threading.Event = field(default_factory=threading.Event)
    notification: dict[str, Any] | None = None
    restart_count: int = 0
    permission_request_to_consume: str = ""
    parent_step_id: str = ""


class _EmptyMcpRuntime:
    def list_tools(self) -> list[Any]:
        return []

    def list_statuses(self) -> list[Any]:
        return []

    def close(self) -> None:
        return None


class MultiAgentRuntime:
    _task_registry: dict[tuple[str, str, str], dict[str, _WorkerTask]] = {}
    _registry_lock = threading.RLock()

    def __init__(
        self,
        *,
        runs_root: Path,
        workspace_root: Path,
        leader_session_id: str,
        model_gateway: ModelGateway,
        path_policy: PathPolicy,
        inherited_allowed_tools: list[str],
        inherited_approval_allowed_tools: list[str],
        inherited_approved_tools: list[str],
        event_sink: Any,
        interaction_handler: HumanInteractionHandler | None,
        enable_web: bool,
        mcp_tool_names: list[str],
        tool_registry: Any,
        mcp_runtime: Any,
        team_root: Path | None = None,
        worker_max_turns: int | None = DEFAULT_INTERACTIVE_MAX_TURNS,
        environ: Mapping[str, str] | None = None,
        gateway_factory: GatewayFactory = gateway_from_profile,
        parent_task_step_id: str = "",
    ) -> None:
        self.runs_root = runs_root
        self.workspace_root = workspace_root
        self.leader_session_id = leader_session_id
        self.model_gateway = model_gateway
        self.path_policy = path_policy
        self.inherited_allowed_tools = list(inherited_allowed_tools)
        self.inherited_approval_allowed_tools = list(inherited_approval_allowed_tools)
        self.inherited_approved_tools = list(inherited_approved_tools)
        self.event_sink = event_sink
        self.interaction_handler = interaction_handler
        self.enable_web = enable_web
        self.mcp_tool_names = list(mcp_tool_names)
        self.tool_registry = tool_registry
        self.mcp_runtime = mcp_runtime
        self.worker_max_turns = worker_max_turns
        self.environ = environ
        self.gateway_factory = gateway_factory
        self.parent_task_step_id = parent_task_step_id
        self.store = TeamStore(team_root or (user_config_dir() / "teams"))
        self._profile_resolver = resolve_worker_profile
        self._scope = (
            str(self.store.root.resolve()),
            str(self.workspace_root.resolve()),
            self.leader_session_id,
        )
        with self._registry_lock:
            self._tasks = self._task_registry.setdefault(self._scope, {})
        self._lock = threading.RLock()

    def spawn_worker(
        self,
        *,
        description: str,
        prompt: str,
        subagent_type: WorkerType,
        team_id: str | None = None,
        model_profile: str | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        try:
            worker_profile = self._profile_resolver(
                profile,
                fallback_subagent_type=subagent_type,
            )
        except ValueError as error:
            return {"is_error": True, "error": str(error)}
        resolved_model_profile = model_profile or worker_profile.model_profile
        team = self.store.ensure_team(
            team_id=team_id or f"team-{self.leader_session_id}",
            workspace_root=self.workspace_root,
            leader_session_id=self.leader_session_id,
        )
        agent_id = f"{worker_profile.subagent_type}-{uuid.uuid4().hex[:8]}"
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        worktree_lease: WorktreeLease | None = None
        worker_workspace_root = self.workspace_root
        try:
            backend = self._backend_for(worker_profile.backend)
            if worker_profile.worktree:
                worktree_lease = create_worktree(
                    self.workspace_root,
                    slug=f"{worker_profile.name}-{agent_id}",
                )
                worker_workspace_root = worktree_lease.worktree_path
            session = self._create_worker_session(
                agent_id=agent_id,
                task_id=task_id,
                team_id=team.team_id,
                workspace_root=worker_workspace_root,
                worker_profile=WorkerProfileRuntime(
                    name=worker_profile.name,
                    subagent_type=worker_profile.subagent_type,
                    system_prompt=worker_profile.system_prompt,
                    model_profile=resolved_model_profile,
                    allowed_tools=worker_profile.allowed_tools,
                    approval_allowed_tools=worker_profile.approval_allowed_tools,
                    approved_tools=worker_profile.approved_tools,
                    max_turns=worker_profile.max_turns,
                    enable_web=worker_profile.enable_web,
                    backend=worker_profile.backend,
                    worktree=worker_profile.worktree,
                ),
            )
        except ValueError as error:
            return {"is_error": True, "error": str(error)}
        record = WorkerRecord(
            agent_id=agent_id,
            task_id=task_id,
            subagent_type=worker_profile.subagent_type,
            description=description,
            status="running",
            session_id=session.session_id,
            profile=worker_profile.name,
            model_profile=resolved_model_profile or "",
            parent_step_id=self.parent_task_step_id,
        )
        self.store.upsert_worker(team.team_id, record)
        worker = _WorkerTask(
            agent_id=agent_id,
            task_id=task_id,
            team_id=team.team_id,
            session=session,
            backend=backend.backend_type,
            worktree_lease=worktree_lease,
            parent_step_id=self.parent_task_step_id,
        )
        with self._lock:
            self._tasks[task_id] = worker
        self._emit_worker_event(
            "worker_started",
            worker,
            status="running",
            subagent_type=worker_profile.subagent_type,
            description=description,
        )
        backend.spawn(self, worker, prompt)
        return {
            "agent_id": agent_id,
            "task_id": task_id,
            "team_id": team.team_id,
            "status": "running",
            "profile": worker_profile.name,
            "backend": backend.backend_type,
            "parent_step_id": self.parent_task_step_id,
            **({"process_id": worker.process.pid} if worker.process is not None else {}),
            **({"worktree_path": str(worktree_lease.worktree_path)} if worktree_lease is not None else {}),
        }

    def send_message(self, to: str, message: str) -> dict[str, Any]:
        worker = self._find_worker(to)
        if worker is None:
            return {"is_error": True, "error": f"unknown agent: {to}"}
        if worker.thread is not None and worker.thread.is_alive():
            if not worker.done.is_set():
                self.store.write_worker_message(
                    worker.team_id,
                    worker.agent_id,
                    WorkerMessage(sender="coordinator", recipient=worker.agent_id, content=message),
                )
                return {
                    "agent_id": worker.agent_id,
                    "task_id": worker.task_id,
                    "status": "queued",
                }
            worker.thread.join(timeout=1)
            if worker.thread.is_alive():
                return {"is_error": True, "error": f"agent {to} is still finishing"}
        self.store.write_mailbox(
            worker.team_id,
            worker.agent_id,
            MailboxMessage.user_message(
                sender="coordinator",
                recipient=worker.agent_id,
                content=message,
            ),
        )
        worker.done.clear()
        worker.notification = None
        worker.restart_count += 1
        record = self._worker_record(worker)
        if record is None:
            return {"is_error": True, "error": f"task record not found for agent: {to}"}
        worker.session = self._create_worker_session(
            agent_id=worker.agent_id,
            task_id=worker.task_id,
            team_id=worker.team_id,
            worker_profile=self._resolve_record_profile(record, model_profile=record.model_profile or None),
            restart_count=worker.restart_count,
        )
        thread = threading.Thread(
            target=self._run_worker,
            args=(worker, message),
            name=f"haagent-worker-{worker.agent_id}",
            daemon=True,
        )
        worker.thread = thread
        self.store.update_worker_status(
            worker.team_id,
            worker.agent_id,
            "running",
            session_id=worker.session.session_id,
            restart_count=worker.restart_count,
            status_note=RESTART_STATUS_NOTE,
        )
        self.store.append_notification(
            worker.team_id,
            self._notification(
                worker,
                status="running",
                summary=RESTART_STATUS_NOTE,
                result_excerpt="",
                error="",
            ),
        )
        self._emit_worker_event(
            "worker_started",
            worker,
            status="running",
            subagent_type=record.subagent_type if record is not None else "",
            description=record.description if record is not None else "",
        )
        thread.start()
        return {
            "agent_id": worker.agent_id,
            "task_id": worker.task_id,
            "status": "running",
            "restarted": True,
            "restart_count": worker.restart_count,
            "status_note": RESTART_STATUS_NOTE,
        }

    def approve_permission(self, request_id: str, *, message: str = "") -> dict[str, Any]:
        try:
            request = self.store.resolve_permission_request(
                request_id,
                approved=True,
                response_message=message,
            )
        except ValueError as error:
            return {"is_error": True, "error": str(error)}
        worker = self._find_worker_by_task_id(request.task_id)
        if worker is None:
            return {"is_error": True, "error": f"unknown task: {request.task_id}"}
        record = self._worker_record(worker)
        if record is None:
            return {"is_error": True, "error": f"task record not found for agent: {request.agent_id}"}
        worker.done.clear()
        worker.notification = None
        worker.restart_count += 1
        try:
            worker.session = self._create_worker_session(
                agent_id=worker.agent_id,
                task_id=worker.task_id,
                team_id=worker.team_id,
                worker_profile=self._resolve_record_profile(record, model_profile=record.model_profile or None),
                restart_count=worker.restart_count,
                approved_tools_extra=[request.tool_name],
            )
        except ValueError as error:
            return {"is_error": True, "error": str(error)}
        worker.permission_request_to_consume = request.request_id
        thread = threading.Thread(
            target=self._run_worker,
            args=(worker, _permission_resolution_prompt(request, approved=True)),
            name=f"haagent-worker-{worker.agent_id}",
            daemon=True,
        )
        worker.thread = thread
        self.store.update_worker_status(
            worker.team_id,
            worker.agent_id,
            "running",
            session_id=worker.session.session_id,
            restart_count=worker.restart_count,
            status_note=f"Permission {request.request_id} approved; worker resumed.",
        )
        self.store.append_notification(
            worker.team_id,
            self._notification(
                worker,
                status="running",
                summary=f"Permission {request.request_id} approved; worker resumed.",
            ),
        )
        self._emit_worker_event(
            "worker_started",
            worker,
            status="running",
            subagent_type=record.subagent_type,
            description=record.description,
        )
        thread.start()
        return {
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "task_id": request.task_id,
            "status": "approved",
        }

    def reject_permission(self, request_id: str, *, message: str = "") -> dict[str, Any]:
        try:
            request = self.store.resolve_permission_request(
                request_id,
                approved=False,
                response_message=message,
            )
        except ValueError as error:
            return {"is_error": True, "error": str(error)}
        worker = self._find_worker_by_task_id(request.task_id)
        if worker is None:
            return {"is_error": True, "error": f"unknown task: {request.task_id}"}
        summary = f"Permission {request.request_id} rejected for tool {request.tool_name}."
        if message.strip():
            summary = f"{summary} {message.strip()}"
        notification = self._notification(
            worker,
            status="failed",
            summary=summary,
            result_excerpt=summary,
            error=summary,
            request_id=request.request_id,
        )
        self.store.update_worker_status(worker.team_id, worker.agent_id, "failed")
        self.store.append_notification(worker.team_id, notification)
        self.store.consume_permission_request(request.request_id)
        worker.notification = notification
        worker.done.set()
        record = self._worker_record(worker)
        self._emit_worker_event(
            "worker_failed",
            worker,
            status="failed",
            subagent_type=record.subagent_type if record is not None else "",
            description=record.description if record is not None else "",
        )
        return {
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "task_id": request.task_id,
            "status": "rejected",
        }

    def stop_task(self, task_id: str, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            worker = self._tasks.get(task_id)
        if worker is None:
            return {"is_error": True, "error": f"unknown task: {task_id}"}
        worker.session.cancel_current_run()
        status = "stopped"
        self.store.update_worker_status(worker.team_id, worker.agent_id, status)
        notification = self._notification(worker, status=status, summary="worker stopped")
        worker.notification = notification
        worker.done.set()
        self.store.append_notification(worker.team_id, notification)
        record = self._worker_record(worker)
        self._emit_worker_event(
            "worker_stopped",
            worker,
            status=status,
            subagent_type=record.subagent_type if record is not None else "",
            description=record.description if record is not None else "",
        )
        return {"agent_id": worker.agent_id, "task_id": task_id, "status": status, "force": force}

    def wait_for_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            worker = self._tasks[task_id]
        worker.done.wait(timeout)
        if worker.thread is not None and worker.done.is_set():
            worker.thread.join(timeout=1)
        return worker.notification or {}

    def list_workers(self) -> list[dict[str, Any]]:
        teams = self.store.list_teams_for_leader(self.leader_session_id)
        result: list[dict[str, Any]] = []
        for team in teams:
            for worker in team.agents:
                result.append(
                    {
                        "team_id": team.team_id,
                        "agent_id": worker.agent_id,
                        "task_id": worker.task_id,
                        "subagent_type": worker.subagent_type,
                        "description": worker.description,
                        "status": worker.status,
                        "episode_path": worker.episode_path,
                        "restart_count": worker.restart_count,
                        "status_note": worker.status_note,
                    },
                )
        return result

    def task_get(self, task_id: str) -> dict[str, Any]:
        found = self._worker_record_by_task_id(task_id)
        if found is None:
            return {"is_error": True, "error": f"unknown task: {task_id}"}
        team_id, record = found
        return {"status": "success", "task": _worker_record_payload(team_id, record)}

    def task_list(self, *, status: str | None = None) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        for team in self.store.list_teams_for_leader(self.leader_session_id):
            for worker in team.agents:
                if status and worker.status != status:
                    continue
                tasks.append(_worker_record_payload(team.team_id, worker))
        return {"status": "success", "tasks": tasks}

    def task_output(self, task_id: str, *, max_chars: int = 12000) -> dict[str, Any]:
        found = self._worker_record_by_task_id(task_id)
        if found is None:
            return {"is_error": True, "error": f"unknown task: {task_id}"}
        _team_id, record = found
        output = _worker_output_text(record)
        if not output:
            output = "(no output)"
        max_chars = min(max(int(max_chars), 1), 50000)
        original_chars = len(output)
        if original_chars > max_chars:
            output = output[-max_chars:]
        return {
            "status": "success",
            "task_id": task_id,
            "agent_id": record.agent_id,
            "task_status": record.status,
            "episode_path": record.episode_path,
            "output": output,
            "truncated": original_chars > max_chars,
        }

    def _run_worker(self, worker: _WorkerTask, prompt: str) -> None:
        try:
            if worker.permission_request_to_consume:
                self.store.consume_permission_request(worker.permission_request_to_consume)
                worker.permission_request_to_consume = ""
            next_prompt = prompt
            while True:
                result = worker.session.run_prompt_events(
                    next_prompt,
                    event_sink=None,
                    include_session_events=False,
                    interaction_handler=self.interaction_handler,
                )
                queued = self.store.consume_worker_messages(worker.team_id, worker.agent_id)
                next_prompt = "\n".join(
                    message.content.strip()
                    for message in queued
                    if message.content.strip()
                )
                if not next_prompt:
                    break
            pending_request_id = _pending_request_id_from_result(result)
            status = (
                "completed"
                if result.status == "completed"
                else "awaiting_approval"
                if pending_request_id
                else "failed"
            )
            failure_summary = _failure_summary_from_episode(result)
            summary = _non_empty_text(result.final_response) or failure_summary or _non_empty_text(result.reason) or status
            notification = self._notification(
                worker,
                status=status,
                summary=summary,
                result_excerpt=summary[:1000],
                episode_path=str(result.episode_path),
                error="" if status == "completed" else failure_summary or _non_empty_text(result.reason) or status,
                request_id=pending_request_id,
            )
        except Exception as error:
            status = "failed"
            notification = self._notification(
                worker,
                status=status,
                summary=str(error),
                result_excerpt="",
                error=str(error),
            )
        self.store.update_worker_status(
            worker.team_id,
            worker.agent_id,
            status,
            episode_path=str(notification.get("episode_path", "")),
            session_id=worker.session.session_id,
            restart_count=worker.restart_count,
        )
        self.store.append_notification(worker.team_id, notification)
        worker.notification = notification
        worker.done.set()
        record = self._worker_record(worker)
        self._emit_worker_event(
            "worker_completed" if status == "completed" else "worker_failed",
            worker,
            status=status,
            subagent_type=record.subagent_type if record is not None else "",
            description=record.description if record is not None else "",
            episode_path=str(notification.get("episode_path", "")),
        )

    def _start_worker_thread(self, worker: _WorkerTask, prompt: str) -> None:
        thread = threading.Thread(
            target=self._run_worker,
            args=(worker, prompt),
            name=f"haagent-worker-{worker.agent_id}",
            daemon=True,
        )
        worker.thread = thread
        thread.start()

    def _backend_for(self, backend_type: str) -> WorkerBackend:
        try:
            return BACKEND_REGISTRY[backend_type]
        except KeyError as error:
            raise ValueError(f"unsupported backend: {backend_type}") from error

    def _notification(
        self,
        worker: _WorkerTask,
        *,
        status: str,
        summary: str,
        result_excerpt: str = "",
        episode_path: str = "",
        error: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        return WorkerNotification(
            event_type="worker_status",
            team_id=worker.team_id,
            agent_id=worker.agent_id,
            task_id=worker.task_id,
            status=status,
            summary=summary[:300],
            result_excerpt=result_excerpt[:1000],
            episode_path=episode_path,
            error=error or "",
            needs_attention=bool(error),
            request_id=request_id,
            parent_step_id=worker.parent_step_id,
            evidence_refs=tuple(_worker_evidence_refs(worker, episode_path=episode_path)),
        ).to_dict()

    def _find_worker(self, agent_id: str) -> _WorkerTask | None:
        with self._lock:
            for worker in self._tasks.values():
                if worker.agent_id == agent_id:
                    return worker
        return None

    def _find_worker_by_task_id(self, task_id: str) -> _WorkerTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _worker_record_by_task_id(self, task_id: str) -> tuple[str, WorkerRecord] | None:
        for team in self.store.list_teams_for_leader(self.leader_session_id):
            for worker in team.agents:
                if worker.task_id == task_id:
                    return team.team_id, worker
        return None

    def _worker_record(self, worker: _WorkerTask) -> WorkerRecord | None:
        team = self.store.load_team(worker.team_id)
        if team is None:
            return None
        for record in team.agents:
            if record.agent_id == worker.agent_id:
                return record
        return None

    def _create_worker_session(
        self,
        *,
        agent_id: str,
        task_id: str,
        team_id: str,
        worker_profile: WorkerProfileRuntime,
        workspace_root: Path | None = None,
        restart_count: int = 0,
        approved_tools_extra: list[str] | None = None,
    ) -> Any:
        policy = worker_tool_policy(
            worker_profile.subagent_type,
            inherited_allowed_tools=self.inherited_allowed_tools,
            inherited_approval_allowed_tools=self.inherited_approval_allowed_tools,
            inherited_approved_tools=self.inherited_approved_tools,
            web_enabled=self.enable_web,
            mcp_tool_names=self.mcp_tool_names,
        )
        allowed_tools = (
            list(worker_profile.allowed_tools)
            if worker_profile.allowed_tools is not None
            else policy.allowed_tools
        )
        approval_allowed_tools = (
            list(worker_profile.approval_allowed_tools)
            if worker_profile.approval_allowed_tools is not None
            else policy.approval_allowed_tools
        )
        approved_tools = (
            list(worker_profile.approved_tools)
            if worker_profile.approved_tools is not None
            else policy.approved_tools
        )
        for tool_name in approved_tools_extra or []:
            if tool_name not in approved_tools:
                approved_tools.append(tool_name)

        from haagent.runtime.session.agent import AgentSession

        suffix = "" if restart_count <= 0 else f"-restart{restart_count}"
        resolved_workspace_root = workspace_root or self.workspace_root
        return AgentSession(
            workspace_root=resolved_workspace_root,
            runs_root=self.runs_root,
            model_gateway=self._worker_gateway(worker_profile.model_profile),
            model_profile_name=worker_profile.model_profile,
            max_turns=worker_profile.max_turns or self.worker_max_turns,
            session_id=f"{self.leader_session_id}-{agent_id}{suffix}",
            memory_extraction_enabled=False,
            enable_web=self.enable_web if worker_profile.enable_web is None else worker_profile.enable_web,
            allowed_tools_override=allowed_tools,
            approval_allowed_tools_override=approval_allowed_tools,
            approved_tools_override=approved_tools,
            mcp_runtime=self.mcp_runtime or _EmptyMcpRuntime(),
            worker_context={
                "agent_id": agent_id,
                "agent_profile": worker_profile.name,
                "system_prompt": worker_profile.system_prompt,
                "leader_session_id": self.leader_session_id,
                "team_id": team_id,
                "inbox_enabled": True,
            },
            worker_permission_requester=lambda tool_name, args, policy_decision: self.create_permission_request(
                team_id=team_id,
                agent_id=agent_id,
                task_id=task_id,
                tool_name=tool_name,
                args=args,
                reason=policy_decision.approval.reason,
            ),
        )

    def create_permission_request(
        self,
        *,
        team_id: str,
        agent_id: str,
        task_id: str,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
    ) -> WorkerPermissionRequest:
        request = WorkerPermissionRequest(
            request_id=f"perm-{uuid.uuid4().hex[:12]}",
            team_id=team_id,
            agent_id=agent_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_args_summary=json.dumps(
                interaction_args_summary(tool_name, args),
                ensure_ascii=False,
                sort_keys=True,
            ),
            reason=reason,
            status="pending",
        )
        self.store.write_permission_request(request)
        self.store.update_worker_status(team_id, agent_id, "awaiting_approval")
        return request

    def _resolve_record_profile(
        self,
        record: WorkerRecord,
        *,
        model_profile: str | None,
    ) -> WorkerProfileRuntime:
        worker_profile = self._profile_resolver(
            record.profile or None,
            fallback_subagent_type=record.subagent_type,
        )
        if model_profile == worker_profile.model_profile:
            return worker_profile
        return WorkerProfileRuntime(
            name=worker_profile.name,
            subagent_type=worker_profile.subagent_type,
            system_prompt=worker_profile.system_prompt,
            model_profile=model_profile,
            allowed_tools=worker_profile.allowed_tools,
            approval_allowed_tools=worker_profile.approval_allowed_tools,
            approved_tools=worker_profile.approved_tools,
            max_turns=worker_profile.max_turns,
            enable_web=worker_profile.enable_web,
            backend=worker_profile.backend,
            worktree=worker_profile.worktree,
        )

    def _worker_gateway(self, model_profile: str | None) -> ModelGateway:
        if model_profile is None:
            return self.model_gateway
        try:
            active_selection = load_active_model_selection(config_dir=user_config_dir())
            profile = load_model_selection_profile(
                ModelSelection(connection_id=model_profile, model=active_selection.model),
                environ=self.environ,
                config_dir=user_config_dir(),
            )
        except ProviderProfileError as error:
            raise ValueError(str(error)) from error
        return self.gateway_factory(profile)

    def _emit_worker_event(
        self,
        event_type: str,
        worker: _WorkerTask,
        *,
        status: str,
        subagent_type: str,
        description: str,
        episode_path: str = "",
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            {
                "event_type": event_type,
                "agent_id": worker.agent_id,
                "task_id": worker.task_id,
                "team_id": worker.team_id,
                "subagent_type": subagent_type,
                "description": description,
                "status": status,
                "parent_step_id": worker.parent_step_id,
                "evidence_refs": _worker_evidence_refs(worker, episode_path=episode_path),
                "episode_path": episode_path,
            },
        )


def _failure_summary_from_episode(result: Any) -> str:
    if getattr(result, "status", "") == "completed":
        return ""
    reason = _non_empty_text(getattr(result, "reason", None))
    if reason:
        return reason
    episode_path_value = getattr(result, "episode_path", None)
    if not episode_path_value:
        return ""
    failure_path = Path(str(episode_path_value)) / "failure.json"
    try:
        data = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    failure = data.get("failure")
    if not isinstance(failure, dict):
        return ""
    evidence = _non_empty_text(failure.get("evidence"))
    category = _non_empty_text(failure.get("category"))
    if category and evidence:
        return f"{category}: {evidence}"
    return evidence or category


def _pending_request_id_from_result(result: Any) -> str:
    reason = _non_empty_text(getattr(result, "reason", None))
    if not reason:
        return ""
    prefix = "worker approval pending: "
    if not reason.startswith(prefix):
        return ""
    return reason[len(prefix):].strip()


def _permission_resolution_prompt(request: WorkerPermissionRequest, *, approved: bool) -> str:
    decision = "approved" if approved else "rejected"
    lines = [
        f"Permission request {request.request_id} was {decision}.",
        f"Tool: {request.tool_name}",
    ]
    if request.response_message.strip():
        lines.append(f"Leader message: {request.response_message.strip()}")
    if approved:
        lines.append("Continue the task from the approval boundary and report the result.")
    else:
        lines.append("Stop the task and report the denial explicitly.")
    return "\n".join(lines)


def _worker_record_payload(team_id: str, worker: WorkerRecord) -> dict[str, Any]:
    return {
        "team_id": team_id,
        "agent_id": worker.agent_id,
        "task_id": worker.task_id,
        "subagent_type": worker.subagent_type,
        "description": worker.description,
        "status": worker.status,
        "session_id": worker.session_id,
        "episode_path": worker.episode_path,
        "restart_count": worker.restart_count,
        "status_note": worker.status_note,
        "profile": worker.profile,
        "model_profile": worker.model_profile,
        "parent_step_id": worker.parent_step_id,
        "updated_at": worker.updated_at,
    }


def _worker_evidence_refs(worker: _WorkerTask, *, episode_path: str = "") -> list[str]:
    refs = [f"worker={worker.agent_id}", f"task={worker.task_id}"]
    if episode_path:
        refs.append(f"episode={episode_path}")
    return refs


def _worker_output_text(worker: WorkerRecord) -> str:
    parts: list[str] = []
    if worker.status_note:
        parts.append(worker.status_note)
    if worker.episode_path:
        episode = Path(worker.episode_path)
        parts.extend(_model_response_texts(episode / "transcript.jsonl"))
        failure = _failure_text(episode / "failure.json")
        if failure:
            parts.append(failure)
    return "\n".join(part for part in parts if part.strip())


def _model_response_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    texts: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "model_response":
            continue
        content = _non_empty_text(record.get("content"))
        if content:
            texts.append(content)
    return texts


def _failure_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    failure = data.get("failure")
    if not isinstance(failure, dict):
        return ""
    category = _non_empty_text(failure.get("category"))
    evidence = _non_empty_text(failure.get("evidence"))
    if category and evidence:
        return f"{category}: {evidence}"
    return evidence or category


def _non_empty_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text
