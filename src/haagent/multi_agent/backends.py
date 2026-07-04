"""
haagent/multi_agent/backends.py - worker 执行后端接口

定义 worker backend 的最小执行契约，并保持默认 in-process 行为。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

from haagent.models.fake import FakeModelGateway


class WorkerBackend(Protocol):
    @property
    def backend_type(self) -> str:
        """返回后端类型。"""

    def spawn(self, runtime: Any, worker: Any, prompt: str) -> None:
        """启动 worker。"""


class InProcessWorkerBackend:
    @property
    def backend_type(self) -> str:
        return "in_process"

    def spawn(self, runtime: Any, worker: Any, prompt: str) -> None:
        runtime._start_worker_thread(worker, prompt)


class SubprocessWorkerBackend:
    @property
    def backend_type(self) -> str:
        return "subprocess"

    def spawn(self, runtime: Any, worker: Any, prompt: str) -> None:
        backend_dir = runtime.store._team_dir(worker.team_id) / "agents" / worker.agent_id / "subprocess"
        backend_dir.mkdir(parents=True, exist_ok=True)
        config_path = backend_dir / "config.json"
        result_path = backend_dir / "result.json"
        config = _subprocess_config(runtime, worker, prompt, result_path)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "haagent.multi_agent.subprocess_worker", str(config_path)],
            cwd=runtime.workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        worker.process = process
        watcher = threading.Thread(
            target=_watch_subprocess_worker,
            args=(runtime, worker, process, result_path),
            name=f"haagent-subprocess-watch-{worker.agent_id}",
            daemon=True,
        )
        worker.thread = watcher
        watcher.start()


BACKEND_REGISTRY: dict[str, WorkerBackend] = {
    "in_process": InProcessWorkerBackend(),
    "subprocess": SubprocessWorkerBackend(),
}


def _subprocess_config(runtime: Any, worker: Any, prompt: str, result_path: Path) -> dict[str, Any]:
    session = worker.session
    return {
        "workspace_root": str(session.workspace_root),
        "runs_root": str(runtime.runs_root),
        "prompt": prompt,
        "result_path": str(result_path),
        "session_id": session.session_id,
        "model_profile": session.model_profile_name,
        "max_turns": session.max_turns,
        "enable_web": session.enable_web,
        "allowed_tools": session._allowed_tools_override,
        "approval_allowed_tools": session._approval_allowed_tools_override,
        "approved_tools": session._approved_tools_override,
        "worker_context": session._worker_context,
        "gateway": _gateway_config(session.model_gateway),
    }


def _gateway_config(gateway: Any) -> dict[str, Any]:
    if isinstance(gateway, FakeModelGateway):
        response = gateway._response
        return {
            "type": "fake_response",
            "content": response.content,
            "tool_calls": [
                {"name": call.name, "args": call.args, "id": call.id}
                for call in response.tool_calls
            ],
        }
    return {
        "type": "provider_profile" if getattr(gateway, "provider_name", "") else "unsupported",
        "provider_name": getattr(gateway, "provider_name", ""),
    }


def _watch_subprocess_worker(runtime: Any, worker: Any, process: subprocess.Popen[str], result_path: Path) -> None:
    stdout, stderr = process.communicate()
    payload = _read_result_payload(result_path)
    if payload is None:
        status = "failed"
        summary = (stderr or stdout or f"subprocess exited with code {process.returncode}").strip()
        episode_path = ""
    else:
        status = str(payload.get("status", "failed"))
        summary = str(payload.get("final_response") or payload.get("reason") or status)
        episode_path = str(payload.get("episode_path", ""))
        if process.returncode not in (0, None) and status == "completed":
            status = "failed"
            summary = (stderr or f"subprocess exited with code {process.returncode}").strip()
    notification = runtime._notification(
        worker,
        status=status,
        summary=summary,
        result_excerpt=summary[:1000],
        episode_path=episode_path,
        error="" if status == "completed" else summary,
    )
    runtime.store.update_worker_status(
        worker.team_id,
        worker.agent_id,
        status,
        episode_path=episode_path,
        session_id=worker.session.session_id,
        restart_count=worker.restart_count,
    )
    runtime.store.append_notification(worker.team_id, notification)
    worker.notification = notification
    worker.done.set()
    record = runtime._worker_record(worker)
    runtime._emit_worker_event(
        "worker_completed" if status == "completed" else "worker_failed",
        worker,
        status=status,
        subagent_type=record.subagent_type if record is not None else "",
        description=record.description if record is not None else "",
    )


def _read_result_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
