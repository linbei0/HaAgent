"""
src/haagent/multi_agent/subprocess_worker.py - subprocess worker 入口

读取父进程写出的 worker 配置，在独立 Python 进程中运行 AgentSession 并回写结果。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from haagent.models.fake import FakeModelGateway
from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.types import McpSettings
from haagent.models.types import ModelResponse, ToolCall
from haagent.models.gateway_registry import gateway_from_profile
from haagent.models.model_connections import (
    ModelSelection,
    load_active_model_selection,
    load_model_selection_profile,
    load_providers_config_snapshot,
    user_config_dir,
)
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.execution.retry import RetryController
from haagent.runtime.settings import load_runtime_settings


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write("usage: python -m haagent.multi_agent.subprocess_worker CONFIG\n")
        return 2
    config_path = Path(args[0])
    result_path: Path | None = None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("worker config must contain a mapping")
        result_path = Path(str(payload["result_path"]))
        result = run_from_payload(payload)
        _write_result(result_path, result)
        return 0 if result["status"] == "completed" else 1
    except Exception as error:
        if result_path is not None:
            _write_result(
                result_path,
                {
                    "status": "failed",
                    "final_response": "",
                    "reason": f"{type(error).__name__}: {error}",
                    "episode_path": "",
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            sys.stderr.write(traceback.format_exc())
        return 1


def run_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session = AgentSession(
        workspace_root=Path(str(payload["workspace_root"])),
        runs_root=Path(str(payload["runs_root"])),
        model_gateway=_build_gateway(payload),
        model_profile_name=_optional_str(payload.get("model_profile")),
        max_turns=_optional_int(payload.get("max_turns")),
        session_id=str(payload["session_id"]),
        memory_extraction_enabled=False,
        enable_web=bool(payload.get("enable_web", False)),
        allowed_tools_override=_optional_str_list(payload.get("allowed_tools")),
        approval_allowed_tools_override=_optional_str_list(payload.get("approval_allowed_tools")),
        approved_tools_override=_optional_str_list(payload.get("approved_tools")),
        mcp_runtime=SyncMcpRuntime(McpSettings()),
        worker_context=_optional_mapping(payload.get("worker_context")),
    )
    result = session.run_prompt_events(
        str(payload["prompt"]),
        event_sink=None,
        include_session_events=False,
        interaction_handler=None,
    )
    return {
        "status": result.status,
        "final_response": result.final_response,
        "reason": result.reason,
        "episode_path": str(result.episode_path),
    }


def _build_gateway(payload: dict[str, Any]):
    gateway = payload.get("gateway")
    if isinstance(gateway, dict) and gateway.get("type") == "fake_response":
        tool_calls = [
            ToolCall(
                name=str(item["name"]),
                args=dict(item.get("args", {})),
                id=str(item.get("id", "")),
            )
            for item in gateway.get("tool_calls", [])
            if isinstance(item, dict)
        ]
        return FakeModelGateway(
            ModelResponse(
                content=str(gateway.get("content", "")),
                tool_calls=tool_calls,
            ),
        )
    model_profile = _optional_str(payload.get("model_profile"))
    if model_profile is None:
        raise ValueError("subprocess worker requires a serializable gateway or model_profile")
    active_selection = load_active_model_selection(config_dir=user_config_dir())
    selection = ModelSelection(connection_id=model_profile, model=active_selection.model)
    snapshot = load_providers_config_snapshot(user_config_dir() / "providers.json")
    return gateway_from_profile(
        load_model_selection_profile(selection, snapshot=snapshot),
        retry_controller=RetryController(load_runtime_settings().model_retry),
    )


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("expected a list of strings")
    return list(value)


def _optional_mapping(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return dict(value)


if __name__ == "__main__":
    raise SystemExit(main())
