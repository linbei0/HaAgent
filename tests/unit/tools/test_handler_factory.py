"""
tests/unit/tools/test_handler_factory.py - 静态工具 handler 组合测试

验证 handler factory 与静态工具注册表保持严格对齐。
"""

from pathlib import Path
from typing import Any

from haagent.runtime.execution.path_policy import default_path_policy
from haagent.tools.registry import TOOL_REGISTRY


def _handler(_: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success"}


def test_static_handler_factory_has_one_handler_for_every_static_tool(tmp_path: Path) -> None:
    from haagent.tools.handler_factory import build_static_tool_handlers

    handlers = build_static_tool_handlers(
        workspace_root=tmp_path,
        path_policy=default_path_policy(tmp_path),
        skill_settings=None,
        cancellation_token=None,
        mcp_runtime=None,
        sandbox_backend=None,
        router_handlers={
            "fake_tool": _handler,
            "load_image_attachment": _handler,
            "agent": _handler,
            "send_message": _handler,
            "task_stop": _handler,
            "task_get": _handler,
            "task_list": _handler,
            "task_output": _handler,
            "request_user_input": _handler,
            "start_memory_update": _handler,
        },
    )

    assert set(handlers) == set(TOOL_REGISTRY)
