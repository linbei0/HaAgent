"""
tests/unit/tools/test_tool_access.py - 工具访问统一快照测试

验证模型可见工具和运行时能力过滤使用同一份 ToolAccessManager 结果。
"""

from pathlib import Path
from types import SimpleNamespace

from haagent.models.capabilities import ModelCapabilities
from haagent.tools.access import ToolAccessManager
from haagent.tools.registry import default_tool_runtime_registry


class _Mcp:
    def __init__(self, state: str) -> None:
        self._state = state

    def list_statuses(self):
        return [SimpleNamespace(state=self._state)]


def _resolve(tmp_path: Path, requested: list[str], *, mcp=None):
    return ToolAccessManager.resolve(
        requested,
        registry=default_tool_runtime_registry(),
        workspace_root=tmp_path,
        mcp_runtime=mcp,
        model_capabilities=ModelCapabilities(vision="supported"),
        skill_catalog=None,
        image_attachment_history=False,
    )


def test_process_tools_remain_visible_and_are_controlled_by_tool_approval(tmp_path: Path) -> None:
    snapshot = _resolve(tmp_path, ["file_read", "shell", "code_run"])

    assert snapshot.allowed_tools == ("file_read", "shell", "code_run")


def test_mcp_tools_require_connected_runtime(tmp_path: Path) -> None:
    unavailable = _resolve(tmp_path, ["list_mcp_resources"], mcp=_Mcp("failed"))
    connected = _resolve(tmp_path, ["list_mcp_resources"], mcp=_Mcp("connected"))

    assert unavailable.allowed_tools == ()
    assert unavailable.denied_tools["list_mcp_resources"] == "mcp_unavailable"
    assert connected.allowed_tools == ("list_mcp_resources",)


def test_unsupported_vision_removes_attachment_tool(tmp_path: Path) -> None:
    snapshot = ToolAccessManager.resolve(
        ["load_image_attachment"],
        registry=default_tool_runtime_registry(),
        workspace_root=tmp_path,
        mcp_runtime=None,
        model_capabilities=ModelCapabilities(vision="unsupported"),
        skill_catalog=None,
        image_attachment_history=True,
    )

    assert snapshot.allowed_tools == ()
    assert snapshot.denied_tools["load_image_attachment"] == "vision_unsupported"
