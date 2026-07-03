"""
tests/unit/multi_agent/test_permissions.py - worker 权限分层测试

验证 explorer、worker 与 verification 的工具集合符合多智能体 v1 约束。
"""

from haagent.multi_agent.permissions import worker_tool_policy


def test_worker_tool_policy_layers_tools_by_agent_type() -> None:
    inherited_tools = [
        "file_list",
        "file_search",
        "file_read",
        "file_write",
        "apply_patch",
        "shell",
        "code_run",
        "web_fetch",
    ]

    explorer = worker_tool_policy(
        "explorer",
        inherited_allowed_tools=inherited_tools,
        inherited_approval_allowed_tools=["file_write", "shell", "code_run"],
        inherited_approved_tools=[],
        web_enabled=True,
        mcp_tool_names=["mcp__docs__search"],
    )
    assert explorer.allowed_tools == [
        "file_list",
        "file_search",
        "file_read",
        "skill_list",
        "skill_read",
        "web_search",
        "web_fetch",
        "mcp__docs__search",
        "list_mcp_resources",
        "read_mcp_resource",
    ]
    assert explorer.approval_allowed_tools == []
    assert explorer.approved_tools == []

    worker = worker_tool_policy(
        "worker",
        inherited_allowed_tools=inherited_tools,
        inherited_approval_allowed_tools=["file_write", "shell", "code_run"],
        inherited_approved_tools=["file_write"],
        web_enabled=False,
        mcp_tool_names=[],
    )
    assert worker.allowed_tools == inherited_tools
    assert worker.approval_allowed_tools == ["file_write", "shell", "code_run"]
    assert worker.approved_tools == ["file_write"]

    verification = worker_tool_policy(
        "verification",
        inherited_allowed_tools=inherited_tools,
        inherited_approval_allowed_tools=["shell", "code_run"],
        inherited_approved_tools=["shell"],
        web_enabled=False,
        mcp_tool_names=[],
    )
    assert verification.allowed_tools == ["file_read", "file_search", "shell", "code_run"]
    assert verification.approval_allowed_tools == ["shell", "code_run"]
    assert verification.approved_tools == ["shell"]
