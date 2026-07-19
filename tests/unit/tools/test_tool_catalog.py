"""
tests/unit/tools/test_tool_catalog.py - 静态 ToolCatalog 契约测试

验证 contribution 同源登记、safety 必填，以及 chat tags 与 handler 对齐。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from haagent.runtime.execution.path_policy import default_path_policy
from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.catalog import ToolCatalog, ToolContribution, ToolRuntimeDeps
from haagent.tools.registry import TOOL_REGISTRY


def _handler(_: dict[str, Any], __: Any = None) -> dict[str, Any]:
    return {"status": "success"}


def test_default_catalog_matches_tool_registry_and_handlers(tmp_path: Path) -> None:
    from haagent.tools.catalog import default_tool_catalog
    from haagent.tools.handler_factory import build_static_tool_handlers

    catalog = default_tool_catalog()
    assert catalog.definitions == TOOL_REGISTRY
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


def test_candidate_tools_uses_catalog_tags() -> None:
    from haagent.tools.access import ToolAccessManager
    from haagent.tools.catalog import default_tool_catalog

    catalog = default_tool_catalog()
    candidate = ToolAccessManager.candidate_tools(
        catalog=catalog,
        enable_web=True,
        include_memory_tool=True,
        image_attachment_history=True,
        mcp_tool_names=[],
    )
    assert candidate[: len(catalog.chat_default_tools())] == catalog.chat_default_tools()
    assert set(catalog.chat_web_tools()).issubset(candidate)
    assert set(catalog.chat_skill_tools()).issubset(candidate)
    assert "load_image_attachment" in candidate


def test_default_candidate_tool_set_preserves_agent_capabilities_and_resource_gates() -> None:
    from haagent.tools.access import ToolAccessManager
    from haagent.tools.catalog import default_tool_catalog

    catalog = default_tool_catalog()
    normal = ToolAccessManager.candidate_tools(
        catalog=catalog,
        enable_web=False,
        include_memory_tool=False,
        image_attachment_history=False,
        mcp_tool_names=[],
    )
    online = ToolAccessManager.candidate_tools(
        catalog=catalog,
        enable_web=True,
        include_memory_tool=False,
        image_attachment_history=False,
        mcp_tool_names=[],
    )

    expected_default = set(catalog.chat_default_tools()) - {"start_memory_update"}
    assert expected_default.issubset(normal)
    assert {"code_run", "apply_patch", "apply_patch_set"}.issubset(normal)
    assert {"agent", "send_message", "task_get", "task_list", "task_output", "task_stop"}.issubset(normal)
    assert {"skill_list", "skill_read", "skill_market_search"}.issubset(normal)
    assert set(online) - set(normal) == {"web_search", "web_fetch"}
    assert "load_image_attachment" not in normal
    assert "start_memory_update" not in normal

    fully_enabled = ToolAccessManager.candidate_tools(
        catalog=catalog,
        enable_web=True,
        include_memory_tool=True,
        image_attachment_history=False,
        mcp_tool_names=[],
    )
    assert "start_memory_update" in fully_enabled


def test_contribution_requires_explicit_replay_safety() -> None:
    with pytest.raises(TypeError):
        ToolContribution(  # type: ignore[call-arg]
            name="missing_replay",
            description="x",
            parameters={"type": "object", "properties": {}, "required": []},
            risk_level="low",
            execution_effect="read_only",
            router_owned=True,
        )


def test_catalog_rejects_missing_binder_and_duplicate() -> None:
    base = dict(
        description="x",
        parameters={"type": "object", "properties": {}, "required": []},
        risk_level="low",
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        router_owned=True,
    )
    with pytest.raises(ValueError, match="duplicate tool contribution"):
        ToolCatalog(
            [
                ToolContribution(name="a", **base),
                ToolContribution(name="a", **base),
            ],
        )
    with pytest.raises(ValueError, match="must declare bind_handler"):
        ToolCatalog(
            [
                ToolContribution(
                    name="no_binder",
                    description="x",
                    parameters={"type": "object", "properties": {}, "required": []},
                    risk_level="low",
                    execution_effect="read_only",
                    replay_safety=ReplaySafety.NEVER_REPLAY,
                ),
            ],
        )


def test_static_tools_declare_explicit_replay_safety() -> None:
    from haagent.tools.catalog import default_tool_catalog

    catalog = default_tool_catalog()
    replayable = {
        "file_list",
        "file_read",
        "grep",
        "web_search",
        "web_fetch",
        "skill_list",
        "skill_read",
        "skill_market_search",
        "list_mcp_resources",
        "read_mcp_resource",
        "load_image_attachment",
    }
    for name in catalog.names():
        contribution = catalog.get(name)
        expected = ReplaySafety.SAFE_TO_REPLAY if name in replayable else ReplaySafety.NEVER_REPLAY
        assert contribution.replay_safety is expected
        assert TOOL_REGISTRY[name].replay_safety is expected


def test_deleting_contribution_removes_schema_and_projector() -> None:
    from haagent.tools.contributions import all_static_contributions

    remaining = [item for item in all_static_contributions() if item.name != "file_read"]
    catalog = ToolCatalog(remaining)
    assert "file_read" not in catalog.definitions
    assert catalog.project_observation("file_read", {}, {}) is None
    deps = ToolRuntimeDeps(
        workspace_root=Path("."),
        path_policy=default_path_policy(Path(".")),
        router_handlers={
            name: _handler
            for name in catalog.names()
            if catalog.get(name).router_owned
        },
    )
    handlers = catalog.build_static_handlers(deps)
    assert "file_read" not in handlers


def test_grep_observation_exposes_partial_state_without_raw_stderr() -> None:
    from haagent.tools.catalog import default_tool_catalog

    observation = default_tool_catalog().project_observation(
        "grep",
        {"pattern": "needle"},
        {
            "status": "success",
            "pattern": "needle",
            "matches": [],
            "partial": True,
            "warnings": [{"type": "permission_denied", "message": "Skipped 1 inaccessible path."}],
            "skipped_paths": [".tmp/pytest"],
            "guidance": "Narrow path or include and retry.",
            "stderr": "large raw diagnostic must not be projected",
        },
    )

    assert observation is not None
    assert observation["partial"] is True
    assert observation["warnings"] == [
        {"type": "permission_denied", "message": "Skipped 1 inaccessible path."},
    ]
    assert observation["skipped_paths"] == [".tmp/pytest"]
    assert "stderr" not in observation


def test_file_write_binder_uses_execution_context_interaction(tmp_path: Path) -> None:
    """写工具 binder 必须把逐次 interaction_handler 传给实现，不能只为集合对齐。"""
    from haagent.runtime.execution.human_interaction import HumanInteractionResponse
    from haagent.tools.catalog import ToolExecutionContext, default_tool_catalog

    catalog = default_tool_catalog()
    contribution = catalog.get("file_write")
    assert contribution.bind_handler is not None
    handler = contribution.bind_handler(
        ToolRuntimeDeps(
            workspace_root=tmp_path,
            path_policy=default_path_policy(tmp_path),
        ),
    )
    requests: list[Any] = []

    def interaction_handler(request: Any) -> HumanInteractionResponse:
        requests.append(request)
        return HumanInteractionResponse(approved=False, answer="no")

    result = handler(
        {"path": "notes.txt", "content": "hello", "mode": "create"},
        ToolExecutionContext(interaction_handler=interaction_handler),
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert len(requests) == 1
    assert requests[0].interaction_type == "edit_diff"
    assert not (tmp_path / "notes.txt").exists()
