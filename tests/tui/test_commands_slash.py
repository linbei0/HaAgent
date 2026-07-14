"""
tests/tui/test_commands_slash.py - HaAgent TUI commands 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.commands import command_registry, parse_slash_command
from haagent.tui.design.renderers import status_line
from haagent.tui.widgets import PromptInput

from tests.tui.support import FakeAssistantService, _all_text, _text

def test_tui_status_renderer_shows_sandbox_state(tmp_path: Path) -> None:
    status = FakeAssistantService(workspace_root=tmp_path / "sandbox").workspace.status()

    line = status_line(status, ui_state="idle", width=140)

    assert "sandbox:degraded" in line

def test_tui_slash_command_registry_parses_known_and_unknown_commands() -> None:
    registry = command_registry()

    result = parse_slash_command("/sessions", registry)
    model = parse_slash_command("/model", registry)
    models = parse_slash_command("/models", registry)
    unknown = parse_slash_command("/wat", registry)
    not_command = parse_slash_command(" /help", registry)

    assert result is not None
    assert result.command is not None
    assert result.command.name == "sessions"
    assert result.argument == ""
    assert model.command.action == "open_models"
    assert model.command.name == "model"
    assert models.command.action == "open_models"
    assert models.command.name == "model"
    assert unknown.command is None
    assert unknown.error == "未知命令：/wat"
    assert not_command is None
    assert parse_slash_command("/review 看看改动", registry) is None
    assert parse_slash_command("/debug", registry) is None
    assert parse_slash_command("/verify", registry) is None
    assert {command.name for command in registry.commands()} >= {
        "help",
        "sessions",
        "compact",
        "memory",
        "skills",
        "skill",
        "sandbox",
        "new",
        "resume",
        "model",
        "mcp",
        "agents",
        "web",
        "permissions",
        "review",
        "debug",
        "verify",
    }

def test_tui_prompt_pack_command_suggestion_fills_prompt_input(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.pause(0.1)
            app.completion_flow.command_overlay.update_query("rev")

            app.action_accept_command_suggestion()

            input_widget = app.query_one("#prompt-input", PromptInput)
            assert app._prompt_value(input_widget) == "/review "
            assert service.prompts == []
            assert "未知命令" not in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_prompt_pack_command_is_submitted_to_chat_runtime(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            app._set_prompt_value(input_widget, "/review 看看改动")

            app._submit_prompt(input_widget)
            await asyncio.to_thread(service.started.wait, 2)

            assert service.prompts == ["/review 看看改动"]
            assert "未知命令" not in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_sandbox_command_shows_status_doctor_and_updates_settings(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")

            input_widget.value = "/sandbox"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "当前沙箱：local_subprocess" in conversation
            assert "haagent sandbox enable docker" in conversation

            input_widget.value = "/sandbox doctor"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "Docker CLI: not_checked" in conversation
            assert "docker sandbox disabled" in conversation

            input_widget.value = "/sandbox enable docker"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert service.sandbox_enabled_count == 1
            assert "Docker 沙箱已启用" in conversation
            assert "新 session 生效" in conversation
            assert "sandbox:docker" in _text(app, "#status-bar")

            input_widget.value = "/sandbox disable"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert service.sandbox_disabled_count == 1
            assert "已恢复 local_subprocess" in conversation
            assert "sandbox:degraded" in _text(app, "#status-bar")

    asyncio.run(run())

def test_tui_agents_command_lists_current_workers(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        agents=[
            {
                "agent_id": "explorer-1",
                "task_id": "task-1",
                "team_id": "team-session-test",
                "subagent_type": "explorer",
                "description": "Inspect project",
                "status": "running",
            },
            {
                "agent_id": "verification-1",
                "task_id": "task-2",
                "team_id": "team-session-test",
                "subagent_type": "verification",
                "description": "Run tests",
                "status": "completed",
            },
        ],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/agents"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "Workers" in conversation
            assert "explorer-1" in conversation
            assert "running" in conversation
            assert "Inspect project" in conversation
            assert "verification-1" in conversation
            assert "completed" in conversation
            assert service.prompts == []

    asyncio.run(run())

def test_tui_slash_command_registry_includes_mcp() -> None:
    registry = command_registry()

    assert registry.get("mcp") is not None
    assert "models" not in {command.name for command in registry.commands()}

def test_tui_skills_command_lists_skills_and_trusts_project_roots(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
        blocked_project_skill_roots=[str(tmp_path / ".haagent" / "skills")],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            before = _text(app, "#conversation")

            input_widget.value = "/skills"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "Skills" in _all_text(app)
            assert "review" in _all_text(app)
            assert _text(app, "#conversation") == before
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == ""
            assert service.prompts == []
            await pilot.press("escape")
            await pilot.pause(0.1)

            input_widget.value = "/skills trust"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.trusted_skills_count == 1
            assert "已信任当前 workspace 的项目 skills" in _text(app, "#conversation")

    asyncio.run(run_test())

def test_tui_skills_search_lists_marketplace_results(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skills_sh-1",
                provider="skills_sh",
                name="analyze-csv",
                source="office",
                summary="Analyze CSV files.",
                detail_url="https://skills.sh/office/analyze-csv",
                installable=True,
                quality={"installs": 1234},
            ),
            SimpleNamespace(
                result_id="skillsmp-2",
                provider="skillsmp",
                name="csv-helper",
                source="data-team",
                summary="Find CSV workflows.",
                detail_url="https://skillsmp.com/skills/csv-helper",
                installable=False,
                quality={"stars": 42},
            ),
        ],
        marketplace_warnings=["skillsmp search failed: HTTP 502"],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills search csv"
            await pilot.press("enter")

            conversation = _text(app, "#conversation")
            assert service.searched_marketplace_queries == [("csv", None, 10)]
            assert "analyze-csv" in conversation
            assert "skills_sh-1" in conversation
            assert "skills_sh" in conversation
            assert "可安装" in conversation
            assert "csv-helper" in conversation
            assert "skillsmp" in conversation
            assert "暂不支持直接安装" in conversation
            assert "skillsmp search failed: HTTP 502" in conversation

    asyncio.run(run_test())

def test_tui_skills_install_installs_cached_marketplace_result(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skills_sh-1",
                provider="skills_sh",
                name="analyze-csv",
                source="office",
                summary="Analyze CSV files.",
                detail_url="https://skills.sh/office/analyze-csv",
                installable=True,
                quality={},
            ),
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills install skills_sh-1"
            await pilot.press("enter")
            await pilot.pause()

            assert service.installed_marketplace_ids == []
            assert "安装远端 skill" in _all_text(app)
            await pilot.press("y")
            await pilot.pause()

            conversation = _text(app, "#conversation")
            assert service.installed_marketplace_ids == ["skills_sh-1"]
            assert "已安装 marketplace skill：analyze-csv" in conversation
            assert "命令：$analyze-csv" in conversation
            assert "https://skills.sh/office/analyze-csv" in conversation

    asyncio.run(run_test())

def test_tui_skills_install_reports_marketplace_errors(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skillsmp-1",
                provider="skillsmp",
                name="csv-helper",
                source="data-team",
                summary="Find CSV workflows.",
                detail_url="https://skillsmp.com/skills/csv-helper",
                installable=False,
                quality={},
            ),
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills install skillsmp-1"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert service.installed_marketplace_ids == ["skillsmp-1"]
            assert "skills 操作失败" in _text(app, "#conversation")
            assert "only skills_sh results are installable" in _text(app, "#conversation")

            input_widget.value = "/skills wat"
            await pilot.press("enter")
            assert "/skills search <query>" in _text(app, "#conversation")
            assert "/skills install <result-id>" in _text(app, "#conversation")

    asyncio.run(run_test())

def test_tui_skill_command_starts_prompt_with_explicit_skill_context(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skill review check this"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.read_skill_names == ["review"]
            assert service.prompts
            assert service.prompts[0].startswith("Use skill review explicitly.")
            assert "Follow this workflow." in service.prompts[0]
            assert "check this" in service.prompts[0]

    asyncio.run(run_test())

def test_tui_skill_command_without_name_opens_skill_picker(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
            {
                "name": "csv-helper",
                "description": "CSV analysis workflow.",
                "source": "user",
                "command_name": "csv-helper",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skill"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "选择 Skill" in _all_text(app)
            assert "2 skills" in _all_text(app)
            assert "review" in _all_text(app)
            assert "csv-helper" in _all_text(app)

            await pilot.press("c", "s", "v")
            await pilot.pause(0.1)
            assert "搜索: csv" in _all_text(app)
            assert "csv-helper" in _all_text(app)
            assert "review" not in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == "/skill csv-helper "
            assert service.prompts == []

    asyncio.run(run_test())


def test_tui_web_command_toggles_networking_inside_app(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "web:off" in _text(app, "#status-bar")

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/web on"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is False
            assert "用法：/web" in _text(app, "#conversation")

            prompt_input.value = "/web"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is True
            assert "web:on" in _text(app, "#status-bar")
            assert "联网已开启" in _text(app, "#conversation")

            prompt_input.value = "/web"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is False
            assert "web:off" in _text(app, "#status-bar")
            assert "联网已关闭" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_status_bar_is_compact_at_80_and_120_columns(tmp_path: Path) -> None:
    long_workspace = tmp_path / "very" / "long" / "workspace-name-that-should-not-fill-the-status-bar"
    long_model = "provider-model-name-with-many-segments-and-context-window-very-long"
    long_session = "session-20260627-abcdef1234567890abcdef1234567890"

    async def run_80() -> None:
        service = FakeAssistantService(
            workspace_root=long_workspace,
            model=long_model,
            current_session_id=long_session,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            status = _text(app, "#status-bar")
            assert len(status) <= 80
            assert "ws:" in status
            assert "profile: local" in status
            assert "openai-chat/" in status
            assert "key: ok" in status
            assert "sid:" in status
            assert "turn:" in status
            assert "state: idle" in status
            assert str(long_workspace) not in status
            assert long_model not in status
            assert long_session not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert list(app.query("#side-bar")) == []

    async def run_120() -> None:
        service = FakeAssistantService(
            workspace_root=long_workspace,
            model=long_model,
            current_session_id=long_session,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            assert len(status) <= 120
            assert "ws:" in status
            assert "profile: local" in status
            assert "key: ok" in status
            assert "sid:" in status
            assert str(long_workspace) not in status
            assert long_model not in status
            assert long_session not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert list(app.query("#side-bar")) == []

    asyncio.run(run_80())
    asyncio.run(run_120())

def test_tui_mcp_command_renders_server_status(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        mcp_status={
            "configured_count": 2,
            "connected_count": 1,
            "failed_count": 1,
            "servers": [
                {
                    "name": "fixture",
                    "state": "connected",
                    "detail": "",
                    "tool_count": 1,
                    "resource_count": 1,
                },
                {
                    "name": "broken",
                    "state": "failed",
                    "detail": "connection refused",
                    "tool_count": 0,
                    "resource_count": 0,
                },
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/mcp"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "MCP servers:" in conversation
            assert "fixture: connected (tools: 1, resources: 1)" in conversation
            assert "broken: failed - connection refused" in conversation

    asyncio.run(run())

