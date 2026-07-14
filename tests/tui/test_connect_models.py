"""
tests/tui/test_connect_models.py - HaAgent TUI connect 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from haagent.models.capabilities import ModelCapabilities
from haagent.models.local_runtime import LocalRuntimeDiscovery, LocalRuntimeModel
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.overlays.models import LocalRuntimeOverlay, ModelCatalogLoadingOverlay
from textual.widgets import OptionList

from tests.tui.support import FakeAssistantService, _all_text, _connection_record, _text

def test_tui_connect_overlay_deletes_connection_after_confirmation(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        ),
        SimpleNamespace(
            name="local",
            provider="openai-chat",
            model="deepseek-chat",
            active=True,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        ),
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("d")
            await pilot.pause(0.1)
            assert "删除模型连接：router" in _all_text(app)

            await pilot.press("n")
            await pilot.pause(0.1)
            assert service.deleted_model_connection is None

            await pilot.press("d")
            await pilot.pause(0.1)
            await pilot.press("y")
            await pilot.pause(0.1)

            assert service.deleted_model_connection == "router"
            text = _all_text(app)
            assert "模型连接已删除：router" in text
            assert all(_connection_record(connection).id != "router" for connection in service.model_connections)
            assert "local" in text

    asyncio.run(run())

def test_tui_connect_wizard_masks_key_saves_connection_and_tests_without_switching_model(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("p", "e", "r", "s", "o", "n", "a", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "t", "e", "s", "t", "-", "s", "e", "c", "r", "e", "t")
            await pilot.pause(0.1)
            assert "sk-test-secret" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.configured_model_connection.id == "requesty-personal"
            assert service.configured_model_connection.name == "personal"
            assert service.configured_model_connection.gateway_provider == "openai-chat"
            assert service.switched_model_connection is None
            assert service.switched_model is None
            assert service.default_model_selection is None
            assert service.tested_model_connection == "requesty-personal"
            assert service.tested_model == "openai/gpt-5.2-chat"
            assert service.configured_api_key == "sk-test-secret"

    asyncio.run(run())

def test_tui_connect_wizard_rejects_api_key_as_connection_name(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="deepseek",
            name="DeepSeek",
            env_names=["DEEPSEEK_API_KEY"],
            api_base_url="https://api.deepseek.com",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="deepseek-chat", name="DeepSeek Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "t", "e", "s", "t", "-", "s", "e", "c", "r", "e", "t")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_connection is None
            assert service.switched_model_connection is None
            assert "连接名不能是 API key" in _all_text(app)

    asyncio.run(run())

def test_tui_connect_keeps_loading_overlay_visible_while_catalog_loads(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_refresh_release = threading.Event()
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            text = _all_text(app)
            assert isinstance(app.screen, ModelCatalogLoadingOverlay)
            assert "模型目录" in text
            assert "正在读取模型目录" in text

            service.catalog_refresh_release.set()
            await pilot.pause(0.2)
            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 0
            assert "provider: Requesty" in _all_text(app)

    asyncio.run(run())

def test_tui_connect_reuses_in_memory_catalog(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.press("n")
            await pilot.pause(0.2)
            await pilot.press("escape")
            await pilot.pause(0.1)
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.press("n")
            await pilot.pause(0.2)

            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 0

    asyncio.run(run())

def test_tui_connect_refreshes_once_when_cached_catalog_has_no_configurable_models(
    tmp_path: Path,
) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.cached_catalog_providers = [
        SimpleNamespace(
            id="fresh",
            name="Fresh",
            env_names=["FRESH_API_KEY"],
            api_base_url="https://fresh.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[],
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.3)

            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 1
            text = _all_text(app)
            assert "provider: Requesty" in text
            assert "模型目录没有可配置模型" not in text

    asyncio.run(run())

def test_tui_connect_reports_empty_catalog_instead_of_opening_empty_wizard(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.2)

            text = _all_text(app)
            assert "没有可配置的目录模型" not in text
            assert "模型目录没有可配置模型" in text
            assert "请刷新目录或检查网络" in text

    asyncio.run(run())

def test_tui_connect_wizard_scrolls_provider_window_with_selection(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id=f"provider-{index:02d}",
            name=f"Provider {index:02d}",
            env_names=[f"PROVIDER_{index:02d}_API_KEY"],
            api_base_url=f"https://provider-{index:02d}.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id=f"model-{index:02d}", name=f"Model {index:02d}")],
        )
        for index in range(15)
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.2)
            for _ in range(13):
                await pilot.press("down")
            await pilot.pause(0.1)

            option_list = app.screen.query_one("#connection-setup-list", OptionList)
            assert option_list.highlighted == 13
            assert "Provider 14/15" in _all_text(app)

    asyncio.run(run())

def test_tui_connect_wizard_scrolls_test_model_window_with_selection(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="many",
            name="Many Models",
            env_names=["MANY_API_KEY"],
            api_base_url="https://many.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[
                SimpleNamespace(id=f"model-{index:02d}", name=f"Model {index:02d}")
                for index in range(20)
            ],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            for _ in range(17):
                await pilot.press("down")
            await pilot.pause(0.1)

            option_list = app.screen.query_one("#connection-setup-list", OptionList)
            assert option_list.highlighted == 17
            assert "模型 18/20" in _all_text(app)

    asyncio.run(run())

def test_tui_connect_overlay_runs_connection_test_without_showing_secret(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.connection_test_result = SimpleNamespace(
        ok=True,
        profile_name="router",
        provider="openai-chat",
        model="openai/gpt-5.2-chat",
        message="OK",
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("t")
            await pilot.pause(0.2)
            assert service.tested_model_connection == "router"
            assert "OK" in _all_text(app)
            assert "sk-test-secret" not in _all_text(app)

    asyncio.run(run())

def test_tui_connect_overlay_refreshes_catalog_via_worker(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            models=[],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("c", "o", "n", "n", "e", "c", "t", "enter")
            await pilot.pause(0.1)
            await pilot.press("r")
            await pilot.pause(0.2)
            assert service.refreshed_catalog_count >= 1
            assert "模型目录已刷新：1 个 provider" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_model_overlay_lists_catalog_models_for_each_configured_connection(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = [
        SimpleNamespace(
            id="requesty-personal",
            name="personal",
            provider_id="requesty",
            provider_name="Requesty",
            gateway_provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            api_key_env="REQUESTY_API_KEY",
            credential_source="keyring",
            credential_available=True,
        ),
        SimpleNamespace(
            id="requesty-work",
            name="work",
            provider_id="requesty",
            provider_name="Requesty",
            gateway_provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            api_key_env="REQUESTY_API_KEY",
            credential_source="keyring",
            credential_available=True,
        ),
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            models=[
                SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat"),
                SimpleNamespace(id="anthropic/claude-sonnet-4.5", name="Claude Sonnet 4.5"),
            ],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.2)

            text = _all_text(app)
            assert "Requesty / personal" in text
            assert "Requesty / work" in text
            assert text.count("openai/gpt-5.2-chat") >= 2
            assert "API key" not in text
            assert "model-api-key" not in text

            await pilot.press("down")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.switched_model_connection == "requesty-work"
            assert service.switched_model == "openai/gpt-5.2-chat"
            assert service.default_model_selection is None

            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("p")
            await pilot.pause(0.1)
            assert service.default_model_selection.connection_id == "requesty-personal"
            assert service.default_model_selection.model == "openai/gpt-5.2-chat"

    asyncio.run(run())

def test_tui_model_overlay_without_connections_guides_to_connect(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_connections = []
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.2)

            assert "请先 /connect" in _all_text(app)
            assert service.switched_model_connection is None

    asyncio.run(run())


def test_local_runtime_overlay_renders_through_textual_widget_pipeline(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    discovery = LocalRuntimeDiscovery(
        runtime_kind="ollama",
        base_url="http://127.0.0.1:11434/v1",
        status="available",
        models=(
            LocalRuntimeModel(
                id="qwen3:1.7b",
                name="qwen3:1.7b",
                loaded=True,
                capabilities=ModelCapabilities(
                    tools="supported",
                    streaming="supported",
                    vision="unsupported",
                    reasoning="supported",
                    tools_mode="native",
                    context_window_tokens=32768,
                    protocols=frozenset({"responses", "chat_completions"}),
                ),
            ),
        ),
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(LocalRuntimeOverlay((discovery,)))
            await pilot.pause(0.1)
            assert "qwen3:1.7b" in _all_text(app)
            assert "tools=native" in _all_text(app)

    asyncio.run(run())

