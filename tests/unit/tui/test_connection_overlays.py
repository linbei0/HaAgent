"""
tests/unit/tui/test_connection_overlays.py - 连接配置与模型切换弹窗测试

验证供应商连接配置和目录模型切换在 TUI 结果层保持解耦。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from haagent.app.assistant_types import AssistantModelConnection
from haagent.models.model_ref import ModelChoice, ModelRef
from haagent.tui.overlays.connections import ConnectionCenterOverlay, ConnectionSetupWizard
from haagent.tui.overlays.models import ModelSwitchOverlay, ModelSwitchState
from textual.app import App
from textual.widgets import OptionList


def test_connection_setup_builds_connection_and_test_model_without_model_selection() -> None:
    provider = SimpleNamespace(
        id="requesty",
        name="Requesty",
        api_base_url="https://router.requesty.ai/v1",
        env_names=["REQUESTY_API_KEY"],
        provider_package="@ai-sdk/openai-compatible",
        models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
    )
    wizard = ConnectionSetupWizard([provider])
    wizard.step = "api_key"
    wizard.connection_name = "personal"

    result = wizard._new_connection_result("sk-test-secret")

    assert result is not None
    assert result.connection.id == "requesty-personal"
    assert result.connection.name == "personal"
    assert result.connection.gateway_provider == "openai-chat"
    assert result.connection.api_key == "sk-test-secret"
    assert result.test_model == "openai/gpt-5.2-chat"
    assert not hasattr(result, "selection")


def test_connection_setup_does_not_submit_empty_api_key() -> None:
    provider = SimpleNamespace(
        id="deepseek",
        name="DeepSeek",
        api_base_url="https://api.deepseek.com",
        env_names=["DEEPSEEK_API_KEY"],
        provider_package="@ai-sdk/openai-compatible",
        models=[SimpleNamespace(id="deepseek-chat", name="DeepSeek Chat")],
    )
    wizard = ConnectionSetupWizard([provider])
    wizard.step = "api_key"
    wizard.connection_name = "work"

    assert wizard._new_connection_result("   ") is None


def test_connection_setup_rejects_secret_like_connection_name() -> None:
    provider = SimpleNamespace(
        id="deepseek",
        name="DeepSeek",
        api_base_url="https://api.deepseek.com",
        env_names=["DEEPSEEK_API_KEY"],
        provider_package="@ai-sdk/openai-compatible",
        models=[SimpleNamespace(id="deepseek-chat", name="DeepSeek Chat")],
    )
    wizard = ConnectionSetupWizard([provider])

    assert wizard._accept_connection_name("sk-test-secret") is False
    assert wizard.step == "provider"
    assert wizard.connection_name == ""


def test_connection_center_overlay_uses_option_list_for_connections() -> None:
    async def run() -> None:
        overlay = ConnectionCenterOverlay([_connection("requesty-personal", "personal", "requesty")])
        app = App()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(overlay)

            assert overlay.query_one(OptionList).option_count == 1
            await pilot.press("t")
            await pilot.pause(0.1)

            assert overlay.state.selected_connection.id == "requesty-personal"

    asyncio.run(run())


def test_connection_center_header_guides_new_connection_for_all_providers() -> None:
    from haagent.tui.overlays.connections import ConnectionCenterState

    # state.render 与 overlay 头部共用引导文案；OptionList 只列已配置连接。
    rendered = ConnectionCenterState(
        connections=[_connection("deepseek-default", "default", "deepseek")]
    ).render()
    assert "已配置" in rendered
    assert "n 新建" in rendered
    assert "全部供应商" in rendered
    assert "下方仅显示已保存的连接" in rendered


def test_model_switch_state_expands_catalog_models_for_each_connection() -> None:
    state = ModelSwitchState(choices=[
        _choice(connection, model)
        for connection in ("requesty-personal", "requesty-work")
        for model in ("openai/gpt-5.2-chat", "anthropic/claude-sonnet-4.5")
    ])

    assert [(row.ref.connection_id, row.ref.model) for row in state.visible_rows] == [
        ("requesty-personal", "openai/gpt-5.2-chat"),
        ("requesty-personal", "anthropic/claude-sonnet-4.5"),
        ("requesty-work", "openai/gpt-5.2-chat"),
        ("requesty-work", "anthropic/claude-sonnet-4.5"),
    ]
    assert "Requesty / personal" in state.render()
    assert "Requesty / work" in state.render()


def test_model_switch_state_renders_only_a_small_visible_window() -> None:
    state = ModelSwitchState(
        choices=[_choice("requesty-personal", f"model-{index:03d}") for index in range(80)],
    )

    rendered = state.render()
    moved = state.move(40).render()

    assert "模型 1/80" in rendered
    assert "model-000" in rendered
    assert "model-079" not in rendered
    assert rendered.count("Requesty / personal") <= 15
    assert "模型 41/80" in moved
    assert "model-040" in moved
    assert "model-000" not in moved


def test_model_switch_overlay_ignores_escape_after_it_is_already_closed() -> None:
    async def run() -> None:
        overlay = ModelSwitchOverlay([_choice("requesty-personal", "model-001")])
        app = App()
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(overlay)
            await pilot.pause(0.1)
            overlay.dismiss(None)
            await pilot.pause(0.1)

            overlay.on_key(_FakeKeyEvent("escape"))

    asyncio.run(run())


def test_connection_setup_ignores_escape_after_it_is_already_closed() -> None:
    """卡顿后 Esc 可能在 screen 已 pop 后仍投递；不得再 dismiss 炸栈。"""

    async def run() -> None:
        overlay = ConnectionSetupWizard([_provider("p1", model_count=3)])
        app = App()
        async with app.run_test(size=(100, 30)) as pilot:
            await app.push_screen(overlay)
            await pilot.pause(0.05)
            overlay.dismiss(None)
            await pilot.pause(0.05)

            overlay.on_key(_FakeKeyEvent("escape"))

    asyncio.run(run())


def test_connection_setup_move_does_not_rebuild_option_list() -> None:
    """143+ provider 时每次 ↑/↓ 全量 set_options 会卡死；移动只改高亮。"""

    async def run() -> None:
        providers = [_provider(f"provider-{index:03d}", model_count=2) for index in range(80)]
        overlay = ConnectionSetupWizard(providers)
        app = App()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(overlay)
            await pilot.pause(0.05)
            option_list = overlay.query_one("#connection-setup-list", OptionList)
            assert option_list.option_count == 80
            calls = {"set_options": 0}
            original_set_options = option_list.set_options

            def counting_set_options(*args, **kwargs):
                calls["set_options"] += 1
                return original_set_options(*args, **kwargs)

            option_list.set_options = counting_set_options  # type: ignore[method-assign]

            overlay.on_key(_FakeKeyEvent("down"))
            await pilot.pause(0.05)

            assert overlay.provider_index == 1
            assert option_list.highlighted == 1
            assert calls["set_options"] == 0
            assert option_list.option_count == 80

    asyncio.run(run())


class _FakeKeyEvent:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _choice(connection_id: str, model: str) -> ModelChoice:
    return ModelChoice(
        ref=ModelRef(connection_id, model),
        connection_name="personal" if connection_id.endswith("personal") else "work",
        provider_name="Requesty",
        model_name=model,
    )


def _connection(connection_id: str, name: str, provider_id: str) -> AssistantModelConnection:
    return AssistantModelConnection(
        id=connection_id,
        name=name,
        provider_id=provider_id,
        provider_name="Requesty",
        gateway_provider="openai-chat",
        base_url="https://router.requesty.ai/v1",
        api_key_env="REQUESTY_API_KEY",
        credential_source="keyring",
        credential_available=True,
        credential_source_used="keyring",
        model_config_diagnostics=(),
    )


def _provider(provider_id: str, *, model_count: int = 1) -> SimpleNamespace:
    models = [
        SimpleNamespace(id=f"{provider_id}-model-{index}", name=f"Model {index}")
        for index in range(model_count)
    ]
    return SimpleNamespace(
        id=provider_id,
        name=provider_id.replace("-", " ").title(),
        api_base_url=f"https://example.com/{provider_id}",
        env_names=[f"{provider_id.upper().replace('-', '_')}_API_KEY"],
        provider_package="@ai-sdk/openai-compatible",
        models=models,
    )
