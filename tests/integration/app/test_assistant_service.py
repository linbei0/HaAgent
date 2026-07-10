"""
tests/integration/app/test_assistant_service.py - AssistantService 应用服务层测试

验证 TUI 前置服务层能复用模型连接、session 和事件流能力，且不暴露真实 API key。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from haagent.app.assistant_service import (
    AssistantService,
    AssistantServiceError,
    ModelConnectionConfigureRequest,
    ModelSelectionRequest,
)
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.credentials import FakeCredentialStore
from haagent.models.types import ModelResponse
from haagent.models.openai_chat import OpenAIChatCompletionsGateway
from haagent.models import model_connections
from haagent.models.gateway_registry import gateway_from_profile
from haagent.models.model_connections import (
    ModelSelection,
    ProviderConnectionRecord,
    list_provider_connection_records,
    load_active_model_selection,
    save_active_model_selection,
    save_provider_connection,
)
from haagent.runtime.events import AssistantMessageEvent, SessionLifecycleEvent, TaskProgressEvent
from haagent.runtime.session.attachments import ImageAttachment
from haagent.skills.marketplace import MarketplaceProvider, MarketplaceSearchResult, MarketplaceSkillCard


class RecordingGateway:
    provider_name = "openai-chat"

    def __init__(self, profile_name: str = "local") -> None:
        self.profile_name = profile_name
        self.model_inputs: list[object] = []

    def generate(self, messages, tool_schemas):
        task_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        self.model_inputs.append(task_content)
        return ModelResponse("done", [])


class BlockingGateway:
    provider_name = "openai-chat"

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release

    def generate(self, messages, tool_schemas):
        self.entered.set()
        self.release.wait(timeout=2)
        return ModelResponse("done", [])


class RecordingSession:
    provider_name = "openai-chat"
    turn_count = 0

    def __init__(
        self,
        *,
        workspace_root: Path,
        runs_root: Path,
        model_gateway,
        model_profile_name: str | None = None,
        model_connection_id: str | None = None,
        model_name: str | None = None,
        model_base_url: str | None = None,
        max_turns: int | None,
        enable_web: bool = False,
    ) -> None:
        self.session_id = "session-from-default-registry"
        self.workspace_root = workspace_root
        self.runs_root = runs_root
        self.model_gateway = model_gateway
        self.model_profile_name = model_profile_name
        self.model_connection_id = model_connection_id
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.max_turns = max_turns
        self.enable_web = enable_web
        self.session_path = runs_root / self.session_id

    def switch_model_gateway(
        self,
        *,
        profile_name: str,
        model_connection_id: str | None = None,
        provider: str,
        model: str,
        base_url: str,
        gateway,
    ) -> None:
        self.model_gateway = gateway
        self.model_profile_name = profile_name
        self.model_connection_id = model_connection_id
        self.model_name = model
        self.model_base_url = base_url

    def compact_current_session(self):
        return type(
            "SessionCompactResult",
            (),
            {
                "applied": True,
                "reason": "applied",
                "original_turn_count": 8,
                "compacted_turn_count": 2,
                "preserved_recent_count": 6,
                "saved_chars": 900,
            },
        )()

    def set_max_turns(self, max_turns: int | None) -> None:
        self.max_turns = max_turns


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _write_user_connection(
    home: Path,
    *,
    name: str = "local",
    provider: str = "openai-chat",
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    api_key_env: str = "DEEPSEEK_API_KEY",
    credential_source: str = "keyring",
) -> None:
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": name,
                        "name": name,
                        "provider_id": name,
                        "provider_name": name,
                        "gateway_provider": provider,
                        "base_url": base_url,
                        "api_key_env": api_key_env,
                        "credential_source": credential_source,
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )
    (config_dir / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": name, "model": model}}),
        encoding="utf-8",
    )


def _write_two_connections(config_dir: Path) -> None:
    save_provider_connection(
        ProviderConnectionRecord(
            id="local",
            name="local",
            provider_id="deepseek",
            provider_name="DeepSeek",
            gateway_provider="openai-chat",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_connection(
        ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_model_selection(ModelSelection("local", "deepseek-chat"), config_dir=config_dir)


def _session_image_attachment(session_path: Path, attachment_id: str) -> ImageAttachment:
    image_path = session_path / "attachments" / f"{attachment_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_png_bytes())
    return ImageAttachment.from_file(
        image_path,
        session_root=session_path,
        attachment_id=attachment_id,
    )


def _image_attachment_paths(content) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        str(part.get("relative_path"))
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_attachment"
    ]


def _text_part(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", ""))
    return ""


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00{@\xe8\xdd\x00\x00\x00\x0cIDATx\x9cc\xfc\xcf"
        b"\x00\x02\x00\x06\x08\x01\x01Z\xcf\x06H\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _service(
    tmp_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    gateway_factory=None,
    config_dir: Path | None = None,
) -> AssistantService:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    if config_dir is not None:
        config_dir.mkdir(parents=True, exist_ok=True)
    return AssistantService(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        environ=environ or {},
        gateway_factory=gateway_factory or (lambda profile: RecordingGateway(profile.name)),
    )


def test_service_lists_model_connections_with_credential_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_connection(
        ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    service = _service(tmp_path, config_dir=config_dir, environ={"OPENROUTER_API_KEY": "sk-router"})

    connections = service.list_model_connections()

    assert len(connections) == 1
    assert connections[0].id == "router"
    assert connections[0].name == "router"
    assert connections[0].provider_name == "OpenRouter"
    assert connections[0].gateway_provider == "openai-chat"


def test_service_lists_no_connections_from_obsolete_provider_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(
        json.dumps({"profiles": [{"name": "old", "model": "old-model"}]}),
        encoding="utf-8",
    )
    service = _service(tmp_path, config_dir=config_dir)

    assert service.list_model_connections() == []


def test_service_configures_connection_over_obsolete_provider_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(
        json.dumps({"profiles": [{"name": "old", "model": "old-model"}]}),
        encoding="utf-8",
    )
    store = FakeCredentialStore()
    monkeypatch.setattr(model_connections, "DEFAULT_CREDENTIAL_STORE", store)
    service = _service(tmp_path, config_dir=config_dir)

    service.configure_model_connection(
        ModelConnectionConfigureRequest(
            id="requesty-work",
            name="work",
            provider_id="requesty",
            provider_name="Requesty",
            gateway_provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            api_key_env="REQUESTY_WORK_API_KEY",
            credential_source="keyring",
            api_key="sk-work",
        )
    )

    provider_config = json.loads((config_dir / "providers.json").read_text(encoding="utf-8"))
    assert "profiles" not in provider_config
    assert provider_config["version"] == 2
    assert provider_config["connections"][0]["id"] == "requesty-work"
    assert store.values["connection:requesty-work"] == "sk-work"


def test_service_searches_skill_marketplace_and_caches_results(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    calls: list[tuple[str, list[str] | None, int]] = []

    def fake_search_marketplace(query: str, *, providers=None, limit: int = 10):
        calls.append((query, providers, limit))
        return MarketplaceSearchResult(
            status="success",
            query=query,
            cards=[
                MarketplaceSkillCard(
                    provider=MarketplaceProvider.SKILLS_SH,
                    result_id="skills_sh-1",
                    remote_id="cowork-os/cowork-os/analyze-csv",
                    name="analyze-csv",
                    source="cowork-os/cowork-os",
                    summary="Analyze CSV files.",
                    detail_url="https://skills.sh/cowork-os/cowork-os/analyze-csv",
                    installable=True,
                ),
                MarketplaceSkillCard(
                    provider=MarketplaceProvider.SKILLSMP,
                    result_id="skillsmp-2",
                    remote_id="openai-csv-workbench",
                    name="csv-workbench",
                    source="openai",
                    summary="Analyze CSV files.",
                    detail_url="https://skillsmp.com/creators/openai/csv-workbench",
                    installable=False,
                ),
            ],
            warnings=[],
        )

    monkeypatch.setattr("haagent.app.assistant_service.search_marketplace", fake_search_marketplace)

    result = service.search_skill_marketplace("csv", providers=["skills_sh"], limit=3)

    assert calls == [("csv", ["skills_sh"], 3)]
    assert result.status == "success"
    assert [item.result_id for item in result.results] == ["skills_sh-1", "skillsmp-2"]
    assert result.results[0].installable is True
    assert result.results[1].installable is False


def test_service_installs_cached_skills_sh_marketplace_result(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    service = _service(tmp_path)
    monkeypatch.setattr(
        "haagent.app.assistant_service.search_marketplace",
        lambda query, *, providers=None, limit=10: MarketplaceSearchResult(
            status="success",
            query=query,
            cards=[
                MarketplaceSkillCard(
                    provider=MarketplaceProvider.SKILLS_SH,
                    result_id="skills_sh-1",
                    remote_id="cowork-os/cowork-os/analyze-csv",
                    name="analyze-csv",
                    source="cowork-os/cowork-os",
                    summary="Analyze CSV files.",
                    detail_url="https://skills.sh/cowork-os/cowork-os/analyze-csv",
                    installable=True,
                ),
            ],
            warnings=[],
        ),
    )
    service.search_skill_marketplace("csv")

    installed = service.install_marketplace_skill("skills_sh-1")

    assert installed.command_name == "analyze-csv"
    assert installed.skill_file.exists()
    assert "source: marketplace" in installed.skill_file.read_text(encoding="utf-8")
    skill_names = [skill["name"] for skill in service.list_skills().skills]
    assert "analyze-csv" in skill_names


def test_service_rejects_unknown_marketplace_result_id(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(AssistantServiceError, match="unknown marketplace result id"):
        service.install_marketplace_skill("skills_sh-404")


def test_service_rejects_skillsmp_marketplace_install(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(
        "haagent.app.assistant_service.search_marketplace",
        lambda query, *, providers=None, limit=10: MarketplaceSearchResult(
            status="success",
            query=query,
            cards=[
                MarketplaceSkillCard(
                    provider=MarketplaceProvider.SKILLSMP,
                    result_id="skillsmp-1",
                    remote_id="openai-csv-workbench",
                    name="csv-workbench",
                    source="openai",
                    summary="Analyze CSV files.",
                    detail_url="https://skillsmp.com/creators/openai/csv-workbench",
                    installable=False,
                ),
            ],
            warnings=[],
        ),
    )
    service.search_skill_marketplace("csv")

    with pytest.raises(AssistantServiceError, match="only skills_sh results are installable"):
        service.install_marketplace_skill("skillsmp-1")


def test_service_sets_default_model_selection_without_switching_current_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_connection(
        ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="env",
        ),
        config_dir=config_dir,
    )
    service = _service(tmp_path, config_dir=config_dir, environ={"OPENROUTER_API_KEY": "sk-router"})

    service.set_default_model_selection(ModelSelectionRequest("router", "openai/gpt-5.2-chat"))

    assert service.current_session() is None
    assert load_active_model_selection(config_dir=config_dir) == ModelSelection("router", "openai/gpt-5.2-chat")


def test_service_compacts_current_session_through_session_boundary(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = AssistantService(
        workspace_root=tmp_path / "workspace",
        runs_root=tmp_path / ".runs",
        environ={"DEEPSEEK_API_KEY": "sk-test"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
        session_cls=RecordingSession,
    )
    service.workspace_root.mkdir()
    service.create_session()

    result = service.compact_current_session()

    assert result.applied is True
    assert result.reason == "applied"
    assert result.original_turn_count == 8
    assert result.compacted_turn_count == 2
    assert result.preserved_recent_count == 6
    assert result.saved_chars == 900


def test_service_configures_model_connection_with_keyring_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    store = FakeCredentialStore()
    monkeypatch.setattr(model_connections, "DEFAULT_CREDENTIAL_STORE", store)
    service = _service(tmp_path)

    record = service.configure_model_connection(
        ModelConnectionConfigureRequest(
            id="requesty-personal",
            name="personal",
            provider_id="requesty",
            provider_name="Requesty",
            gateway_provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            api_key_env="REQUESTY_API_KEY",
            credential_source="keyring",
            api_key="sk-test-secret",
        )
    )

    providers_text = (home / ".haagent" / "providers.json").read_text(encoding="utf-8")
    assert record.id == "requesty-personal"
    assert store.values["connection:requesty-personal"] == "sk-test-secret"
    assert "sk-test-secret" not in providers_text


def test_service_selects_models_from_named_connection_without_rewriting_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    store = FakeCredentialStore()
    monkeypatch.setattr(model_connections, "DEFAULT_CREDENTIAL_STORE", store)
    service = _service(tmp_path)

    connection = service.configure_model_connection(
        ModelConnectionConfigureRequest(
            id="requesty-personal",
            name="personal",
            provider_id="requesty",
            provider_name="Requesty",
            gateway_provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            api_key_env="REQUESTY_API_KEY",
            credential_source="keyring",
            api_key="sk-requesty",
        )
    )
    service.set_default_model_selection(
        ModelSelectionRequest(
            connection_id="requesty-personal",
            model="openai/gpt-5.2-chat",
        )
    )

    status = service.switch_current_session_model_selection(
        ModelSelectionRequest(
            connection_id="requesty-personal",
            model="anthropic/claude-sonnet-4.5",
        )
    )
    settings = json.loads((home / ".haagent" / "settings.json").read_text(encoding="utf-8"))
    providers_text = (home / ".haagent" / "providers.json").read_text(encoding="utf-8")

    assert connection.id == "requesty-personal"
    assert status.model_connection_id == "requesty-personal"
    assert status.model == "anthropic/claude-sonnet-4.5"
    assert settings["active_model"] == {
        "connection_id": "requesty-personal",
        "model": "openai/gpt-5.2-chat",
    }
    assert store.values == {"connection:requesty-personal": "sk-requesty"}
    assert "sk-requesty" not in providers_text


def test_service_deletes_model_connection_and_refreshes_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    _write_two_connections(config_dir)
    service = _service(tmp_path)

    service.delete_model_connection("local")

    assert [
        connection.id
        for connection in list_provider_connection_records(config_path=config_dir / "providers.json")
    ] == ["router"]
    assert load_active_model_selection(config_dir=config_dir).connection_id == "router"


def test_service_refreshes_model_catalog(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    provider = ModelCatalogProvider(
        id="requesty",
        name="Requesty",
        env_names=["REQUESTY_API_KEY"],
        api_base_url="https://router.requesty.ai/v1",
        provider_package="@ai-sdk/openai-compatible",
        documentation_url="https://requesty.ai/models",
        models=[],
    )
    service = _service(tmp_path)

    result = service.refresh_model_catalog(
        transport=lambda: {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "api": "https://router.requesty.ai/v1",
                "npm": "@ai-sdk/openai-compatible",
                "doc": "https://requesty.ai/models",
                "models": {},
            }
        }
    )

    assert result.providers == [provider]
    assert result.used_cache is False


def test_service_gets_model_catalog_from_fresh_cache_without_refreshing(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    provider = ModelCatalogProvider(
        id="requesty",
        name="Requesty",
        env_names=["REQUESTY_API_KEY"],
        api_base_url="https://router.requesty.ai/v1",
        provider_package=None,
        documentation_url=None,
        models=[],
    )
    service = _service(tmp_path)
    service.refresh_model_catalog(
        transport=lambda: {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "api": "https://router.requesty.ai/v1",
                "models": {},
            }
        }
    )

    result = service.get_model_catalog(
        transport=lambda: (_ for _ in ()).throw(AssertionError("fresh cache should be used"))
    )

    assert result.providers == [provider]
    assert result.used_cache is True


def test_service_refresh_model_catalog_forces_network_refresh(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    service = _service(tmp_path)
    service.refresh_model_catalog(
        transport=lambda: {
            "cached": {
                "id": "cached",
                "name": "Cached",
                "env": ["CACHED_API_KEY"],
                "api": "https://cached.example/v1",
                "models": {},
            }
        }
    )

    result = service.refresh_model_catalog(
        transport=lambda: {
            "fresh": {
                "id": "fresh",
                "name": "Fresh",
                "env": ["FRESH_API_KEY"],
                "api": "https://fresh.example/v1",
                "models": {},
            }
        }
    )

    assert result.providers[0].id == "fresh"
    assert result.used_cache is False


def test_service_connection_test_uses_gateway_factory_and_redacts_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    class ConnectionGateway:
        provider_name = "openai-chat"

        def generate(self, messages, tool_schemas):
            calls.append((messages, tool_schemas))
            return ModelResponse(content="OK")

    def gateway_factory(profile):
        assert profile.api_key == "sk-test-secret"
        assert profile.model == "openai/gpt-5.2-chat"
        return ConnectionGateway()

    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(
        Path.home(),
        name="router",
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/old-default",
        api_key_env="OPENROUTER_API_KEY",
    )
    service = _service(
        tmp_path,
        gateway_factory=gateway_factory,
        environ={"OPENROUTER_API_KEY": "sk-test-secret"},
    )

    result = service.test_model_connection("router", model="openai/gpt-5.2-chat")

    assert result.ok is True
    assert result.message == "OK"
    assert calls
    assert "sk-test-secret" not in repr(result)


def test_active_model_status_reports_api_key_available(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "sk-secret"})

    status = service.get_workspace_status()

    assert status.workspace_root == (tmp_path / "workspace").resolve()
    assert status.runs_root == tmp_path / ".runs"
    assert status.profile_name == "local"
    assert status.provider == "openai-chat"
    assert status.base_url == "https://api.deepseek.com"
    assert status.model == "deepseek-chat"
    assert status.api_key_env == "DEEPSEEK_API_KEY"
    assert status.api_key_available is True
    assert status.credential_source_configured == "keyring"
    assert status.credential_source_used == "env"
    assert status.credential_store_available is True
    assert status.image_input_supported is False
    assert status.profile_error is None
    assert "sk-secret" not in repr(status)


def test_workspace_status_reports_default_sandbox_status(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.sandbox_status.backend == "local_subprocess"
    assert status.sandbox_status.degraded is True
    assert status.sandbox_status.reason == "docker sandbox disabled"


def test_service_enables_docker_sandbox_and_preserves_settings(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = _service(tmp_path)

    status = service.enable_docker_sandbox()
    saved = json.loads((home / ".haagent" / "settings.json").read_text(encoding="utf-8"))

    assert status.backend == "docker"
    assert status.degraded is False
    assert saved["active_model"] == {"connection_id": "local", "model": "deepseek-chat"}
    assert saved["sandbox"]["enabled"] is True
    assert saved["sandbox"]["backend"] == "docker"
    assert saved["sandbox"]["fail_if_unavailable"] is True


def test_service_disables_sandbox(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = _service(tmp_path)
    service.enable_docker_sandbox()

    status = service.disable_sandbox()
    saved = json.loads((home / ".haagent" / "settings.json").read_text(encoding="utf-8"))

    assert status.backend == "local_subprocess"
    assert status.degraded is True
    assert saved["active_model"] == {"connection_id": "local", "model": "deepseek-chat"}
    assert saved["sandbox"]["enabled"] is False


def test_service_reports_sandbox_doctor(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = _service(tmp_path)
    monkeypatch.setattr("haagent.runtime.sandbox.status.shutil.which", lambda name: None)

    report = service.get_sandbox_doctor_report()

    assert report.ready is False
    assert report.docker_cli == "missing"
    assert "Install Docker Desktop" in report.next_action


def test_active_model_status_reports_missing_api_key_env(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    monkeypatch.setattr(model_connections, "DEFAULT_CREDENTIAL_STORE", FakeCredentialStore({}))
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.profile_name == "local"
    assert status.api_key_env == "DEEPSEEK_API_KEY"
    assert status.api_key_available is False
    assert status.credential_source_configured == "keyring"
    assert status.credential_source_used is None
    assert status.profile_error is None


def test_active_model_status_reports_keyring_api_key_available(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    monkeypatch.setattr(
        model_connections,
        "DEFAULT_CREDENTIAL_STORE",
        FakeCredentialStore({"connection:local": "keyring-secret"}),
    )
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.api_key_available is True
    assert status.credential_source_used == "keyring"
    assert status.credential_store_available is True
    assert "keyring-secret" not in repr(status)


def test_active_model_status_reports_keyring_unavailable(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    monkeypatch.setattr(
        model_connections,
        "DEFAULT_CREDENTIAL_STORE",
        FakeCredentialStore(available=False, error="backend unavailable"),
    )
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.api_key_available is False
    assert status.credential_store_available is False
    assert status.credential_store_error == "backend unavailable"
    assert status.profile_error is None


def test_active_model_status_reports_missing_selection(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    Path.home().mkdir()
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.profile_name is None
    assert status.api_key_available is False
    assert status.profile_error is not None
    assert "/connect" in status.profile_error


def test_create_new_session_sets_current_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})

    session = service.create_session()

    assert session.session_id
    assert session.turn_count == 0
    assert session.workspace_root == (tmp_path / "workspace").resolve()
    assert session.permission_mode == "request_approval"
    assert service.current_session().session_id == session.session_id


def test_create_new_session_uses_registry_as_default_gateway_factory(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = AssistantService(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        environ={"DEEPSEEK_API_KEY": "secret"},
        session_cls=RecordingSession,
    )

    service.create_session()

    assert service.gateway_factory is gateway_from_profile
    assert isinstance(service._session.model_gateway, OpenAIChatCompletionsGateway)


def test_create_and_resume_session_inject_same_retry_policy_into_compatible_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    settings_path = Path.home() / ".haagent" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["model_retry"] = {"max_attempts": 2}
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    controllers = []

    def gateway_factory(profile, *, retry_controller):
        del profile
        controllers.append(retry_controller)
        return RecordingGateway("retry")

    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"}, gateway_factory=gateway_factory)
    created = service.create_session()
    service.resume_session(created.session_path)

    assert [controller.policy.max_attempts for controller in controllers] == [2, 2]
    assert controllers[0] is not controllers[1]


def test_service_web_flag_is_reported_and_passed_to_new_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = AssistantService(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        environ={"DEEPSEEK_API_KEY": "secret"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
        session_cls=RecordingSession,
        enable_web=True,
    )

    status = service.get_workspace_status()
    session = service.create_session()

    assert status.web_enabled is True
    assert session.web_enabled is True
    assert service._session.enable_web is True


def test_assistant_service_mcp_status_without_session(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    service = AssistantService(workspace_root=tmp_path, runs_root=tmp_path / ".runs")

    status = service.get_mcp_status()

    assert status["configured_count"] == 0
    assert status["connected_count"] == 0
    assert status["failed_count"] == 0
    assert status["servers"] == []


def test_assistant_service_reads_configured_mcp_servers_without_session(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "exa": {
                        "type": "http",
                        "url": "https://mcp.exa.ai/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    service = AssistantService(workspace_root=tmp_path, runs_root=tmp_path / ".runs")

    status = service.get_mcp_status()

    assert status["configured_count"] == 1
    assert status["connected_count"] == 0
    assert status["failed_count"] == 0
    assert status["servers"] == [
        {
            "name": "exa",
            "state": "configured",
            "detail": "not loaded; create or resume a session to connect",
            "tool_count": 0,
            "resource_count": 0,
        }
    ]


def test_service_can_toggle_web_for_current_and_future_sessions(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = AssistantService(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        environ={"DEEPSEEK_API_KEY": "secret"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
        session_cls=RecordingSession,
    )

    service.set_web_enabled(True)
    session = service.create_session()
    service.set_web_enabled(False)

    assert session.web_enabled is True
    assert service.get_workspace_status().web_enabled is False
    assert service.current_session().web_enabled is False
    assert service._session.enable_web is False


def test_service_saves_interactive_turn_limit_and_updates_current_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "secret"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
    )
    service.session_cls = RecordingSession
    service.create_session()

    status = service.set_interactive_max_turns(80)

    assert status.current_max_turns == 80
    assert status.configured_interactive_max_turns == 80
    assert service.current_session().max_turns == 80
    assert service._session.max_turns == 80
    saved = json.loads((home / ".haagent" / "settings.json").read_text(encoding="utf-8"))
    assert saved["active_model"] == {"connection_id": "local", "model": "deepseek-chat"}
    assert saved["interactive_max_turns"] == 80


def test_service_sets_current_session_unlimited_without_persisting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_connection(home)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "secret"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
    )
    service.session_cls = RecordingSession
    service.create_session()

    status = service.set_current_turns_unlimited()

    assert status.current_max_turns is None
    assert status.configured_interactive_max_turns == 200
    assert service.current_session().max_turns is None
    assert service._session.max_turns is None
    saved = json.loads((home / ".haagent" / "settings.json").read_text(encoding="utf-8"))
    assert saved["active_model"] == {"connection_id": "local", "model": "deepseek-chat"}


def test_service_switches_current_session_model_without_changing_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    _write_two_connections(config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )
    service.create_session()

    status = service.switch_current_session_model_selection(
        ModelSelectionRequest(connection_id="router", model="openai/gpt-5.2-chat")
    )
    workspace_status = service.get_workspace_status()

    assert status.model_connection_id == "router"
    assert service.current_session().model_connection_id == "router"
    assert workspace_status.profile_name == "router"
    assert workspace_status.model == "openai/gpt-5.2-chat"
    assert workspace_status.base_url == "https://openrouter.ai/api/v1"
    assert workspace_status.api_key_env == "OPENROUTER_API_KEY"
    assert workspace_status.api_key_available is True
    assert load_active_model_selection(config_dir=config_dir) == ModelSelection("local", "deepseek-chat")


def test_service_switch_before_session_uses_profile_for_next_session_without_creating_one(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    _write_two_connections(config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )

    pending = service.switch_current_session_model_selection(
        ModelSelectionRequest(connection_id="router", model="openai/gpt-5.2-chat")
    )

    assert pending.session_id == "pending"
    assert pending.model_connection_id == "router"
    assert service.current_session() is None

    created = service.create_session()

    assert created.model_connection_id == "router"
    assert load_active_model_selection(config_dir=config_dir) == ModelSelection("local", "deepseek-chat")


def test_resume_session_preserves_switched_model_profile_when_default_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    _write_two_connections(config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )
    created = service.create_session()
    switched = service.switch_current_session_model_selection(
        ModelSelectionRequest(connection_id="router", model="openai/gpt-5.2-chat")
    )

    resumed = service.resume_session(created.session_path)

    assert switched.model_connection_id == "router"
    assert resumed.model_connection_id == "router"
    assert load_active_model_selection(config_dir=config_dir) == ModelSelection("local", "deepseek-chat")


def test_service_rejects_model_switch_while_current_run_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    _write_two_connections(config_dir)
    entered = threading.Event()
    release = threading.Event()

    def gateway_factory(profile):
        if profile.name == "local:deepseek-chat":
            return BlockingGateway(entered, release)
        return RecordingGateway(profile.name)

    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
        gateway_factory=gateway_factory,
    )
    service.create_session()
    thread = threading.Thread(target=lambda: service.run_prompt_events("hello"))
    thread.start()
    assert entered.wait(timeout=2)

    try:
        with pytest.raises(AssistantServiceError, match="current task is running"):
            service.switch_current_session_model_selection(
                ModelSelectionRequest(connection_id="router", model="openai/gpt-5.2-chat")
            )
    finally:
        release.set()
        thread.join(timeout=3)


def test_resume_session_sets_current_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    created = service.create_session()
    service.run_prompt_events("remember this")

    resumed = service.resume_session(created.session_path)

    assert resumed.session_id == created.session_id
    assert resumed.turn_count == 1
    assert service.current_session().session_id == created.session_id
    history = service.current_session_history()
    assert len(history) == 1
    assert history[0].request == "remember this"
    assert "remember this" in history[0].summary
    assert history[0].status == "completed"


def test_external_root_authorization_is_saved_and_restored_with_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    external = tmp_path / "external"
    external.mkdir()
    created = service.create_session()

    service.add_external_root(external, "read")
    restored = service.resume_session(created.session_path)

    assert restored.external_roots == [
        {
            "path": str(external.resolve()),
            "access": "read",
            "source": "user",
        },
    ]
    assert service.get_workspace_status().external_roots == restored.external_roots


def test_permission_mode_is_saved_and_restored_with_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    created = service.create_session()

    changed = service.set_permission_mode("full_access")
    restored = service.resume_session(created.session_path)

    assert changed.permission_mode == "full_access"
    assert restored.permission_mode == "full_access"
    assert service.get_workspace_status().permission_mode == "full_access"


def test_switch_project_root_preserves_permission_mode(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    new_project = tmp_path / "new-project"
    new_project.mkdir()
    service.create_session()
    service.set_permission_mode("full_access")

    status = service.switch_project_root(new_project)

    assert status.workspace_root == new_project.resolve()
    assert status.external_roots == []
    assert status.permission_mode == "full_access"


def test_switch_project_root_clears_external_authorizations(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    external = tmp_path / "external"
    new_project = tmp_path / "new-project"
    external.mkdir()
    new_project.mkdir()
    service.create_session()
    service.add_external_root(external, "full")

    status = service.switch_project_root(new_project)

    assert status.workspace_root == new_project.resolve()
    assert status.external_roots == []


def test_continue_latest_session_uses_current_workspace_only(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    old_session = service.create_session()
    service.run_prompt_events("old")
    latest_session = service.create_session()
    service.run_prompt_events("latest")

    restored = service.continue_latest_session()

    assert restored.session_id == latest_session.session_id
    assert restored.session_id != old_session.session_id
    assert restored.turn_count == 1


def test_list_sessions_only_lists_current_workspace(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    service.run_prompt_events("current workspace")
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    other_service = AssistantService(
        workspace_root=other_workspace,
        runs_root=tmp_path / ".runs",
        environ={"DEEPSEEK_API_KEY": "secret"},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
    )
    other_service.run_prompt_events("other workspace")

    sessions = service.list_sessions()

    assert [item.first_request for item in sessions] == ["current workspace"]


def test_run_prompt_events_forwards_chat_events(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    events = []

    result = service.run_prompt_events("send events", event_sink=events.append)

    assert result.status == "completed"
    assert [event.state for event in events if isinstance(event, SessionLifecycleEvent)] == [
        "session_started",
        "turn_started",
        "turn_finished",
        "session_finished",
    ]
    assert any(isinstance(event, AssistantMessageEvent) for event in events)
    assert [event.event_name for event in events if isinstance(event, TaskProgressEvent)] == [
        "task_step_started",
        "task_plan_created",
        "task_step_progress",
        "task_step_finished",
    ]


def test_service_reuses_last_sent_image_attachments_for_followup_turn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    gateways: list[RecordingGateway] = []

    def gateway_factory(profile):
        gateway = RecordingGateway(profile.name)
        gateways.append(gateway)
        return gateway

    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"}, gateway_factory=gateway_factory)
    created = service.create_session()
    attachment = _session_image_attachment(created.session_path, "img-one")

    service.run_prompt_events("描述图片", attachments=[attachment])
    service.run_prompt_events("继续分析")

    assert len(gateways[0].model_inputs) == 2
    first_content = gateways[0].model_inputs[0]
    second_content = gateways[0].model_inputs[1]
    assert _image_attachment_paths(first_content) == [attachment.relative_path]
    assert _image_attachment_paths(second_content) == [attachment.relative_path]


def test_service_replaces_auto_reused_images_when_new_images_are_sent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    gateways: list[RecordingGateway] = []

    def gateway_factory(profile):
        gateway = RecordingGateway(profile.name)
        gateways.append(gateway)
        return gateway

    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"}, gateway_factory=gateway_factory)
    created = service.create_session()
    first = _session_image_attachment(created.session_path, "img-one")
    second = _session_image_attachment(created.session_path, "img-two")

    service.run_prompt_events("描述第一张", attachments=[first])
    service.run_prompt_events("描述第二张", attachments=[second])
    service.run_prompt_events("继续分析")

    assert _image_attachment_paths(gateways[0].model_inputs[1]) == [second.relative_path]
    assert _image_attachment_paths(gateways[0].model_inputs[2]) == [second.relative_path]
    followup_text = _text_part(gateways[0].model_inputs[2])
    assert "Image Attachment History:" in followup_text
    assert first.relative_path in followup_text
    assert second.relative_path in followup_text


def test_resumed_session_reuses_last_sent_image_attachments_for_followup_turn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    gateways: list[RecordingGateway] = []

    def gateway_factory(profile):
        gateway = RecordingGateway(profile.name)
        gateways.append(gateway)
        return gateway

    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"}, gateway_factory=gateway_factory)
    created = service.create_session()
    attachment = _session_image_attachment(created.session_path, "img-one")
    service.run_prompt_events("描述图片", attachments=[attachment])

    service.resume_session(created.session_path)
    service.run_prompt_events("继续分析")

    assert len(gateways) == 2
    assert _image_attachment_paths(gateways[1].model_inputs[0]) == [attachment.relative_path]


def test_session_creation_requires_usable_active_model(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_connection(Path.home())
    monkeypatch.setattr(model_connections, "DEFAULT_CREDENTIAL_STORE", FakeCredentialStore({}))
    service = _service(tmp_path)

    with pytest.raises(AssistantServiceError, match="DEEPSEEK_API_KEY"):
        service.create_session()


def test_service_lists_trusts_and_reads_skills(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    project_skill_dir = workspace / ".haagent" / "skills" / "project-flow"
    project_skill_dir.mkdir(parents=True)
    (project_skill_dir / "SKILL.md").write_text("# Project Flow\nProject-only workflow.\n", encoding="utf-8")
    user_skill_dir = home / ".haagent" / "skills" / "grill-me"
    user_skill_dir.mkdir(parents=True)
    (user_skill_dir / "SKILL.md").write_text(
        "---\nname: grill-me\ndescription: User-only skill.\ndisable-model-invocation: true\n---\n\n# Grill Me\nAsk sharp questions.\n",
        encoding="utf-8",
    )
    service = AssistantService(workspace_root=workspace, gateway_factory=lambda profile: RecordingGateway())

    initial = service.list_skills()
    trusted = service.trust_project_skills()
    content = service.read_skill_for_user("grill-me")

    assert [skill["name"] for skill in initial.skills] == ["grill-me"]
    assert initial.blocked_project_skill_roots == [str((workspace / ".haagent" / "skills").resolve())]
    assert {skill["name"] for skill in trusted.skills} == {"grill-me", "Project Flow"}
    assert content.command_name == "grill-me"
    assert "Ask sharp questions." in content.content
