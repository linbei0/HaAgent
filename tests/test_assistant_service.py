"""
tests/test_assistant_service.py - AssistantService 应用服务层测试

验证 TUI 前置服务层能复用 profile、session 和事件流能力，且不暴露真实 API key。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from haagent.app.assistant_service import AssistantService, AssistantServiceError, ModelProfileConfigureRequest
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.credentials import FakeCredentialStore
from haagent.models.gateway import ModelResponse, OpenAIChatCompletionsGateway
from haagent.models import provider_profile
from haagent.models.gateway_registry import gateway_from_profile
from haagent.models.provider_profile import ProviderProfileRecord, save_active_profile, save_provider_profile


class RecordingGateway:
    provider_name = "openai-chat"

    def __init__(self, profile_name: str = "local") -> None:
        self.profile_name = profile_name
        self.model_inputs: list[str] = []

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
        model_name: str | None = None,
        model_base_url: str | None = None,
        max_turns: int,
        enable_web: bool = False,
    ) -> None:
        self.session_id = "session-from-default-registry"
        self.workspace_root = workspace_root
        self.runs_root = runs_root
        self.model_gateway = model_gateway
        self.model_profile_name = model_profile_name
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.max_turns = max_turns
        self.enable_web = enable_web
        self.session_path = runs_root / self.session_id

    def switch_model_gateway(
        self,
        *,
        profile_name: str,
        provider: str,
        model: str,
        base_url: str,
        gateway,
    ) -> None:
        self.model_gateway = gateway
        self.model_profile_name = profile_name
        self.model_name = model
        self.model_base_url = base_url


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _write_user_profile(
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
                "profiles": [
                    {
                        "name": name,
                        "provider": provider,
                        "base_url": base_url,
                        "model": model,
                        "api_key_env": api_key_env,
                        "credential_source": credential_source,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (config_dir / "settings.json").write_text(
        json.dumps({"active_profile": name}),
        encoding="utf-8",
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


def test_service_lists_model_profiles_with_active_and_credential_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("router", config_dir=config_dir)
    service = _service(tmp_path, config_dir=config_dir)

    profiles = service.list_model_profiles()

    assert len(profiles) == 1
    assert profiles[0].name == "router"
    assert profiles[0].active is True
    assert profiles[0].capability.status == "runnable"


def test_service_sets_default_model_profile_without_switching_current_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    service = _service(tmp_path, config_dir=config_dir)

    service.set_default_model_profile("router")

    assert service.current_session() is None
    assert provider_profile.load_active_profile_name(settings_path=config_dir / "settings.json") == "router"


def test_service_configures_model_profile_with_keyring_without_writing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    store = FakeCredentialStore()
    monkeypatch.setattr(provider_profile, "DEFAULT_CREDENTIAL_STORE", store)
    service = _service(tmp_path)

    record = service.configure_model_profile(
        ModelProfileConfigureRequest(
            name="requesty-openai-gpt-5-2-chat",
            provider="openai-chat",
            base_url="https://router.requesty.ai/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="REQUESTY_API_KEY",
            credential_source="keyring",
            api_key="sk-test-secret",
        )
    )

    providers_text = (home / ".haagent" / "providers.json").read_text(encoding="utf-8")
    assert record.name == "requesty-openai-gpt-5-2-chat"
    assert store.values["profile:requesty-openai-gpt-5-2-chat"] == "sk-test-secret"
    assert "sk-test-secret" not in providers_text


def test_service_deletes_model_profile_and_refreshes_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("local", config_dir=config_dir)
    service = _service(tmp_path)

    service.delete_model_profile("local")

    assert [profile.name for profile in service.list_model_profiles()] == ["router"]
    assert provider_profile.load_active_profile_name(settings_path=config_dir / "settings.json") == "router"


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
        return ConnectionGateway()

    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    _write_user_profile(
        Path.home(),
        name="router",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
    )
    service = _service(
        tmp_path,
        gateway_factory=gateway_factory,
        environ={"OPENROUTER_API_KEY": "sk-test-secret"},
    )

    result = service.test_model_profile("router")

    assert result.ok is True
    assert result.message == "OK"
    assert calls
    assert "sk-test-secret" not in repr(result)


def test_active_profile_status_reports_api_key_available(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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
    assert status.profile_error is None
    assert "sk-secret" not in repr(status)


def test_active_profile_status_reports_missing_api_key_env(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    monkeypatch.setattr(provider_profile, "DEFAULT_CREDENTIAL_STORE", FakeCredentialStore({}))
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.profile_name == "local"
    assert status.api_key_env == "DEEPSEEK_API_KEY"
    assert status.api_key_available is False
    assert status.credential_source_configured == "keyring"
    assert status.credential_source_used is None
    assert status.profile_error is None


def test_active_profile_status_reports_keyring_api_key_available(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    monkeypatch.setattr(
        provider_profile,
        "DEFAULT_CREDENTIAL_STORE",
        FakeCredentialStore({"profile:local": "keyring-secret"}),
    )
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.api_key_available is True
    assert status.credential_source_used == "keyring"
    assert status.credential_store_available is True
    assert "keyring-secret" not in repr(status)


def test_active_profile_status_reports_keyring_unavailable(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    monkeypatch.setattr(
        provider_profile,
        "DEFAULT_CREDENTIAL_STORE",
        FakeCredentialStore(available=False, error="backend unavailable"),
    )
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.api_key_available is False
    assert status.credential_store_available is False
    assert status.credential_store_error == "backend unavailable"
    assert status.profile_error is None


def test_active_profile_status_reports_missing_profile(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    Path.home().mkdir()
    service = _service(tmp_path)

    status = service.get_workspace_status()

    assert status.profile_name is None
    assert status.api_key_available is False
    assert status.profile_error is not None
    assert "/model" in status.profile_error


def test_create_new_session_sets_current_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})

    session = service.create_session()

    assert session.session_id
    assert session.turn_count == 0
    assert session.workspace_root == (tmp_path / "workspace").resolve()
    assert session.permission_mode == "request_approval"
    assert service.current_session().session_id == session.session_id


def test_create_new_session_uses_registry_as_default_gateway_factory(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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


def test_service_web_flag_is_reported_and_passed_to_new_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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


def test_service_can_toggle_web_for_current_and_future_sessions(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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


def test_service_switches_current_session_model_without_changing_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("local", config_dir=config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )
    service.create_session()

    status = service.switch_current_session_model("router")

    assert status.model_profile_name == "router"
    assert service.current_session().model_profile_name == "router"
    assert provider_profile.load_active_profile_name(settings_path=config_dir / "settings.json") == "local"


def test_service_switch_before_session_uses_profile_for_next_session_without_creating_one(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("local", config_dir=config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )

    pending = service.switch_current_session_model("router")

    assert pending.session_id == "pending"
    assert pending.model_profile_name == "router"
    assert service.current_session() is None

    created = service.create_session()

    assert created.model_profile_name == "router"
    assert provider_profile.load_active_profile_name(settings_path=config_dir / "settings.json") == "local"


def test_resume_session_preserves_switched_model_profile_when_default_differs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("local", config_dir=config_dir)
    service = _service(
        tmp_path,
        environ={"DEEPSEEK_API_KEY": "local-secret", "OPENROUTER_API_KEY": "router-secret"},
    )
    created = service.create_session()
    switched = service.switch_current_session_model("router")

    resumed = service.resume_session(created.session_path)

    assert switched.model_profile_name == "router"
    assert resumed.model_profile_name == "router"
    assert provider_profile.load_active_profile_name(settings_path=config_dir / "settings.json") == "local"


def test_service_rejects_model_switch_while_current_run_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    config_dir = home / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=config_dir,
    )
    save_active_profile("local", config_dir=config_dir)
    entered = threading.Event()
    release = threading.Event()

    def gateway_factory(profile):
        if profile.name == "local":
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
            service.switch_current_session_model("router")
    finally:
        release.set()
        thread.join(timeout=3)


def test_resume_session_sets_current_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    created = service.create_session()

    changed = service.set_permission_mode("full_access")
    restored = service.resume_session(created.session_path)

    assert changed.permission_mode == "full_access"
    assert restored.permission_mode == "full_access"
    assert service.get_workspace_status().permission_mode == "full_access"


def test_switch_project_root_preserves_permission_mode(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
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
    _write_user_profile(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})
    events = []

    result = service.run_prompt_events("send events", event_sink=events.append)

    assert result.status == "completed"
    assert [event.event_type for event in events] == [
        "session_started",
        "turn_started",
        "assistant_message",
        "turn_finished",
        "session_finished",
    ]


def test_session_creation_requires_usable_active_profile(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    monkeypatch.setattr(provider_profile, "DEFAULT_CREDENTIAL_STORE", FakeCredentialStore({}))
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
