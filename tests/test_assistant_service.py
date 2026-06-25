"""
tests/test_assistant_service.py - AssistantService 应用服务层测试

验证 TUI 前置服务层能复用 profile、session 和事件流能力，且不暴露真实 API key。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from haagent.app.assistant_service import AssistantService, AssistantServiceError
from haagent.models.credentials import FakeCredentialStore
from haagent.models.gateway import ModelResponse
from haagent.models import provider_profile


class RecordingGateway:
    provider_name = "openai-chat"

    def __init__(self, profile_name: str = "local") -> None:
        self.profile_name = profile_name
        self.model_inputs: list[str] = []

    def generate(self, messages, tool_schemas):
        task_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        self.model_inputs.append(task_content)
        return ModelResponse("done", [])


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


def _service(tmp_path: Path, *, environ: dict[str, str] | None = None) -> AssistantService:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AssistantService(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        environ=environ or {},
        gateway_factory=lambda profile: RecordingGateway(profile.name),
    )


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
    assert "haagent setup" in status.profile_error


def test_create_new_session_sets_current_session(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    service = _service(tmp_path, environ={"DEEPSEEK_API_KEY": "secret"})

    session = service.create_session()

    assert session.session_id
    assert session.turn_count == 0
    assert session.workspace_root == (tmp_path / "workspace").resolve()
    assert service.current_session().session_id == session.session_id


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
