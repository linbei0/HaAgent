"""
tests/test_cli_personal_assistant.py - 个人助手启动体验测试

验证用户级模型配置、默认 haagent 入口和目录会话恢复。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent import cli
from haagent.models.credentials import FakeCredentialStore
from haagent.models.gateway import ModelResponse
from haagent.models import provider_profile
from haagent.runtime.chat_session import AgentSession
from haagent.runtime.task_contract import load_task


class RecordingGateway:
    provider_name = "recording"

    def __init__(self) -> None:
        self.model_inputs: list[str] = []

    def generate(self, messages, tool_schemas):
        model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        self.model_inputs.append(model_input)
        return ModelResponse(f"done: {' '.join(m.get('content', '') for m in messages if m.get('role') == 'user')}", [])


class FakeProfileGateway:
    provider_name = "openai-chat"

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, messages, tool_schemas):
        return ModelResponse(f"profile done: {' '.join(m.get('content', '') for m in messages if m.get('role') == 'user')}", [])


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _write_user_profile(home: Path, *, api_key_env: str = "CHAT_SECRET") -> None:
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "local",
                        "provider": "openai-chat",
                        "base_url": "https://api.example/v1",
                        "model": "chat-test",
                        "api_key_env": api_key_env,
                        "credential_source": "keyring",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (config_dir / "settings.json").write_text(
        json.dumps({"active_profile": "local"}),
        encoding="utf-8",
    )


def test_setup_writes_user_level_provider_settings_and_keyring_secret(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_home(monkeypatch, tmp_path)
    store = FakeCredentialStore({})
    monkeypatch.setattr(provider_profile, "DEFAULT_CREDENTIAL_STORE", store)
    answers = iter(
        [
            "local",
            "openai-chat",
            "https://api.example/v1",
            "chat-test",
            "CHAT_SECRET",
            "",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-setup-secret")

    exit_code = cli.main(["setup"])

    providers = json.loads((tmp_path / ".haagent" / "providers.json").read_text(encoding="utf-8"))
    settings = json.loads((tmp_path / ".haagent" / "settings.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert providers == {
        "profiles": [
            {
                "name": "local",
                "provider": "openai-chat",
                "base_url": "https://api.example/v1",
                "model": "chat-test",
                "api_key_env": "CHAT_SECRET",
                "credential_source": "keyring",
            },
        ],
    }
    assert settings == {"active_profile": "local"}
    assert "sk-" not in json.dumps(providers, ensure_ascii=False)
    assert store.get_password("haagent", "profile:local") == "sk-setup-secret"
    assert "active_profile=local" in output
    assert "keyring" in output


def test_setup_keyring_failure_does_not_write_insecure_file(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_home(monkeypatch, tmp_path)
    store = FakeCredentialStore(available=False, error="backend unavailable")
    monkeypatch.setattr(provider_profile, "DEFAULT_CREDENTIAL_STORE", store)
    answers = iter(
        [
            "local",
            "openai-chat",
            "https://api.example/v1",
            "chat-test",
            "CHAT_SECRET",
            "",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-setup-secret")

    exit_code = cli.main(["setup"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "backend unavailable" in output
    assert not (tmp_path / ".haagent" / "insecure_credentials.json").exists()


def test_setup_insecure_file_requires_explicit_choice(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_home(monkeypatch, tmp_path)
    answers = iter(
        [
            "local",
            "openai-chat",
            "https://api.example/v1",
            "chat-test",
            "CHAT_SECRET",
            "insecure_file",
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-plain-secret")

    exit_code = cli.main(["setup"])

    providers = json.loads((tmp_path / ".haagent" / "providers.json").read_text(encoding="utf-8"))
    insecure_file = (tmp_path / ".haagent" / "insecure_credentials.json").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert exit_code == 0
    assert providers["profiles"][0]["credential_source"] == "insecure_file"
    assert "sk-plain-secret" not in json.dumps(providers, ensure_ascii=False)
    assert "sk-plain-secret" in insecure_file
    assert "明文" in output


def test_chat_uses_active_user_profile_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CHAT_SECRET", "sk-secret-value-that-must-not-leak")
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "chat_gateway_cls", FakeProfileGateway)

    exit_code = cli.main(["chat", "Use default profile"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "provider=openai-chat" in output
    assert "sk-secret-value-that-must-not-leak" not in output


def test_chat_without_config_explains_setup_first(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["chat", "Hello"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "请先运行 haagent setup" in output


def test_haagent_without_subcommand_enters_chat_repl(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CHAT_SECRET", "secret")
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "chat_gateway_cls", FakeProfileGateway)
    inputs = iter([":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "session_id=" in output
    assert f"workspace_root={workspace.resolve()}" in output
    assert "provider=openai-chat" in output


def test_haagent_default_workspace_is_current_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    inputs = iter(["List this folder", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["--provider", "fake"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    task = load_task(episode_path / "task.yaml")
    assert exit_code == 0
    assert task.workspace_root == str(workspace.resolve())


def test_sessions_lists_only_current_workspace_sessions(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other"
    workspace.mkdir()
    other_workspace.mkdir()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        model_gateway=RecordingGateway(),
    )
    session.run_prompt("Summarize these documents")
    other = AgentSession(
        workspace_root=other_workspace,
        runs_root=tmp_path / ".runs",
        model_gateway=RecordingGateway(),
    )
    other.run_prompt("Other workspace task")
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["sessions", "--workspace-root", str(workspace), "--runs-root", str(tmp_path / ".runs")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert session.session_id in output
    assert "Summarize these documents" in output
    assert other.session_id not in output
    assert "Other workspace task" not in output


def test_continue_restores_latest_current_workspace_session(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    old_session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
    )
    old_session.run_prompt("old")
    latest_session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
    )
    latest_session.run_prompt("latest")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "build_run_model_gateway", lambda args: RecordingGateway())
    inputs = iter([":status", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["--continue", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"session_id={latest_session.session_id}" in output
    assert f"session_id={old_session.session_id}" not in output
    assert "turn_count=1" in output


def test_profile_api_key_is_not_written_to_config_session_or_episode(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    secret = "sk-secret-value-that-must-not-leak"
    _set_home(monkeypatch, tmp_path / "home")
    _write_user_profile(Path.home())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CHAT_SECRET", secret)
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "chat_gateway_cls", FakeProfileGateway)

    exit_code = cli.main(["chat", "Check secret handling"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    session_path = next((workspace / ".runs" / "sessions").glob("session-*"))
    checked_files = [
        Path.home() / ".haagent" / "providers.json",
        Path.home() / ".haagent" / "settings.json",
        session_path / "session.json",
        session_path / "turns.jsonl",
        session_path / "working_state.json",
        episode_path / "episode.json",
        episode_path / "transcript.jsonl",
        episode_path / "tool-calls.jsonl",
    ]
    combined = output + "\n" + "\n".join(path.read_text(encoding="utf-8") for path in checked_files)
    assert exit_code == 0
    assert secret not in combined
