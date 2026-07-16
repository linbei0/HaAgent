"""
tests/unit/app/test_workspace_status_cache.py - workspace.status 热路径缓存

token 流式刷新不得每次查询 keyring / 重读 providers；只有 session、模型、
凭据或配置变化时才强制重建状态。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from haagent.app.workspace_usecases import AssistantWorkspace
from haagent.models.model_connections import ProviderConnectionRecord, ProvidersConfigSnapshot


class _CountingCredentialStatus:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return SimpleNamespace(
            api_key_available=True,
            credential_source_configured="env",
            credential_source_used="env",
            credential_store_available=True,
            credential_store_error=None,
        )


def _workspace(tmp_path: Path, monkeypatch, credential_counter: _CountingCredentialStatus) -> AssistantWorkspace:
    connection = ProviderConnectionRecord(
        id="conn-1",
        name="test",
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://example.test/v1",
        api_key_env="TEST_API_KEY",
    )
    # 只测 status 缓存边界，不依赖完整 AssistantContext 构造。
    context = SimpleNamespace(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        environ={},
        enable_web=False,
        max_turns=16,
        session=None,
        status_generation=0,
        providers_snapshot=ProvidersConfigSnapshot(
            path=tmp_path / "config" / "providers.json",
            records=(connection,),
            digest="test",
        ),
    )
    workspace = AssistantWorkspace(context)

    monkeypatch.setattr(
        "haagent.app.workspace_usecases.load_active_model_selection",
        lambda config_dir=None: SimpleNamespace(connection_id="conn-1", model="gpt-test", variant=None),
    )
    monkeypatch.setattr(
        "haagent.app.workspace_usecases.provider_connection_credential_status",
        credential_counter,
    )
    monkeypatch.setattr(
        "haagent.app.workspace_usecases.sandbox_status",
        lambda: SimpleNamespace(backend="local_subprocess", degraded=True, reason="test"),
    )
    return workspace


def test_workspace_status_reuses_credential_lookup_on_hot_path(tmp_path: Path, monkeypatch) -> None:
    counter = _CountingCredentialStatus()
    workspace = _workspace(tmp_path, monkeypatch, counter)

    first = workspace.status()
    second = workspace.status()

    assert first.model == "gpt-test"
    assert second.model == "gpt-test"
    assert counter.calls == 1


def test_set_web_enabled_invalidates_status_cache(tmp_path: Path, monkeypatch) -> None:
    counter = _CountingCredentialStatus()
    workspace = _workspace(tmp_path, monkeypatch, counter)

    first = workspace.status()
    assert first.web_enabled is False
    second = workspace.set_web_enabled(True)

    assert second.web_enabled is True
    assert counter.calls == 2
