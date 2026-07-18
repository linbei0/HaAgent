"""
tests/unit/models/test_model_connections.py - providers v4 存储与解析测试
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from haagent.models.capabilities import ModelCapabilities, effective_input_window_tokens
from haagent.models.catalog import CatalogFetchResult, ModelCatalogModel, ModelCatalogProvider
from haagent.models.config.config_store import ModelConfigStore
from haagent.models.config.connections import ProviderConnectionRecord, ProviderProfileError
from haagent.models.config.credentials import CredentialError
from haagent.models.model_options import ModelParameterConfig
from haagent.models.model_ref import ModelRef
from haagent.models.model_runtime import ModelRuntime, model_capabilities_from_catalog
from haagent.models.config.selection_store import ModelSelectionStore
from haagent.models.local_runtime import LocalRuntimeDiscovery, LocalRuntimeModel


def _record(*, connection_id: str = "main", models=None, source: str = "none") -> ProviderConnectionRecord:
    return ProviderConnectionRecord(
        id=connection_id,
        name=connection_id,
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="" if source == "none" else "OPENAI_API_KEY",
        credential_source=source,
        models=models or {},
    )


def test_store_saves_two_connections_and_preserves_models(tmp_path) -> None:
    store = ModelConfigStore(tmp_path / "providers.json")
    first = _record(models={"gpt-5": ModelParameterConfig({"temperature": 0.2}, {"deep": {"temperature": 0.8}})})
    snapshot = store.save_connection(first, expected_digest=store.load().digest)
    snapshot = store.save_connection(_record(connection_id="backup"), expected_digest=snapshot.digest)
    snapshot = store.save_connection(_record(), expected_digest=snapshot.digest)
    assert [item.id for item in snapshot.records] == ["main", "backup"]
    assert snapshot.connection("main").models["gpt-5"].variants == {"deep": {"temperature": 0.8}}
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 4


def test_store_round_trips_per_model_context_limit(tmp_path) -> None:
    store = ModelConfigStore(tmp_path / "providers.json")
    record = _record(
        models={"gpt-5": ModelParameterConfig({}, {}, max_context_tokens=400_000)},
    )

    store.save_connection(record, expected_digest=store.load().digest)
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["connections"][0]["models"]["gpt-5"]["max_context_tokens"] == 400_000
    assert store.load().connection("main").models["gpt-5"].max_context_tokens == 400_000


def test_store_rejects_unknown_v4_field_and_digest_conflict(tmp_path) -> None:
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"version": 4, "connections": [], "legacy": True}), encoding="utf-8")
    snapshot = ModelConfigStore(path).load()
    with pytest.raises(ProviderProfileError, match="unknown field"):
        snapshot.require_valid()
    path.write_text(json.dumps({"version": 4, "connections": []}), encoding="utf-8")
    store = ModelConfigStore(path)
    old = store.load()
    path.write_text(json.dumps({"version": 4, "connections": [], "$schema": "x"}), encoding="utf-8")
    with pytest.raises(ProviderProfileError, match="changed"):
        store.save_connection(_record(), expected_digest=old.digest)


def test_snapshot_binds_catalog_and_lists_choices_in_order(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(
        _record(models={"gpt-5": ModelParameterConfig({}, {"fast": {}, "deep": {}})}),
        available_models={"main": {"gpt-5", "gpt-4.1"}},
    )
    choices = runtime.list_choices()
    assert [choice.ref.model for choice in choices] == ["gpt-4.1", "gpt-5"]
    assert choices[1].variants == ("fast", "deep")


def test_catalog_capabilities_prefer_input_limit_and_ignore_invalid_values() -> None:
    capabilities = model_capabilities_from_catalog(
        ModelCatalogModel(
            id="gpt-test",
            limits={"input": 180_000, "context": 200_000},
        ),
    )
    invalid = model_capabilities_from_catalog(
        ModelCatalogModel(
            id="invalid",
            limits={"input": True, "context": "200000"},
        ),
    )

    assert capabilities == ModelCapabilities(
        context_window_tokens=200_000,
        input_window_tokens=180_000,
    )
    assert effective_input_window_tokens(capabilities) == 180_000
    assert effective_input_window_tokens(ModelCapabilities(context_window_tokens=128_000)) == 128_000
    assert effective_input_window_tokens(invalid) is None


def test_runtime_passes_bound_catalog_capabilities_to_route(tmp_path, monkeypatch) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(_record(models={"gpt-test": ModelParameterConfig({}, {})}))
    runtime.set_active(ModelRef("main", "gpt-test"))
    runtime.bind_remote_catalog(
        CatalogFetchResult(
            providers=[
                ModelCatalogProvider(
                    id="openai",
                    models=[
                        ModelCatalogModel(
                            id="gpt-test",
                            limits={"input": 180_000, "context": 200_000},
                        ),
                    ],
                ),
            ],
            source="test",
            fetched_at="2026-07-18T00:00:00Z",
        ),
    )
    direct_gateway = runtime.create_gateway(ModelRef("main", "gpt-test"))
    try:
        assert direct_gateway.capabilities().input_window_tokens == 180_000
        assert direct_gateway.capabilities().context_window_tokens == 200_000
    finally:
        direct_gateway.close()
    captured: dict[str, object] = {}

    def fake_gateway_from_route(primary, **kwargs):
        captured["primary"] = primary
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("haagent.models.model_runtime.gateway_from_route", fake_gateway_from_route)

    runtime.create_route_gateway()

    assert captured["primary_capabilities"] == ModelCapabilities(
        context_window_tokens=200_000,
        input_window_tokens=180_000,
    )


def test_runtime_applies_user_context_limit_after_catalog_capabilities(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(
        _record(
            models={"gpt-test": ModelParameterConfig({}, {}, max_context_tokens=400_000)},
        ),
    )
    runtime.set_active(ModelRef("main", "gpt-test"))
    runtime.bind_remote_catalog(
        CatalogFetchResult(
            providers=[
                ModelCatalogProvider(
                    id="openai",
                    models=[
                        ModelCatalogModel(id="gpt-test", limits={"input": 1_000_000, "context": 1_000_000}),
                    ],
                ),
            ],
            source="test",
            fetched_at="2026-07-18T00:00:00Z",
        ),
    )

    gateway = runtime.create_gateway(ModelRef("main", "gpt-test"))
    try:
        assert gateway.capabilities().context_window_tokens == 400_000
        assert gateway.capabilities().input_window_tokens == 400_000
    finally:
        gateway.close()


def test_runtime_applies_user_context_limit_to_discovered_local_capabilities(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(
        ProviderConnectionRecord(
            id="local-ollama",
            name="Ollama",
            provider_id="ollama",
            provider_name="Ollama",
            gateway_provider="openai",
            base_url="http://127.0.0.1:11434/v1",
            api_key_env="",
            credential_source="none",
            runtime_kind="ollama",
            models={"llama3": ModelParameterConfig({}, {}, max_context_tokens=400_000)},
        ),
    )
    runtime.bind_local_discoveries(
        [
            LocalRuntimeDiscovery(
                runtime_kind="ollama",
                base_url="http://127.0.0.1:11434/v1",
                status="available",
                models=(
                    LocalRuntimeModel(
                        id="llama3",
                        name="llama3",
                        loaded=True,
                        capabilities=ModelCapabilities(
                            context_window_tokens=1_000_000,
                            input_window_tokens=1_000_000,
                        ),
                    ),
                ),
            ),
        ],
    )

    gateway = runtime.create_gateway(ModelRef("local-ollama", "llama3"))
    try:
        assert gateway.capabilities().context_window_tokens == 400_000
        assert gateway.metadata().context_window_tokens == 400_000
    finally:
        gateway.close()


def test_bind_available_models_merges_remote_and_local_in_any_order(tmp_path) -> None:
    store = ModelConfigStore(tmp_path / "providers.json")
    snapshot = store.save_connection(
        _record(connection_id="remote", models={"gpt-5": ModelParameterConfig({}, {})}),
        expected_digest=store.load().digest,
    )
    local = ProviderConnectionRecord(
        id="local-ollama",
        name="local-ollama",
        provider_id="ollama",
        provider_name="Ollama",
        gateway_provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",
        credential_source="none",
        runtime_kind="ollama",
        models={"llama3": ModelParameterConfig({}, {})},
    )
    snapshot = store.save_connection(local, expected_digest=snapshot.digest)

    remote_only = {"remote": {"gpt-5", "gpt-4.1"}}
    local_only = {"local-ollama": {"llama3", "qwen2.5"}}

    catalog_first = snapshot.bind_available_models(remote_only, source="remote").bind_available_models(
        local_only,
        source="local",
    )
    local_first = snapshot.bind_available_models(local_only, source="local").bind_available_models(
        remote_only,
        source="remote",
    )

    assert dict(catalog_first.available_models) == dict(local_first.available_models)
    assert catalog_first.available_models == {
        "remote": ("gpt-4.1", "gpt-5"),
        "local-ollama": ("llama3", "qwen2.5"),
    }


def test_bind_available_models_source_refresh_clears_stale_results(tmp_path) -> None:
    store = ModelConfigStore(tmp_path / "providers.json")
    snapshot = store.save_connection(
        _record(connection_id="remote", models={"gpt-5": ModelParameterConfig({}, {})}),
        expected_digest=store.load().digest,
    )
    local = ProviderConnectionRecord(
        id="local-ollama",
        name="local-ollama",
        provider_id="ollama",
        provider_name="Ollama",
        gateway_provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",
        credential_source="none",
        runtime_kind="ollama",
        models={"llama3": ModelParameterConfig({}, {})},
    )
    snapshot = store.save_connection(local, expected_digest=snapshot.digest)
    bound = snapshot.bind_available_models(
        {"remote": {"gpt-5", "gpt-4.1"}},
        source="remote",
    ).bind_available_models(
        {"local-ollama": {"llama3", "qwen2.5"}},
        source="local",
    )

    # local 下线后空映射必须清掉旧 Ollama 结果，且不影响 remote。
    after_local_down = bound.bind_available_models({}, source="local")
    assert "local-ollama" not in after_local_down.available_models
    assert after_local_down.available_models["remote"] == ("gpt-4.1", "gpt-5")

    # remote catalog 移除 provider 后必须清掉旧 remote 结果，且不影响 local。
    after_remote_gone = bound.bind_available_models({}, source="remote")
    assert "remote" not in after_remote_gone.available_models
    assert after_remote_gone.available_models["local-ollama"] == ("llama3", "qwen2.5")


def test_runtime_resolves_variant_and_unconfigured_model_keeps_defaults(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(
        _record(
            models={
                "gpt-5": ModelParameterConfig(
                    {"reasoning": {"effort": "low"}},
                    {"deep": {"reasoning": {"effort": "high"}}},
                )
            }
        ),
    )
    resolved = runtime.resolve(ModelRef("main", "gpt-5", "deep"))
    assert resolved.settings.options == {"reasoning": {"effort": "high"}}
    assert runtime.resolve(ModelRef("main", "unconfigured")).settings.configured is False
    with pytest.raises(ProviderProfileError, match="not available"):
        runtime.resolve(ModelRef("main", "gpt-5", "gone"))


def test_runtime_public_api_hides_snapshot_and_selection_store(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(_record(connection_id="main", models={"gpt-5": ModelParameterConfig({}, {})}))
    runtime.save_connection(_record(connection_id="backup", models={"gpt-4.1": ModelParameterConfig({}, {})}))
    runtime.set_active(ModelRef("main", "gpt-5", "deep"))
    runtime.set_fallback(ModelRef("backup", "gpt-4.1"), cloud_consent=True)

    # 信息隐藏：调用方只能用公开方法，不能直接读写 snapshot/store。
    assert "snapshot" not in runtime.__dict__
    assert "config_store" not in runtime.__dict__
    assert "selection_store" not in runtime.__dict__
    assert not hasattr(runtime, "snapshot")
    assert not hasattr(runtime, "config_store")
    assert not hasattr(runtime, "selection_store")

    assert runtime.load_active() == ModelRef("main", "gpt-5", "deep")
    assert runtime.load_route().fallback == ModelRef("backup", "gpt-4.1")
    assert runtime.connection("main").id == "main"
    assert runtime.ref_for_connection("backup").model == "gpt-5"

    runtime.bind_available_models({"main": {"gpt-5", "gpt-4.1"}}, source="remote")
    assert [choice.ref.model for choice in runtime.list_choices() if choice.ref.connection_id == "main"] == [
        "gpt-4.1",
        "gpt-5",
    ]

    runtime.delete_connection("backup")
    with pytest.raises(ProviderProfileError, match="not found"):
        runtime.connection("backup")
    assert runtime.load_route().fallback is None


def test_selection_store_round_trip_and_delete_connection(tmp_path) -> None:
    store = ModelSelectionStore(tmp_path)
    active = ModelRef("main", "gpt-5", "deep")
    fallback = ModelRef("backup", "gpt-4.1")
    store.save_active(active)
    store.save_fallback(fallback, cloud_consent=True)
    assert store.load_route().primary == active
    assert store.load_route().fallback == fallback
    assert store.load_route().cloud_fallback_consent is True
    store.remove_connection("backup", ["main"])
    assert store.load_route().fallback is None


def test_runtime_save_connection_rolls_back_when_credential_write_fails(tmp_path, monkeypatch) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.save_connection(_record(connection_id="existing", models={"gpt-5": ModelParameterConfig({}, {})}))
    runtime.bind_available_models({"existing": {"gpt-5", "gpt-4.1"}}, source="remote")
    before_choices = [choice.ref.model for choice in runtime.list_choices()]
    record = _record(source="keyring")

    def _fail_save(*args, **kwargs):
        del args, kwargs
        raise CredentialError("keyring unavailable")

    monkeypatch.setattr(
        "haagent.models.model_runtime.save_connection_api_key",
        _fail_save,
    )
    with pytest.raises(CredentialError, match="keyring unavailable"):
        runtime.save_connection(record, api_key="secret-key")
    # 凭据失败后不得留下半提交连接，且保留 catalog 内存态。
    assert [item.id for item in runtime.list_connections()] == ["existing"]
    assert [choice.ref.model for choice in runtime.list_choices()] == before_choices
    assert "main" not in {item.id for item in runtime.list_connections()}


def test_runtime_save_connection_reports_when_rollback_also_fails(tmp_path, monkeypatch) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    record = _record(source="keyring")

    monkeypatch.setattr(
        "haagent.models.model_runtime.save_connection_api_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(CredentialError("keyring unavailable")),
    )

    def _fail_rollback(*args, **kwargs):
        del args, kwargs
        raise ProviderProfileError("digest conflict during rollback")

    monkeypatch.setattr(runtime._config_store, "delete_connection", _fail_rollback)
    with pytest.raises(ProviderProfileError, match="config rollback also failed"):
        runtime.save_connection(record, api_key="secret-key")
