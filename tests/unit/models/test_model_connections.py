"""
tests/unit/models/test_model_connections.py - providers v4 存储与解析测试
"""

from __future__ import annotations

import json

import pytest

from haagent.models.config.config_store import ModelConfigStore
from haagent.models.config.connections import ProviderConnectionRecord, ProviderProfileError
from haagent.models.model_options import ModelParameterConfig
from haagent.models.model_ref import ModelRef
from haagent.models.model_runtime import ModelRuntime
from haagent.models.config.selection_store import ModelSelectionStore


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
    runtime.snapshot = runtime.config_store.save_connection(
        _record(models={"gpt-5": ModelParameterConfig({}, {"fast": {}, "deep": {}})}),
        expected_digest=runtime.snapshot.digest,
    ).bind_available_models({"main": {"gpt-5", "gpt-4.1"}})
    choices = runtime.list_choices()
    assert [choice.ref.model for choice in choices] == ["gpt-4.1", "gpt-5"]
    assert choices[1].variants == ("fast", "deep")


def test_runtime_resolves_variant_and_unconfigured_model_keeps_defaults(tmp_path) -> None:
    runtime = ModelRuntime.load(config_dir=tmp_path, environ={})
    runtime.snapshot = runtime.config_store.save_connection(
        _record(models={"gpt-5": ModelParameterConfig({"reasoning": {"effort": "low"}}, {"deep": {"reasoning": {"effort": "high"}}})}),
        expected_digest=runtime.snapshot.digest,
    )
    resolved = runtime.resolve(ModelRef("main", "gpt-5", "deep"))
    assert resolved.settings.options == {"reasoning": {"effort": "high"}}
    assert runtime.resolve(ModelRef("main", "unconfigured")).settings.configured is False
    with pytest.raises(ProviderProfileError, match="not available"):
        runtime.resolve(ModelRef("main", "gpt-5", "gone"))


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
