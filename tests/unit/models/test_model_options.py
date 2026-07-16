"""
tests/unit/models/test_model_options.py - 模型参数 options/variants 单元测试

覆盖深合并、保留字段、secret 拒绝、variant 解析与 payload 合并契约。
"""

from __future__ import annotations

import pytest

from haagent.models.model_options import (
    ModelOptionsError,
    deep_merge_options,
    merge_provider_payload,
    parse_connection_models,
    redact_model_options,
    validate_options_object,
)
from haagent.models.model_settings import ModelSettings


def test_deep_merge_objects_recursive_scalars_and_arrays_replace() -> None:
    base = {"temperature": 0.2, "reasoning": {"effort": "low", "enabled": True}, "tags": ["a"]}
    overlay = {"temperature": 0.8, "reasoning": {"effort": "high"}, "tags": ["b", "c"], "top_p": 0.9}
    assert deep_merge_options(base, overlay) == {
        "temperature": 0.8,
        "reasoning": {"effort": "high", "enabled": True},
        "tags": ["b", "c"],
        "top_p": 0.9,
    }


def test_deep_merge_null_is_explicit_value() -> None:
    assert deep_merge_options({"x": 1}, {"x": None}) == {"x": None}


def test_reserved_top_level_fields_rejected() -> None:
    with pytest.raises(ModelOptionsError, match="managed by HaAgent"):
        validate_options_object({"temperature": 0.5, "messages": []}, path="options")


def test_secret_like_field_names_rejected() -> None:
    with pytest.raises(ModelOptionsError, match="managed by HaAgent|secret"):
        validate_options_object({"api_key": "sk-x"}, path="options")
    with pytest.raises(ModelOptionsError, match="secret"):
        validate_options_object({"client_secret": "x"}, path="options")


def test_token_budget_parameter_names_are_not_misclassified_as_secrets() -> None:
    options = {
        "max_output_tokens": 16000,
        "max_tokens": 4096,
        "budget_tokens": 8000,
        "token_budget": 2000,
    }

    assert validate_options_object(options, path="options") == options


def test_explicit_null_models_options_and_variants_are_rejected() -> None:
    with pytest.raises(ModelOptionsError, match="models must be a JSON object"):
        parse_connection_models(None, path="models")
    with pytest.raises(ModelOptionsError, match="options must be a JSON object"):
        parse_connection_models({"m1": {"options": None}}, path="models")
    with pytest.raises(ModelOptionsError, match="variants must be a JSON object"):
        parse_connection_models({"m1": {"variants": None}}, path="models")


@pytest.mark.parametrize(
    "field",
    [
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "auth_token",
        "session_token",
        "bearer_token",
        "oauth_token",
        "custom_token",
        "auth_tokens",
        "session_tokens",
        "access_tokens",
    ],
)
def test_credential_token_fields_are_rejected(field: str) -> None:
    with pytest.raises(ModelOptionsError, match="secret"):
        validate_options_object({field: "credential"}, path="options")


def test_nested_credential_token_fields_are_rejected() -> None:
    with pytest.raises(ModelOptionsError, match="secret"):
        validate_options_object(
            {"auth": {"session_token": "x"}},
            path="options",
        )
    with pytest.raises(ModelOptionsError, match="secret"):
        validate_options_object(
            {"providers": [{"id_token": "x"}]},
            path="options",
        )
    with pytest.raises(ModelOptionsError, match="secret"):
        validate_options_object(
            {"items": [[{"session_token": "secret"}]]},
            path="options",
        )


def test_model_settings_resolve_and_empty() -> None:
    empty = ModelSettings.empty()
    assert empty.configured is False
    assert empty.options == {}

    models = parse_connection_models(
        {
            "m1": {
                "options": {"temperature": 0.1},
                "variants": {"fast": {"temperature": 0.9}},
            },
        },
        path="models",
    )
    resolved = ModelSettings.from_options(models["m1"].options).resolve(models["m1"].variants["fast"])
    assert resolved.configured is True
    assert resolved.options == {"temperature": 0.9}


def test_merge_provider_payload_skips_empty_and_blocks_reserved() -> None:
    base = {"model": "gpt", "messages": [], "temperature": 0.0}
    assert merge_provider_payload(base, {}) is base
    merged = merge_provider_payload(base, {"temperature": 0.7, "top_p": 0.5})
    assert merged["temperature"] == 0.7
    assert merged["top_p"] == 0.5
    assert merged["model"] == "gpt"
    with pytest.raises(ModelOptionsError, match="managed by HaAgent"):
        merge_provider_payload(base, {"model": "hijack"})


def test_audit_summary_includes_digest_and_key_paths() -> None:
    models = parse_connection_models(
        {"m1": {"options": {"temperature": 0.2, "reasoning": {"effort": "low"}}}},
        path="models",
    )
    resolved = ModelSettings.from_options(models["m1"].options)
    summary = resolved.to_traceable_dict()
    assert summary["configured"] is True
    assert summary["options_digest"]
    assert "temperature" in summary["options_key_paths"]
    assert redact_model_options({"temperature": 0.2}) == {"temperature": 0.2}
