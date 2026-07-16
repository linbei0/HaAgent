"""
src/haagent/models/model_options.py - 模型原生请求参数解析与合并

负责 providers.json v4 的 options/variants 深合并、保留字段校验、
secret-like 字段拒绝，以及解析结果摘要。配置层纯函数，不触碰 gateway。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from haagent.runtime.execution.command import redact_secret_like_text


RESERVED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "input",
        "messages",
        "contents",
        "system",
        "systemInstruction",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "previous_response_id",
        "conversation",
        "store",
        "api_key",
        "apiKey",
        "authorization",
        "headers",
        "base_url",
        "baseURL",
        "endpoint",
    }
)

_SECRET_FIELD_PATTERN = re.compile(
    r"(api[_-]?key|apikey|authorization|credential|password|secret)",
    re.IGNORECASE,
)
_VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_DEFAULT_VARIANT_NAME = "default"


class ModelOptionsError(ValueError):
    """模型参数配置结构或安全边界错误。"""


@dataclass(frozen=True)
class ModelParameterConfig:
    """单个 model_id 的 options 与命名 variants。"""

    options: dict[str, Any]
    variants: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ResolvedModelRequestConfig:
    """绑定到当前 ModelSelection 的不可变有效请求参数。"""

    connection_id: str
    model_id: str
    variant: str | None
    configured: bool
    options: dict[str, Any]
    options_digest: str

    def audit_summary(self) -> dict[str, Any]:
        """episode 可复现字段：选择状态、digest 与脱敏参数摘要。"""

        redacted = redact_model_options(self.options)
        return {
            "connection_id": self.connection_id,
            "model": self.model_id,
            "variant": self.variant,
            "configured": self.configured,
            "options_digest": self.options_digest,
            "options_key_paths": sorted(_json_key_paths(redacted)),
            "options_summary": redacted,
        }


def empty_resolved_config(
    *,
    connection_id: str,
    model_id: str,
    variant: str | None = None,
) -> ResolvedModelRequestConfig:
    return ResolvedModelRequestConfig(
        connection_id=connection_id,
        model_id=model_id,
        variant=variant,
        configured=False,
        options={},
        options_digest=options_digest({}),
    )


def deep_merge_options(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """object 递归合并；scalar 与 array 整体替换。null 是显式值，不表示删除。"""

    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge_options(existing, value)
        else:
            result[key] = value
    return result


def options_digest(options: Mapping[str, Any]) -> str:
    canonical = json.dumps(options, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def redact_model_options(value: Any) -> Any:
    """记录前脱敏；结构保留，可疑标量替换为占位符。"""

    if isinstance(value, Mapping):
        return {str(key): redact_model_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_model_options(item) for item in value]
    if isinstance(value, str):
        redacted, changed = redact_secret_like_text(value)
        return redacted if changed else value
    return value


def validate_options_object(
    options: object,
    *,
    path: str,
) -> dict[str, Any]:
    if not isinstance(options, dict):
        raise ModelOptionsError(f"{path} must be a JSON object")
    _reject_reserved_and_secrets(options, path=path)
    return dict(options)


def validate_variant_name(name: object, *, path: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ModelOptionsError(f"{path} variant name must be a non-empty string")
    if name == _DEFAULT_VARIANT_NAME:
        raise ModelOptionsError(f"{path}: variant name 'default' is reserved")
    if not _VARIANT_NAME_PATTERN.fullmatch(name):
        raise ModelOptionsError(
            f"{path}: variant name must match [A-Za-z0-9._-] and be at most 64 characters",
        )
    return name


def parse_model_parameter_config(
    raw: object,
    *,
    path: str,
) -> ModelParameterConfig:
    if not isinstance(raw, dict):
        raise ModelOptionsError(f"{path} must be a JSON object")
    unknown = sorted(str(key) for key in raw if key not in {"options", "variants"})
    if unknown:
        raise ModelOptionsError(f"{path} contains unknown field: {unknown[0]}")
    options = validate_options_object(
        raw.get("options", {}),
        path=f"{path}.options",
    )
    variants_raw = raw.get("variants", {})
    if not isinstance(variants_raw, dict):
        raise ModelOptionsError(f"{path}.variants must be a JSON object")
    variants: dict[str, dict[str, Any]] = {}
    for name, body in variants_raw.items():
        variant_name = validate_variant_name(name, path=f"{path}.variants")
        if not isinstance(body, dict):
            raise ModelOptionsError(f"{path}.variants.{variant_name} must be a JSON object")
        variants[variant_name] = validate_options_object(
            body,
            path=f"{path}.variants.{variant_name}",
        )
    return ModelParameterConfig(options=options, variants=variants)


def parse_connection_models(
    raw: object,
    *,
    path: str,
) -> dict[str, ModelParameterConfig]:
    if not isinstance(raw, dict):
        raise ModelOptionsError(f"{path} must be a JSON object")
    result: dict[str, ModelParameterConfig] = {}
    for model_id, body in raw.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelOptionsError(f"{path} model id must be a non-empty string")
        result[model_id] = parse_model_parameter_config(
            body,
            path=f"{path}.{model_id}",
        )
    return result


def resolve_model_request_config(
    *,
    connection_id: str,
    model_id: str,
    variant: str | None,
    model_config: ModelParameterConfig | None,
) -> ResolvedModelRequestConfig:
    """按 空对象 <- options <- variant 合并；未配置时 configured=false。"""

    if model_config is None:
        if variant is not None:
            raise ModelOptionsError(
                f"model variant is not available: connection={connection_id} "
                f"model={model_id} variant={variant}",
            )
        return empty_resolved_config(connection_id=connection_id, model_id=model_id)

    if variant is not None:
        if variant not in model_config.variants:
            raise ModelOptionsError(
                f"model variant is not available: connection={connection_id} "
                f"model={model_id} variant={variant}",
            )
        merged = deep_merge_options(model_config.options, model_config.variants[variant])
    else:
        merged = dict(model_config.options)

    return ResolvedModelRequestConfig(
        connection_id=connection_id,
        model_id=model_id,
        variant=variant,
        configured=bool(model_config.options or model_config.variants),
        options=merged,
        options_digest=options_digest(merged),
    )


def merge_provider_payload(
    payload: dict[str, Any],
    options: Mapping[str, Any],
    *,
    reserved: frozenset[str] | None = None,
) -> dict[str, Any]:
    """把已校验 options 合并进 provider payload；保留字段禁止被 options 覆盖。"""

    if not options:
        return payload
    blocked = reserved or RESERVED_TOP_LEVEL_FIELDS
    for key in options:
        if key in blocked:
            raise ModelOptionsError(f"options.{key} is managed by HaAgent")
    return deep_merge_options(payload, options)


def _reject_reserved_and_secrets(
    options: Mapping[str, Any],
    *,
    path: str,
) -> None:
    for key, value in options.items():
        key_path = f"{path}.{key}"
        if key in RESERVED_TOP_LEVEL_FIELDS:
            raise ModelOptionsError(f"{key_path} is managed by HaAgent")
        normalized_key = str(key).replace("-", "_").casefold()
        if (
            _SECRET_FIELD_PATTERN.search(str(key))
            or normalized_key == "token"
            or normalized_key.endswith("_token")
        ):
            raise ModelOptionsError(f"{key_path} looks like a secret field and is not allowed")
        if isinstance(value, dict):
            _reject_reserved_and_secrets(value, path=key_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    _reject_reserved_and_secrets(
                        item,
                        path=f"{key_path}[{index}]",
                    )


def _json_key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        if not value and prefix:
            paths.append(prefix)
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_json_key_paths(item, child))
    elif isinstance(value, list):
        if not value and prefix:
            paths.append(prefix)
        for index, item in enumerate(value):
            paths.extend(_json_key_paths(item, f"{prefix}[{index}]"))
    elif prefix:
        paths.append(prefix)
    return paths
