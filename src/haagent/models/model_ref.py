"""
haagent/models/model_ref.py - 模型选择与运行时绑定值对象

提供跨配置、session、TUI 和 gateway 传递的不可变模型身份及解析结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from haagent.models.model_settings import ModelSettings


@dataclass(frozen=True)
class ModelRef:
    """模型选择身份；不包含凭据和 provider 请求 payload。"""

    connection_id: str
    model: str
    variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "connection_id": self.connection_id,
            "model": self.model,
        }
        if self.variant is not None:
            payload["variant"] = self.variant
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, field_name: str = "model_ref") -> "ModelRef":
        connection_id = value.get("connection_id")
        model = value.get("model")
        if not isinstance(connection_id, str) or not connection_id.strip():
            raise ValueError(f"{field_name}.connection_id is required")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{field_name}.model is required")
        variant = value.get("variant")
        if variant is not None and (not isinstance(variant, str) or not variant.strip()):
            raise ValueError(f"{field_name}.variant must be a non-empty string when present")
        return cls(connection_id=connection_id, model=model, variant=variant)


@dataclass(frozen=True)
class ModelChoice:
    """TUI 可直接消费的完整模型选择项。"""

    ref: ModelRef
    connection_name: str
    provider_name: str
    model_name: str
    variants: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedModel:
    """已解析的模型运行时绑定；api_key 不进入 repr。"""

    ref: ModelRef
    provider: str
    base_url: str
    runtime_kind: str
    settings: "ModelSettings"
    credential: "ResolvedCredential"


@dataclass(frozen=True)
class ResolvedCredential:
    api_key: str = field(repr=False)
    api_key_env: str
    source: str
    source_used: str


@dataclass(frozen=True)
class ModelInvocation:
    """一次模型调用的统一输入；provider-specific payload 由 adapter 构造。"""

    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    settings: "ModelSettings"
