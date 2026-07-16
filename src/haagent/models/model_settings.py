"""
haagent/models/model_settings.py - provider 原生模型参数值对象

集中负责 options 合并、digest 和安全审计摘要；provider adapter 不解析配置文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from haagent.models.model_options import (
    deep_merge_options,
    options_digest,
    redact_model_options,
    _json_key_paths,
)


@dataclass(frozen=True)
class ModelSettings:
    options: Mapping[str, Any]
    configured: bool
    digest: str

    @classmethod
    def empty(cls) -> "ModelSettings":
        return cls(options={}, configured=False, digest=options_digest({}))

    @classmethod
    def from_options(cls, options: Mapping[str, Any], *, configured: bool = True) -> "ModelSettings":
        value = dict(options)
        return cls(options=value, configured=configured, digest=options_digest(value))

    def resolve(self, override: Mapping[str, Any]) -> "ModelSettings":
        merged = deep_merge_options(self.options, override)
        return self.from_options(merged, configured=self.configured or bool(override))

    def to_traceable_dict(self) -> dict[str, Any]:
        redacted = redact_model_options(self.options)
        return {
            "configured": self.configured,
            "options_digest": self.digest,
            "options_key_paths": sorted(_json_key_paths(redacted)),
            "options_summary": redacted,
        }
