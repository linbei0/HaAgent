"""
agentfoundry/context/manifest.py - Context Manifest 数据结构

描述每次模型调用使用了哪些上下文来源，并支持 JSON 序列化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextSource:
    source_type: str
    name: str
    description: str
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        source = {
            "source_type": self.source_type,
            "name": self.name,
            "description": self.description,
        }
        if self.status is not None:
            source["status"] = self.status
        return source


@dataclass(frozen=True)
class ContextIndex:
    context_id: str
    model_input_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "model_input_path": self.model_input_path,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class ContextManifest:
    context_id: str
    provider: str
    workspace_root: str
    generated_at: str
    sources: list[ContextSource]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "provider": self.provider,
            "workspace_root": self.workspace_root,
            "generated_at": self.generated_at,
            "sources": [source.to_dict() for source in self.sources],
        }
