"""
haagent/context/manifest.py - Context Manifest 数据结构

描述每次模型调用使用的初始消息，支持 JSON 序列化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextIndex:
    context_id: str
    model_input_path: str
    manifest_path: str
    budget: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        index: dict[str, Any] = {
            "context_id": self.context_id,
            "model_input_path": self.model_input_path,
            "manifest_path": self.manifest_path,
        }
        if self.budget is not None:
            index["budget"] = self.budget
        return index


@dataclass(frozen=True)
class ContextManifest:
    context_id: str
    provider: str
    workspace_root: str
    generated_at: str
    message_count: int
    system_chars: int
    task_chars: int
    next_action: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "context_id": self.context_id,
            "provider": self.provider,
            "workspace_root": self.workspace_root,
            "generated_at": self.generated_at,
            "message_count": self.message_count,
            "system_chars": self.system_chars,
            "task_chars": self.task_chars,
        }
        if self.next_action is not None:
            manifest["next_action"] = self.next_action
        if self.memory is not None:
            manifest["memory"] = self.memory
        return manifest
