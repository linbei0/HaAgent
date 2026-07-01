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
    compaction: dict[str, Any] | None = None
    source_diagnostics: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    compact_readiness: dict[str, Any] | None = None
    auto_compact_trigger: dict[str, Any] | None = None
    session_compaction: dict[str, Any] | None = None
    full_compact_contract: dict[str, Any] | None = None
    full_compact: dict[str, Any] | None = None

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
        if self.compaction is not None:
            manifest["compaction"] = self.compaction
        if self.source_diagnostics is not None:
            manifest["source_diagnostics"] = self.source_diagnostics
        if self.selection is not None:
            manifest["selection"] = self.selection
        if self.compact_readiness is not None:
            manifest["compact_readiness"] = self.compact_readiness
        if self.auto_compact_trigger is not None:
            manifest["auto_compact_trigger"] = self.auto_compact_trigger
        if self.session_compaction is not None:
            manifest["session_compaction"] = self.session_compaction
        if self.full_compact_contract is not None:
            manifest["full_compact_contract"] = self.full_compact_contract
        if self.full_compact is not None:
            manifest["full_compact"] = self.full_compact
        return manifest
