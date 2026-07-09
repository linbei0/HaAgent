"""
haagent/context/sources.py - 上下文来源类型

定义 ContextSelection 与压缩层共用的候选、section 和决策数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ContextPlacement = Literal["system", "task"]


@dataclass(frozen=True)
class ContextCandidate:
    source_type: str
    source_id: str
    placement: ContextPlacement
    title: str
    content: str
    reason: str
    priority: int
    hard_required: bool = False
    skip_reason: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSection:
    """选择层与压缩层共用的上下文段落。

    key/source/kind/priority 供压缩预算与诊断使用；
    source_type/source_id/placement/chars 供选择审计与 placement 分组使用。
    """

    key: str
    title: str
    content: str
    source: str
    priority: int
    kind: str
    recent_rank: int | None = None
    hard_required: bool = False
    source_type: str | None = None
    source_id: str | None = None
    placement: ContextPlacement | None = None
    chars: int | None = None


@dataclass(frozen=True)
class ContextDecision:
    source_type: str
    source_id: str
    title: str
    reason: str
    placement: ContextPlacement | None
    priority: int
    chars: int
    selected: bool
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "reason": self.reason,
            "placement": self.placement,
            "priority": self.priority,
            "chars": self.chars,
            "selected": self.selected,
        }
        if self.skip_reason is not None:
            item["skip_reason"] = self.skip_reason
        if self.metadata:
            item["metadata"] = dict(self.metadata)
        return item
