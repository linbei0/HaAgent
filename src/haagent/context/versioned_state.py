"""
haagent/context/versioned_state.py - 版本化模型上下文状态

用不可变快照和顶层 section 增量描述运行时状态，保证同一 epoch 内只追加消息。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


CONTEXT_STATE_SCHEMA_VERSION = 1
_HASH_PREFIX = "sha256:"


class ContextStateError(ValueError):
    """上下文状态无法通过严格 schema 或基线校验。"""


def canonical_json(value: object) -> str:
    """返回跨字段顺序稳定的 JSON；不接受不可序列化的运行时对象。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContextStateError(f"context state is not canonical JSON: {error}") from error


def content_digest(value: object) -> str:
    return _HASH_PREFIX + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _snapshot_digest(
    *,
    sections: Mapping[str, object],
    source_digests: Mapping[str, str],
) -> str:
    # snapshot_id 表示内容本身，不把 epoch/revision 混入 hash；同一内容跨版本仍可复用。
    payload = {"sections": dict(sections), "source_digests": dict(source_digests)}
    return content_digest(payload)


def _validate_version(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContextStateError(f"{field} must be a non-negative integer")
    return value


def _normalize_sections(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContextStateError("sections must be an object")
    sections = dict(value)
    for key in sections:
        if not isinstance(key, str) or not key:
            raise ContextStateError("section names must be non-empty strings")
    # Canonicalization is also the schema check for nested values.
    canonical_json(sections)
    return sections


def _normalize_source_digests(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContextStateError("source_digests must be an object")
    result = dict(value)
    if any(not isinstance(key, str) or not key for key in result):
        raise ContextStateError("source digest names must be non-empty strings")
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ContextStateError("source digests must be non-empty strings")
    return result


@dataclass(frozen=True)
class ContextStateSnapshot:
    schema_version: int
    epoch: int
    revision: int
    snapshot_id: str
    sections: Mapping[str, object]
    source_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != CONTEXT_STATE_SCHEMA_VERSION
        ):
            raise ContextStateError(f"unsupported context state schema_version: {self.schema_version}")
        _validate_version(self.epoch, "epoch")
        _validate_version(self.revision, "revision")
        sections = _normalize_sections(self.sections)
        source_digests = _normalize_source_digests(self.source_digests)
        expected = _snapshot_digest(
            sections=sections,
            source_digests=source_digests,
        )
        if self.snapshot_id != expected:
            raise ContextStateError(
                f"snapshot_id does not match canonical state: expected {expected}, got {self.snapshot_id}"
            )
        object.__setattr__(self, "sections", MappingProxyType(sections))
        object.__setattr__(self, "source_digests", MappingProxyType(source_digests))

    @classmethod
    def create(
        cls,
        *,
        epoch: int = 0,
        revision: int = 0,
        sections: Mapping[str, object] | None = None,
        source_digests: Mapping[str, str] | None = None,
    ) -> "ContextStateSnapshot":
        normalized_sections = _normalize_sections(sections or {})
        normalized_sources = _normalize_source_digests(source_digests or {})
        return cls(
            schema_version=CONTEXT_STATE_SCHEMA_VERSION,
            epoch=epoch,
            revision=revision,
            snapshot_id=_snapshot_digest(
                sections=normalized_sections,
                source_digests=normalized_sources,
            ),
            sections=normalized_sections,
            source_digests=normalized_sources,
        )

    @classmethod
    def from_dict(cls, raw: object) -> "ContextStateSnapshot":
        if not isinstance(raw, dict):
            raise ContextStateError("context state snapshot must be an object")
        required = {"schema_version", "epoch", "revision", "snapshot_id", "sections", "source_digests"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ContextStateError(f"context state snapshot missing fields: {', '.join(missing)}")
        if not isinstance(raw["snapshot_id"], str) or not raw["snapshot_id"].startswith(_HASH_PREFIX):
            raise ContextStateError("snapshot_id must be a sha256 digest")
        return cls(
            schema_version=raw["schema_version"],
            epoch=raw["epoch"],
            revision=raw["revision"],
            snapshot_id=raw["snapshot_id"],
            sections=raw["sections"],
            source_digests=raw["source_digests"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "revision": self.revision,
            "snapshot_id": self.snapshot_id,
            "sections": dict(self.sections),
            "source_digests": dict(self.source_digests),
        }


@dataclass(frozen=True)
class ContextStateDelta:
    schema_version: int
    epoch: int
    base_revision: int
    revision: int
    base_snapshot_id: str
    snapshot_id: str
    changed: Mapping[str, object]
    removed: list[str]
    source_digests: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != CONTEXT_STATE_SCHEMA_VERSION
        ):
            raise ContextStateError(f"unsupported context state schema_version: {self.schema_version}")
        _validate_version(self.epoch, "epoch")
        _validate_version(self.base_revision, "base_revision")
        _validate_version(self.revision, "revision")
        if self.revision <= self.base_revision:
            raise ContextStateError("revision must advance beyond base_revision")
        if not isinstance(self.base_snapshot_id, str) or not self.base_snapshot_id.startswith(_HASH_PREFIX):
            raise ContextStateError("base_snapshot_id must be a sha256 digest")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.startswith(_HASH_PREFIX):
            raise ContextStateError("snapshot_id must be a sha256 digest")
        changed = _normalize_sections(self.changed)
        removed = list(self.removed)
        if any(not isinstance(key, str) or not key for key in removed):
            raise ContextStateError("removed section names must be non-empty strings")
        if len(set(removed)) != len(removed):
            raise ContextStateError("removed section names must be unique")
        if set(changed) & set(removed):
            raise ContextStateError("a section cannot be changed and removed in the same delta")
        sources = _normalize_source_digests(self.source_digests)
        object.__setattr__(self, "changed", MappingProxyType(changed))
        object.__setattr__(self, "removed", removed)
        object.__setattr__(self, "source_digests", MappingProxyType(sources))

    @classmethod
    def from_dict(cls, raw: object) -> "ContextStateDelta":
        if not isinstance(raw, dict):
            raise ContextStateError("context state delta must be an object")
        required = {
            "schema_version", "epoch", "base_revision", "revision", "base_snapshot_id",
            "snapshot_id", "changed", "removed",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ContextStateError(f"context state delta missing fields: {', '.join(missing)}")
        return cls(
            **{key: raw[key] for key in required},
            source_digests=raw.get("source_digests", {}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "context_state_delta",
            "epoch": self.epoch,
            "base_revision": self.base_revision,
            "revision": self.revision,
            "base_snapshot_id": self.base_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "changed": dict(self.changed),
            "removed": list(self.removed),
            "source_digests": dict(self.source_digests),
        }


def diff_snapshots(base: ContextStateSnapshot, target: ContextStateSnapshot) -> ContextStateDelta | None:
    if base.epoch != target.epoch:
        raise ContextStateError("cannot diff snapshots from different epochs")
    if base.sections == target.sections and base.source_digests == target.source_digests:
        return None
    if target.revision <= base.revision:
        raise ContextStateError("target snapshot revision must advance beyond base")
    changed = {
        key: value
        for key, value in target.sections.items()
        if base.sections.get(key, object()) != value
    }
    removed = sorted(set(base.sections) - set(target.sections))
    return ContextStateDelta(
        schema_version=target.schema_version,
        epoch=target.epoch,
        base_revision=base.revision,
        revision=target.revision,
        base_snapshot_id=base.snapshot_id,
        snapshot_id=target.snapshot_id,
        changed=changed,
        removed=removed,
        source_digests=dict(target.source_digests),
    )


def apply_delta(base: ContextStateSnapshot, delta: ContextStateDelta) -> ContextStateSnapshot:
    if delta.epoch != base.epoch:
        raise ContextStateError("delta epoch does not match base snapshot")
    if delta.base_revision != base.revision:
        raise ContextStateError(
            f"stale context state delta: expected base revision {base.revision}, got {delta.base_revision}"
        )
    if delta.base_snapshot_id != base.snapshot_id:
        raise ContextStateError("delta base_snapshot_id does not match base snapshot")
    sections = dict(base.sections)
    sections.update(delta.changed)
    for key in delta.removed:
        sections.pop(key, None)
    snapshot = ContextStateSnapshot.create(
        epoch=base.epoch,
        revision=delta.revision,
        sections=sections,
        source_digests=delta.source_digests or base.source_digests,
    )
    if snapshot.snapshot_id != delta.snapshot_id:
        raise ContextStateError("applied delta does not produce declared snapshot_id")
    return snapshot


def next_snapshot(
    base: ContextStateSnapshot,
    sections: Mapping[str, object],
    source_digests: Mapping[str, str] | None = None,
) -> tuple[ContextStateSnapshot, ContextStateDelta | None]:
    normalized_sections = _normalize_sections(sections)
    if source_digests is None:
        normalized_sources = (
            dict(base.source_digests)
            if normalized_sections == base.sections
            else {key: content_digest(value) for key, value in normalized_sections.items()}
        )
    else:
        normalized_sources = _normalize_source_digests(source_digests)
    if normalized_sections == base.sections and normalized_sources == base.source_digests:
        return base, None
    target = ContextStateSnapshot.create(
        epoch=base.epoch,
        revision=base.revision + 1,
        sections=normalized_sections,
        source_digests=normalized_sources,
    )
    return target, diff_snapshots(base, target)


def reset_epoch(snapshot: ContextStateSnapshot, epoch: int) -> ContextStateSnapshot:
    if epoch <= snapshot.epoch:
        raise ContextStateError("new epoch must be greater than current epoch")
    return ContextStateSnapshot.create(epoch=epoch, revision=1, sections=snapshot.sections, source_digests=snapshot.source_digests)
