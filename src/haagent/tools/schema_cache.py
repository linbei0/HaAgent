"""
src/haagent/tools/schema_cache.py - 工具模型 schema 缓存

按 registry.schema_version + 名称集合缓存 canonical UTF-8 JSON，
导出时 json.loads 深拷贝；hash 用 sort_keys，模型字段顺序保持声明顺序。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from haagent.tools.registry import ToolRuntimeRegistry, allowed_tool_definitions


class ToolSchemaCache:
    def __init__(self) -> None:
        # (schema_version, names_key) -> utf-8 json bytes of schema list
        self._cache: dict[tuple[str, tuple[str, ...]], bytes] = {}

    def export(
        self,
        names: list[str],
        registry: ToolRuntimeRegistry,
        *,
        diagnostics_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> list[dict[str, Any]]:
        key = self._key(names, registry)
        payload = self._cache.get(key)
        status = "hit" if payload is not None else "miss"
        if payload is None:
            schemas = [definition.to_model_schema() for definition in allowed_tool_definitions(names, registry=registry)]
            # 保留 properties/required/enum 声明顺序；不在序列化时 sort_keys。
            payload = json.dumps(schemas, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._cache[key] = payload
        if diagnostics_sink is not None:
            diagnostics_sink(
                {
                    "status": status,
                    "count": len(names),
                    "bytes": len(payload),
                    "fingerprint": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                },
            )
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _key(names: list[str], registry: ToolRuntimeRegistry) -> tuple[str, tuple[str, ...]]:
        return (registry.schema_version, tuple(names))


_DEFAULT_SCHEMA_CACHE = ToolSchemaCache()


def default_tool_schema_cache() -> ToolSchemaCache:
    return _DEFAULT_SCHEMA_CACHE
