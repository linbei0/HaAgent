"""
src/haagent/tools/schema_validation.py - 工具参数 JSON Schema 子集校验

递归校验静态工具实际使用的对象、数组、枚举与数值边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaValidationIssue:
    path: str
    message: str
    expected: object | None = None
    actual: object | None = None


def validate_json_value(value: object, schema: dict[str, Any], *, path: str = "") -> SchemaValidationIssue | None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_json_type(value, expected_type):
        label = path or "arguments"
        return SchemaValidationIssue(label, f"argument {label} must be {expected_type}", schema, type(value).__name__)
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        label = path or "arguments"
        return SchemaValidationIssue(label, f"argument {label} must be one of: {', '.join(map(str, enum))}", schema, value)
    if isinstance(value, dict):
        return _validate_object(value, schema, path=path)
    if isinstance(value, list):
        return _validate_array(value, schema, path=path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _validate_number(value, schema, path=path)
    return None


def _validate_object(value: dict[object, object], schema: dict[str, Any], *, path: str) -> SchemaValidationIssue | None:
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    for name in required if isinstance(required, list) else []:
        if name not in value:
            field = _join_path(path, str(name))
            return SchemaValidationIssue(field, f"missing required argument: {field}", properties.get(name))
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                field = _join_path(path, str(name))
                return SchemaValidationIssue(field, f"unexpected argument: {field}", actual=value[name])
    for name, item in value.items():
        item_schema = properties.get(name)
        if isinstance(item_schema, dict):
            issue = validate_json_value(item, item_schema, path=_join_path(path, str(name)))
            if issue is not None:
                return issue
    return None


def _validate_array(value: list[object], schema: dict[str, Any], *, path: str) -> SchemaValidationIssue | None:
    label = path or "arguments"
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if isinstance(minimum, int) and len(value) < minimum:
        return SchemaValidationIssue(label, f"argument {label} must contain at least {minimum} items", schema, len(value))
    if isinstance(maximum, int) and len(value) > maximum:
        return SchemaValidationIssue(label, f"argument {label} must contain at most {maximum} items", schema, len(value))
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            issue = validate_json_value(item, item_schema, path=f"{label}[{index}]")
            if issue is not None:
                return issue
    return None


def _validate_number(value: int | float, schema: dict[str, Any], *, path: str) -> SchemaValidationIssue | None:
    label = path or "arguments"
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        return SchemaValidationIssue(label, f"argument {label} must be >= {minimum}", schema, value)
    if isinstance(maximum, (int, float)) and value > maximum:
        return SchemaValidationIssue(label, f"argument {label} must be <= {maximum}", schema, value)
    return None


def _join_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True
