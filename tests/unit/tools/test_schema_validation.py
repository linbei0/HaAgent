"""
tests/unit/tools/test_schema_validation.py - 递归工具参数 schema 校验测试

验证嵌套对象、数组边界和未知字段都由统一校验器处理。
"""

from haagent.tools.schema_validation import SchemaValidationIssue, validate_json_value


QUESTION_SCHEMA = {
    "type": "object",
    "required": ["questions"],
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["id", "header", "question"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "header": {"type": "string"},
                    "question": {"type": "string"},
                    "multiple": {"type": "boolean"},
                },
            },
        },
    },
}


def test_recursive_schema_validation_accepts_nested_questions() -> None:
    issue = validate_json_value(
        {"questions": [{"id": "storage", "header": "存储", "question": "选哪个？"}]},
        QUESTION_SCHEMA,
    )

    assert issue is None


def test_recursive_schema_validation_reports_nested_unknown_field() -> None:
    issue = validate_json_value(
        {
            "questions": [
                {"id": "storage", "header": "存储", "question": "选哪个？", "secret": True},
            ],
        },
        QUESTION_SCHEMA,
    )

    assert issue == SchemaValidationIssue(
        path="questions[0].secret",
        message="unexpected argument: questions[0].secret",
        expected=None,
        actual=True,
    )


def test_recursive_schema_validation_enforces_array_bounds() -> None:
    issue = validate_json_value({"questions": []}, QUESTION_SCHEMA)

    assert issue is not None
    assert issue.path == "questions"
    assert issue.message == "argument questions must contain at least 1 items"
