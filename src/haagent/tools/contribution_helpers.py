"""
haagent/tools/contribution_helpers.py - 静态工具 contribution 共用摘要 helper

供 contributions 登记 args/result/observation 投影，不参与 policy。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.command import redact_secret_like_text
from haagent.runtime.execution.guardrails import GuardrailResult, SECRET_TOKEN_PATTERN


def summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def interaction_summary_value(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def changed_files_summary(result: dict[str, object]) -> list[dict[str, object]]:
    changed_files = result.get("changed_files")
    if not isinstance(changed_files, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in changed_files:
        if not isinstance(item, dict):
            continue
        summary: dict[str, object] = {
            "path": summary_value(str(item.get("path", "")), 160),
            "change_type": str(item.get("change_type", "modified")),
            "additions": item.get("additions"),
            "deletions": item.get("deletions"),
        }
        if "bytes_written" in item:
            summary["bytes_written"] = item.get("bytes_written")
        if "replacements" in item:
            summary["replacements"] = item.get("replacements")
        summaries.append(summary)
    return summaries


def first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def string_value(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def first_present_string(*values: object) -> str:
    return string_value(first_present(*values))


def compact_excerpt(value: str) -> tuple[str, bool]:
    from haagent.context.compression.sections import collapse_text_head_tail

    redacted, changed = redact_secret_like_text(value)
    compacted, collapsed_chars = collapse_text_head_tail(
        redacted,
        max_chars=600,
        head_chars=300,
        tail_chars=160,
    )
    return compacted, changed or collapsed_chars > 0


def format_search_match(match: object) -> str:
    if isinstance(match, dict):
        path = first_present_string(match.get("path"), match.get("file"))
        line = match.get("line") or match.get("line_number")
        text = compact_excerpt(
            first_present_string(match.get("text"), match.get("line_text")),
        )[0]
        return f"{path}:{line}: {text}"
    return compact_excerpt(str(match))[0]


def secret_write_guardrail(tool_name: str, args: dict[str, Any]) -> GuardrailResult | None:
    text = "\n".join(str(args.get(field, "")) for field in ["content", "old_text", "new_text"])
    if SECRET_TOKEN_PATTERN.search(text):
        return GuardrailResult(
            status="blocked",
            scope="tool_input",
            rule_id=f"{tool_name}_secret_write",
            message=f"{tool_name} arguments contain a secret-like token",
            severity="high",
        )
    return None


def patch_set_secret_guardrail(args: dict[str, Any]) -> GuardrailResult | None:
    replacements = args.get("replacements")
    if not isinstance(replacements, list):
        return None
    for replacement in replacements:
        if not isinstance(replacement, dict):
            continue
        text = "\n".join(
            str(replacement.get(field, "")) for field in ["old_text", "new_text"]
        )
        if SECRET_TOKEN_PATTERN.search(text):
            return GuardrailResult(
                status="blocked",
                scope="tool_input",
                rule_id="apply_patch_set_secret_write",
                message="apply_patch_set arguments contain a secret-like token",
                severity="high",
            )
    return None


def shell_guardrail(args: dict[str, Any]) -> GuardrailResult | None:
    command = str(args.get("command", ""))
    lowered = command.lower()
    if any(
        pattern in lowered
        for pattern in ["~/.ssh", "id_rsa", "api_key", "api key", "secret_key"]
    ):
        return GuardrailResult(
            status="blocked",
            scope="tool_input",
            rule_id="shell_secret_exfiltration",
            message="shell command attempts to read or print secrets",
            severity="high",
        )
    if any(
        pattern in lowered
        for pattern in ["rm -rf /", "del /f /s /q c:\\", "format c:"]
    ):
        return GuardrailResult(
            status="blocked",
            scope="tool_input",
            rule_id="shell_destructive_outside_workspace",
            message="shell command is destructive outside workspace scope",
            severity="high",
        )
    return None


def code_run_guardrail(args: dict[str, Any]) -> GuardrailResult | None:
    code = str(args.get("code", ""))
    lowered = code.lower()
    if any(pattern in lowered for pattern in ["id_rsa", "api_key", "api key", "secret_key"]):
        return GuardrailResult(
            status="blocked",
            scope="tool_input",
            rule_id="code_run_secret_access",
            message="python code attempts to read or print secrets",
            severity="high",
        )
    return None
