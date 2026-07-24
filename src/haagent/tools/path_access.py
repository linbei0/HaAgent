"""
haagent/tools/path_access.py - 工具路径权限协调

把纯 PathPolicy 判定转换为同一次工具调用内可暂停、可恢复的外部目录授权。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from haagent.runtime.execution.human_interaction import ToolPermissionRequest
from haagent.runtime.execution.path_policy import (
    PathAccess,
    PathPolicy,
    authorize_external_root,
    classify_path_access,
)
from haagent.tools.base import ToolExecutionContext, tool_error


def _is_haagent_own_artifact(path: Path) -> bool:
    """识别 HaAgent 自身 episode artifacts/tool-results 路径。

    HaAgent 把超长工具结果落盘到 ~/.haagent/runs/episodes/**/artifacts/tool-results/，
    并主动引导模型用 file_read 回读。该目录是 HaAgent 自己写出的内容，回读属于内部
    可信读取，不应触发外部目录权限审批（对齐 OpenCode 对自身截断输出目录的无条件放行）。
    """
    resolved = path.resolve()
    parts = resolved.parts
    lowered = [part.casefold() for part in parts]
    try:
        idx = lowered.index(".haagent")
    except ValueError:
        return False
    tail = [p.casefold() for p in parts[idx + 1 :]]
    # 需要包含 runs/<...>/artifacts/tool-results 结构
    return (
        len(tail) >= 4
        and tail[0] == "runs"
        and "artifacts" in tail
        and "tool-results" in tail
    )


def resolve_tool_paths(
    paths: Iterable[str | Path],
    policy: PathPolicy,
    access: PathAccess,
    context: ToolExecutionContext | None,
) -> list[Path] | dict[str, object]:
    """一次判定全部路径，必要时只发出一个外部目录权限请求。"""
    decisions = [classify_path_access(str(path), policy, access) for path in paths]
    for decision in decisions:
        if decision.status == "denied":
            return tool_error(decision.error_type, decision.message, retryable=False)

    pending_roots = _unique_paths(
        decision.authorization_root
        for decision, path in zip(decisions, paths)
        if decision.status == "approval_required"
        # HaAgent 自身 artifacts 目录的只读回读免审批；写入仍走正常审批。
        and not (access == "read" and _is_haagent_own_artifact(Path(path)))
    )
    if pending_roots:
        patterns = tuple(_directory_pattern(root) for root in pending_roots)
        request = ToolPermissionRequest(
            permission="external_directory",
            patterns=patterns,
            always=patterns,
            metadata={
                "directories": [str(root) for root in pending_roots],
                "directory": str(pending_roots[0]),
                "access": access,
            },
            question=_permission_question(pending_roots),
            reason="当前工具请求访问 workspace 之外的目录",
            risk_level="medium" if access == "read" else "high",
        )
        response = context.ask(request) if context is not None else None
        if response is None:
            return tool_error(
                "approval_required",
                "访问外部目录需要用户确认",
                retryable=False,
                directories=[str(root) for root in pending_roots],
                access=access,
            )
        if not response.approved:
            return tool_error(
                "approval_denied",
                "用户拒绝访问外部目录",
                retryable=False,
                directories=[str(root) for root in pending_roots],
                access=access,
            )
        if response.answer == "always":
            for root in pending_roots:
                authorize_external_root(policy, root, access)

    resolved = [decision.path for decision in decisions]
    if any(path is None for path in resolved):
        return tool_error("path_policy_denied", "路径判定缺少解析结果", retryable=False)
    return [path for path in resolved if path is not None]


def _unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _directory_pattern(root: Path) -> str:
    return (root.resolve() / "*").as_posix()


def _permission_question(roots: list[Path]) -> str:
    if len(roots) == 1:
        return f"允许访问外部目录 {roots[0]} 吗？"
    return f"允许访问这 {len(roots)} 个外部目录吗？"
