"""
src/haagent/runtime/session/chat_task_contract.py - 自然语言临时任务契约

为 TUI 自然语言 turn 生成确定性的 TaskSpec 约束与验收标准。
"""

from __future__ import annotations

from pathlib import Path


def build_chat_task_contract(
    *,
    goal: str,
    workspace_root: Path,
    target_paths: list[str],
) -> dict[str, list[str]]:
    """基于已结构化的目标路径生成临时任务契约，不推断自然语言意图。"""
    del goal, workspace_root
    acceptance_criteria = ["Address the stated user goal."]
    if target_paths:
        acceptance_criteria.append(
            "Use explicitly referenced target paths as task context: "
            + ", ".join(target_paths)
            + ".",
        )
    return {
        "constraints": ["Keep file and command operations within the workspace root."],
        "acceptance_criteria": acceptance_criteria,
    }
