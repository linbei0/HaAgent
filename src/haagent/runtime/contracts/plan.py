"""
src/haagent/runtime/contracts/plan.py - Agent Plan Trace 构建模块

从 task.yaml 的确定性字段生成 planning 阶段 trace，不发起额外模型调用。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.contracts.task import TaskSpec


def build_plan(task: TaskSpec) -> dict[str, Any]:
    """从 TaskSpec 生成 Agent Plan Trace v0。"""
    return {
        "goal": task.goal,
        "allowed_tools": list(task.allowed_tools),
        "acceptance_criteria": list(task.acceptance_criteria),
        "verification_commands": list(task.verification_commands),
        "planned_steps": _planned_steps(task),
    }


def _planned_steps(task: TaskSpec) -> list[str]:
    steps = ["Clarify the task goal and constraints from task.yaml."]
    if task.allowed_tools:
        steps.append(f"Use allowed tools: {', '.join(task.allowed_tools)}.")
    else:
        steps.append("Proceed without tool calls.")
    if task.acceptance_criteria:
        steps.append(f"Check acceptance criteria: {'; '.join(task.acceptance_criteria)}.")
    else:
        steps.append("Check completion against the task goal.")
    steps.append("Run verification commands if provided.")
    return steps
