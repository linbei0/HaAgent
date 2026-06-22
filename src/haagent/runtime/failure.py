"""
haagent/runtime/failure.py - Failure Taxonomy v1

集中定义 run 失败归因 category，避免 orchestrator 中散落字符串。
"""

from __future__ import annotations

from enum import StrEnum


class FailureCategory(StrEnum):
    TASK_SPEC = "Task Spec Failure"
    CONTEXT = "Context Failure"
    MODEL = "Model Failure"
    MODEL_CALL = "Model Call Failure"
    TOOL_INTERFACE = "Tool Interface Failure"
    TOOL_ARGUMENT = "Tool Argument Failure"
    USER_DENIED = "User Denied Failure"
    VERIFICATION = "Verification Failure"
    LOOP_LIMIT = "Loop Limit Failure"
    RUNTIME = "Runtime Failure"
