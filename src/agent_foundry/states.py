"""
agent_foundry/states.py - Run 状态枚举

定义 MVP Run Orchestrator 允许出现的状态集合。
"""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
