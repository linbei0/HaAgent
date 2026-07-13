"""
src/haagent/runtime/execution/safety_guard.py - 安全防护层（迁移残留）

ProgressGuard 已接管循环/停滞检测。本模块仅保留连续失败警告能力，
供尚未迁移的调用方与单元测试使用；不再对相同参数调用强制 abort。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SafetyViolation:
    type: str
    message: str
    should_abort: bool
    recovery_suggestion: str


@dataclass
class SafetyGuardState:
    consecutive_failures: int = 0
    recent_tool_signatures: list[str] = field(default_factory=list)


class SafetyGuard:
    """检测连续失败；相同参数循环终止已迁至 ProgressGuard。"""

    CONSECUTIVE_FAILURE_WARN_THRESHOLD = 3

    def __init__(self) -> None:
        self._state = SafetyGuardState()

    @property
    def state(self) -> SafetyGuardState:
        return self._state

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> SafetyViolation | None:
        signature = _call_signature(tool_name, args)
        self._state.recent_tool_signatures.append(signature)

        if result.get("status") == "error":
            self._state.consecutive_failures += 1
            if self._state.consecutive_failures >= self.CONSECUTIVE_FAILURE_WARN_THRESHOLD:
                return SafetyViolation(
                    type="repeated_failure",
                    message=(
                        f"{self._state.consecutive_failures} consecutive tool failures. "
                        f"Last: {tool_name}"
                    ),
                    should_abort=False,
                    recovery_suggestion=(
                        "Multiple failures in a row. Consider: (1) use file_read/file_list "
                        "to inspect current state, (2) use request_user_input to ask for "
                        "guidance, or (3) try a completely different approach."
                    ),
                )
            return None

        self._state.consecutive_failures = 0
        # 相同参数循环终止已由 ProgressGuard 基于 action+observation 处理
        return None


def _call_signature(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{sorted(args.items())}"
