"""
src/haagent/runtime/execution/safety_guard.py - 安全防护层

只检测真正的异常（死循环、连续失败），不猜测任务意图。
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
    """检测死循环和连续失败，不干预正常的探索行为。"""

    IDENTICAL_CALL_ABORT_THRESHOLD = 3
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

        if result.get("status") == "error":
            self._state.consecutive_failures += 1
            self._state.recent_tool_signatures.append(signature)
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
        self._state.recent_tool_signatures.append(signature)

        # 检测完全相同参数的连续重复调用
        recent = self._state.recent_tool_signatures
        threshold = self.IDENTICAL_CALL_ABORT_THRESHOLD
        if len(recent) >= threshold and len(set(recent[-threshold:])) == 1:
            return SafetyViolation(
                type="tool_loop",
                message=(
                    f"Detected infinite loop: {tool_name} called with identical "
                    f"arguments {threshold} times in a row."
                ),
                should_abort=True,
                recovery_suggestion=(
                    "This appears to be a stuck loop. Try a different tool, inspect "
                    "current state with file_read/file_list, or ask the user with "
                    "request_user_input."
                ),
            )

        return None


def _call_signature(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{sorted(args.items())}"
