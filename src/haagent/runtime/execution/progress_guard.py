"""
src/haagent/runtime/execution/progress_guard.py - 任务进展与循环检测状态机

观察已完成 model-tool 轮次，识别高置信循环与低置信停滞；
running/approval/user-input/streaming 不进入窗口，避免误伤合法等待。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class _ProgressGuardState(StrEnum):
    HEALTHY = "healthy"
    WARNED = "warned"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProgressDecision:
    level: Literal["none", "warn", "block"]
    reason: str = ""
    pattern: str = ""


# pairs 项: (tool_name, args, observation, status)
ProgressPair = tuple[str, dict[str, Any], str, str]


@dataclass(frozen=True)
class ProgressFrame:
    """单次已完成模型工具轮次的聚合帧（并行工具合并为一帧）。"""

    pairs: tuple[ProgressPair, ...]
    workspace_changed: bool = False
    verification_progressed: bool = False
    context_chars: int = 0
    has_running_tool: bool = False
    has_running_worker: bool = False
    waiting_approval: bool = False
    waiting_user_input: bool = False


_NOISE_ARG_KEYS = frozenset(
    {
        "tool_call_id",
        "call_id",
        "response_id",
        "timestamp",
        "duration",
        "duration_ms",
        "episode_path",
        "episode_id",
    }
)

# observation 中常见运行噪声，签名前剔除
_NOISE_OBS_PATTERNS = (
    re.compile(r"\bcall_[A-Za-z0-9_-]+\b"),
    re.compile(r"\bresp_[A-Za-z0-9_-]+\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*ms\b", re.IGNORECASE),
)


class ProgressGuard:
    """有界 action-observation 窗口上的进展状态机。"""

    WINDOW_SIZE = 20
    IDENTICAL_PAIR_THRESHOLD = 3
    ERROR_LOOP_THRESHOLD = 3
    AB_CYCLE_LENGTH = 6
    STAGNATION_TURNS = 4

    def __init__(self) -> None:
        self._state = _ProgressGuardState.HEALTHY
        self._signatures: deque[str] = deque(maxlen=self.WINDOW_SIZE)
        self._action_signatures: deque[str] = deque(maxlen=self.WINDOW_SIZE)
        self._error_flags: deque[bool] = deque(maxlen=self.WINDOW_SIZE)
        self._context_sizes: deque[int] = deque(maxlen=self.WINDOW_SIZE)

    def reset(self) -> None:
        """用户确认继续或重规划后清空旧循环证据。"""

        self._clear()
        self._state = _ProgressGuardState.HEALTHY

    def observe(self, frame: ProgressFrame) -> ProgressDecision:
        if self._is_excluded(frame):
            return ProgressDecision(level="none")

        if frame.workspace_changed or frame.verification_progressed:
            # 明确进展：清空窗口并以当前帧作为新起点
            self._clear()
            self._append_frame(frame)
            self._state = _ProgressGuardState.HEALTHY
            return ProgressDecision(level="none", reason="progress", pattern="")

        self._append_frame(frame)

        high = self._detect_high_confidence()
        if high is not None:
            return self._apply_high_confidence(high)

        if self._detect_stagnation():
            return self._apply_stagnation()

        return ProgressDecision(level="none")

    def _clear(self) -> None:
        self._signatures.clear()
        self._action_signatures.clear()
        self._error_flags.clear()
        self._context_sizes.clear()

    def _append_frame(self, frame: ProgressFrame) -> None:
        self._signatures.append(_frame_pair_signature(frame))
        self._action_signatures.append(_frame_action_signature(frame))
        self._error_flags.append(_frame_is_error(frame))
        self._context_sizes.append(frame.context_chars)

    def _is_excluded(self, frame: ProgressFrame) -> bool:
        # 合法等待不进入停滞窗口
        return bool(
            frame.has_running_tool
            or frame.has_running_worker
            or frame.waiting_approval
            or frame.waiting_user_input
        )

    def _detect_high_confidence(self) -> ProgressDecision | None:
        # 全 error 时优先 error_loop，避免与 identical_pair 语义重叠
        error_loop = self._error_loop_hit()
        if error_loop is not None:
            return error_loop
        identical = self._identical_pair_hit()
        if identical is not None:
            return identical
        ab = self._ab_cycle_hit()
        if ab is not None:
            return ab
        return None

    def _identical_pair_hit(self) -> ProgressDecision | None:
        n = self.IDENTICAL_PAIR_THRESHOLD
        if len(self._signatures) < n:
            return None
        recent = list(self._signatures)[-n:]
        if len(set(recent)) == 1:
            return ProgressDecision(
                level="warn",
                reason="相同 action 与 observation 连续重复，可能陷入循环。",
                pattern="identical_pair",
            )
        return None

    def _error_loop_hit(self) -> ProgressDecision | None:
        n = self.ERROR_LOOP_THRESHOLD
        if len(self._action_signatures) < n:
            return None
        actions = list(self._action_signatures)[-n:]
        errors = list(self._error_flags)[-n:]
        if all(errors) and len(set(actions)) == 1:
            return ProgressDecision(
                level="warn",
                reason="相同 action 连续失败，建议换策略或检查输入。",
                pattern="error_loop",
            )
        return None

    def _ab_cycle_hit(self) -> ProgressDecision | None:
        n = self.AB_CYCLE_LENGTH
        if len(self._signatures) < n:
            return None
        recent = list(self._signatures)[-n:]
        a, b = recent[0], recent[1]
        if a == b:
            return None
        expected = [a, b] * (n // 2)
        if recent == expected:
            return ProgressDecision(
                level="warn",
                reason="检测到 A-B 交替循环，可能未取得实质进展。",
                pattern="ab_cycle",
            )
        return None

    def _detect_stagnation(self) -> bool:
        n = self.STAGNATION_TURNS
        if len(self._context_sizes) < n:
            return False
        contexts = list(self._context_sizes)[-n:]
        # 上下文持续增长且无 workspace/verification 证据
        return contexts[-1] > contexts[0]

    def _apply_high_confidence(self, decision: ProgressDecision) -> ProgressDecision:
        if self._state == _ProgressGuardState.BLOCKED:
            return ProgressDecision(
                level="block",
                reason=decision.reason,
                pattern=decision.pattern,
            )
        if self._state == _ProgressGuardState.HEALTHY:
            self._state = _ProgressGuardState.WARNED
            return decision
        # warned + 再次命中高置信 → block
        self._state = _ProgressGuardState.BLOCKED
        return ProgressDecision(
            level="block",
            reason=decision.reason,
            pattern=decision.pattern,
        )

    def _apply_stagnation(self) -> ProgressDecision:
        if self._state == _ProgressGuardState.BLOCKED:
            return ProgressDecision(level="none")
        if self._state == _ProgressGuardState.HEALTHY:
            self._state = _ProgressGuardState.WARNED
            return ProgressDecision(
                level="warn",
                reason="连续多轮无 workspace/verification 进展且上下文增长，建议 replan。",
                pattern="stagnation",
            )
        # warned + 低置信停滞：不重复刷屏、不 block
        return ProgressDecision(level="none")


def _frame_pair_signature(frame: ProgressFrame) -> str:
    parts = [
        f"{_action_signature(name, args)}|{_observation_signature(obs)}|{status}"
        for name, args, obs, status in frame.pairs
    ]
    return _signature_digest(parts)


def _frame_action_signature(frame: ProgressFrame) -> str:
    parts = [_action_signature(name, args) for name, args, _obs, _status in frame.pairs]
    return _signature_digest(parts)


def _signature_digest(parts: list[str]) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _frame_is_error(frame: ProgressFrame) -> bool:
    if not frame.pairs:
        return False
    return all(status == "error" for _n, _a, _o, status in frame.pairs)


def _action_signature(tool_name: str, args: dict[str, Any]) -> str:
    cleaned = {
        key: value
        for key, value in args.items()
        if key not in _NOISE_ARG_KEYS
    }
    encoded = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=str)
    return f"{tool_name}:{encoded}"


def _observation_signature(observation: str) -> str:
    text = observation if isinstance(observation, str) else str(observation)
    for pattern in _NOISE_OBS_PATTERNS:
        text = pattern.sub("", text)
    return " ".join(text.split())
