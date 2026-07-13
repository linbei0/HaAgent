"""
tests/unit/runtime/test_progress_guard.py - ProgressGuard 纯状态机合同

覆盖高置信循环、低置信停滞、豁免、重置与状态转换。
"""

from __future__ import annotations

import pytest

from haagent.runtime.execution.progress_guard import (
    ProgressFrame,
    ProgressGuard,
)


def _pair(tool: str, args: dict, obs: str, *, status: str = "success") -> dict:
    return {
        "tool_name": tool,
        "args": args,
        "observation": obs,
        "status": status,
    }


def _frame(
    *,
    pairs: list[dict] | None = None,
    workspace_changed: bool = False,
    verification_progressed: bool = False,
    context_chars: int = 100,
    has_running_tool: bool = False,
    has_running_worker: bool = False,
    waiting_approval: bool = False,
    waiting_user_input: bool = False,
) -> ProgressFrame:
    return ProgressFrame(
        pairs=tuple(
            (
                p["tool_name"],
                p["args"],
                p["observation"],
                p.get("status", "success"),
            )
            for p in (pairs or [_pair("file_read", {"path": "a.md"}, "content-a")])
        ),
        workspace_changed=workspace_changed,
        verification_progressed=verification_progressed,
        context_chars=context_chars,
        has_running_tool=has_running_tool,
        has_running_worker=has_running_worker,
        waiting_approval=waiting_approval,
        waiting_user_input=waiting_user_input,
    )


def test_identical_action_observation_three_times_warns() -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "README.md"}, "same body")]
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    decision = guard.observe(_frame(pairs=pairs))
    assert decision.level == "warn"
    assert decision.pattern == "identical_pair"


def test_warned_then_same_high_confidence_blocks() -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "x.md"}, "body")]
    for _ in range(3):
        guard.observe(_frame(pairs=pairs))
    decision = guard.observe(_frame(pairs=pairs))
    assert decision.level == "block"
    assert decision.pattern == "identical_pair"


def test_same_action_consecutive_errors_warns() -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "missing.md"}, "err", status="error")]
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    decision = guard.observe(_frame(pairs=pairs))
    assert decision.level == "warn"
    assert decision.pattern == "error_loop"


def test_ab_ab_ab_pattern_warns() -> None:
    guard = ProgressGuard()
    a = [_pair("file_read", {"path": "a.md"}, "A")]
    b = [_pair("file_read", {"path": "b.md"}, "B")]
    levels = []
    for pairs in [a, b, a, b, a, b]:
        levels.append(guard.observe(_frame(pairs=pairs)).level)
    assert levels[:5] == ["none", "none", "none", "none", "none"]
    assert levels[5] == "warn"


def test_noise_ids_do_not_affect_signature() -> None:
    guard = ProgressGuard()
    for index in range(3):
        pairs = [
            _pair(
                "file_read",
                {"path": "README.md", "tool_call_id": f"call_{index}"},
                f"same call_{index} {index + 1}ms",
            ),
        ]
        decision = guard.observe(_frame(pairs=pairs))
    assert decision.level == "warn"


def test_low_confidence_stagnation_warns_not_blocks() -> None:
    guard = ProgressGuard()
    # 4 个不同 pair，无 workspace/verification，context 增长
    decisions = []
    for turn in range(1, 5):
        pairs = [_pair("file_list", {"max_depth": 1}, f"listing-{turn}")]
        decisions.append(
            guard.observe(
                _frame(
                    pairs=pairs,
                    context_chars=100 + turn * 50,
                    workspace_changed=False,
                    verification_progressed=False,
                ),
            ).level,
        )
    assert decisions[:3] == ["none", "none", "none"]
    assert decisions[3] == "warn"
    # 再来一轮低置信停滞：不重复 warn，不 block
    again = guard.observe(
        _frame(
            pairs=[_pair("file_list", {"max_depth": 1}, "listing-5")],
            context_chars=400,
        ),
    )
    assert again.level == "none"


def test_parallel_batch_is_one_frame_not_multi_turn_stagnation() -> None:
    guard = ProgressGuard()
    # 一轮内多个工具聚合为一个 frame；observation 每轮不同，避免 identical_pair
    for turn in range(1, 4):
        multi = [
            _pair("file_read", {"path": "a.md"}, f"A-{turn}"),
            _pair("file_read", {"path": "b.md"}, f"B-{turn}"),
            _pair("grep", {"pattern": "x"}, f"hits-{turn}"),
        ]
        assert guard.observe(_frame(pairs=multi, context_chars=100 + turn)).level == "none"
    # 第 4 个完成轮次才触发低置信停滞（不是把并行工具拆成多轮）
    multi4 = [
        _pair("file_read", {"path": "a.md"}, "A-4"),
        _pair("file_read", {"path": "b.md"}, "B-4"),
        _pair("grep", {"pattern": "x"}, "hits-4"),
    ]
    d = guard.observe(_frame(pairs=multi4, context_chars=200))
    assert d.level == "warn"
    assert d.pattern == "stagnation"


@pytest.mark.parametrize(
    "wait_flag",
    ["has_running_tool", "has_running_worker", "waiting_approval", "waiting_user_input"],
)
def test_running_pending_and_waiting_excluded_from_window(wait_flag: str) -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "x.md"}, "body")]
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    assert guard.observe(_frame(pairs=pairs)).level == "none"
    assert guard.observe(_frame(pairs=pairs, **{wait_flag: True})).level == "none"
    assert guard.observe(_frame(pairs=pairs)).level == "warn"


def test_progress_resets_to_healthy() -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "x.md"}, "same")]
    for _ in range(3):
        guard.observe(_frame(pairs=pairs))
    progress = guard.observe(
        _frame(
            pairs=[_pair("file_write", {"path": "out.md"}, "written")],
            workspace_changed=True,
        ),
    )
    assert progress.level == "none"


def test_reset_from_blocked_with_user_continue() -> None:
    guard = ProgressGuard()
    pairs = [_pair("file_read", {"path": "x.md"}, "same")]
    for _ in range(4):
        guard.observe(_frame(pairs=pairs))
    guard.reset()
    assert guard.observe(_frame(pairs=pairs)).level == "none"


def test_verification_progress_resets_stagnation() -> None:
    guard = ProgressGuard()
    for turn in range(1, 4):
        guard.observe(
            _frame(
                pairs=[_pair("shell", {"command": "pytest"}, f"out-{turn}")],
                context_chars=100 + turn * 20,
            ),
        )
    d = guard.observe(
        _frame(
            pairs=[_pair("shell", {"command": "pytest"}, "passed")],
            verification_progressed=True,
            context_chars=200,
        ),
    )
    assert d.level == "none"
