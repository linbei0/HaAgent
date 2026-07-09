"""
tests/unit/runtime/test_lazy_multi_agent.py - MultiAgentRuntime 懒加载护栏

普通 run 在 allowed_tools 不含 agent/task 工具时不应构造 MultiAgentRuntime。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from haagent.models.types import ModelResponse
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.orchestration.state import RunStatus


class _DoneGateway:
    provider_name = "lazy-agent-test"

    def generate(self, messages, tool_schemas):
        del messages, tool_schemas
        return ModelResponse("done without workers", [])


def test_orchestrator_skips_multi_agent_runtime_when_agent_tools_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class TrackingRuntime:
        def __init__(self, **kwargs) -> None:
            del kwargs
            created.append(self)

    # 当前实现 eager import；实现后改为局部 import，两边都 patch 防漂移。
    monkeypatch.setattr(
        "haagent.runtime.orchestration.orchestrator.MultiAgentRuntime",
        TrackingRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        "haagent.multi_agent.runtime.MultiAgentRuntime",
        TrackingRuntime,
    )

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: plain file work without workers
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=_DoneGateway(),
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert created == []


def test_orchestrator_builds_multi_agent_runtime_when_agent_tools_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class TrackingRuntime:
        def __init__(self, **kwargs) -> None:
            del kwargs
            created.append(self)

        def wait_for_task(self, task_id: str, timeout: float | None = None):
            del task_id, timeout
            return {}

    monkeypatch.setattr(
        "haagent.runtime.orchestration.orchestrator.MultiAgentRuntime",
        TrackingRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        "haagent.multi_agent.runtime.MultiAgentRuntime",
        TrackingRuntime,
    )

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: may spawn workers
constraints: []
allowed_tools:
  - file_read
  - agent
  - task_list
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=_DoneGateway(),
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert len(created) == 1
