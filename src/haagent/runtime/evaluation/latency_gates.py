"""
src/haagent/runtime/evaluation/latency_gates.py - 交互热路径性能门禁

只保留需要本机计时的微基准；连接、缓存正确性、SSE 和 ProgressGuard
行为由 pytest 覆盖，避免在生产模块中再嵌入一套测试框架。
"""

from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from haagent.context.builder import ContextBuilder
from haagent.context.instruction_cache import InstructionCache
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.events import AssistantDeltaEvent
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.registry import export_tool_schemas
from haagent.tools.schema_cache import ToolSchemaCache
from haagent.tui.application.runtime_events import handle_runtime_ui_event


SUBMIT_TO_REQUEST_START_P95_MS = 150.0
DELTA_HANDLER_P95_MS = 5.0
DELTA_HANDLER_SAMPLES = 200
LOCAL_PREP_SAMPLES = 20


@dataclass(frozen=True)
class LatencyGateResult:
    name: str
    status: Literal["passed", "failed"]
    metric: str
    threshold: str
    actual: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metric": self.metric,
            "threshold": self.threshold,
            "actual": self.actual,
            "detail": self.detail,
        }


def run_interactive_latency_gates(*, work_dir: Path | None = None) -> list[LatencyGateResult]:
    """运行需要本机计时的两个交互热路径门禁。"""

    if work_dir is not None:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        return _run_gates(root)
    with tempfile.TemporaryDirectory(prefix="haagent-latency-gates-") as temp_dir:
        return _run_gates(Path(temp_dir))


def _run_gates(root: Path) -> list[LatencyGateResult]:
    return [
        _gate_submit_to_request_start_p95(root / "prep"),
        _gate_delta_handler_p95(),
    ]


def latency_gates_check_summary(results: list[LatencyGateResult]) -> dict[str, Any]:
    failed = [item for item in results if item.status != "passed"]
    return {
        "name": "interactive_latency",
        "status": "passed" if not failed else "failed",
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "gates": [item.to_dict() for item in results],
    }


def _result(
    passed: bool,
    name: str,
    metric: str,
    threshold: str,
    actual: str,
) -> LatencyGateResult:
    return LatencyGateResult(
        name=name,
        status="passed" if passed else "failed",
        metric=metric,
        threshold=threshold,
        actual=actual,
    )


def _p95(samples_ms: list[float]) -> float:
    ordered = sorted(samples_ms)
    index = max(0, min(len(ordered) - 1, (len(ordered) * 95 + 99) // 100 - 1))
    return ordered[index]


def _write_skill(root: Path) -> None:
    skill_dir = root / "alpha"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("# alpha\nalpha guidance\n", encoding="utf-8")


def _make_writer(root: Path) -> EpisodeWriter:
    root.mkdir(parents=True, exist_ok=True)
    task_path = root / "task.yaml"
    task_path.write_text("goal: latency-gate\n", encoding="utf-8")
    writer = EpisodeWriter.create(root / ".runs", task_path)
    writer.write_plan(
        {
            "goal": "latency-gate",
            "constraints": [],
            "acceptance_criteria": [],
            "verification_commands": [],
            "planned_steps": ["Use allowed tools."],
        },
    )
    return writer


def _task(root: Path) -> TaskSpec:
    return TaskSpec(
        goal="measure warmed request preparation",
        workspace_root=str(root),
        allowed_tools=["file_read", "skill_list", "skill_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )


def _gate_submit_to_request_start_p95(root: Path) -> LatencyGateResult:
    """用已预热的 context build + schema export 近似 request-start 前耗时。"""

    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / ".haagent"
    _write_skill(config_dir / "skills")
    (root / "AGENTS.md").write_text("# agents\nprep\n", encoding="utf-8")
    instruction_cache = InstructionCache()
    skill_catalog = SkillCatalogService(config_dir=config_dir)
    schema_cache = ToolSchemaCache()
    names = ["file_read", "skill_list", "skill_read"]
    task = _task(root)
    writer = _make_writer(root / "episode")

    def build_context() -> None:
        ContextBuilder(
            task=task,
            workspace_root=root,
            provider_name="latency-gate",
            episode_writer=writer,
            instruction_cache=instruction_cache,
            skill_catalog=skill_catalog,
        ).build()
        export_tool_schemas(names, cache=schema_cache)

    build_context()
    samples: list[float] = []
    for _ in range(LOCAL_PREP_SAMPLES):
        started = time.perf_counter()
        build_context()
        samples.append((time.perf_counter() - started) * 1000.0)

    p95 = _p95(samples)
    actual = f"p95={p95:.3f}ms mean={statistics.mean(samples):.3f}ms"
    threshold = f"≤{SUBMIT_TO_REQUEST_START_P95_MS}"
    return _result(
        p95 <= SUBMIT_TO_REQUEST_START_P95_MS,
        "submit_to_request_start_p95",
        "submit_to_request_start_p95_ms",
        threshold,
        actual,
    )


class _FakeConversation:
    def __init__(self, app: _FakeRuntimeEventApp) -> None:
        self._app = app

    def merge_assistant_delta(self, turn_index: int, model_turn: int | None, delta: str) -> None:
        del model_turn
        self._app.assistant_deltas.append((turn_index, delta))


class _FakeRuntimeEventApp:
    """只实现 delta 热路径需要的最小 UI 接口。"""

    def __init__(self) -> None:
        self.refreshes = 0
        self.streaming_refresh_schedules = 0
        self.assistant_deltas: list[tuple[int, str]] = []
        self._conversation = _FakeConversation(self)

    def _refresh(self) -> None:
        self.refreshes += 1

    def _schedule_streaming_refresh(self) -> None:
        self.streaming_refresh_schedules += 1


def _gate_delta_handler_p95() -> LatencyGateResult:
    app = _FakeRuntimeEventApp()
    samples: list[float] = []
    for index in range(DELTA_HANDLER_SAMPLES):
        event = AssistantDeltaEvent("session-latency", 1, 1, f"t{index}")
        started = time.perf_counter()
        handle_runtime_ui_event(app, event)
        samples.append((time.perf_counter() - started) * 1000.0)
    p95 = _p95(samples)
    actual = f"p95={p95:.3f}ms samples={len(samples)} schedules={app.streaming_refresh_schedules}"
    passed = p95 <= DELTA_HANDLER_P95_MS and app.refreshes == 0
    return _result(
        passed,
        "delta_handler_p95",
        "delta_handler_p95_ms",
        f"≤{DELTA_HANDLER_P95_MS}",
        actual,
    )
