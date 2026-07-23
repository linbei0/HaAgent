"""
src/haagent/runtime/evaluation/latency_gates.py - 交互热路径性能门禁

覆盖本地准备、规模数据与真实 Textual widget 路径；连接、SSE 和 ProgressGuard
正确性仍由 pytest 覆盖，避免在生产模块中嵌入重复的行为测试框架。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from textual.app import App, ComposeResult

from haagent.context.builder import ContextBuilder
from haagent.context.instruction_cache import InstructionCache
from haagent.memory.retrieval import MemoryRetrievalRequest, MemoryRetriever
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.events import AssistantDeltaEvent
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.registry import export_tool_schemas
from haagent.tools.schema_cache import ToolSchemaCache
from haagent.tui.application.runtime_events import handle_runtime_ui_event
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch
from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.timeline_models import TimelineItem


SUBMIT_TO_REQUEST_START_P95_MS = 150.0
DELTA_HANDLER_P95_MS = 5.0
DELTA_HANDLER_SAMPLES = 200
LOCAL_PREP_SAMPLES = 20
FILE_REFERENCE_QUERY_P95_MS = 75.0
# GitHub hosted runner 的稳定 p95 约 91ms；125ms 仍显著低于审计前的约 393ms。
MEMORY_RETRIEVAL_P95_MS = 125.0
TIMELINE_MOUNT_MS = 750.0
TIMELINE_SHIFT_MS = 500.0
DELTA_PAINT_P95_MS = 150.0
SCALE_SAMPLES = 20
DELTA_PAINT_SAMPLES = 20


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
    """运行本地准备、规模数据与真实 Textual 渲染门禁。"""

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
        _gate_file_reference_query_p95(root / "file-refs"),
        _gate_memory_retrieval_p95(root / "memory"),
        *_gate_textual_timeline(),
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


def _gate_file_reference_query_p95(root: Path) -> LatencyGateResult:
    root.mkdir(parents=True, exist_ok=True)
    files = tuple(
        FileReferenceMatch(
            path=root / f"folder-{index % 997}" / f"match-{index:06}.txt",
            display_path=f"folder-{index % 997}/match-{index:06}.txt",
        )
        for index in range(100_000)
    )
    index = FileReferenceIndex(root=root, files=files)
    samples: list[float] = []
    for _ in range(SCALE_SAMPLES):
        for query in ("match", "definitely-not-present"):
            started = time.perf_counter()
            matches = index.matches(query, limit=20)
            samples.append((time.perf_counter() - started) * 1000.0)
    p95 = _p95(samples)
    return _result(
        p95 <= FILE_REFERENCE_QUERY_P95_MS and not matches,
        "file_reference_query_p95",
        "file_reference_query_p95_ms_100k",
        f"≤{FILE_REFERENCE_QUERY_P95_MS}",
        f"p95={p95:.3f}ms paths={len(files)} broad_and_missing_queries=true",
    )


def _gate_memory_retrieval_p95(root: Path) -> LatencyGateResult:
    workspace = root / "workspace"
    user_memory_root = root / "user-memory"
    _write_memory_scale_data(workspace, count=10_000)
    retriever = MemoryRetriever()
    request = MemoryRetrievalRequest(
        query="latency marker",
        workspace_root=workspace,
        user_memory_root=user_memory_root,
    )
    retriever.retrieve(request)
    samples: list[float] = []
    selected_count = 0
    for _ in range(SCALE_SAMPLES):
        started = time.perf_counter()
        selected_count = len(retriever.retrieve(request).memories)
        samples.append((time.perf_counter() - started) * 1000.0)
    p95 = _p95(samples)
    return _result(
        p95 <= MEMORY_RETRIEVAL_P95_MS and selected_count > 0,
        "memory_retrieval_p95",
        "memory_retrieval_p95_ms_10k",
        f"≤{MEMORY_RETRIEVAL_P95_MS}",
        f"p95={p95:.3f}ms records=10000 selected={selected_count}",
    )


def _write_memory_scale_data(workspace: Path, *, count: int) -> None:
    memory_root = workspace / ".haagent" / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    records: list[str] = []
    for index in range(count):
        memory_id = f"mem_latency_{index:06}"
        title = f"Latency marker {index}"
        body = f"Latency marker body {index}"
        items.append(
            {
                "id": memory_id,
                "category": "facts",
                "title": title,
                "summary": body,
                "tags": ["latency"],
                "updated_at": "2026-07-22T00:00:00+00:00",
                "status": "active",
            },
        )
        records.append(
            json.dumps(
                {
                    "memory_id": memory_id,
                    "scope": "workspace",
                    "category": "facts",
                    "title": title,
                    "body": body,
                    "evidence": {
                        "source_type": "file",
                        "evidence_summary": "latency gate fixture",
                    },
                    "source_candidate_id": f"candidate_{index:06}",
                    "content_hash": f"hash_{index:06}",
                    "created_at": "2026-07-22T00:00:00+00:00",
                    "updated_at": "2026-07-22T00:00:00+00:00",
                    "tags": ["latency"],
                    "status": "active",
                },
                ensure_ascii=False,
            ),
        )
    (memory_root / "facts.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    (memory_root / "index.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "updated_at": "2026-07-22T00:00:00+00:00",
                "source": "workspace",
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class _TextualGateConversation:
    def __init__(self, app: "_TextualGateApp") -> None:
        self._app = app

    def merge_assistant_delta(self, turn_index: int, model_turn: int | None, delta: str) -> None:
        del model_turn
        self._app.timeline.update_assistant_delta(turn_index, delta)


class _TextualGateApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self._conversation = _TextualGateConversation(self)
        self._streaming_timer = None

    def compose(self) -> ComposeResult:
        yield ConversationTimeline(id="conversation")

    @property
    def timeline(self) -> ConversationTimeline:
        return self.query_one(ConversationTimeline)

    def on_mount(self) -> None:
        timeline = self.timeline
        for turn_index in range(500):
            timeline._append_item(
                TimelineItem(
                    item_id=next(timeline._ids),
                    role="user",
                    turn_index=turn_index,
                    content=f"prompt {turn_index}",
                ),
            )
        timeline._sync_blocks()

    def _schedule_streaming_refresh(self) -> None:
        if self._streaming_timer is not None:
            return
        # 与生产 TUI 相同，delta 在约 33ms 窗口内合并后再触发 Markdown/layout。
        self._streaming_timer = self.set_timer(0.033, self._flush_streaming_refresh)

    def _flush_streaming_refresh(self) -> None:
        self._streaming_timer = None
        self.timeline.flush_pending_assistant_delta()


def _gate_textual_timeline() -> list[LatencyGateResult]:
    return asyncio.run(_measure_textual_timeline())


async def _measure_textual_timeline() -> list[LatencyGateResult]:
    app = _TextualGateApp()
    mount_started = time.perf_counter()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        mount_ms = (time.perf_counter() - mount_started) * 1000.0
        timeline = app.timeline

        shift_started = time.perf_counter()
        timeline.watch_scroll_y(10, 0)
        await pilot.pause()
        shift_ms = (time.perf_counter() - shift_started) * 1000.0

        timeline.set_stick_to_bottom(True)
        await pilot.pause()
        timeline.start_assistant_response(turn_index=501)
        delta_samples: list[float] = []
        delta_count = 0
        expected_chunks: list[str] = []
        for sample_index in range(DELTA_PAINT_SAMPLES):
            started = time.perf_counter()
            for delta_index in range(5):
                delta_count += 1
                chunk = f"t{sample_index}-{delta_index}"
                expected_chunks.append(chunk)
                handle_runtime_ui_event(
                    app,
                    AssistantDeltaEvent("session-latency", 501, 1, chunk),
                )
            await pilot.pause(0.05)
            delta_samples.append((time.perf_counter() - started) * 1000.0)
        delta_p95 = _p95(delta_samples)
        painted_all_deltas = timeline._assistant_items_by_turn[501].content == "".join(expected_chunks)

    return [
        _result(
            mount_ms <= TIMELINE_MOUNT_MS,
            "timeline_initial_mount",
            "timeline_initial_mount_ms_500_items",
            f"≤{TIMELINE_MOUNT_MS}",
            f"mount={mount_ms:.3f}ms blocks=50",
        ),
        _result(
            shift_ms <= TIMELINE_SHIFT_MS,
            "timeline_window_shift",
            "timeline_window_shift_ms_50_items",
            f"≤{TIMELINE_SHIFT_MS}",
            f"shift={shift_ms:.3f}ms reused=25",
        ),
        _result(
            delta_p95 <= DELTA_PAINT_P95_MS and painted_all_deltas,
            "delta_to_textual_paint_p95",
            "delta_to_textual_paint_p95_ms",
            f"≤{DELTA_PAINT_P95_MS}",
            f"p95={delta_p95:.3f}ms samples={len(delta_samples)} deltas={delta_count}",
        ),
    ]
