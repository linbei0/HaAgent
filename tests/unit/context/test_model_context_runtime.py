"""
tests/unit/context/test_model_context_runtime.py - Model Context Runtime 契约测试

覆盖 projection、生命周期、严格聚合校验和原子持久化失败边界。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from haagent.context.compression.budget import CompressionBudget
from haagent.context.model_context_runtime import (
    ModelContextLifecycleError,
    ModelContextPackageError,
    ModelContextPersistenceError,
    ModelContextRuntime,
    ModelContextTurnSeed,
)
from haagent.context.projection import (
    ModelContextFacts,
    MutableContextFactsSource,
    ProjectedModelContext,
    project_model_context,
)
from haagent.runtime.session.planning_state import empty_planning_state
from haagent.runtime.session.task_ledger import empty_task_ledger


class _Writer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.transcript: list[dict[str, object]] = []

    def append_transcript(self, record: dict[str, object]) -> None:
        self.transcript.append(record)

    def write_tool_artifact(self, tool_name: str, content: str, *, suffix: str = ".txt") -> str:
        path = self.path / "artifacts" / f"{tool_name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.as_posix()


def _budget() -> CompressionBudget:
    return CompressionBudget(
        context_window_tokens=64_000,
        reserved_output_tokens=4_000,
        safety_buffer_tokens=2_000,
        available_input_tokens=58_000,
        context_builder_max_tokens=16_000,
    )


def _seed(writer: _Writer, projection: ProjectedModelContext | None = None) -> ModelContextTurnSeed:
    return ModelContextTurnSeed(
        context_id="context-1",
        stable_messages=({"role": "system", "content": "stable"},),
        user_request_message={"role": "user", "content": "request"},
        initial_projection=projection or ProjectedModelContext.create({}),
        compression_budget=_budget(),
        episode_writer=writer,
    )


def test_projection_is_deterministic_and_removes_terminal_facts() -> None:
    facts = ModelContextFacts(
        session_summary="  summary  ",
        working_state={"key_findings": ["found"], "last_updated_turn": 2},
        task_ledger=empty_task_ledger(),
        planning_state=empty_planning_state(),
    )

    first = project_model_context(facts)
    second = project_model_context(facts)

    assert first == second
    assert set(first.sections) == {"session_summary", "working_state"}
    assert first.sections["session_summary"] == "summary"


def test_runtime_appends_delta_and_final_state_before_terminal_commit(tmp_path: Path) -> None:
    source = MutableContextFactsSource(ModelContextFacts(session_summary="one"))
    writer = _Writer(tmp_path)
    runtime = ModelContextRuntime.create_or_restore(tmp_path, source)
    turn = runtime.begin_turn(_seed(writer, project_model_context(source())))

    first = turn.before_model_call()
    source.update(ModelContextFacts(session_summary="two"))
    turn.complete_model_step(({"role": "assistant", "content": "done"},), terminal=True)

    assert first.messages[0]["role"] == "system"
    aggregate = json.loads((tmp_path / "model-context.json").read_text(encoding="utf-8"))
    assert aggregate["snapshot"]["revision"] == 2
    assert set(aggregate) == {"schema_version", "messages", "snapshot", "rebuild_required"}
    assert aggregate["schema_version"] == 2
    assert aggregate["snapshot"]["sections"]["session_summary"] == "two"


def test_frame_cannot_mutate_runtime_messages(tmp_path: Path) -> None:
    writer = _Writer(tmp_path)
    runtime = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    turn = runtime.begin_turn(_seed(writer))

    frame = turn.before_model_call()
    frame.messages[0]["content"] = "tampered"

    persisted = json.loads((tmp_path / "model-context.json").read_text(encoding="utf-8"))
    assert persisted["messages"][0]["content"] == "stable"
    turn.complete_model_step(terminal=True)


def test_require_rebuild_persists_intent_and_increments_epoch(tmp_path: Path) -> None:
    writer = _Writer(tmp_path)
    runtime = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    turn = runtime.begin_turn(_seed(writer))
    turn.before_model_call()
    turn.complete_model_step(terminal=True)
    previous_epoch = json.loads(
        (tmp_path / "model-context.json").read_text(encoding="utf-8"),
    )["snapshot"]["epoch"]

    runtime.require_rebuild("manual")
    restored = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    next_turn = restored.begin_turn(_seed(writer))
    next_turn.before_model_call()

    assert json.loads(
        (tmp_path / "model-context.json").read_text(encoding="utf-8"),
    )["snapshot"]["epoch"] == previous_epoch + 1
    next_turn.complete_model_step(terminal=True)


def test_rejects_invalid_schema_and_tool_pairing(tmp_path: Path) -> None:
    runtime = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    writer = _Writer(tmp_path)
    turn = runtime.begin_turn(_seed(writer))
    turn._append_messages(({"role": "tool", "tool_call_id": "missing", "content": "bad"},))

    with pytest.raises(ModelContextPackageError, match="no matching"):
        turn.before_model_call()
    turn.abort()

    path = tmp_path / "model-context.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ModelContextPackageError, match="schema_version"):
        ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())


def test_atomic_replace_failure_keeps_previous_file(tmp_path: Path, monkeypatch) -> None:
    runtime = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    path = tmp_path / "model-context.json"
    turn = runtime.begin_turn(_seed(_Writer(tmp_path)))
    turn.before_model_call()
    turn.complete_model_step(terminal=True)
    original = path.read_bytes()
    original_aggregate = json.loads(path.read_text(encoding="utf-8"))

    def fail_replace(source, target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("haagent.context.model_context_runtime.os.replace", fail_replace)
    with pytest.raises(ModelContextPersistenceError, match="atomically write"):
        runtime.require_rebuild("manual")

    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8")) == original_aggregate


def test_rejects_parallel_active_turns(tmp_path: Path) -> None:
    runtime = ModelContextRuntime.create_or_restore(tmp_path, MutableContextFactsSource())
    turn = runtime.begin_turn(_seed(_Writer(tmp_path)))
    with pytest.raises(ModelContextLifecycleError, match="active turn"):
        runtime.begin_turn(_seed(_Writer(tmp_path)))
    turn.abort()
