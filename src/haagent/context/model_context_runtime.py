"""
src/haagent/context/model_context_runtime.py - 模型上下文生命周期运行时

独占模型消息链、版本化 snapshot、delta、checkpoint、rebuild、校验和原子持久化。
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from haagent.context.compression.budget import CompressionBudget
from haagent.context.compression.checkpoint import maybe_checkpoint_messages
from haagent.context.messages import (
    build_context_state_delta_message,
    build_context_state_snapshot_message,
    context_state_payload,
    is_context_state_message,
)
from haagent.context.projection import (
    ContextFactsSource,
    ModelContextFacts,
    ProjectedModelContext,
    merge_session_projection,
    project_model_context,
)
from haagent.context.versioned_state import (
    ContextStateDelta,
    ContextStateError,
    ContextStateSnapshot,
    apply_delta,
    next_snapshot,
    reset_epoch,
)
from haagent.runtime.episodes.writer import EpisodeWriter


MODEL_CONTEXT_SCHEMA_VERSION = 2


class ModelContextError(RuntimeError):
    """Model Context Runtime 的公共错误基类。"""


class ModelContextLifecycleError(ModelContextError):
    """调用顺序违反单活动 turn 约束。"""


class ModelContextPackageError(ModelContextError):
    """持久化聚合 schema 或消息链损坏。"""


class ModelContextPersistenceError(ModelContextError):
    """原子持久化失败。"""


@dataclass(frozen=True)
class ModelContextAggregate:
    messages: tuple[dict[str, Any], ...]
    snapshot: ContextStateSnapshot
    rebuild_required: bool = False


@dataclass(frozen=True)
class ModelContextTurnSeed:
    context_id: str
    stable_messages: tuple[dict[str, Any], ...]
    user_request_message: dict[str, Any]
    initial_projection: ProjectedModelContext
    compression_budget: CompressionBudget
    episode_writer: EpisodeWriter


@dataclass(frozen=True)
class ModelCallFrame:
    messages: tuple[dict[str, Any], ...]
    context_id: str
    epoch: int
    revision: int
    snapshot_id: str


class ModelContextRuntime:
    """一个 session 唯一的模型上下文所有者。"""

    def __init__(
        self,
        *,
        session_path: Path | None,
        facts_source: ContextFactsSource,
        aggregate: ModelContextAggregate,
    ) -> None:
        self._session_path = session_path
        self._facts_source = facts_source
        self._committed = aggregate
        self._candidate = aggregate
        self._active_turn: ModelContextTurn | None = None

    @classmethod
    def create_or_restore(
        cls,
        session_path: Path,
        facts_source: ContextFactsSource,
    ) -> "ModelContextRuntime":
        path = session_path / "model-context.json"
        if path.exists():
            aggregate = _read_aggregate(path)
        else:
            aggregate = ModelContextAggregate((), ContextStateSnapshot.create(), False)
            _write_aggregate(path, aggregate)
        return cls(session_path=session_path, facts_source=facts_source, aggregate=aggregate)

    @classmethod
    def create_transient(cls, facts_source: ContextFactsSource) -> "ModelContextRuntime":
        """无 session 的高级 run 使用同一状态机，但不制造 session package。"""

        return cls(
            session_path=None,
            facts_source=facts_source,
            aggregate=ModelContextAggregate((), ContextStateSnapshot.create(), False),
        )

    def _bind_facts_source(self, facts_source: ContextFactsSource) -> None:
        """生命周期装配完成后绑定 session-backed adapter。"""

        if self._active_turn is not None:
            raise ModelContextLifecycleError("cannot bind facts source during an active turn")
        self._facts_source = facts_source

    def begin_turn(self, seed: ModelContextTurnSeed) -> "ModelContextTurn":
        if self._active_turn is not None:
            raise ModelContextLifecycleError("model context already has an active turn")
        turn = ModelContextTurn(self, seed)
        self._active_turn = turn
        turn._begin()
        return turn

    def require_rebuild(self, reason: str) -> None:
        if self._active_turn is not None:
            raise ModelContextLifecycleError("cannot request rebuild during an active turn")
        self._candidate = ModelContextAggregate(
            self._candidate.messages,
            self._candidate.snapshot,
            True,
        )
        self._commit()

    def _commit(self) -> None:
        try:
            _validate_aggregate(self._candidate)
            if self._session_path is not None:
                _write_aggregate(self._session_path / "model-context.json", self._candidate)
        except ModelContextError:
            # durable commit 失败时进程内状态也必须回到旧聚合，避免后续调用观察到未落盘状态。
            self._candidate = self._committed
            raise
        self._committed = self._candidate

    def _finish_turn(self, turn: "ModelContextTurn") -> None:
        if self._active_turn is not turn:
            raise ModelContextLifecycleError("model context turn is not active")
        self._active_turn = None


class ModelContextTurn:
    """单轮 context handle；所有写入都回到所属 runtime。"""

    def __init__(self, runtime: ModelContextRuntime, seed: ModelContextTurnSeed) -> None:
        self._runtime = runtime
        self._seed = seed
        self._closed = False
        self._interaction_records: tuple[Mapping[str, object], ...] = ()

    def __enter__(self) -> "ModelContextTurn":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            if exc_type is None:
                self.complete_model_step((), terminal=True)
            else:
                self._runtime._candidate = self._runtime._committed
                self._closed = True
                self._runtime._finish_turn(self)

    def _begin(self) -> None:
        aggregate = self._runtime._candidate
        previous = list(deepcopy(aggregate.messages))
        stable = list(deepcopy(self._seed.stable_messages))
        stable_matches = bool(previous) and _system_prefix(previous) == stable
        force_rebuild = aggregate.rebuild_required or self._seed.initial_projection.requires_rebuild
        desired = self._seed.initial_projection
        if stable_matches and not force_rebuild and aggregate.snapshot.revision > 0:
            snapshot, delta = next_snapshot(
                aggregate.snapshot,
                desired.sections,
                desired.source_digests,
            )
            messages = previous
            if delta is not None:
                messages.append(build_context_state_delta_message(delta))
                self._record_delta(delta, snapshot, turn=0, source="session_turn_boundary")
        else:
            epoch = aggregate.snapshot.epoch + 1 if previous or force_rebuild else 0
            snapshot = ContextStateSnapshot.create(
                epoch=epoch,
                revision=1,
                sections=desired.sections,
                source_digests=desired.source_digests,
            )
            messages = [*stable, build_context_state_snapshot_message(snapshot)]
            if previous or force_rebuild:
                self._seed.episode_writer.append_transcript(
                    {
                        "event": "context_epoch_rebuilt",
                        "previous_epoch": aggregate.snapshot.epoch,
                        "epoch": snapshot.epoch,
                        "revision": snapshot.revision,
                        "snapshot_id": snapshot.snapshot_id,
                        "sections": sorted(snapshot.sections),
                        "reason": (
                            "session_memory_compacted"
                            if self._seed.initial_projection.requires_rebuild
                            else "rebuild_required"
                            if aggregate.rebuild_required
                            else "stable_prefix_changed"
                        ),
                    },
                )
        messages.append(deepcopy(self._seed.user_request_message))
        self._runtime._candidate = ModelContextAggregate(tuple(messages), snapshot, False)

    def before_model_call(
        self,
        *,
        pending_messages: Sequence[dict[str, Any]] = (),
        interaction_records: Sequence[Mapping[str, object]] | None = None,
        turn: int = 0,
    ) -> ModelCallFrame:
        self._ensure_open()
        if interaction_records is not None:
            self._interaction_records = tuple(interaction_records)
        self._append_messages(pending_messages)
        self._sync_projection(turn=turn)
        aggregate = self._runtime._candidate
        checkpoint = maybe_checkpoint_messages(
            messages=list(deepcopy(aggregate.messages)),
            budget=self._seed.compression_budget,
            epoch=aggregate.snapshot.epoch,
            artifact_writer=self._seed.episode_writer.write_tool_artifact,
        )
        if checkpoint.applied:
            snapshot = reset_epoch(aggregate.snapshot, checkpoint.epoch)
            messages = _insert_fresh_context_snapshot(checkpoint.messages, snapshot)
            self._runtime._candidate = ModelContextAggregate(tuple(messages), snapshot, False)
            self._seed.episode_writer.append_transcript(
                {
                    "event": "context_checkpoint",
                    "turn": turn,
                    "context_revision": snapshot.revision,
                    "context_snapshot_id": snapshot.snapshot_id,
                    "changed_sections": sorted(snapshot.sections),
                    **checkpoint.diagnostic,
                },
            )
        self._runtime._commit()
        aggregate = self._runtime._candidate
        snapshot = aggregate.snapshot
        return ModelCallFrame(
            messages=tuple(deepcopy(aggregate.messages)),
            context_id=self._seed.context_id,
            epoch=snapshot.epoch,
            revision=snapshot.revision,
            snapshot_id=snapshot.snapshot_id,
        )

    def complete_model_step(
        self,
        messages: Sequence[dict[str, Any]] = (),
        *,
        terminal: bool = False,
        interaction_records: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        self._ensure_open()
        if interaction_records is not None:
            self._interaction_records = tuple(interaction_records)
        self._append_messages(messages)
        self._sync_projection(turn=0)
        self._runtime._commit()
        if terminal:
            self._closed = True
            self._runtime._finish_turn(self)

    def abort(self) -> None:
        """丢弃未提交 candidate，并释放 active turn；错误本身由调用者继续传播。"""

        self._ensure_open()
        self._runtime._candidate = self._runtime._committed
        self._closed = True
        self._runtime._finish_turn(self)

    def _sync_projection(self, *, turn: int) -> None:
        facts = self._runtime._facts_source()
        if not isinstance(facts, ModelContextFacts):
            raise ModelContextPackageError("context facts source must return ModelContextFacts")
        if self._interaction_records:
            facts = ModelContextFacts(
                session_summary=facts.session_summary,
                working_state=facts.working_state,
                task_ledger=facts.task_ledger,
                planning_state=facts.planning_state,
                interaction_records=self._interaction_records,
            )
        aggregate = self._runtime._candidate
        desired = merge_session_projection(aggregate.snapshot.sections, project_model_context(facts))
        snapshot, delta = next_snapshot(
            aggregate.snapshot,
            desired.sections,
            desired.source_digests,
        )
        if delta is None:
            return
        self._runtime._candidate = ModelContextAggregate(
            tuple([*aggregate.messages, build_context_state_delta_message(delta)]),
            snapshot,
            aggregate.rebuild_required,
        )
        self._record_delta(delta, snapshot, turn=turn, source=None)

    def _append_messages(self, messages: Sequence[dict[str, Any]]) -> None:
        self._ensure_open()
        if not messages:
            return
        aggregate = self._runtime._candidate
        self._runtime._candidate = ModelContextAggregate(
            tuple([*deepcopy(aggregate.messages), *deepcopy(list(messages))]),
            aggregate.snapshot,
            aggregate.rebuild_required,
        )

    def _record_delta(
        self,
        delta: ContextStateDelta,
        snapshot: ContextStateSnapshot,
        *,
        turn: int,
        source: str | None,
    ) -> None:
        record: dict[str, object] = {
            "event": "context_state_delta",
            "turn": turn,
            "epoch": snapshot.epoch,
            "base_revision": delta.base_revision,
            "revision": snapshot.revision,
            "base_snapshot_id": delta.base_snapshot_id,
            "snapshot_id": snapshot.snapshot_id,
            "changed_sections": sorted(delta.changed),
            "removed_sections": list(delta.removed),
            "changed_chars": sum(len(str(value)) for value in delta.changed.values()),
        }
        if source is not None:
            record["source"] = source
        self._seed.episode_writer.append_transcript(record)

    def _ensure_open(self) -> None:
        if self._closed or self._runtime._active_turn is not self:
            raise ModelContextLifecycleError("model context turn is closed")


def _read_aggregate(path: Path) -> ModelContextAggregate:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelContextPackageError(f"invalid model-context.json: {path}") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "messages", "snapshot", "rebuild_required"}:
        raise ModelContextPackageError("invalid model-context.json aggregate fields")
    if raw.get("schema_version") != MODEL_CONTEXT_SCHEMA_VERSION:
        raise ModelContextPackageError(
            f"unsupported model context schema_version: {raw.get('schema_version')}",
        )
    if not isinstance(raw.get("messages"), list) or not isinstance(raw.get("rebuild_required"), bool):
        raise ModelContextPackageError("invalid model-context.json aggregate types")
    try:
        snapshot = ContextStateSnapshot.from_dict(raw.get("snapshot"))
    except ContextStateError as error:
        raise ModelContextPackageError(f"invalid model context snapshot: {error}") from error
    messages = tuple(_validate_message(value, index) for index, value in enumerate(raw["messages"], start=1))
    aggregate = ModelContextAggregate(messages, snapshot, raw["rebuild_required"])
    _validate_aggregate(aggregate)
    return aggregate


def _write_aggregate(path: Path, aggregate: ModelContextAggregate) -> None:
    _validate_aggregate(aggregate)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MODEL_CONTEXT_SCHEMA_VERSION,
        "messages": [_sanitize_message(message) for message in aggregate.messages],
        "snapshot": aggregate.snapshot.to_dict(),
        "rebuild_required": aggregate.rebuild_required,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ModelContextPersistenceError(f"failed to atomically write {path}: {error}") from error


def _validate_aggregate(aggregate: ModelContextAggregate) -> None:
    _validate_message_protocol(aggregate.messages)
    current: ContextStateSnapshot | None = None
    try:
        for message in aggregate.messages:
            payload = context_state_payload(message)
            if payload is None:
                continue
            if payload.get("kind") == "context_state_snapshot":
                current = ContextStateSnapshot.from_dict(payload)
            elif payload.get("kind") == "context_state_delta":
                if current is None:
                    raise ContextStateError("context delta appears before snapshot")
                current = apply_delta(current, ContextStateDelta.from_dict(payload))
    except ContextStateError as error:
        raise ModelContextPackageError(f"invalid model context state: {error}") from error
    if not aggregate.messages and aggregate.snapshot.revision == 0:
        return
    if current is None or current != aggregate.snapshot:
        raise ModelContextPackageError("model context messages do not match aggregate snapshot")


def _validate_message_protocol(messages: Sequence[dict[str, Any]]) -> None:
    pending_tool_calls: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict) or not isinstance(tool_call.get("id"), str):
                        raise ModelContextPackageError("assistant tool call requires a string id")
                    call_id = str(tool_call["id"])
                    if call_id in pending_tool_calls:
                        raise ModelContextPackageError(f"duplicate assistant tool call id: {call_id}")
                    pending_tool_calls.add(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending_tool_calls:
                raise ModelContextPackageError("tool result has no matching assistant tool call")
            pending_tool_calls.remove(call_id)
    if pending_tool_calls:
        raise ModelContextPackageError("assistant tool call is missing a tool result")


def _validate_message(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("role"), str):
        raise ModelContextPackageError(f"invalid model context message at index {index}")
    return deepcopy(value)


def _sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(message)
    sanitized.pop("provider_turn_state", None)
    return _validate_message(sanitized, 0)


def _system_prefix(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            break
        prefix.append(deepcopy(message))
    return prefix


def _insert_fresh_context_snapshot(
    messages: Sequence[dict[str, Any]],
    snapshot: ContextStateSnapshot,
) -> list[dict[str, Any]]:
    filtered = [deepcopy(message) for message in messages if not is_context_state_message(message)]
    index = 0
    while index < len(filtered) and filtered[index].get("role") == "system":
        index += 1
    if index < len(filtered) and filtered[index].get("role") == "user":
        index += 1
    if index < len(filtered) and _is_checkpoint_message(filtered[index]):
        index += 1
    filtered.insert(index, build_context_state_snapshot_message(snapshot))
    return filtered


def _is_checkpoint_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return False
    try:
        payload = json.loads(message["content"])
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("kind") == "context_checkpoint"
