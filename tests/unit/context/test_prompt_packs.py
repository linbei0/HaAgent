"""
tests/unit/context/test_prompt_packs.py - Prompt Pack 上下文注入测试

验证显式选择的内置提示词包会作为原子 system 上下文进入模型输入和 manifest。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.context.compaction import ContextBudget
from haagent.prompts import packs
from haagent.prompts.packs import PromptPack
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter


def test_context_builder_injects_selected_prompt_pack_and_manifest_record(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)

    context = ContextBuilder(
        task=_task("review changes", prompt_pack_ids=["code-review"]),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    manifest = _read_manifest(writer, context.context_id)

    assert "Prompt Packs:" in context.model_input
    assert "Task mode: code review." in context.model_input
    prompt_pack_records = [
        record for record in manifest["selection"]["selected"] if record["source_type"] == "prompt_pack"
    ]
    assert len(prompt_pack_records) == 1
    assert prompt_pack_records[0]["metadata"]["selected"] == [
        {
            "id": "code-review",
            "title": "Code Review",
            "chars": len(packs.get_prompt_pack("code-review").content),
        },
    ]


def test_prompt_pack_is_not_partially_truncated_by_compaction(tmp_path: Path, monkeypatch) -> None:
    content = "Task mode: oversized prompt pack.\n" + "A" * 1200
    monkeypatch.setitem(
        packs._PACKS,
        "oversized-test",
        PromptPack(
            id="oversized-test",
            title="Oversized Test",
            content=content,
            max_chars=2000,
        ),
    )
    writer = _make_writer(tmp_path)

    context = ContextBuilder(
        task=_task("use oversized prompt pack", prompt_pack_ids=["oversized-test"]),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        compaction_budget=ContextBudget(max_total_chars=5000, max_section_chars=100),
    ).build()

    assert content in context.model_input
    assert "[collapsed" not in context.model_input


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)
    writer.write_plan(
        {
            "goal": "test",
            "constraints": [],
            "acceptance_criteria": [],
            "verification_commands": [],
            "planned_steps": ["Use allowed tools."],
        },
    )
    return writer


def _task(goal: str, *, prompt_pack_ids: list[str]) -> TaskSpec:
    return TaskSpec(
        goal=goal,
        workspace_root=".",
        allowed_tools=["file_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        prompt_pack_ids=prompt_pack_ids,
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )


def _read_manifest(writer: EpisodeWriter, context_id: str) -> dict:
    manifest_path = writer.path / "contexts" / f"{context_id}-manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
