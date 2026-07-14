"""
tests/unit/context/test_soul_context.py - ContextBuilder Soul 接线测试

验证主 Agent 注入/热重载、worker 跳过与读取失败边界。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from haagent.context.builder import ContextBuildError, ContextBuilder
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter


def test_context_builder_injects_audits_and_reloads_soul(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    global_soul = config_dir / "SOUL.md"
    workspace_soul = workspace / "SOUL.md"
    global_soul.write_text("GLOBAL-CALM", encoding="utf-8")
    workspace_soul.write_text("WORKSPACE-DIRECT", encoding="utf-8")
    (config_dir / "settings.json").write_text(
        json.dumps(
            {"soul": {"trusted_workspace_roots": [str(workspace)]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)
    writer = _make_writer(workspace)

    first = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=workspace,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    assert first.model_input.index("GLOBAL-CALM") < first.model_input.index("WORKSPACE-DIRECT")
    first_manifest = json.loads(
        (writer.path / "contexts" / f"{first.context_id}-manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    soul_decision = next(
        item
        for item in first_manifest["selection"]["selected"]
        if item["source_type"] == "soul"
    )
    assert [source["status"] for source in soul_decision["metadata"]["sources"]] == [
        "loaded",
        "loaded",
    ]

    workspace_soul.write_text("WORKSPACE-UPDATED", encoding="utf-8")
    second = ContextBuilder(
        task=_task("summarize project again"),
        workspace_root=workspace,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    assert "WORKSPACE-UPDATED" in second.model_input
    assert "WORKSPACE-DIRECT" not in second.model_input


def test_context_builder_skips_soul_for_worker_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "SOUL.md").write_text("MAIN-ONLY-SOUL", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)

    def unexpected_soul_call(*args, **kwargs) -> None:
        raise AssertionError("worker attempted to load Soul settings or files")

    monkeypatch.setattr(
        "haagent.context.builder.load_runtime_settings",
        unexpected_soul_call,
    )
    monkeypatch.setattr("haagent.context.builder.load_soul", unexpected_soul_call)
    writer = _make_writer(tmp_path)
    task = TaskSpec(
        goal="worker task",
        constraints=[],
        allowed_tools=["file_read"],
        acceptance_criteria=[],
        verification_commands=[],
        workspace_root=".",
        worker_context={"system_prompt": "WORKER-PROFILE"},
    )

    context = ContextBuilder(
        task=task,
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    assert "MAIN-ONLY-SOUL" not in context.model_input
    manifest = json.loads(
        (writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    soul_decision = next(
        item
        for item in manifest["selection"]["skipped"]
        if item["source_type"] == "soul"
    )
    assert soul_decision["skip_reason"] == "worker_context"


def test_context_builder_wraps_soul_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_text("present", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    original_open = Path.open

    def denied_open(self: Path, *args, **kwargs):
        if self == global_soul:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    writer = _make_writer(tmp_path)

    with pytest.raises(ContextBuildError, match="cannot build Soul context"):
        ContextBuilder(
            task=_task("summarize project"),
            workspace_root=tmp_path,
            provider_name="test-provider",
            episode_writer=writer,
        ).build()


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


def _task(goal: str) -> TaskSpec:
    return TaskSpec(
        goal=goal,
        workspace_root=".",
        allowed_tools=["file_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )
