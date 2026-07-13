"""
tests/unit/context/test_shared_cache_services.py - Task 9 共享缓存与稳定前缀

同一 cache 实例跨两次 context build / schema export 复用；
model_input 与 tool schemas UTF-8 字节稳定；catalog 与直载行为等价。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.context.instruction_cache import InstructionCache
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.skills.catalog import SkillCatalogService
from haagent.skills.settings import SkillSettings
from haagent.tools.registry import ToolRuntimeRegistry, export_tool_schemas
from haagent.tools.schema_cache import ToolSchemaCache
from haagent.tools.skills import skill_list, skill_read


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _task(goal: str, *, allowed_tools: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        goal=goal,
        workspace_root=".",
        allowed_tools=allowed_tools or ["file_read", "skill_list", "skill_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )


def _write_skill(root: Path, name: str, body: str | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(body or f"# {name}\n{name} guidance for tests\n", encoding="utf-8")
    return skill_path


def test_shared_caches_reuse_across_two_context_builds(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "alpha")
    (tmp_path / "AGENTS.md").write_text("# agents\nstable instruction\n", encoding="utf-8")

    instruction_cache = InstructionCache()
    skill_catalog = SkillCatalogService(config_dir=config_dir)
    settings = SkillSettings()

    load_calls = {"registry": 0, "agents_reads": 0}
    original_load = skill_catalog._load_registry

    def counting_load(**kwargs):
        load_calls["registry"] += 1
        return original_load(**kwargs)

    skill_catalog._load_registry = counting_load  # type: ignore[method-assign]

    original_read = Path.read_text

    def counting_read(self, *args, **kwargs):
        if self.name == "AGENTS.md":
            load_calls["agents_reads"] += 1
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)

    writer1 = _make_writer(tmp_path / "ep1")
    first = ContextBuilder(
        task=_task("round-1"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer1,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()

    writer2 = _make_writer(tmp_path / "ep2")
    second = ContextBuilder(
        task=_task("round-2"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer2,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()

    assert load_calls["registry"] == 1
    assert load_calls["agents_reads"] == 1
    assert "stable instruction" in first.model_input
    assert "stable instruction" in second.model_input
    assert "alpha" in first.model_input
    assert "alpha" in second.model_input
    first_cache = first.manifest.source_diagnostics["cache"]
    second_cache = second.manifest.source_diagnostics["cache"]
    assert first_cache["instructions"]["status"] == "miss"
    assert first_cache["skills"]["status"] == "miss"
    assert second_cache["instructions"]["status"] == "hit"
    assert second_cache["skills"]["status"] == "hit"
    assert second_cache["skills"]["count"] == 1
    assert second_cache["skills"]["chars"] > 0

    # metadata 变化后必须 reload
    (tmp_path / "AGENTS.md").write_text("# agents\nupdated instruction\n", encoding="utf-8")
    writer3 = _make_writer(tmp_path / "ep3")
    third = ContextBuilder(
        task=_task("round-3"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer3,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()
    assert load_calls["agents_reads"] == 2
    assert "updated instruction" in third.model_input
    assert third.manifest.source_diagnostics["cache"]["instructions"]["status"] == "reload"


def test_stable_prefix_model_input_and_schemas_identical_twice(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "alpha")
    (tmp_path / "AGENTS.md").write_text("# agents\nprefix body\n", encoding="utf-8")

    instruction_cache = InstructionCache()
    skill_catalog = SkillCatalogService(config_dir=config_dir)
    schema_cache = ToolSchemaCache()
    names = ["file_read", "skill_list", "skill_read"]

    writer_a = _make_writer(tmp_path / "stable-a")
    built_a = ContextBuilder(
        task=_task("stable", allowed_tools=names),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer_a,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()
    writer_b = _make_writer(tmp_path / "stable-b")
    built_b = ContextBuilder(
        task=_task("stable", allowed_tools=names),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer_b,
        instruction_cache=instruction_cache,
        skill_catalog=skill_catalog,
    ).build()

    input_a = built_a.model_input.encode("utf-8")
    input_b = built_b.model_input.encode("utf-8")
    assert input_a == input_b

    registry = ToolRuntimeRegistry(
        static_tools={},
        dynamic_tools={},
    )
    # use default registry via export_tool_schemas without empty custom registry
    schemas_a = export_tool_schemas(names, cache=schema_cache)
    schemas_b = export_tool_schemas(names, cache=schema_cache)
    assert schemas_a == schemas_b
    assert schemas_a is not schemas_b
    bytes_a = json.dumps(schemas_a, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    bytes_b = json.dumps(schemas_b, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert bytes_a == bytes_b
    del registry


def test_catalog_path_behavior_matches_direct_skill_load(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "alpha", "# alpha\nfull body line\n")
    settings = SkillSettings()
    catalog = SkillCatalogService(config_dir=config_dir)

    listed_direct = skill_list({}, workspace_root=tmp_path)
    listed_catalog = skill_list({}, workspace_root=tmp_path, skill_catalog=catalog)
    assert listed_catalog == listed_direct

    read_direct = skill_read({"name": "alpha"}, workspace_root=tmp_path)
    read_catalog = skill_read({"name": "alpha"}, workspace_root=tmp_path, skill_catalog=catalog)
    assert read_catalog == read_direct
    assert "full body line" in str(read_catalog.get("content", ""))

    writer = _make_writer(tmp_path / "eq")
    with_catalog = ContextBuilder(
        task=_task("eq"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        skill_catalog=catalog,
        instruction_cache=InstructionCache(),
    ).build()
    writer2 = _make_writer(tmp_path / "eq2")
    without = ContextBuilder(
        task=_task("eq"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer2,
        instruction_cache=InstructionCache(),
    ).build()
    # skills block 成员应一致（Available Skills 段）
    assert "Available Skills:" in with_catalog.model_input
    assert "Available Skills:" in without.model_input
    catalog_skills = with_catalog.model_input.split("Available Skills:")[1]
    direct_skills = without.model_input.split("Available Skills:")[1]
    assert catalog_skills == direct_skills
