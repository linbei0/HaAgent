"""
tests/unit/skills/test_skills_loader.py - Skills 加载行为测试

覆盖 HaAgent 本地 Markdown skills 的发现、元数据解析和项目信任边界。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import logging

from haagent.skills import SkillSettings, load_skill_registry
from haagent.skills.loader import discover_project_skill_dirs, parse_skill_metadata


def _write_skill(root: Path, name: str, body: str | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(body or f"# {name}\n{name} guidance\n", encoding="utf-8")
    return skill_path


def test_parse_skill_metadata_supports_yaml_block_scalars() -> None:
    content = textwrap.dedent(
        """\
        ---
        name: Review Flow
        description: >
          Review changed code for correctness,
          tests, and project fit.
        aliases:
          - review-code
        user-invocable: false
        disable-model-invocation: true
        ---

        # Review Flow
        """
    )

    metadata = parse_skill_metadata("review-flow", content)

    assert metadata.name == "Review Flow"
    assert metadata.description == "Review changed code for correctness, tests, and project fit.\n"
    assert metadata.aliases == ("review-code",)
    assert metadata.user_invocable is False
    assert metadata.disable_model_invocation is True


def test_parse_malformed_skill_frontmatter_falls_back_without_warning(caplog) -> None:
    content = "---\nname: [bad yaml\n---\n\n# Fallback\nBody description.\n"

    with caplog.at_level(logging.WARNING, logger="haagent.skills.loader"):
        metadata = parse_skill_metadata("fallback", content)

    assert metadata.name == "Fallback"
    assert metadata.description == "Body description."
    assert not caplog.records


def test_load_skill_registry_includes_user_and_compatibility_dirs(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "personal")
    _write_skill(home / ".agents" / "skills", "agents-flow")
    _write_skill(home / ".claude" / "skills", "claude-flow")

    registry = load_skill_registry(config_dir=config_dir)

    assert registry.get("personal").source == "user"  # type: ignore[union-attr]
    assert registry.get("agents-flow").source == "user"  # type: ignore[union-attr]
    assert registry.get("claude-flow").source == "user"  # type: ignore[union-attr]


def test_builtin_config_skill_is_loaded_and_reserved_name_cannot_be_overridden(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(
        config_dir / "skills",
        "external-config",
        "---\nname: haagent-config\ndescription: external\n---\n\n# External\n",
    )

    registry = load_skill_registry(config_dir=config_dir)

    skill = registry.get("haagent-config")
    assert skill is not None
    assert skill.source == "builtin"
    assert "Configuration layers" in skill.content


def test_project_skills_require_explicit_trust(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    repo = tmp_path / "repo"
    workspace = repo / "packages" / "app"
    workspace.mkdir(parents=True)
    (repo / ".git").mkdir()
    _write_skill(repo / ".haagent" / "skills", "repo-skill")

    untrusted = load_skill_registry(
        workspace_root=workspace,
        config_dir=home / ".haagent",
        settings=SkillSettings(version=1, trusted_project_roots=()),
    )
    trusted = load_skill_registry(
        workspace_root=workspace,
        config_dir=home / ".haagent",
        settings=SkillSettings(version=1, trusted_project_roots=(str(repo.resolve()),)),
    )

    assert untrusted.get("repo-skill") is None
    assert trusted.get("repo-skill").source == "project"  # type: ignore[union-attr]


def test_project_skill_discovery_stops_at_git_root_and_nearer_skill_overrides(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    workspace = repo / "packages" / "app"
    workspace.mkdir(parents=True)
    (repo / ".git").mkdir()
    root_skill = _write_skill(repo / ".haagent" / "skills", "deploy", "# deploy\nroot\n")
    near_skill = _write_skill(workspace / ".agents" / "skills", "deploy", "# deploy\nnear\n")

    dirs = discover_project_skill_dirs(workspace)
    registry = load_skill_registry(
        workspace_root=workspace,
        user_skill_dirs=[],
        settings=SkillSettings(version=1, trusted_project_roots=(str(repo.resolve()),)),
    )

    assert root_skill.parent.parent.resolve() in dirs
    assert near_skill.parent.parent.resolve() in dirs
    assert "near" in registry.get("deploy").content  # type: ignore[union-attr]
