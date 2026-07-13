"""
tests/unit/skills/test_skill_catalog.py - SkillCatalogService 缓存合同测试

覆盖 hit/miss、source fingerprint 失效、trust/settings 隔离与 reload 失败边界。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from haagent.skills.catalog import SkillCatalogService
from haagent.skills.settings import SkillSettings


def _write_skill(root: Path, name: str, body: str | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(body or f"# {name}\n{name} guidance\n", encoding="utf-8")
    return skill_path


def test_skill_catalog_hit_reuses_same_snapshot(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    skill_root = config_dir / "skills"
    skill_path = _write_skill(skill_root, "alpha")
    settings = SkillSettings(version=1, trusted_project_roots=())
    service = SkillCatalogService(config_dir=config_dir)
    load_calls = {"count": 0}
    original = service._load_registry

    def counting_load(**kwargs):
        load_calls["count"] += 1
        return original(**kwargs)

    service._load_registry = counting_load  # type: ignore[method-assign]

    first = service.snapshot(tmp_path / "ws", settings)
    second = service.snapshot(tmp_path / "ws", settings)

    assert first is second
    assert load_calls["count"] == 1
    assert first.skills == tuple(sorted(first.skills, key=lambda item: (item.name, item.source, item.path or "")))
    assert first.skills[0].name == "alpha"


def test_skill_catalog_misses_when_skill_content_mtime_changes(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    skill_path = _write_skill(config_dir / "skills", "alpha", "# alpha\nold\n")
    settings = SkillSettings()
    service = SkillCatalogService(config_dir=config_dir)

    first = service.snapshot(tmp_path / "ws", settings)
    time.sleep(0.02)
    skill_path.write_text("# alpha\nnew body\n", encoding="utf-8")
    second = service.snapshot(tmp_path / "ws", settings)

    assert first is not second
    assert "new body" in second.skills[0].content


def test_skill_catalog_misses_when_skill_added(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    skill_root = config_dir / "skills"
    skill_path = _write_skill(skill_root, "alpha")
    settings = SkillSettings()
    service = SkillCatalogService(config_dir=config_dir)

    first = service.snapshot(tmp_path / "ws", settings)
    _write_skill(skill_root, "beta")
    # 根目录 mtime 在某些文件系统上可能延迟；强制 touch 根目录以改变 fingerprint。
    skill_root.touch()
    second = service.snapshot(tmp_path / "ws", settings)

    assert first is not second
    assert {skill.name for skill in second.skills} == {"alpha", "beta"}


def test_skill_catalog_trust_isolation(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    repo = tmp_path / "repo"
    workspace = repo / "pkg"
    workspace.mkdir(parents=True)
    (repo / ".git").mkdir()
    _write_skill(repo / ".haagent" / "skills", "project-skill")
    service = SkillCatalogService(config_dir=config_dir)

    untrusted = service.snapshot(workspace, SkillSettings(trusted_project_roots=()))
    trusted = service.snapshot(
        workspace,
        SkillSettings(trusted_project_roots=(str(repo.resolve()),)),
    )

    assert untrusted is not trusted
    assert all(skill.name != "project-skill" for skill in untrusted.skills)
    assert any(skill.name == "project-skill" for skill in trusted.skills)


def test_skill_catalog_invalidate_workspace_forces_reload(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "alpha")
    settings = SkillSettings()
    service = SkillCatalogService(config_dir=config_dir)
    first = service.snapshot(tmp_path / "ws", settings)
    service.invalidate_workspace(tmp_path / "ws")
    second = service.snapshot(tmp_path / "ws", settings)
    assert first is not second
    assert first.skills == second.skills


def test_skill_catalog_reload_failure_does_not_return_stale(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    _write_skill(config_dir / "skills", "alpha")
    settings = SkillSettings()
    service = SkillCatalogService(config_dir=config_dir)
    first = service.snapshot(tmp_path / "ws", settings)
    assert first.skills

    def boom(**_kwargs):
        raise OSError("permission denied")

    service._load_registry = boom  # type: ignore[method-assign]
    service.invalidate_workspace(tmp_path / "ws")
    with pytest.raises(OSError, match="permission denied"):
        service.snapshot(tmp_path / "ws", settings)


def test_skill_catalog_metadata_error_is_explicit(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = home / ".haagent"
    skill_root = config_dir / "skills"
    skill_path = _write_skill(skill_root, "alpha")
    original_stat = Path.stat

    def denied_stat(self, *args, **kwargs):
        if self == skill_path:
            raise PermissionError("skill metadata denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)

    with pytest.raises(OSError, match="SKILL.md metadata.*skill metadata denied"):
        SkillCatalogService(config_dir=config_dir).snapshot(
            tmp_path / "ws",
            SkillSettings(),
        )
