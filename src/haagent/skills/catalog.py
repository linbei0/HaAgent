"""
src/haagent/skills/catalog.py - Skill 目录缓存服务

按完整 cache key 与 source fingerprint 缓存不可变 SkillCatalogSnapshot，
避免每轮重复扫描/解析 SKILL.md；reload 失败不返回陈旧 snapshot。
"""

from __future__ import annotations

import hashlib
import json
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from haagent.skills.loader import (
    discover_project_skill_dirs,
    get_user_skill_dirs,
    load_skill_registry,
)
from haagent.skills.registry import SkillRegistry
from haagent.skills.settings import SkillSettings, is_project_root_trusted
from haagent.skills.types import SkillDefinition


CacheDiagnosticsSink = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class SkillCatalogKey:
    workspace_root: str
    user_roots: tuple[str, ...]
    project_roots: tuple[str, ...]
    project_trusted: bool
    settings_fingerprint: str


@dataclass(frozen=True)
class SkillCatalogSnapshot:
    """不可变 skill 目录快照；不暴露可变 registry。"""

    key: SkillCatalogKey
    source_fingerprint: str
    skills: tuple[SkillDefinition, ...]

    def as_registry(self) -> SkillRegistry:
        # 每次新建 registry，调用方修改不能污染 snapshot。
        registry = SkillRegistry()
        for skill in self.skills:
            registry.register(skill)
        return registry


class SkillCatalogService:
    """Skill 发现结果的进程内缓存；loader 本身保持无状态。"""

    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        load_registry: Callable[..., SkillRegistry] | None = None,
    ) -> None:
        self._config_dir = config_dir
        self._load_registry = load_registry or load_skill_registry
        self._cache: dict[SkillCatalogKey, SkillCatalogSnapshot] = {}

    def snapshot(
        self,
        workspace_root: Path | None,
        settings: SkillSettings,
        *,
        user_skill_dirs: list[Path] | None = None,
        diagnostics_sink: CacheDiagnosticsSink | None = None,
    ) -> SkillCatalogSnapshot:
        resolved_workspace = (
            None if workspace_root is None else Path(workspace_root).expanduser().resolve()
        )
        # loader 会对 user roots mkdir；先对齐目录存在性，避免首次 load 改变 mtime 导致假 miss。
        self._ensure_user_skill_dirs(user_skill_dirs)
        # 先用当前磁盘元数据组 key（只 stat，不读正文）；命中则不再 load/parse。
        key, source_fingerprint = self._build_key(
            resolved_workspace,
            settings,
            user_skill_dirs=user_skill_dirs,
        )
        cached = self._cache.get(key)
        if cached is not None and cached.source_fingerprint == source_fingerprint:
            snapshot = cached
            self._publish(diagnostics_sink, "hit", snapshot)
            return snapshot
        status = "reload" if cached is not None else "miss"

        # reload 失败必须向上抛，禁止回退陈旧 snapshot。
        registry = self._load_registry(
            workspace_root=resolved_workspace,
            config_dir=self._config_dir,
            user_skill_dirs=user_skill_dirs,
            settings=settings,
        )
        skills = tuple(
            sorted(
                registry.list_skills(),
                key=lambda item: (item.name, item.source, item.path or ""),
            )
        )
        snapshot = SkillCatalogSnapshot(
            key=key,
            source_fingerprint=source_fingerprint,
            skills=skills,
        )
        self._cache[key] = snapshot
        self._publish(diagnostics_sink, status, snapshot)
        return snapshot

    def invalidate_workspace(self, workspace_root: Path) -> None:
        resolved = str(Path(workspace_root).expanduser().resolve())
        for key in tuple(self._cache):
            if key.workspace_root == resolved:
                del self._cache[key]

    @staticmethod
    def _publish(
        sink: CacheDiagnosticsSink | None,
        status: str,
        snapshot: SkillCatalogSnapshot,
    ) -> None:
        if sink is None:
            return
        summary_chars = sum(
            len(skill.name) + len(skill.source) + len(skill.description)
            for skill in snapshot.skills
        )
        sink(
            {
                "status": status,
                "count": len(snapshot.skills),
                "chars": summary_chars,
                "fingerprint": f"sha256:{snapshot.source_fingerprint}",
            },
        )

    def _ensure_user_skill_dirs(self, user_skill_dirs: list[Path] | None) -> None:
        roots = (
            user_skill_dirs
            if user_skill_dirs is not None
            else get_user_skill_dirs(config_dir=self._config_dir)
        )
        for root in roots:
            path = Path(root).expanduser()
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                # 创建失败留给后续 load 显式抛错；探测阶段不吞成假命中。
                continue

    def _build_key(
        self,
        workspace_root: Path | None,
        settings: SkillSettings,
        *,
        user_skill_dirs: list[Path] | None,
    ) -> tuple[SkillCatalogKey, str]:
        user_roots = tuple(
            str(path.expanduser().resolve())
            for path in (
                user_skill_dirs
                if user_skill_dirs is not None
                else get_user_skill_dirs(config_dir=self._config_dir)
            )
        )
        project_roots: tuple[str, ...] = ()
        project_trusted = False
        if workspace_root is not None:
            project_trusted = is_project_root_trusted(workspace_root, settings)
            if project_trusted:
                project_roots = tuple(
                    str(path.resolve()) for path in discover_project_skill_dirs(workspace_root)
                )
        settings_fingerprint = hashlib.sha256(
            json.dumps(settings.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return (
            SkillCatalogKey(
                workspace_root="" if workspace_root is None else str(workspace_root),
                user_roots=user_roots,
                project_roots=project_roots,
                project_trusted=project_trusted,
                settings_fingerprint=settings_fingerprint,
            ),
            self._source_fingerprint(user_roots + project_roots),
        )

    def _source_fingerprint(self, roots: tuple[str, ...]) -> str:
        """只收集 root/SKILL.md 元数据；不读取正文，避免把 cache 探测变成全量解析。"""
        root_entries: list[tuple[str, int | None]] = []
        skill_entries: list[tuple[str, int | None, int | None]] = []
        for root in roots:
            path = Path(root)
            try:
                root_stat = path.stat()
            except FileNotFoundError:
                root_entries.append((root, None))
                continue
            except OSError as error:
                raise OSError(f"failed to read skill root metadata: {path}: {error}") from error
            mtime_ns = root_stat.st_mtime_ns
            root_entries.append((root, mtime_ns))
            if not stat_module.S_ISDIR(root_stat.st_mode):
                continue
            try:
                children = sorted(path.iterdir())
            except OSError as error:
                raise OSError(f"failed to list skill root: {path}: {error}") from error
            for child in children:
                try:
                    child_stat = child.stat()
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise OSError(f"failed to read skill directory metadata: {child}: {error}") from error
                if not stat_module.S_ISDIR(child_stat.st_mode):
                    continue
                skill_path = child / "SKILL.md"
                try:
                    skill_stat = skill_path.stat()
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise OSError(f"failed to read SKILL.md metadata: {skill_path}: {error}") from error
                if stat_module.S_ISREG(skill_stat.st_mode):
                    skill_entries.append(
                        (str(skill_path.resolve()), skill_stat.st_mtime_ns, skill_stat.st_size),
                    )
        skill_entries.sort(key=lambda item: item[0])
        root_entries.sort(key=lambda item: item[0])
        payload = {
            "roots": root_entries,
            "skills": skill_entries,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
