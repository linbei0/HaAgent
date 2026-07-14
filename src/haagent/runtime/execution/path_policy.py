"""
src/haagent/runtime/execution/path_policy.py - 路径信任策略

集中表达项目根和外部授权目录的读写执行边界，供工具、会话和 TUI 复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


PathAccess = Literal["read", "full"]
PermissionMode = Literal["request_approval", "auto_approve", "full_access"]
PathSource = Literal["user", "project"]
PERMISSION_MODES: set[str] = {"request_approval", "auto_approve", "full_access"}


@dataclass(frozen=True)
class ExternalRoot:
    path: Path
    access: PathAccess
    source: PathSource = "user"
    created_at: str = ""

    def resolved(self) -> "ExternalRoot":
        return ExternalRoot(
            path=self.path.resolve(),
            access=self.access,
            source=self.source,
            created_at=self.created_at,
        )


@dataclass(frozen=True)
class PathPolicy:
    project_root: Path
    external_roots: list[ExternalRoot] = field(default_factory=list)
    permission_mode: PermissionMode = "request_approval"

    def resolved(self) -> "PathPolicy":
        return PathPolicy(
            project_root=self.project_root.resolve(),
            external_roots=[root.resolved() for root in self.external_roots],
            permission_mode=self.permission_mode,
        )


def default_path_policy(project_root: Path) -> PathPolicy:
    return PathPolicy(project_root=project_root.resolve())


def resolve_path_for_access(path: str, policy: PathPolicy, required_access: PathAccess) -> Path | str:
    """解析文件路径，并按 read/full 权限返回可用路径或中文错误。"""
    candidate_result = _resolve_candidate(path, policy.project_root)
    if isinstance(candidate_result, str):
        return candidate_result
    if policy.permission_mode == "full_access":
        return candidate_result
    matched = _matching_root(candidate_result, policy)
    if matched is None:
        return "目录未授权：该路径不在当前项目根或已授权外部目录内"
    if required_access == "full" and matched.access != "full":
        return "外部目录只读：需要完全信任后才能修改该路径"
    return candidate_result


def resolve_cwd_for_execution(cwd_arg: str | None, policy: PathPolicy) -> Path | str:
    """解析执行 cwd；执行只允许项目根或完全信任的外部目录。"""
    path_arg = "." if cwd_arg in (None, ".") else cwd_arg
    candidate_result = _resolve_candidate(path_arg, policy.project_root)
    if isinstance(candidate_result, str):
        return candidate_result
    if policy.permission_mode == "full_access":
        if not candidate_result.exists():
            return "执行目录不存在"
        if not candidate_result.is_dir():
            return "执行目录必须是目录"
        return candidate_result
    matched = _matching_root(candidate_result, policy)
    if matched is None:
        return "目录未授权：执行目录不在当前项目根或已授权外部目录内"
    if matched.access != "full":
        return "需要完全信任：只读外部目录不能作为执行目录"
    if not candidate_result.exists():
        return "执行目录不存在"
    if not candidate_result.is_dir():
        return "执行目录必须是目录"
    return candidate_result


def serialize_path_policy(policy: PathPolicy) -> dict[str, Any]:
    resolved = policy.resolved()
    return {
        "project_root": str(resolved.project_root),
        "permission_mode": resolved.permission_mode,
        "external_roots": [
            {
                "path": str(root.path),
                "access": root.access,
                "source": root.source,
                "created_at": root.created_at,
            }
            for root in resolved.external_roots
        ],
    }


def load_path_policy(raw: Any) -> PathPolicy:
    if not isinstance(raw, dict):
        raise ValueError("path_policy must be a mapping")
    project_root = raw.get("project_root")
    if not isinstance(project_root, str):
        raise ValueError("path_policy.project_root must be a string")
    permission_mode = raw.get("permission_mode", "request_approval")
    if permission_mode not in PERMISSION_MODES:
        raise ValueError("path_policy.permission_mode must be request_approval, auto_approve, or full_access")
    roots_raw = raw.get("external_roots", [])
    if not isinstance(roots_raw, list):
        raise ValueError("path_policy.external_roots must be a list")
    external_roots: list[ExternalRoot] = []
    for item in roots_raw:
        if not isinstance(item, dict):
            raise ValueError("path_policy.external_roots items must be mappings")
        path = item.get("path")
        access = item.get("access")
        source = item.get("source", "user")
        created_at = item.get("created_at", "")
        if not isinstance(path, str):
            raise ValueError("external root path must be a string")
        if access not in {"read", "full"}:
            raise ValueError("external root access must be read or full")
        if source not in {"user", "project"}:
            raise ValueError("external root source must be user or project")
        if not isinstance(created_at, str):
            raise ValueError("external root created_at must be a string")
        external_roots.append(
            ExternalRoot(
                path=Path(path).resolve(),
                access=access,
                source=source,
                created_at=created_at,
            ),
        )
    return PathPolicy(
        project_root=Path(project_root).resolve(),
        external_roots=external_roots,
        permission_mode=permission_mode,
    )


def _resolve_candidate(path: str, project_root: Path) -> Path | str:
    if not isinstance(path, str):
        return "path must be a string"
    root = project_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class _MatchedRoot:
    path: Path
    access: PathAccess


def _matching_root(path: Path, policy: PathPolicy) -> _MatchedRoot | None:
    resolved = policy.resolved()
    candidates = [_MatchedRoot(resolved.project_root, "full")]
    candidates.extend(_MatchedRoot(root.path, root.access) for root in resolved.external_roots)
    matches = [root for root in candidates if path == root.path or root.path in path.parents]
    if not matches:
        return None
    return max(matches, key=lambda root: len(root.path.parts))
