"""
src/haagent/runtime/execution/path_policy.py - 路径信任策略

集中表达项目根和外部授权目录的读写执行边界，供工具、会话和 TUI 复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Literal

PathAccess = Literal["read", "full"]
PermissionMode = Literal["request_approval", "auto_approve", "full_access"]
PathSource = Literal["user", "project"]
PathAccessStatus = Literal["allowed", "approval_required", "denied"]
PERMISSION_MODES: set[str] = {"request_approval", "auto_approve", "full_access"}
_PERCENT_ENVIRONMENT_VARIABLE = re.compile(r"%([^%]+)%")


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


@dataclass(frozen=True)
class PathAccessDecision:
    """路径访问判定；未授权外部目录可以进入显式审批，而非直接失败。"""

    status: PathAccessStatus
    path: Path | None = None
    authorization_root: Path | None = None
    message: str = ""
    error_type: str = "path_policy_denied"


def default_path_policy(project_root: Path) -> PathPolicy:
    return PathPolicy(project_root=project_root.resolve())


def resolve_path_for_access(path: str, policy: PathPolicy, required_access: PathAccess) -> Path | str:
    """解析文件路径，并按 read/full 权限返回可用路径或中文错误。"""
    decision = classify_path_access(path, policy, required_access)
    if decision.status == "allowed" and decision.path is not None:
        return decision.path
    return decision.message


def classify_path_access(path: str, policy: PathPolicy, required_access: PathAccess) -> PathAccessDecision:
    """返回允许、需外部目录审批或拒绝，供文件/命令工具统一处理。"""
    candidate_result = _resolve_candidate(path, policy.project_root)
    if isinstance(candidate_result, str):
        return PathAccessDecision(status="denied", message=candidate_result)
    if policy.permission_mode == "full_access":
        return PathAccessDecision(status="allowed", path=candidate_result)
    matched = _matching_root(candidate_result, policy)
    if matched is None:
        if policy.permission_mode == "request_approval":
            return PathAccessDecision(
                status="approval_required",
                path=candidate_result,
                authorization_root=_authorization_root(candidate_result),
                message="目录未授权：需要用户确认后访问外部目录",
                error_type="approval_required",
            )
        return PathAccessDecision(
            status="denied",
            path=candidate_result,
            message="目录未授权：该路径不在当前项目根或已授权外部目录内",
        )
    if required_access == "full" and matched.access != "full":
        if policy.permission_mode == "request_approval":
            return PathAccessDecision(
                status="approval_required",
                path=candidate_result,
                authorization_root=matched.path,
                message="外部目录只读：需要用户确认后提升为完全访问",
                error_type="approval_required",
            )
        return PathAccessDecision(
            status="denied",
            path=candidate_result,
            message="外部目录只读：需要完全信任后才能修改该路径",
        )
    return PathAccessDecision(status="allowed", path=candidate_result)


def authorize_external_root(policy: PathPolicy, path: Path, access: PathAccess) -> None:
    """把用户选择“始终允许”的目录加入本次会话共享策略。"""
    root = path.resolve()
    roots = [item for item in policy.external_roots if item.path.resolve() != root]
    roots.append(ExternalRoot(path=root, access=access, source="user"))
    # PathPolicy 保持不可替换；列表是运行时共享状态，原地更新让已绑定 handler 同步看到授权。
    policy.external_roots[:] = roots


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
    try:
        # `~` 与 Windows 的 `%USERPROFILE%` 均先展开为真实用户目录。
        # 展开不授予权限，后续仍按 project/external-root/full_access 策略判断。
        candidate = Path(_expand_environment_variables(path)).expanduser()
    except RuntimeError:
        return "无法解析用户目录：请使用明确的绝对路径"
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _expand_environment_variables(path: str) -> str:
    """展开当前进程环境变量，并跨平台支持 Windows 的 `%NAME%` 语法。"""

    expanded = os.path.expandvars(path)
    return _PERCENT_ENVIRONMENT_VARIABLE.sub(
        lambda match: os.environ.get(match.group(1), match.group(0)),
        expanded,
    )


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


def _authorization_root(path: Path) -> Path:
    candidate = path if path.is_dir() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.resolve()
