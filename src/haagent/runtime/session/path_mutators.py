"""
src/haagent/runtime/session/path_mutators.py - session 路径策略变更

对 PathPolicy 做纯变更，由 AgentSession 负责写回 session metadata。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from haagent.runtime.execution.path_policy import (
    ExternalRoot,
    PathAccess,
    PathPolicy,
    PermissionMode,
)
from haagent.runtime.session.package import ChatSessionError


def with_external_root_added(policy: PathPolicy, workspace_root: Path, path: Path, access: PathAccess) -> PathPolicy:
    resolved_path = path.resolve()
    roots = [root for root in policy.external_roots if root.path.resolve() != resolved_path]
    roots.append(
        ExternalRoot(
            path=resolved_path,
            access=access,
            source="user",
            created_at=datetime.now(UTC).isoformat(),
        ),
    )
    return PathPolicy(
        project_root=workspace_root,
        external_roots=roots,
        permission_mode=policy.permission_mode,
    ).resolved()


def with_external_root_removed(policy: PathPolicy, workspace_root: Path, path: Path) -> PathPolicy:
    resolved_path = path.resolve()
    roots = [root for root in policy.external_roots if root.path.resolve() != resolved_path]
    return PathPolicy(
        project_root=workspace_root,
        external_roots=roots,
        permission_mode=policy.permission_mode,
    ).resolved()


def with_external_root_access(
    policy: PathPolicy,
    workspace_root: Path,
    path: Path,
    access: PathAccess,
) -> PathPolicy:
    resolved_path = path.resolve()
    roots: list[ExternalRoot] = []
    found = False
    for root in policy.external_roots:
        if root.path.resolve() == resolved_path:
            roots.append(
                ExternalRoot(
                    path=resolved_path,
                    access=access,
                    source=root.source,
                    created_at=root.created_at,
                ),
            )
            found = True
        else:
            roots.append(root)
    if not found:
        roots.append(
            ExternalRoot(
                path=resolved_path,
                access=access,
                source="user",
                created_at=datetime.now(UTC).isoformat(),
            ),
        )
    return PathPolicy(
        project_root=workspace_root,
        external_roots=roots,
        permission_mode=policy.permission_mode,
    ).resolved()


def with_external_roots_cleared(policy: PathPolicy, workspace_root: Path) -> PathPolicy:
    return PathPolicy(
        project_root=workspace_root,
        permission_mode=policy.permission_mode,
    ).resolved()


def with_project_root(policy: PathPolicy, path: Path) -> tuple[Path, PathPolicy]:
    workspace_root = path.resolve()
    return workspace_root, PathPolicy(
        project_root=workspace_root,
        permission_mode=policy.permission_mode,
    ).resolved()


def with_permission_mode(policy: PathPolicy, workspace_root: Path, mode: PermissionMode) -> PathPolicy:
    if mode not in {"request_approval", "auto_approve", "full_access"}:
        raise ChatSessionError("permission mode must be request_approval, auto_approve, or full_access")
    return PathPolicy(
        project_root=workspace_root,
        external_roots=policy.external_roots,
        permission_mode=mode,
    ).resolved()
