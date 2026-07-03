"""
tests/unit/runtime/test_path_policy.py - 路径信任策略测试

验证项目根、外部只读目录和外部完全信任目录的统一路径解析边界。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.execution.path_policy import (
    ExternalRoot,
    PathPolicy,
    load_path_policy,
    resolve_cwd_for_execution,
    resolve_path_for_access,
    serialize_path_policy,
)


def test_project_root_allows_read_write_and_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    policy = PathPolicy(project_root=project)

    assert resolve_path_for_access("notes.txt", policy, "read") == target.resolve()
    assert resolve_path_for_access("notes.txt", policy, "full") == target.resolve()
    assert resolve_cwd_for_execution(".", policy) == project.resolve()


def test_untrusted_external_directory_is_denied(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    policy = PathPolicy(project_root=project)

    result = resolve_path_for_access(str(external), policy, "read")

    assert isinstance(result, str)
    assert "目录未授权" in result


def test_external_read_root_allows_read_but_rejects_full_and_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    policy = PathPolicy(
        project_root=project,
        external_roots=[ExternalRoot(path=external, access="read", source="user", created_at="now")],
    )

    assert resolve_path_for_access(str(external), policy, "read") == external.resolve()
    write_result = resolve_path_for_access(str(external), policy, "full")
    cwd_result = resolve_cwd_for_execution(str(external), policy)

    assert isinstance(write_result, str)
    assert "外部目录只读" in write_result
    assert isinstance(cwd_result, str)
    assert "需要完全信任" in cwd_result


def test_external_full_root_allows_write_and_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    policy = PathPolicy(
        project_root=project,
        external_roots=[ExternalRoot(path=external, access="full", source="user", created_at="now")],
    )

    assert resolve_path_for_access(str(external), policy, "full") == external.resolve()
    assert resolve_cwd_for_execution(str(external), policy) == external.resolve()


def test_auto_approve_mode_does_not_expand_path_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    policy = PathPolicy(project_root=project, permission_mode="auto_approve")

    result = resolve_path_for_access(str(external), policy, "read")

    assert isinstance(result, str)
    assert "目录未授权" in result


def test_full_access_mode_allows_external_read_write_and_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    target = external / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    policy = PathPolicy(project_root=project, permission_mode="full_access")

    assert resolve_path_for_access(str(target), policy, "read") == target.resolve()
    assert resolve_path_for_access(str(target), policy, "full") == target.resolve()
    assert resolve_cwd_for_execution(str(external), policy) == external.resolve()


def test_most_specific_external_root_wins(tmp_path: Path) -> None:
    project = tmp_path / "project"
    parent = tmp_path / "external"
    child = parent / "child"
    project.mkdir()
    child.mkdir(parents=True)
    policy = PathPolicy(
        project_root=project,
        external_roots=[
            ExternalRoot(path=parent, access="read", source="user", created_at="older"),
            ExternalRoot(path=child, access="full", source="user", created_at="newer"),
        ],
    )

    assert resolve_path_for_access(str(child), policy, "full") == child.resolve()


def test_path_policy_serializes_and_loads_resolved_roots(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    policy = PathPolicy(
        project_root=project,
        external_roots=[ExternalRoot(path=external, access="read", source="user", created_at="now")],
    )

    loaded = load_path_policy(serialize_path_policy(policy))

    assert loaded.project_root == project.resolve()
    assert loaded.permission_mode == "request_approval"
    assert loaded.external_roots == [
        ExternalRoot(path=external.resolve(), access="read", source="user", created_at="now"),
    ]


def test_path_policy_serializes_and_loads_permission_mode(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    policy = PathPolicy(project_root=project, permission_mode="full_access")

    raw = serialize_path_policy(policy)
    loaded = load_path_policy(raw)

    assert raw["permission_mode"] == "full_access"
    assert loaded.permission_mode == "full_access"
