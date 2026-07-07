"""
haagent/multi_agent/profiles.py - 多智能体 profile 定义与加载

定义 worker 角色配置，供 MultiAgentRuntime 用明确配置创建后台帮手。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haagent.models.model_connections import user_config_dir


VALID_SUBAGENT_TYPES = {"explorer", "worker", "verification"}
SUPPORTED_BACKENDS = {"in_process", "subprocess"}


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    subagent_type: str
    system_prompt: str
    model_profile: str | None = None
    allowed_tools: list[str] | None = None
    approval_allowed_tools: list[str] | None = None
    approved_tools: list[str] | None = None
    max_turns: int | None = None
    enable_web: bool | None = None
    backend: str | None = None
    worktree: bool | None = None


@dataclass(frozen=True)
class WorkerProfileRuntime:
    name: str
    subagent_type: str
    system_prompt: str
    model_profile: str | None
    allowed_tools: list[str] | None
    approval_allowed_tools: list[str] | None
    approved_tools: list[str] | None
    max_turns: int | None
    enable_web: bool | None
    backend: str
    worktree: bool


def load_builtin_agent_profiles() -> dict[str, AgentProfile]:
    return {
        "explorer": AgentProfile(
            name="explorer",
            description="只读探索文件、资料和项目结构。",
            subagent_type="explorer",
            system_prompt="你是只读探索助手。读取资料、总结发现，不修改文件。",
            allowed_tools=["file_list", "grep", "file_read", "skill_list", "skill_read"],
        ),
        "worker": AgentProfile(
            name="worker",
            description="按主助手授权执行普通任务。",
            subagent_type="worker",
            system_prompt="你是执行助手。按任务要求完成工作，并清楚汇报结果。",
        ),
        "verification": AgentProfile(
            name="verification",
            description="运行验证、读取结果并指出风险。",
            subagent_type="verification",
            system_prompt="你是验证助手。运行检查、解释结果，不做无关修改。",
            allowed_tools=["file_read", "grep", "shell", "code_run"],
        ),
    }


def load_user_agent_profiles(config_dir: Path) -> dict[str, AgentProfile]:
    agents_dir = config_dir / "agents"
    if not agents_dir.exists():
        return {}
    profiles: dict[str, AgentProfile] = {}
    for path in sorted(agents_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = _profile_from_dict(raw, source=str(path))
        profiles[profile.name] = profile
    return profiles


def get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile:
    profiles = load_builtin_agent_profiles()
    if config_dir is not None:
        profiles.update(load_user_agent_profiles(config_dir))
    if name not in profiles:
        raise ValueError(f"unknown agent profile: {name}")
    return profiles[name]


def resolve_worker_profile(
    name: str | None,
    *,
    fallback_subagent_type: str,
    config_dir: Path | None = None,
) -> WorkerProfileRuntime:
    base_dir = config_dir or user_config_dir()
    profile = get_agent_profile(name or fallback_subagent_type, config_dir=base_dir)
    backend = profile.backend or "in_process"
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    return WorkerProfileRuntime(
        name=profile.name,
        subagent_type=profile.subagent_type,
        system_prompt=profile.system_prompt,
        model_profile=profile.model_profile,
        allowed_tools=profile.allowed_tools,
        approval_allowed_tools=profile.approval_allowed_tools,
        approved_tools=profile.approved_tools,
        max_turns=profile.max_turns,
        enable_web=profile.enable_web,
        backend=backend,
        worktree=bool(profile.worktree),
    )


def _profile_from_dict(raw: dict[str, Any], *, source: str) -> AgentProfile:
    name = _required_string(raw, "name", source=source)
    description = _required_string(raw, "description", source=source)
    subagent_type = _required_string(raw, "subagent_type", source=source)
    if subagent_type not in VALID_SUBAGENT_TYPES:
        raise ValueError(f"invalid subagent_type in {source}: {subagent_type}")
    system_prompt = _required_string(raw, "system_prompt", source=source)
    return AgentProfile(
        name=name,
        description=description,
        subagent_type=subagent_type,
        system_prompt=system_prompt,
        allowed_tools=_optional_string_list(raw, "allowed_tools", source=source),
        approval_allowed_tools=_optional_string_list(raw, "approval_allowed_tools", source=source),
        approved_tools=_optional_string_list(raw, "approved_tools", source=source),
        model_profile=_optional_string(raw, "model_profile", source=source),
        max_turns=_optional_positive_int(raw, "max_turns", source=source),
        enable_web=_optional_bool(raw, "enable_web", source=source),
        backend=_optional_string(raw, "backend", source=source),
        worktree=_optional_bool(raw, "worktree", source=source),
    )


def _required_string(raw: dict[str, Any], field_name: str, *, source: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required string {field_name} in {source}")
    return value


def _optional_string(raw: dict[str, Any], field_name: str, *, source: str) -> str | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid string {field_name} in {source}")
    return value


def _optional_string_list(raw: dict[str, Any], field_name: str, *, source: str) -> list[str] | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"invalid string list {field_name} in {source}")
    return list(value)


def _optional_positive_int(raw: dict[str, Any], field_name: str, *, source: str) -> int | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid positive integer {field_name} in {source}")
    return value


def _optional_bool(raw: dict[str, Any], field_name: str, *, source: str) -> bool | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"invalid boolean {field_name} in {source}")
    return value
