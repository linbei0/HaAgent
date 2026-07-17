"""
haagent/tools/catalog.py - 静态工具 Deep Catalog

将 definition、handler binder、展示/observation 投影、guardrail 与 chat tags
收拢为 ToolContribution，由 ToolCatalog 统一校验与查询。
动态 MCP 工具不进入本目录，仍由 ToolRouter 的 mcp__ 分支处理。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.guardrails import GuardrailResult
from haagent.runtime.execution.path_policy import PathPolicy
from haagent.runtime.execution.retry import ReplaySafety
from haagent.runtime.sandbox.base import SandboxBackend
from haagent.skills import SkillSettings
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.registry import (
    ALLOWED_EXECUTION_EFFECTS,
    ALLOWED_RISK_LEVELS,
    ExecutionEffect,
    ToolDefinition,
    validate_tool_registry,
)

# 供 contribution / router / 测试统一导入。
__all__ = [
    "ToolCatalog",
    "ToolContribution",
    "ToolExecutionContext",
    "ToolRuntimeDeps",
    "default_tool_catalog",
    "reset_default_tool_catalog_for_tests",
]

ArgsSummaryFn = Callable[[dict[str, Any]], dict[str, object]]
ResultSummaryFn = Callable[[dict[str, Any]], dict[str, object]]
ObservationProjector = Callable[[dict[str, Any], dict[str, Any]], dict[str, object]]
GuardrailFn = Callable[[dict[str, Any]], GuardrailResult | None]
HandlerBinder = Callable[["ToolRuntimeDeps"], ToolHandler]


@dataclass
class ToolRuntimeDeps:
    """构造静态 handler 时注入的运行时依赖；不含 policy/approval。"""

    workspace_root: Path
    path_policy: PathPolicy
    skill_settings: SkillSettings | None = None
    cancellation_token: CancellationToken | None = None
    mcp_runtime: Any | None = None
    sandbox_backend: SandboxBackend | None = None
    skill_catalog: SkillCatalogService | None = None
    router_handlers: dict[str, ToolHandler] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContribution:
    """单个静态工具的完整登记单元；缺 safety 字段必须失败。"""

    name: str
    description: str
    parameters: dict[str, Any]
    risk_level: str
    execution_effect: ExecutionEffect
    replay_safety: ReplaySafety
    tags: frozenset[str] = frozenset()
    router_owned: bool = False
    bind_handler: HandlerBinder | None = None
    summarize_args: ArgsSummaryFn | None = None
    summarize_result: ResultSummaryFn | None = None
    interaction_args_summary: ArgsSummaryFn | None = None
    project_observation: ObservationProjector | None = None
    guardrail: GuardrailFn | None = None
    display_name_zh: str | None = None
    observation_long_text_keys: tuple[str, ...] = ()

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            risk_level=self.risk_level,
            parameters=self.parameters,
            execution_effect=self.execution_effect,
            replay_safety=self.replay_safety,
        )


# chat 可见性 tag 的产品顺序；成员资格仍由 contribution.tags 决定。
_CHAT_DEFAULT_ORDER = (
    "file_list",
    "grep",
    "file_read",
    "request_user_input",
    "start_memory_update",
    "file_write",
    "code_run",
    "apply_patch",
    "apply_patch_set",
    "shell",
    "agent",
    "send_message",
    "task_stop",
    "task_get",
    "task_list",
    "task_output",
)
_CHAT_WEB_ORDER = ("web_search", "web_fetch", "skill_market_search")
_CHAT_SKILL_ORDER = ("skill_list", "skill_read")
_CHAT_APPROVAL_ORDER = (
    "file_write",
    "code_run",
    "apply_patch",
    "apply_patch_set",
    "shell",
)


class ToolCatalog:
    """静态工具唯一登记表：定义、绑定、投影与 tag 查询。"""

    def __init__(self, contributions: Sequence[ToolContribution]) -> None:
        by_name: dict[str, ToolContribution] = {}
        for contribution in contributions:
            _validate_contribution(contribution)
            if contribution.name in by_name:
                raise ValueError(f"duplicate tool contribution: {contribution.name}")
            by_name[contribution.name] = contribution
        self._contributions = by_name
        self._definitions = {
            name: item.to_definition() for name, item in by_name.items()
        }
        validate_tool_registry(self._definitions)

    @property
    def definitions(self) -> dict[str, ToolDefinition]:
        return dict(self._definitions)

    def names(self) -> frozenset[str]:
        return frozenset(self._contributions)

    def get(self, name: str) -> ToolContribution:
        try:
            return self._contributions[name]
        except KeyError as error:
            raise KeyError(f"unknown tool contribution: {name}") from error

    def has(self, name: str) -> bool:
        return name in self._contributions

    def names_with_tag(self, tag: str) -> list[str]:
        return [
            name
            for name, item in self._contributions.items()
            if tag in item.tags
        ]

    def _ordered_tagged(self, tag: str, order: tuple[str, ...]) -> list[str]:
        tagged = {name for name, item in self._contributions.items() if tag in item.tags}
        ordered = [name for name in order if name in tagged]
        extras = [name for name in self.names_with_tag(tag) if name not in set(order)]
        return ordered + extras

    def chat_default_tools(self) -> list[str]:
        return self._ordered_tagged("chat_default", _CHAT_DEFAULT_ORDER)

    def chat_web_tools(self) -> list[str]:
        return self._ordered_tagged("chat_web", _CHAT_WEB_ORDER)

    def chat_skill_tools(self) -> list[str]:
        return self._ordered_tagged("chat_skill", _CHAT_SKILL_ORDER)

    def chat_approval_tools(self) -> list[str]:
        return self._ordered_tagged("chat_approval", _CHAT_APPROVAL_ORDER)

    def build_static_handlers(self, deps: ToolRuntimeDeps) -> dict[str, ToolHandler]:
        handlers: dict[str, ToolHandler] = {}
        for name, contribution in self._contributions.items():
            if contribution.router_owned:
                try:
                    handlers[name] = deps.router_handlers[name]
                except KeyError as error:
                    raise ValueError(
                        f"router-owned tool missing handler: {name}",
                    ) from error
                continue
            if contribution.bind_handler is None:
                raise ValueError(f"tool contribution missing binder: {name}")
            handlers[name] = contribution.bind_handler(deps)
        return handlers

    def summarize_args(self, tool_name: str, args: Mapping[str, Any]) -> dict[str, object]:
        contribution = self._contributions.get(tool_name)
        payload = dict(args)
        if contribution is not None:
            if contribution.interaction_args_summary is not None:
                return contribution.interaction_args_summary(payload)
            if contribution.summarize_args is not None:
                return contribution.summarize_args(payload)
        return {"args_keys": sorted(str(key) for key in payload)}

    def summarize_result(self, tool_name: str, result: Mapping[str, Any]) -> dict[str, object]:
        contribution = self._contributions.get(tool_name)
        payload = dict(result)
        if contribution is not None and contribution.summarize_result is not None:
            return contribution.summarize_result(payload)
        return {
            "status": str(payload.get("status", "unknown")),
            "result_keys": sorted(str(key) for key in payload),
        }

    def interaction_args_summary(
        self,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> dict[str, object]:
        contribution = self._contributions.get(tool_name)
        payload = dict(args)
        if contribution is not None and contribution.interaction_args_summary is not None:
            return contribution.interaction_args_summary(payload)
        return {"args_keys": sorted(str(key) for key in payload)}

    def project_observation(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, object] | None:
        contribution = self._contributions.get(tool_name)
        if contribution is None or contribution.project_observation is None:
            return None
        return contribution.project_observation(dict(args), dict(result))

    def check_guardrail(
        self,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> GuardrailResult | None:
        contribution = self._contributions.get(tool_name)
        if contribution is None or contribution.guardrail is None:
            return None
        return contribution.guardrail(dict(args))

    def display_name_zh(self, tool_name: str, default: str = "运行工具") -> str:
        contribution = self._contributions.get(tool_name)
        if contribution is None or not contribution.display_name_zh:
            return default
        return contribution.display_name_zh

    def observation_long_text_keys(self, tool_name: str) -> tuple[str, ...]:
        contribution = self._contributions.get(tool_name)
        if contribution is None:
            return ()
        return contribution.observation_long_text_keys


_DEFAULT_CATALOG: ToolCatalog | None = None


def default_tool_catalog() -> ToolCatalog:
    """进程级默认静态 catalog；首次访问时加载全部 contribution。"""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        from haagent.tools.contributions import all_static_contributions

        _DEFAULT_CATALOG = ToolCatalog(all_static_contributions())
    return _DEFAULT_CATALOG


def reset_default_tool_catalog_for_tests() -> None:
    """测试专用：清空默认 catalog 缓存。"""
    global _DEFAULT_CATALOG
    _DEFAULT_CATALOG = None


def _validate_contribution(contribution: ToolContribution) -> None:
    if not contribution.name:
        raise ValueError("tool contribution name is required")
    if contribution.risk_level not in ALLOWED_RISK_LEVELS:
        raise ValueError(
            f"{contribution.name} risk_level is invalid: {contribution.risk_level}",
        )
    if contribution.execution_effect not in ALLOWED_EXECUTION_EFFECTS:
        raise ValueError(
            f"{contribution.name} execution_effect is invalid: "
            f"{contribution.execution_effect}",
        )
    if not isinstance(contribution.replay_safety, ReplaySafety):
        raise ValueError(
            f"{contribution.name} replay_safety is required and must be ReplaySafety",
        )
    if contribution.router_owned and contribution.bind_handler is not None:
        raise ValueError(
            f"{contribution.name} cannot be both router_owned and bind_handler",
        )
    if not contribution.router_owned and contribution.bind_handler is None:
        raise ValueError(
            f"{contribution.name} must declare bind_handler or router_owned=True",
        )
