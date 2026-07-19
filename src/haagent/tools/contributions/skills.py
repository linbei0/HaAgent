"""
haagent/tools/contributions/skills.py - 技能静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.skill_market import skill_market_search
from haagent.tools.skills import skill_list, skill_read


def _bind_skill_list(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return skill_list(
            args,
            deps.workspace_root,
            deps.skill_settings,
            skill_catalog=deps.skill_catalog,
        )

    return handler


def _bind_skill_read(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return skill_read(
            args,
            deps.workspace_root,
            deps.skill_settings,
            skill_catalog=deps.skill_catalog,
        )

    return handler


def _bind_skill_market_search(_deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return skill_market_search(args)

    return handler


SKILL_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="skill_list",
        description=(
            "List compact metadata for built-in, user, and project skills without loading their bodies. Use when "
            "the current prompt does not already identify a matching skill or when the available list must be "
            "filtered. Do not use it as a substitute for skill_read after an exact matching skill is known."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "optional text filter matched against skill name and description",
                },
                "source": {
                    "type": "string",
                    "description": "optional source filter such as builtin, user, or project",
                },
                "max_results": {
                    "type": "integer",
                    "description": "optional maximum number of skills to return; defaults to 20",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_skill"}),
        bind_handler=_bind_skill_list,
    ),
    ToolContribution(
        name="skill_read",
        description=(
            "Load one skill's full instructions only when its name and description clearly match the task. Pass "
            "the exact skill name returned by the available-skills context or skill_list. Follow referenced paths "
            "relative to that skill's directory; if no skill matches, continue without inventing a name."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "exact skill name, command name, or alias from the available skill metadata",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_skill"}),
        bind_handler=_bind_skill_read,
    ),
    ToolContribution(
        name="skill_market_search",
        description=(
            "Search remote skill marketplaces only when the user asks to find, compare, or install a new skill, "
            "or when local skills are explicitly insufficient and marketplace discovery is part of the request. "
            "Do not search the marketplace for ordinary work that existing tools or local skills can perform."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "marketplace search query; English keywords usually work best",
                },
                "providers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["skills_sh", "skillsmp"]},
                    "description": "optional provider filter; defaults to both skills_sh and skillsmp",
                },
                "limit": {
                    "type": "integer",
                    "description": "maximum results to return; defaults to 10 and must be between 1 and 10",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"skill_market"}),
        bind_handler=_bind_skill_market_search,
    ),
]
