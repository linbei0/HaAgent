"""
haagent/tools/contributions/web.py - 联网静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.web import web_fetch, web_search


def _bind_web_search(_deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return web_search(args)

    return handler


def _bind_web_fetch(_deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return web_fetch(args)

    return handler


WEB_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="web_search",
        description="search the public web using the configured search provider and return sourced compact results",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "maximum results to return; defaults to 5 and must be between 1 and 10",
                },
                "provider": {
                    "type": "string",
                    "enum": ["tavily", "brave"],
                    "description": "optional search provider; defaults to HAAGENT_WEB_SEARCH_PROVIDER or tavily",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "optional Tavily topic",
                },
                "freshness": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "optional recency filter",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_web"}),
        bind_handler=_bind_web_search,
        display_name_zh="联网搜索",
    ),
    ToolContribution(
        name="web_fetch",
        description="fetch one public HTTP(S) URL and return compact readable external content",
        risk_level="medium",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "public HTTP or HTTPS URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "maximum returned content characters; defaults to 12000 and must be between 500 and 50000",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_web"}),
        bind_handler=_bind_web_fetch,
        display_name_zh="读取网页",
    ),
]
