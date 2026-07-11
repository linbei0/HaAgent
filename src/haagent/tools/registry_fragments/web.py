"""
haagent/tools/registry_fragments/web.py - 联网工具注册表

定义公开网页搜索和抓取工具。
"""

from haagent.tools.registry import ToolDefinition


WEB_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "web_search": ToolDefinition(
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
    ),
    "web_fetch": ToolDefinition(
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
    ),
}
