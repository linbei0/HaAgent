"""
haagent/tools/registry_fragments/skills.py - 技能工具注册表

定义本地技能读取与远程技能市场检索工具。
"""

from haagent.tools.registry import ToolDefinition


SKILL_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "skill_list": ToolDefinition(
        name="skill_list",
        description="list available local skills as compact metadata without loading skill bodies",
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
                    "description": "optional source filter: user or project",
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
    ),
    "skill_read": ToolDefinition(
        name="skill_read",
        description="read one local skill body by name after choosing it from skill_list or available skills",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "skill name, command name, or alias",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
    ),
    "skill_market_search": ToolDefinition(
        name="skill_market_search",
        description="search the remote skill marketplace providers skills_sh and skillsmp as compact external metadata",
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
    ),
}
