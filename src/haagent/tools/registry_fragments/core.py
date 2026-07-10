"""
haagent/tools/registry_fragments/core.py - 核心会话工具注册表

定义测试、附件、用户补充输入与记忆更新工具。
"""

from haagent.memory.prompts import START_MEMORY_UPDATE_TOOL_DESCRIPTION
from haagent.tools.registry import ToolDefinition


CORE_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "fake_tool": ToolDefinition(
        name="fake_tool",
        description="deterministic test tool",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        },
    ),
    "load_image_attachment": ToolDefinition(
        name="load_image_attachment",
        description=(
            "load a previously attached session image by image_id so the next model call "
            "can inspect it as visual input"
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "id from Image Attachment History, for example img-123abc",
                },
            },
            "required": ["image_id"],
            "additionalProperties": False,
        },
    ),
    "request_user_input": ToolDefinition(
        name="request_user_input",
        description="ask the user for missing information before continuing the task",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "question to ask the user",
                },
                "reason": {
                    "type": "string",
                    "description": "short reason why the information is needed",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
    "start_memory_update": ToolDefinition(
        name="start_memory_update",
        description=START_MEMORY_UPDATE_TOOL_DESCRIPTION,
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "short reason describing the durable information that may be worth settlement",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
}
