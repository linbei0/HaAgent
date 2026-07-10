"""
haagent/tools/registry_fragments/agent.py - 后台智能体工具注册表

定义后台任务创建、查询、消息与停止工具。
"""

from haagent.tools.registry import ToolDefinition


AGENT_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "agent": ToolDefinition(
        name="agent",
        description="spawn a background worker agent for delegated research, implementation, or verification",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
                "subagent_type": {
                    "type": "string",
                    "enum": ["explorer", "worker", "verification"],
                },
                "team": {"type": "string"},
                "model_profile": {"type": "string"},
                "profile": {
                    "type": "string",
                    "description": "agent profile name; defaults to subagent_type when omitted",
                },
            },
            "required": ["description", "prompt", "subagent_type"],
            "additionalProperties": False,
        },
    ),
    "send_message": ToolDefinition(
        name="send_message",
        description="send a follow-up message to an existing worker agent",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["to", "message"],
            "additionalProperties": False,
        },
    ),
    "task_stop": ToolDefinition(
        name="task_stop",
        description="request a running worker task to stop",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "force": {"type": "boolean"},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    ),
    "task_get": ToolDefinition(
        name="task_get",
        description="get status and metadata for one background worker task",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    ),
    "task_list": ToolDefinition(
        name="task_list",
        description="list background worker tasks for the current session",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["queued", "running", "idle", "completed", "failed", "stopped"],
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    "task_output": ToolDefinition(
        name="task_output",
        description="read bounded output from a background worker task episode",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "description": "maximum output characters to return; capped at 50000",
                },
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    ),
}
