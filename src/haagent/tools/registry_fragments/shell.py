"""
haagent/tools/registry_fragments/shell.py - 命令执行工具注册表

定义 shell 与临时 Python 脚本执行工具。
"""

from haagent.tools.registry import ToolDefinition


SHELL_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "code_run": ToolDefinition(
        name="code_run",
        description="run a multiline Python script from a temporary workspace file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to write to a temporary script and execute",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds; defaults to 60 and must be <= 120",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'working directory relative to workspace_root; use "." or omit '
                        "for workspace root"
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
    ),
    "shell": ToolDefinition(
        name="shell",
        description="run a shell command with timeout and captured output",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'working directory relative to workspace_root; use "." or omit '
                        "for workspace root"
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds; defaults to 60 and must be <= 120",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
    ),
}
