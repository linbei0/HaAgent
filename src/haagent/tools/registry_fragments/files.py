"""
haagent/tools/registry_fragments/files.py - 工作区文件工具注册表

定义文件浏览、检索、读取与写入补丁工具。
"""

from haagent.tools.registry import ToolDefinition


FILE_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "file_list": ToolDefinition(
        name="file_list",
        description="list a compact workspace file tree for project discovery",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": 'optional workspace-relative directory to list; defaults to "."',
                },
                "max_depth": {
                    "type": "integer",
                    "description": "optional maximum directory depth; defaults to 2",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "optional maximum entries to return; defaults to 100",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    "grep": ToolDefinition(
        name="grep",
        description="search file contents with a regular expression using ripgrep when available",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "regular expression to search for in workspace files",
                },
                "root": {
                    "type": "string",
                    "description": "optional workspace-relative directory or file to search",
                },
                "file_glob": {
                    "type": "string",
                    "description": "optional file glob for directory roots; defaults to **/*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "optional case sensitivity flag; defaults to true",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "optional total match limit; defaults to 200",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    ),
    "file_read": ToolDefinition(
        name="file_read",
        description="read a workspace text file with offset, limit, or keyword context",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "offset": {
                    "type": "integer",
                    "description": "optional zero-based line offset",
                },
                "limit": {
                    "type": "integer",
                    "description": "optional maximum number of lines",
                },
                "keyword": {
                    "type": "string",
                    "description": "optional keyword; read lines near the first match",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    "file_write": ToolDefinition(
        name="file_write",
        description="create, overwrite, or append a workspace text file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "content": {
                    "type": "string",
                    "description": "text content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "write mode: create, overwrite, or append",
                },
            },
            "required": ["path", "content", "mode"],
            "additionalProperties": False,
        },
    ),
    "apply_patch": ToolDefinition(
        name="apply_patch",
        description="replace unique text inside a workspace file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "old_text": {
                    "type": "string",
                    "description": "unique text to replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    ),
    "apply_patch_set": ToolDefinition(
        name="apply_patch_set",
        description=(
            "apply multiple unique text replacements atomically after reading current file context; "
            "no files are written if any replacement does not match exactly once. "
            "Prefer this over repeated apply_patch calls for related multi-file or multi-site edits"
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "replacements": {
                    "type": "array",
                    "description": (
                        "non-empty list of replacements; each item has workspace-relative path, "
                        "old_text, and new_text"
                    ),
                },
            },
            "required": ["replacements"],
            "additionalProperties": False,
        },
    ),
}
