"""
src/haagent/tui/design/copy.py - TUI 中文文案单一来源

集中维护面板、弹窗和常见状态文案，避免 TUI 各模块出现中英标题漂移。
"""

from __future__ import annotations

PANEL_TITLES = {
    "conversation": "对话",
    "sessions": "会话",
    "memory": "记忆候选",
    "search": "搜索",
    "commands": "快捷命令",
    "file_refs": "文件引用",
    "workspace": "工作区",
    "profile": "模型配置",
    "current_session": "当前会话",
}

MODAL_TITLES = {
    "help": "HaAgent 帮助",
    "approval": "工具审批",
    "edit_diff": "文件改动审批",
    "sessions": "会话",
    "search": "搜索",
    "commands": "快捷命令",
    "file_refs": "文件引用",
}

EMPTY_LABELS = {
    "none": "无",
    "missing": "缺失",
    "available": "可用",
    "no_pending_candidates": "暂无待确认候选",
    "no_matching_sessions": "无匹配会话",
    "no_matching_files": "无匹配文件",
    "no_matching_commands": "无匹配命令",
}

BLOCK_TITLES = {
    "You": "你",
    "Assistant": "HaAgent",
    "Config": "配置",
    "Command": "命令",
    "Failure": "失败",
    "Cancel": "取消",
    "Memory": "记忆",
    "Memory warning": "记忆警告",
    "Session warning": "会话警告",
    "Model warning": "模型提醒",
    "Answer required": "需要补充",
}

TOOL_DISPLAY_NAMES = {
    "file_read": "读取文件",
    "file_write": "写入文件",
    "apply_patch": "修改文件",
    "apply_patch_set": "修改文件",
    "shell": "运行命令",
    "code_run": "运行代码",
    "web_search": "联网搜索",
    "web_fetch": "读取网页",
}


def tool_display_name(tool_name: str) -> str:
    """普通界面使用中文动作名；原始标识只在展开详情中展示。"""

    return TOOL_DISPLAY_NAMES.get(tool_name, "运行工具")
