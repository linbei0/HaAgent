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
    "Answer required": "需要补充",
}
