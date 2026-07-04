"""
haagent/tui/keys.py - TUI 键位与帮助单一来源

集中维护实际绑定、footer 文案和帮助 modal 内容，降低上下文键位漂移风险。
"""

from __future__ import annotations

from typing import Literal

from textual.binding import Binding

KeyContext = Literal["chat", "memory_list", "memory_detail", "pending_input", "approval", "edit_diff", "too_small"]

APP_BINDINGS = [
    ("ctrl+q", "quit", "退出"),
    ("ctrl+f", "open_search", "搜索"),
    Binding("ctrl+t", "toggle_theme", "主题", priority=True),
    Binding("ctrl+p", "open_permissions", "权限", priority=True),
    ("?", "help", "帮助"),
    ("escape", "cancel_interaction", "取消"),
    ("ctrl+x", "cancel_current_task", "取消任务"),
    ("pageup", "conversation_page_up", "上翻"),
    ("pagedown", "conversation_page_down", "下翻"),
    ("end", "conversation_end", "回到底部"),
]

APPROVAL_BINDINGS = [
    ("y", "allow", "允许"),
    ("n", "deny", "拒绝"),
    ("escape", "deny", "拒绝"),
    ("?", "help", "帮助"),
]

EDIT_DIFF_BINDINGS = [
    ("y", "allow_once", "允许本次"),
    ("a", "allow_always", "始终允许"),
    ("n", "deny", "拒绝"),
    ("escape", "deny", "拒绝"),
    ("?", "help", "帮助"),
]

HELP_DISMISS_BINDINGS = [("escape", "dismiss_help", "关闭")]

_HELP_LINES: dict[KeyContext, list[tuple[str, str]]] = {
    "chat": [
        ("Enter", "发送当前输入"),
        ("Shift+Enter", "插入换行"),
        ("PgUp/PgDn", "滚动对话"),
        ("End", "回到底部"),
        ("/", "打开快捷命令"),
        ("/cancel", "取消当前任务"),
        ("Ctrl+F", "搜索当前对话"),
        ("Ctrl+T", "切换主题"),
        ("Ctrl+P", "打开权限设置"),
        ("/sessions", "打开 session 列表"),
        ("/memory", "打开记忆候选审查"),
        ("?", "打开此帮助"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "memory_list": [
        ("↑/↓", "移动选中项"),
        ("g/G", "跳到首项/末项"),
        ("Enter", "查看当前候选详情"),
        ("a 或 y", "确认当前候选"),
        ("r", "拒绝当前候选"),
        ("Esc", "返回聊天模式"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "memory_detail": [
        ("Esc", "返回列表，并保留当前选中项"),
        ("a 或 y", "确认当前候选"),
        ("r", "拒绝当前候选"),
        ("?", "打开此帮助"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "pending_input": [
        ("Enter", "提交回答并继续同一轮任务"),
        ("Shift+Enter", "插入换行"),
        ("Esc", "取消回答"),
        ("?", "打开此帮助"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "approval": [
        ("y", "允许当前工具调用"),
        ("n", "拒绝当前工具调用"),
        ("Esc", "拒绝并关闭审批"),
        ("?", "打开此帮助"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "edit_diff": [
        ("y", "允许本次文件改动"),
        ("a", "始终允许当前会话内同类文件改动"),
        ("n", "拒绝文件改动"),
        ("Esc", "拒绝并关闭审批"),
        ("?", "打开此帮助"),
        ("Ctrl+Q", "退出 TUI"),
    ],
    "too_small": [
        ("Ctrl+Q", "退出 TUI"),
    ],
}

_HELP_TITLES: dict[KeyContext, str] = {
    "chat": "聊天模式",
    "memory_list": "记忆候选列表",
    "memory_detail": "记忆候选详情",
    "pending_input": "等待补充输入",
    "approval": "审批确认",
    "edit_diff": "文件改动审批",
    "too_small": "终端尺寸过小",
}

_FOOTER_KEYS: dict[KeyContext, list[str]] = {
    "chat": ["Enter", "Shift+Enter", "/", "Ctrl+F", "Ctrl+P", "Ctrl+T", "/memory", "?", "Ctrl+Q"],
    "memory_list": ["↑/↓", "g/G", "Enter", "a/y", "r", "Esc", "?", "Ctrl+Q"],
    "memory_detail": ["Esc", "a/y", "r", "?", "Ctrl+Q"],
    "pending_input": ["Enter", "Shift+Enter", "Esc", "?", "Ctrl+Q"],
    "approval": ["y", "n", "Esc", "?", "Ctrl+Q"],
    "edit_diff": ["y", "a", "n", "Esc", "?", "Ctrl+Q"],
    "too_small": ["Ctrl+Q"],
}


def normalize_context(context: str) -> KeyContext:
    if context in _HELP_LINES:
        return context  # type: ignore[return-value]
    return "chat"


def key_help_lines(context: str, *, footer_only: bool = False, include_footer_only: bool = True) -> list[tuple[str, str]]:
    normalized = normalize_context(context)
    lines = _HELP_LINES[normalized]
    if not footer_only:
        return lines
    footer_keys = set(_FOOTER_KEYS[normalized])
    result: list[tuple[str, str]] = []
    matched_footer_keys: set[str] = set()
    for key, description in lines:
        compact_key = key.replace(" 或 ", " ")
        if key in footer_keys or compact_key in footer_keys:
            result.append((compact_key, description))
            matched_footer_keys.add(key)
            matched_footer_keys.add(compact_key)
            continue
        for footer_key in footer_keys:
            if _same_key_group(key, footer_key):
                result.append((footer_key, description))
                matched_footer_keys.add(footer_key)
                break
    if include_footer_only:
        result.extend((key, "") for key in _FOOTER_KEYS[normalized] if key not in matched_footer_keys)
    return result


def footer_text(context: str) -> str:
    normalized = normalize_context(context)
    pieces = []
    for key, description in key_help_lines(normalized, footer_only=True):
        if description:
            pieces.append(f"[{key}]{_footer_label(description)}")
        else:
            pieces.append(f"[{key}]")
    return " ".join(pieces)


def help_body(context: str) -> str:
    normalized = normalize_context(context)
    lines = [_HELP_TITLES[normalized], ""]
    for key, description in key_help_lines(normalized, include_footer_only=False):
        lines.append(f"{key:<12} {description}")
    return "\n".join(lines)


def _footer_label(description: str) -> str:
    if description.startswith("打开此帮助"):
        return "帮助"
    if description.startswith("退出"):
        return "退出"
    if description.startswith("滚动"):
        return "滚动"
    if description.startswith("打开记忆"):
        return "记忆"
    if description.startswith("打开快捷"):
        return "命令"
    if description.startswith("搜索"):
        return "搜索"
    if description.startswith("切换主题"):
        return "主题"
    if description.startswith("打开权限"):
        return "权限"
    if description.startswith("打开 session"):
        return "会话"
    if description.startswith("切换"):
        return "焦点"
    if description.startswith("发送"):
        return "发送"
    if description.startswith("插入换行"):
        return "换行"
    if description.startswith("提交"):
        return "提交回答"
    if description.startswith("取消"):
        return "取消"
    if description.startswith("返回聊天"):
        return "返回聊天"
    if description.startswith("返回列表"):
        return "返回列表"
    if description.startswith("查看"):
        return "详情"
    if description.startswith("确认"):
        return "确认"
    if description.startswith("拒绝"):
        return "拒绝"
    if description.startswith("允许"):
        return "允许"
    if description.startswith("始终"):
        return "始终"
    if description.startswith("跳到"):
        return "首尾"
    if description.startswith("移动"):
        return "移动"
    return description


def _same_key_group(help_key: str, footer_key: str) -> bool:
    if help_key.startswith("/") or footer_key.startswith("/"):
        return help_key == footer_key
    help_parts = {part.strip() for part in help_key.replace(" 或 ", "/").split("/")}
    footer_parts = {part.strip() for part in footer_key.replace(" 或 ", "/").split("/")}
    return bool(help_parts & footer_parts)
