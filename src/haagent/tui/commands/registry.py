"""
haagent/tui/commands.py - TUI 结构化命令注册表

定义 slash command 的稳定命令边界，供输入区和未来命令面板复用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    action: str

    @property
    def token(self) -> str:
        return f"/{self.name}"


@dataclass(frozen=True)
class SlashCommandResult:
    command: SlashCommand | None
    argument: str = ""
    error: str | None = None


class CommandRegistry:
    def __init__(self, commands: list[SlashCommand]) -> None:
        self._commands = {command.name: command for command in commands}

    def commands(self) -> list[SlashCommand]:
        return list(self._commands.values())

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def require(self, name: str) -> SlashCommand:
        command = self.get(name)
        if command is None:
            raise KeyError(name)
        return command


def command_registry() -> CommandRegistry:
    return CommandRegistry(
        [
            SlashCommand("help", "打开上下文帮助", "help"),
            SlashCommand("sessions", "打开会话列表", "sessions"),
            SlashCommand("model", "打开模型中心", "open_models"),
            SlashCommand("mcp", "查看 MCP 连接状态", "mcp"),
            SlashCommand("memory", "打开记忆候选审查", "memory"),
            SlashCommand("skills", "查看、信任、搜索和安装 skills", "skills"),
            SlashCommand("skill", "显式使用一个 skill", "skill"),
            SlashCommand("web", "切换联网工具", "web"),
            SlashCommand("permissions", "管理外部目录权限", "permissions"),
            SlashCommand("cancel", "取消当前任务", "cancel_task"),
            SlashCommand("new", "新建 session", "new_session"),
            SlashCommand("resume", "继续最新 session", "resume_latest"),
            SlashCommand("details", "显示或隐藏工具详情", "toggle_details"),
            SlashCommand("compact", "智能压缩当前会话", "compact_session"),
        ],
    )


def parse_slash_command(text: str, registry: CommandRegistry) -> SlashCommandResult | None:
    if not text.startswith("/"):
        return None
    command_text, _, argument = text[1:].partition(" ")
    if not command_text:
        return SlashCommandResult(command=None, error="请输入命令名")
    if command_text == "models":
        command_text = "model"
    command = registry.get(command_text)
    if command is None:
        return SlashCommandResult(command=None, error=f"未知命令：/{command_text}")
    return SlashCommandResult(command=command, argument=argument.strip())
