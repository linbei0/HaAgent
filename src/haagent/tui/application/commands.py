"""
src/haagent/tui/application/commands.py - TUI slash 命令分发器

把结构化命令 action 映射到 App/flow/handler 方法，避免主 App 堆叠长分支。
命令逻辑本体在 ChatCommandHandlers 和各 flow 中，这里只做路由。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from haagent.tui.commands import SlashCommandResult


CommandHandler = Callable[[SlashCommandResult], None]


class CommandDispatcher:
    """根据 slash command action 调用宿主 App 的命令处理方法。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        # 所有 flow/handler 访问都在调用时惰性求值，构造期不触碰它们，
        # 便于用最小 app 桩测试分发逻辑。
        self._handlers: dict[str, CommandHandler] = {
            "help": lambda result: app.action_help(),
            "sessions": lambda result: app.session_flow.open_sessions(),
            "compact_session": lambda result: app._command_handlers.compact(),
            "open_connections": lambda result: app.model_flow.open_connections(),
            "open_models": lambda result: app.model_flow.open_models(),
            "open_channels": lambda result: app.channel_flow.open_channels(),
            "mcp": lambda result: app._command_handlers.mcp(),
            "agents": lambda result: app._command_handlers.agents(),
            "memory": lambda result: self._open_memory_if_closed(),
            "toggle_details": lambda result: app._command_handlers.toggle_tool_details(),
            "skills": lambda result: app._handle_skills_command(result.argument),
            "skill": lambda result: app._handle_skill_command(result.argument),
            "sandbox": lambda result: app._command_handlers.sandbox(result.argument),
            "web": lambda result: app._command_handlers.web(result.argument),
            "turns": lambda result: app._command_handlers.turns(result.argument),
            "permissions": lambda result: app._show_permissions(),
            "cancel_task": lambda result: app.action_cancel_current_task(),
            "new_session": lambda result: app.session_flow.new_session(),
            "resume_latest": lambda result: app.session_flow.resume_latest(),
        }

    def dispatch(self, result: SlashCommandResult) -> bool:
        if result.error:
            self._app._conversation.append_block("Command", result.error)
            self._app._refresh()
            return True
        command = result.command
        if command is None:
            return False
        handler = self._handlers.get(command.action)
        if handler is None:
            return False
        handler(result)
        return True

    def _open_memory_if_closed(self) -> None:
        if not self._app.memory_flow.mode:
            self._app.memory_flow.toggle()
