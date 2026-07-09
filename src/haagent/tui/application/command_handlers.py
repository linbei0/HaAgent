"""
src/haagent/tui/application/command_handlers.py - TUI slash 命令处理体

集中承载 /turns、/sandbox、/web、/mcp、/agents、/compact 等命令的具体逻辑，
让主 App 只保留薄入口。全部操作经 AssistantService，不绕过 runtime。
"""

from __future__ import annotations

from typing import Any

from haagent.app.assistant_service import AssistantServiceError


class ChatCommandHandlers:
    """承载与会话运行时相关的 slash 命令处理逻辑。"""

    def __init__(self, app: Any) -> None:
        self._app = app

    # ── /turns ───────────────────────────────────────────────────────────
    def turns(self, argument: str) -> None:
        usage = "用法：/turns [show|unlimited|COUNT]"
        parts = argument.strip().split()
        if not parts or parts == ["show"]:
            status = self._app.service.get_turn_limit_status()
            current = "unlimited" if status.current_max_turns is None else str(status.current_max_turns)
            self._block(
                f"当前 session turn 限制：{current}\n"
                f"已保存交互默认值：{status.configured_interactive_max_turns}\n"
                f"{usage}"
            )
            return
        if parts == ["unlimited"]:
            try:
                self._app.service.set_current_turns_unlimited()
            except AssistantServiceError as error:
                self._block(str(error))
            else:
                self._block("当前 session turn 限制已设为 unlimited；不会写入全局配置。")
            return
        count_text = parts[0] if len(parts) == 1 else parts[1] if parts[0] == "set" and len(parts) == 2 else ""
        if not count_text.isdigit() or int(count_text) <= 0:
            self._block(usage)
            return
        self._app.service.set_interactive_max_turns(int(count_text))
        self._block(f"已保存交互默认 turn 限制：{int(count_text)}；当前 session 已同步。")

    # ── /sandbox ─────────────────────────────────────────────────────────
    def sandbox(self, argument: str) -> None:
        parts = argument.strip().split()
        usage = "用法：/sandbox [status|doctor|enable docker [--allow-fallback]|disable]"
        try:
            if not parts or parts == ["status"]:
                self._block(sandbox_status_text(self._app.service.get_sandbox_status()))
            elif parts == ["doctor"]:
                self._block(sandbox_doctor_text(self._app.service.get_sandbox_doctor_report()))
            elif parts[:2] == ["enable", "docker"]:
                self._enable_docker(parts[2:], usage)
            elif parts == ["disable"]:
                self._disable_sandbox()
            else:
                self._block(usage)
        except Exception as error:
            self._block(f"沙箱设置失败：{error}")

    def _enable_docker(self, extra: list[str], usage: str) -> None:
        # --allow-fallback 是显式选择的降级路径：用户主动接受在 Docker 不可用时
        # 回退到 local_subprocess，而不是静默降级。默认仍要求 Docker 可用。
        if any(item not in {"--allow-fallback", "--fail-if-unavailable"} for item in extra):
            self._block(usage)
            return
        allow_fallback = "--allow-fallback" in extra
        status = self._app.service.enable_docker_sandbox(fail_if_unavailable=not allow_fallback)
        self._app._sandbox_status = _sandbox_state(status)
        self._block(f"Docker 沙箱已启用；新 session 生效。\n{sandbox_status_text(status)}")

    def _disable_sandbox(self) -> None:
        status = self._app.service.disable_sandbox()
        self._app._sandbox_status = _sandbox_state(status)
        self._block(f"已恢复 local_subprocess；后续新 session 会使用本机执行。\n{sandbox_status_text(status)}")

    # ── /web ─────────────────────────────────────────────────────────────
    def web(self, argument: str) -> None:
        if argument.strip():
            self._block("用法：/web")
            return
        status = self._app.service.get_workspace_status()
        enabled = not status.web_enabled
        self._app.service.set_web_enabled(enabled)
        state = "开启" if enabled else "关闭"
        self._block(f"联网已{state}；后续任务可使用 web_search / web_fetch。")

    # ── /mcp ─────────────────────────────────────────────────────────────
    def mcp(self) -> None:
        status = self._app.service.get_mcp_status()
        servers = status.get("servers", [])
        if not isinstance(servers, list) or not servers:
            self._block("No MCP servers configured.")
            return
        lines = ["MCP servers:"]
        for item in servers:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "unknown"))
            state = str(item.get("state", "configured"))
            detail = str(item.get("detail", "")).strip()
            if state == "connected":
                lines.append(
                    f"- {name}: connected (tools: {int(item.get('tool_count', 0))}, "
                    f"resources: {int(item.get('resource_count', 0))})"
                )
            elif detail:
                lines.append(f"- {name}: {state} - {detail}")
            else:
                lines.append(f"- {name}: {state}")
        self._block("\n".join(lines))

    # ── /agents ──────────────────────────────────────────────────────────
    def agents(self) -> None:
        try:
            agents = self._app.service.list_agents()
        except Exception as error:
            self._app._append_block("Agents", f"读取 worker 状态失败：{error}")
            self._app._refresh()
            return
        if not agents:
            self._app._append_block("Agents", "当前 session 没有 worker。")
            self._app._refresh()
            return
        lines = ["Workers:"]
        for item in agents:
            agent_id = str(item.get("agent_id", "unknown"))
            status = str(item.get("status", "unknown"))
            subagent_type = str(item.get("subagent_type", "worker"))
            description = str(item.get("description", "")).strip()
            suffix = f" - {description}" if description else ""
            lines.append(f"- {agent_id} [{subagent_type}] {status}{suffix}")
        self._app._append_block("Agents", "\n".join(lines))
        self._app._refresh()

    # ── /compact ─────────────────────────────────────────────────────────
    def compact(self) -> None:
        try:
            result = self._app.service.compact_current_session()
        except Exception as error:
            self._block(f"压缩当前会话失败：{error}")
            return
        if result.applied:
            self._block(
                "已压缩当前会话："
                f"压缩 {result.compacted_turn_count} 轮，"
                f"保留最近 {result.preserved_recent_count} 轮，"
                f"节省约 {result.saved_chars} 字符。"
            )
        else:
            self._block(f"当前会话无需压缩：{result.reason}")

    def toggle_tool_details(self) -> None:
        enabled = not self._app._tool_details_enabled
        self._app._tool_details_enabled = enabled
        self._app._conversation.set_tool_details(enabled)
        state = "开启" if enabled else "关闭"
        self._block(f"工具详情已{state}")

    def _block(self, body: str) -> None:
        self._app._append_block("Command", body)
        self._app._refresh()


def _sandbox_state(status: object) -> dict[str, object]:
    return {
        "backend": getattr(status, "backend", "unknown"),
        "availability": {
            "degraded": getattr(status, "degraded", True),
            "reason": getattr(status, "reason", ""),
        },
    }


def sandbox_status_text(status: object) -> str:
    backend = getattr(status, "backend", "unknown")
    degraded = bool(getattr(status, "degraded", True))
    reason = str(getattr(status, "reason", "") or "")
    lines = [f"当前沙箱：{backend}", f"degraded={str(degraded).lower()}"]
    if reason:
        lines.append(f"reason={reason}")
    if backend != "docker":
        lines.append("开启 Docker 隔离：haagent sandbox enable docker")
    else:
        lines.append("检查 Docker 可用性：haagent sandbox doctor")
    return "\n".join(lines)


def sandbox_doctor_text(report: object) -> str:
    lines = [
        f"当前沙箱：{getattr(report, 'backend', 'unknown')}",
        f"ready={str(bool(getattr(report, 'ready', False))).lower()}",
        f"Docker CLI: {getattr(report, 'docker_cli', 'unknown')}",
        f"Docker daemon: {getattr(report, 'docker_daemon', 'unknown')}",
        f"image={getattr(report, 'image', 'unknown')}",
        f"auto_build_image={str(bool(getattr(report, 'auto_build_image', False))).lower()}",
    ]
    reason = str(getattr(report, "reason", "") or "")
    next_action = str(getattr(report, "next_action", "") or "")
    if reason:
        lines.append(f"reason={reason}")
    if next_action:
        lines.append(f"next_action={next_action}")
    return "\n".join(lines)
