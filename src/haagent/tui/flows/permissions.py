"""
haagent/tui/flows/permissions.py - TUI 权限交互流程

封装权限设置、外部目录授权和完全信任确认等交互流程，减少主应用类的分支职责。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from haagent.tui.overlays.modals import ConfirmModal, PermissionsModal

if TYPE_CHECKING:
    from haagent.tui.application.app import HaAgentTuiApp


def is_wide_external_root(path: Path) -> bool:
    """识别需二次确认的高风险外部目录。"""
    resolved = path.resolve()
    home = Path.home().resolve()
    risky_roots = {home, home / "Desktop", home / "Downloads", home / "Documents"}
    if resolved.parent == resolved:
        return True
    if resolved.anchor and str(resolved) == resolved.anchor:
        return True
    return resolved in {root.resolve() for root in risky_roots}


def show_permissions(app: "HaAgentTuiApp") -> None:
    status = app.service.workspace.status()
    app.push_screen(
        PermissionsModal(
            status.workspace_root,
            status.external_roots or [],
            status.permission_mode,
        ),
        app._handle_permissions_result,
    )


def handle_permissions_result(app: "HaAgentTuiApp", result: dict[str, object] | None) -> None:
    if result is None:
        app._restore_prompt_focus()
        return
    action = result.get("action")
    try:
        if action == "clear":
            app.push_screen(
                ConfirmModal("清空外部目录授权", "清空后本会话将只保留当前项目根权限。确认？"),
                app._handle_clear_external_roots_confirmed,
            )
            return
        if action == "set_mode":
            mode = str(result.get("mode", "request_approval"))
            if mode == "full_access":
                app.push_screen(
                    ConfirmModal(
                        "启用完全访问权限",
                        "启用后，本会话不再限制工作区外文件读写和执行目录。\n你将承担该模式下的本地文件和命令风险。确认？",
                    ),
                    lambda confirmed, mode=mode: app._handle_permission_mode_confirmed(mode, confirmed),
                )
                return
            set_permission_mode(app, mode)
            return
        path = Path(str(result.get("path", "")))
        if action == "remove":
            app.service.sessions.permissions.remove_external_root(path)
            app._conversation.append_block("Permissions", f"已移除外部目录：{path}")
        elif action == "set_access":
            access = str(result.get("access"))
            if access == "full" and is_wide_external_root(path):
                app.push_screen(
                    ConfirmModal("完全信任高风险目录", f"将允许读取、修改并在该目录执行命令：\n{path}\n确认？"),
                    lambda confirmed, path=path: app._handle_set_full_access_confirmed(path, confirmed),
                )
                return
            app.service.sessions.permissions.set_external_root_access(path, access)
            label = "只读参考" if access == "read" else "完全信任"
            app._conversation.append_block("Permissions", f"已设为{label}：{path}")
    except Exception as error:
        app._conversation.append_block("Permissions warning", f"权限操作失败：{error}")
    app._refresh()
    app._restore_prompt_focus()


def set_permission_mode(app: "HaAgentTuiApp", mode: str) -> None:
    try:
        app.service.sessions.permissions.set_mode(mode)
    except Exception as error:
        app._conversation.append_block("Permissions warning", f"权限模式切换失败：{error}")
    else:
        label = {
            "auto_approve": "自动批准",
            "full_access": "完全访问权限",
        }.get(mode, "请求批准")
        app._conversation.append_block("Permissions", f"权限模式已切换为：{label}")
    app._refresh()
    app._restore_prompt_focus()


def handle_permission_mode_confirmed(app: "HaAgentTuiApp", mode: str, confirmed: bool) -> None:
    if confirmed:
        set_permission_mode(app, mode)
        return
    app._conversation.append_block("Permissions", "已取消启用完全访问权限。")
    app._refresh()
    app._restore_prompt_focus()


def handle_clear_external_roots_confirmed(app: "HaAgentTuiApp", confirmed: bool) -> None:
    if confirmed:
        try:
            app.service.sessions.permissions.clear_external_roots()
        except Exception as error:
            app._conversation.append_block("Permissions warning", f"清空外部目录授权失败：{error}")
        else:
            app._conversation.append_block("Permissions", "已清空本会话外部目录授权。")
    app._refresh()
    app._restore_prompt_focus()


def handle_set_full_access_confirmed(app: "HaAgentTuiApp", path: Path, confirmed: bool) -> None:
    if confirmed:
        try:
            app.service.sessions.permissions.set_external_root_access(path, "full")
        except Exception as error:
            app._conversation.append_block("Permissions warning", f"权限操作失败：{error}")
        else:
            app._conversation.append_block("Permissions", f"已完全信任：{path}")
    app._refresh()
    app._restore_prompt_focus()

