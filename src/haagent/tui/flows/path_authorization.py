"""
haagent/tui/path_authorization_flow.py - TUI 路径授权判断与交互编排

封装外部绝对路径提及检测、工作区关系判断，以及自然语言 prompt 触发的外部目录授权流程，减少主应用类的分支职责。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from haagent.tui.overlays.modals import ConfirmModal, ExternalDirectoryDecisionModal

if TYPE_CHECKING:
    from haagent.tui.application.app import HaAgentTuiApp


WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r'(?:"([A-Za-z]:[\\/][^"\r\n]+)"|([A-Za-z]:[\\/][^\s"\']+))')
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(r'(?:"(/[^"\r\n]+)"|(?<!\S)(/[^ \t\r\n"\']+))')


def handle_prompt_path_authorization(app: "HaAgentTuiApp", prompt: str) -> bool:
    status = app.service.workspace.status()
    untrusted_paths = find_untrusted_absolute_paths(
        prompt,
        project_root=status.workspace_root,
        external_roots=status.external_roots or [],
    )
    if not untrusted_paths:
        return False
    target_path = untrusted_paths[0]
    if getattr(status, "permission_mode", "request_approval") == "full_access":
        app._set_next_turn_target_path(target_path)
        app._conversation.append_block("Permissions", "完全访问权限已启用；本轮不会限制工作区外路径。")
        app._start_prompt(prompt)
        return True
    app._pending_external_prompt = prompt
    app._pending_external_path = target_path
    app.push_screen(ExternalDirectoryDecisionModal(target_path), app._handle_external_directory_decision)
    return True


def handle_external_directory_decision(app: "HaAgentTuiApp", decision: str | None) -> None:
    prompt = app._pending_external_prompt
    path = app._pending_external_path
    app._pending_external_prompt = None
    app._pending_external_path = None
    if prompt is None or path is None:
        app._restore_prompt_focus()
        return
    try:
        if decision == "read":
            app.service.sessions.permissions.add_external_root(path, "read")
            app._set_next_turn_target_path(path)
            app._conversation.append_block("Permissions", f"已作为只读参考加入：{path}")
        elif decision == "full":
            if is_wide_external_root(path):
                app._pending_full_trust_prompt = prompt
                app._pending_full_trust_path = path
                app.push_screen(
                    ConfirmModal("完全信任高风险目录", f"将允许读取、修改并在该目录执行命令：\n{path}\n确认？"),
                    app._handle_external_full_trust_confirmed,
                )
                return
            app.service.sessions.permissions.add_external_root(path, "full")
            app._set_next_turn_target_path(path)
            app._conversation.append_block("Permissions", f"已完全信任：{path}")
        elif decision == "switch":
            app.service.sessions.permissions.switch_project_root(path)
            app._conversation.append_block("Permissions", f"已切换工作区：{path}")
        else:
            app._conversation.append_block("Permissions", f"已取消外部目录授权：{path}")
            app._refresh()
            app._restore_prompt_focus()
            return
    except Exception as error:
        app._conversation.append_block("Permissions warning", f"外部目录授权失败：{error}")
        app._refresh()
        app._restore_prompt_focus()
        return
    app._refresh()
    app._start_prompt(prompt)


def handle_external_full_trust_confirmed(app: "HaAgentTuiApp", confirmed: bool) -> None:
    prompt = app._pending_full_trust_prompt
    path = app._pending_full_trust_path
    app._pending_full_trust_prompt = None
    app._pending_full_trust_path = None
    if prompt is None or path is None:
        app._restore_prompt_focus()
        return
    if not confirmed:
        app._conversation.append_block("Permissions", f"已取消完全信任：{path}")
        app._refresh()
        app._restore_prompt_focus()
        return
    try:
        app.service.sessions.permissions.add_external_root(path, "full")
        app._set_next_turn_target_path(path)
    except Exception as error:
        app._conversation.append_block("Permissions warning", f"外部目录授权失败：{error}")
        app._refresh()
        app._restore_prompt_focus()
        return
    app._conversation.append_block("Permissions", f"已完全信任：{path}")
    app._refresh()
    app._start_prompt(prompt)


def find_untrusted_absolute_paths(
    text: str,
    *,
    project_root: Path,
    external_roots: list[dict[str, str]] | None = None,
) -> list[Path]:
    roots = [project_root.resolve()]
    for item in external_roots or []:
        raw_path = item.get("path")
        if raw_path:
            roots.append(Path(raw_path).resolve())
    matches: list[Path] = []
    for candidate in _absolute_path_candidates(text):
        resolved = candidate.resolve()
        if any(resolved == root or root in resolved.parents for root in roots):
            continue
        if resolved not in matches:
            matches.append(resolved)
    return matches


def is_wide_external_root(path: Path) -> bool:
    resolved = path.resolve()
    home = Path.home().resolve()
    risky_roots = {home, home / "Desktop", home / "Downloads", home / "Documents"}
    if resolved.parent == resolved:
        return True
    if resolved.anchor and str(resolved) == resolved.anchor:
        return True
    return resolved in {root.resolve() for root in risky_roots}


def _absolute_path_candidates(text: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in (WINDOWS_ABSOLUTE_PATH_PATTERN, POSIX_ABSOLUTE_PATH_PATTERN):
        for match in pattern.finditer(text):
            raw = next((group for group in match.groups() if group), "")
            if raw:
                paths.append(Path(raw.rstrip(".,;，。；")))
    return paths
