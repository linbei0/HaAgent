"""
haagent/tui/application/channel_flow.py - TUI 渠道配置流程

打开 /channels overlay，处理新增微信 QR 登录、重登、启停、删除与连接测试。
QR 与网络操作在 worker 线程执行，不阻塞 UI。
"""

from __future__ import annotations

import asyncio
from typing import Any

from haagent.tui.overlays.channels import ChannelsOverlay, ChannelsOverlayResult
from haagent.tui.overlays.modals import ConfirmModal


def _render_qr_ascii(url: str) -> str | None:
    """可选依赖 qrcode：失败时返回 None，调用方回退 URL 文本。"""
    try:
        import qrcode  # type: ignore[import-untyped]
    except Exception:
        return None
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        lines: list[str] = []
        for row in matrix:
            lines.append("".join("██" if cell else "  " for cell in row))
        return "\n".join(lines)
    except Exception:
        return None


class ChannelFlow:
    """封装渠道配置交互；所有写操作经 AssistantService.channels。"""

    def __init__(self, app: Any) -> None:
        self._app = app

    def open_channels(self) -> None:
        if self._app._prompt_has_pending_text():
            return
        try:
            instances = self._app.service.channels.list_instances()
        except Exception as error:
            self._app._conversation.append_block("Channels", f"读取渠道配置失败：{error}")
            self._app._refresh()
            return
        self._app.push_screen(ChannelsOverlay(instances), self.handle_channels_result)

    def handle_channels_result(self, result: ChannelsOverlayResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        try:
            if result.action == "add_weixin":
                self._app._run_channel_weixin_login(None)
                return
            if result.action == "relogin" and result.instance_id:
                self._app._run_channel_weixin_login(result.instance_id)
                return
            if result.action == "enable" and result.instance_id:
                self._app.service.channels.set_enabled(result.instance_id, True)
                self._app._conversation.append_line(f"渠道已启用：{result.instance_id}")
            elif result.action == "disable" and result.instance_id:
                self._app.service.channels.set_enabled(result.instance_id, False)
                self._app._conversation.append_line(f"渠道已停用：{result.instance_id}")
            elif result.action == "test" and result.instance_id:
                self._app._run_channel_connection_test(result.instance_id)
                return
            elif result.action == "pair" and result.instance_id:
                self._issue_pairing_code(result.instance_id)
                return
            elif result.action == "workspace" and result.instance_id:
                # 打开路径选择器；若 result 已带路径则直接应用（测试/快捷路径）。
                if result.workspace_root:
                    self._apply_workspace(result.instance_id, result.workspace_root)
                else:
                    self._open_workspace_picker(result.instance_id)
                return
            elif result.action == "delete" and result.instance_id:
                self._confirm_delete(result.instance_id)
                return
        except Exception as error:
            self._app._conversation.append_block("Channels", f"渠道操作失败：{error}")
        self._app._refresh()
        self.open_channels()

    def _issue_pairing_code(self, instance_id: str) -> None:
        try:
            code = self._app.service.channels.issue_pairing_code(instance_id)
        except Exception as error:
            self._app._conversation.append_block("Channels", f"配对码签发失败：{error}")
            self._app._refresh()
            self.open_channels()
            return
        self._app._conversation.append_block(
            "Channels",
            f"配对码（10 分钟内有效，只显示一次）：{code}\n"
            f"请在微信向机器人发送：/pair {code}",
        )
        self._app._refresh()
        self.open_channels()

    def _open_workspace_picker(self, instance_id: str) -> None:
        from pathlib import Path

        from haagent.tui.overlays.workspace_picker import WorkspacePickerOverlay

        # 默认从该实例当前 workspace 或 TUI 当前 workspace 开始浏览。
        start = self._app.service.workspace.status().workspace_root
        try:
            for item in self._app.service.channels.list_instances():
                if str(getattr(item, "id", "")) == instance_id:
                    root = getattr(item, "workspace_root", None)
                    if root:
                        start = Path(root)
                    break
        except Exception:
            pass
        self._app.push_screen(
            WorkspacePickerOverlay(Path(start)),
            lambda path, iid=instance_id: self._handle_workspace_picked(iid, path),
        )

    def _handle_workspace_picked(self, instance_id: str, path: str | None) -> None:
        if not path:
            self.open_channels()
            return
        self._apply_workspace(instance_id, path)

    def _apply_workspace(self, instance_id: str, workspace_root: str | Path) -> None:
        from pathlib import Path

        try:
            updated = self._app.service.channels.set_workspace_root(
                instance_id, Path(workspace_root)
            )
        except Exception as error:
            self._app._conversation.append_block("Channels", f"更新 workspace 失败：{error}")
            self._app._refresh()
            self.open_channels()
            return
        self._app._conversation.append_block(
            "Channels",
            f"渠道 {instance_id} workspace 已设为：{updated.workspace_root}",
        )
        self._app._refresh()
        self.open_channels()

    def _confirm_delete(self, instance_id: str) -> None:
        self._app.push_screen(
            ConfirmModal(
                f"删除渠道：{instance_id}",
                "将删除本地配置、keyring 凭据与渠道动态状态，不删除 HaAgent session。确认？",
            ),
            lambda confirmed, iid=instance_id: self.handle_delete_result(iid, confirmed),
        )

    def handle_delete_result(self, instance_id: str, confirmed: bool | None) -> None:
        if not confirmed:
            self.open_channels()
            return
        try:
            self._app.service.channels.delete_instance(instance_id)
            self._app._conversation.append_line(f"渠道已删除：{instance_id}")
        except Exception as error:
            self._app._conversation.append_block("Channels", f"删除失败：{error}")
        self._app._refresh()
        self.open_channels()

    def run_weixin_login(self, instance_id: str | None) -> None:
        # 整个 start+poll 必须在同一次 asyncio.run 内完成。
        # 跨 run 复用 httpx.AsyncClient 会触发 Event loop is closed。
        target_id = instance_id or "weixin-default"
        workspace_root = self._app.service.workspace.status().workspace_root
        try:
            outcome = asyncio.run(self._weixin_login_async(target_id, workspace_root))
        except Exception as error:
            self._app.call_from_thread(self._login_failed, str(error))
            return
        kind = outcome.get("kind")
        if kind == "success":
            self._app.call_from_thread(
                self._login_succeeded,
                outcome["instance_id"],
                outcome.get("pairing_code"),
            )
            return
        self._app.call_from_thread(self._login_failed, outcome.get("message") or "登录失败")

    async def _weixin_login_async(self, target_id: str, workspace_root: Any) -> dict[str, Any]:
        channels = self._app.service.channels
        start = await channels.start_weixin_qr_login(
            workspace_root=workspace_root,
            instance_id=target_id,
        )
        # 先展示二维码；轮询仍在本协程/同一 loop 内继续。
        self._app.call_from_thread(
            self._show_qr_started,
            start.instance_id,
            start.qrcode_id,
            start.qrcode_url,
        )
        # 测试可注入较短间隔；生产默认 1.5s。
        interval = float(getattr(self, "_poll_interval_seconds", 1.5))
        for _ in range(120):
            poll = await channels.poll_weixin_qr_login(
                instance_id=start.instance_id,
                qrcode_id=start.qrcode_id,
            )
            if poll.status == "confirmed":
                return {
                    "kind": "success",
                    "instance_id": poll.instance_id,
                    "pairing_code": getattr(poll, "pairing_code", None),
                }
            if poll.status in {"expired", "failed"}:
                return {
                    "kind": "failed",
                    "message": poll.message or f"登录状态：{poll.status}",
                }
            # 同 loop 内等待，避免拆成多次 asyncio.run。
            await asyncio.sleep(interval)
        # 超时：显式取消，关闭 HTTP client，避免泄漏。
        try:
            await channels.cancel_weixin_qr_login(start.instance_id)
        except Exception:
            pass
        return {"kind": "failed", "message": "登录超时"}

    def run_connection_test(self, instance_id: str) -> None:
        try:
            result = asyncio.run(self._app.service.channels.test_connection(instance_id))
        except Exception as error:
            self._app.call_from_thread(self._test_failed, instance_id, str(error))
            return
        self._app.call_from_thread(self._test_done, result.ok, result.instance_id, result.message)

    def _show_qr_started(self, instance_id: str, qrcode_id: str, qrcode_url: str) -> None:
        # 不把 token 写入对话；优先 ASCII 二维码，否则展示 URL。
        body_lines = [
            f"微信登录 {instance_id}",
            f"二维码 ID：{qrcode_id}",
            "请在手机微信扫码：",
        ]
        ascii_qr = _render_qr_ascii(qrcode_url)
        if ascii_qr:
            body_lines.append(ascii_qr)
        body_lines.append(qrcode_url)
        self._app._conversation.append_block("Channels", "\n".join(body_lines))
        self._app._refresh()

    def _login_succeeded(self, instance_id: str, pairing_code: str | None = None) -> None:
        lines = [f"微信渠道登录成功：{instance_id}"]
        if pairing_code:
            # 配对码只展示一次；用户需在微信发送 /pair <码> 绑定 owner。
            lines.append(f"配对码（10 分钟内有效，只显示一次）：{pairing_code}")
            lines.append(f"请在微信向机器人发送：/pair {pairing_code}")
        self._app._conversation.append_block("Channels", "\n".join(lines))
        self._app._refresh()
        # 成功后回到列表，便于启用/测试；配对码已在对话区展示。
        self.open_channels()

    def _login_failed(self, message: str) -> None:
        # 失败时不重开列表 overlay，避免遮住已展示的二维码链接。
        self._app._conversation.append_block("Channels", f"微信登录失败：{message}")
        self._app._refresh()
        # 测试或卸载过程中 prompt 可能尚未挂载，焦点恢复失败不能吞错误消息。
        try:
            self._app._defer_prompt_focus()
        except Exception:
            pass

    def _test_done(self, ok: bool, instance_id: str, message: str) -> None:
        label = "成功" if ok else "失败"
        self._app._conversation.append_block(
            "Channels",
            f"连接测试{label}：{instance_id}\n{message}",
        )
        self._app._refresh()
        self.open_channels()

    def _test_failed(self, instance_id: str, message: str) -> None:
        self._app._conversation.append_block(
            "Channels",
            f"连接测试失败：{instance_id}\n{message}",
        )
        self._app._refresh()
        self.open_channels()
