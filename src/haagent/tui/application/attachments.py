"""
src/haagent/tui/application/attachments.py - TUI 图片附件输入协调

管理 prompt 中的 [image N] token 与待发送 ImageAttachment 的对应关系，
以及从剪贴板粘贴图片到输入框的流程。图片解析和落盘仍由 AssistantService 负责。
"""

from __future__ import annotations

import re
from typing import Any

from haagent.runtime.session.attachments import ImageAttachment, MAX_IMAGE_ATTACHMENTS

IMAGE_TOKEN_PATTERN = re.compile(r"\[image\s+(\d+)\]")


class AttachmentController:
    """维护待发送图片附件与 prompt token 的同步。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.pending: list[ImageAttachment] = []
        self.tokens: dict[int, ImageAttachment] = {}
        self.session_image_count = 0

    def reset(self) -> None:
        self.pending = []
        self.tokens = {}
        self.session_image_count = 0

    def attachments_from_prompt(self, prompt: str) -> list[ImageAttachment]:
        attachments: list[ImageAttachment] = []
        seen: set[int] = set()
        for match in IMAGE_TOKEN_PATTERN.finditer(prompt):
            image_number = int(match.group(1))
            if image_number in seen:
                continue
            attachment = self.tokens.get(image_number)
            if attachment is not None:
                attachments.append(attachment)
                seen.add(image_number)
        return attachments

    def sync_with_prompt(self, prompt: str) -> None:
        referenced = {int(match.group(1)) for match in IMAGE_TOKEN_PATTERN.finditer(prompt)}
        self.tokens = {number: attachment for number, attachment in self.tokens.items() if number in referenced}
        self.pending = list(self.tokens.values())

    def paste_from_clipboard(self) -> None:
        if self._app._state in {"running", "cancelling", "waiting approval"}:
            self._app._conversation.append_block("Command", "运行中不能修改待发送附件。")
            self._app._refresh()
            return
        if len(self.pending) >= MAX_IMAGE_ATTACHMENTS:
            self._app._conversation.append_block("Command", f"每条消息最多 {MAX_IMAGE_ATTACHMENTS} 张图片。")
            self._app._refresh()
            return
        try:
            attachment = self._app.service.sessions.paste_clipboard_image(existing=list(self.pending))
        except Exception as error:
            self._app._conversation.append_block("Command", f"添加图片附件失败：{error}")
            self._app._refresh()
            return
        self.session_image_count += 1
        image_number = self.session_image_count
        self.pending.append(attachment)
        self.tokens[image_number] = attachment
        prompt_input = self._app._prompt_input()
        current = self._app._prompt_value(prompt_input)
        token = f"[image {image_number}]"
        prefix = f"{current} " if current and not current.endswith((" ", "\n")) else current
        self._app._set_prompt_value(prompt_input, f"{prefix}{token}")


def prompt_without_image_tokens(prompt: str) -> str:
    text = IMAGE_TOKEN_PATTERN.sub(" ", prompt)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()
