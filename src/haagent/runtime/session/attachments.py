"""
src/haagent/runtime/session/attachments.py - 图片附件保存与校验

保存 TUI 剪贴板图片，并提供可审计的附件元数据结构。
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MAX_IMAGE_ATTACHMENTS = 5
MAX_IMAGE_ATTACHMENT_BYTES = 20 * 1024 * 1024
ATTACHMENTS_DIR = "attachments"


class AttachmentError(RuntimeError):
    """图片附件处理失败时抛出。"""


class AttachmentLimitError(AttachmentError):
    """图片附件超过数量或大小限制时抛出。"""


@dataclass(frozen=True)
class ImageAttachment:
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    relative_path: str
    base_path: str | None = None

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        session_root: Path,
        attachment_id: str | None = None,
    ) -> "ImageAttachment":
        data = path.read_bytes()
        mime_type, width, height = _inspect_image(data)
        return cls(
            id=attachment_id or f"img-{uuid.uuid4().hex[:12]}",
            filename=path.name,
            mime_type=mime_type,
            size_bytes=len(data),
            width=width,
            height=height,
            sha256=hashlib.sha256(data).hexdigest(),
            relative_path=path.relative_to(session_root).as_posix(),
            base_path=str(session_root.resolve()),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ImageAttachment":
        return cls(
            id=_required_str(raw, "id"),
            filename=_required_str(raw, "filename"),
            mime_type=_required_str(raw, "mime_type"),
            size_bytes=_required_int(raw, "size_bytes"),
            width=_required_int(raw, "width"),
            height=_required_int(raw, "height"),
            sha256=_required_str(raw, "sha256"),
            relative_path=_validate_relative_path(_required_str(raw, "relative_path")),
            base_path=_optional_base_path(raw.get("base_path")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "relative_path": self.relative_path,
        }

    def with_base_path(self, base_path: Path) -> "ImageAttachment":
        return ImageAttachment(
            id=self.id,
            filename=self.filename,
            mime_type=self.mime_type,
            size_bytes=self.size_bytes,
            width=self.width,
            height=self.height,
            sha256=self.sha256,
            relative_path=self.relative_path,
            base_path=str(base_path.resolve()),
        )

    def with_absolute_path(self, fallback_root: Path) -> dict[str, object]:
        data = self.to_dict()
        data["type"] = "image_attachment"
        root = Path(self.base_path) if self.base_path else fallback_root
        data["path"] = str((root / self.relative_path).resolve())
        return data


def save_clipboard_image(
    image_bytes: bytes,
    *,
    session_path: Path,
    existing: list[ImageAttachment],
) -> ImageAttachment:
    if len(existing) >= MAX_IMAGE_ATTACHMENTS:
        raise AttachmentLimitError("每条消息最多 5 张图片。")
    if len(image_bytes) > MAX_IMAGE_ATTACHMENT_BYTES:
        raise AttachmentLimitError("单张图片最大 20MB。")
    mime_type, _width, _height = _inspect_image(image_bytes)
    suffix = ".png" if mime_type == "image/png" else ".jpg"
    attachment_id = f"img-{uuid.uuid4().hex[:12]}"
    attachments_dir = session_path / ATTACHMENTS_DIR
    attachments_dir.mkdir(parents=True, exist_ok=True)
    path = attachments_dir / f"{attachment_id}{suffix}"
    path.write_bytes(image_bytes)
    return ImageAttachment.from_file(path, session_root=session_path, attachment_id=attachment_id)


def image_attachments_from_raw(raw: object) -> list[ImageAttachment]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("attachments must be a list")
    return [ImageAttachment.from_dict(item) if isinstance(item, dict) else _raise_attachment_item() for item in raw]


def _inspect_image(image_bytes: bytes) -> tuple[str, int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        fallback = _inspect_image_header(image_bytes)
        if fallback is not None:
            return fallback
        raise AttachmentError("Pillow is required for image attachments") from error
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except UnidentifiedImageError as error:
        raise AttachmentError("剪贴板内容不是可识别的图片。") from error
    if image_format == "PNG":
        return "image/png", int(width), int(height)
    if image_format in {"JPEG", "JPG"}:
        return "image/jpeg", int(width), int(height)
    raise AttachmentError(f"不支持的图片格式：{image_format or 'unknown'}")


def _inspect_image_header(image_bytes: bytes) -> tuple[str, int, int] | None:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
        width = int.from_bytes(image_bytes[16:20], "big")
        height = int.from_bytes(image_bytes[20:24], "big")
        return "image/png", width, height
    if image_bytes.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(image_bytes):
            if image_bytes[index] != 0xFF:
                index += 1
                continue
            marker = image_bytes[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(image_bytes):
                break
            segment_length = int.from_bytes(image_bytes[index:index + 2], "big")
            if segment_length < 2 or index + segment_length > len(image_bytes):
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3} and segment_length >= 7:
                height = int.from_bytes(image_bytes[index + 3:index + 5], "big")
                width = int.from_bytes(image_bytes[index + 5:index + 7], "big")
                return "image/jpeg", width, height
            index += segment_length
    return None


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ATTACHMENTS_DIR:
        raise ValueError("attachments relative_path must stay under attachments/")
    return path.as_posix()


def _required_str(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"attachments.{field} must be a non-empty string")
    return value


def _required_int(raw: dict[str, Any], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"attachments.{field} must be a non-negative integer")
    return value


def _optional_base_path(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("attachments.base_path must be a non-empty string when present")
    return str(Path(value).resolve())


def _raise_attachment_item() -> ImageAttachment:
    raise ValueError("attachments entries must be mappings")
