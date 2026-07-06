"""
tests/unit/runtime/test_image_attachments.py - 图片附件元数据测试

验证剪贴板图片进入 runtime 前后的保存、校验与 task contract 形状。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from haagent.context.builder import ContextBuilder
from haagent.runtime.contracts.task import TaskLoadError, TaskSpec, load_task
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.session.attachments import (
    AttachmentLimitError,
    ImageAttachment,
    save_clipboard_image,
)
from haagent.runtime.session.turn import write_chat_task_yaml
from haagent.tools.registry import default_tool_runtime_registry


def test_save_clipboard_image_records_metadata_and_session_relative_path(tmp_path: Path) -> None:
    image_bytes = _png_bytes()
    session_path = tmp_path / ".runs" / "sessions" / "session-1"

    attachment = save_clipboard_image(
        image_bytes,
        session_path=session_path,
        existing=[],
    )

    saved_path = session_path / attachment.relative_path
    assert saved_path.exists()
    assert saved_path.read_bytes() == image_bytes
    assert attachment.mime_type == "image/png"
    assert attachment.size_bytes == len(image_bytes)
    assert attachment.width == 2
    assert attachment.height == 1
    assert attachment.sha256
    assert attachment.relative_path.startswith("attachments/")


def test_save_clipboard_image_enforces_count_and_size_limits(tmp_path: Path) -> None:
    existing = [
        ImageAttachment(
            id=f"img-{index}",
            filename=f"img-{index}.png",
            mime_type="image/png",
            size_bytes=10,
            width=1,
            height=1,
            sha256="abc",
            relative_path=f"attachments/img-{index}.png",
        )
        for index in range(5)
    ]

    with pytest.raises(AttachmentLimitError, match="最多 5 张"):
        save_clipboard_image(_png_bytes(), session_path=tmp_path, existing=existing)

    with pytest.raises(AttachmentLimitError, match="20MB"):
        save_clipboard_image(
            _png_bytes() + (b"x" * (20 * 1024 * 1024)),
            session_path=tmp_path,
            existing=[],
        )


def test_write_and_load_chat_task_yaml_preserves_attachment_metadata(tmp_path: Path) -> None:
    attachment = _attachment()
    task_path = tmp_path / "task.yaml"

    write_chat_task_yaml(
        task_path,
        "describe image",
        tmp_path,
        attachments=[attachment],
    )
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    task = load_task(task_path)

    assert raw["attachments"] == [attachment.to_dict()]
    assert [item.to_dict() for item in task.attachments] == [attachment.to_dict()]
    assert task.attachments[0].base_path == str(tmp_path.resolve())


def test_load_task_rejects_attachment_path_outside_controlled_directory(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "goal": "bad attachment",
                "workspace_root": str(tmp_path),
                "attachments": [
                    {
                        **_attachment().to_dict(),
                        "relative_path": "../outside.png",
                    }
                ],
                "constraints": [],
                "allowed_tools": ["file_read"],
                "acceptance_criteria": ["done"],
                "verification_commands": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskLoadError, match="attachments"):
        load_task(task_path)


def test_context_builder_message_contains_attachment_metadata_without_base64_snapshot(tmp_path: Path) -> None:
    image_path = tmp_path / "attachments" / "image.png"
    image_path.parent.mkdir()
    image_path.write_bytes(_png_bytes())
    attachment = ImageAttachment.from_file(
        image_path,
        session_root=tmp_path,
        attachment_id="img-test",
    )
    writer = _make_writer(tmp_path)

    context = ContextBuilder(
        task=TaskSpec(
            goal="describe image",
            constraints=[],
            allowed_tools=["file_read"],
            acceptance_criteria=["done"],
            verification_commands=[],
            workspace_root=str(tmp_path),
            attachments=[attachment],
        ),
        workspace_root=tmp_path,
        provider_name="openai-chat",
        episode_writer=writer,
        tool_registry=default_tool_runtime_registry(),
    ).build()

    user_message = context.messages[-1]
    content = user_message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": content[0]["text"]}
    assert content[1]["type"] == "image_attachment"
    assert content[1]["relative_path"] == attachment.relative_path
    snapshot = (writer.path / "contexts" / "0001.json").read_text(encoding="utf-8")
    assert "base64" not in snapshot
    assert attachment.relative_path in snapshot


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00{@\xe8\xdd\x00\x00\x00\x0cIDATx\x9cc\xfc\xcf"
        b"\x00\x02\x00\x06\x08\x01\x01Z\xcf\x06H\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _attachment() -> ImageAttachment:
    return ImageAttachment(
        id="img-test",
        filename="img-test.png",
        mime_type="image/png",
        size_bytes=10,
        width=2,
        height=1,
        sha256="a" * 64,
        relative_path="attachments/img-test.png",
    )


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "source-task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)
    (writer.path / "plan.json").write_text(
        json.dumps({"planned_steps": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    return writer
