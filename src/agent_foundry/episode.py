"""
agent_foundry/episode.py - Episode Package 写入器

负责为每次 run 创建可复盘的证据包，并追加 transcript/tool trace。
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeWriter:
    path: Path

    @classmethod
    def create(cls, runs_root: Path, task_path: Path) -> "EpisodeWriter":
        """创建新的 episode 目录，并初始化本阶段要求的核心文件。"""
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        episode_path = runs_root / run_id
        episode_path.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(task_path, episode_path / "task.yaml")
        (episode_path / "transcript.jsonl").write_text("", encoding="utf-8")
        (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
        return cls(path=episode_path)

    def append_transcript(self, record: dict[str, Any]) -> None:
        self._append_jsonl("transcript.jsonl", record)

    def append_tool_call(self, record: dict[str, Any]) -> None:
        self._append_jsonl("tool-calls.jsonl", record)

    def write_context_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json("context-manifest.json", manifest)

    def write_environment(self) -> None:
        self._write_json(
            "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def write_failure_attribution(self, failure: dict[str, Any] | None) -> None:
        """写入失败归因；成功 run 也保留文件，方便测试和审计稳定读取。"""
        if failure is None:
            content = "# Failure Attribution\n\n未失败。\n"
        else:
            content = (
                "# Failure Attribution\n\n"
                f"- stage: {failure.get('stage')}\n"
                f"- category: {failure.get('category')}\n"
                f"- evidence: {failure.get('evidence')}\n"
            )
        (self.path / "failure-attribution.md").write_text(content, encoding="utf-8")

    def _append_jsonl(self, name: str, record: dict[str, Any]) -> None:
        with (self.path / name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_json(self, name: str, value: dict[str, Any]) -> None:
        (self.path / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
