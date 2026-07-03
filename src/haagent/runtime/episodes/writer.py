"""
src/haagent/runtime/episodes/writer.py - Episode Package 写入器

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

from haagent.runtime.orchestration.failure import FailureCategory


EPISODE_VERSION = "1.0"


@dataclass(frozen=True)
class EpisodeWriter:
    path: Path
    task_path: Path

    @classmethod
    def create(cls, runs_root: Path, task_path: Path) -> "EpisodeWriter":
        """创建新的 episode 目录，并初始化本阶段要求的核心文件。"""
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        episode_path = runs_root / run_id
        episode_path.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(task_path, episode_path / "task.yaml")
        (episode_path / "transcript.jsonl").write_text("", encoding="utf-8")
        (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
        return cls(path=episode_path, task_path=task_path)

    def append_transcript(self, record: dict[str, Any]) -> None:
        self._append_jsonl("transcript.jsonl", record)

    def append_tool_call(self, record: dict[str, Any]) -> None:
        self._append_jsonl("tool-calls.jsonl", record)

    def append_interaction_event(self, event_type: str, record: dict[str, Any]) -> None:
        self.append_transcript({"event": event_type, **record})

    def write_context_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json("context-manifest.json", manifest)

    def write_plan(self, plan: dict[str, Any]) -> None:
        self._write_json("plan.json", plan)

    def write_episode_metadata(
        self,
        status: str,
        provider: str | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        """写入 episode 根 schema，terminal state 会复写 status。"""
        metadata_path = self.path / "episode.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "episode_version": EPISODE_VERSION,
                "created_at": metadata.get("created_at", datetime.now(UTC).isoformat()),
                "task_path": str(self.task_path),
                "status": status,
                "provider": provider if provider is not None else metadata.get("provider"),
                "workspace_root": (
                    str(workspace_root)
                    if workspace_root is not None
                    else metadata.get("workspace_root")
                ),
            },
        )
        self._write_json("episode.json", metadata)

    def write_environment(self, workspace_root: Path | None = None) -> None:
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        if workspace_root is not None:
            environment["workspace_root"] = str(workspace_root)
        self._write_json("environment.json", environment)

    def write_sandbox_metadata(
        self,
        workspace_root: Path,
        command_timeout_seconds: int | float,
    ) -> None:
        self._write_json(
            "sandbox.json",
            {
                "workspace_root": str(workspace_root),
                "filesystem_boundary": "workspace_root",
                "network_policy": "unrestricted",
                "process_policy": "local_subprocess",
                "credential_policy": "inherit_environment",
                "resource_limits": {
                    "command_timeout_seconds": command_timeout_seconds,
                },
            },
        )

    def write_workspace_preflight(self, preflight: dict[str, Any]) -> None:
        workspace_dir = self.path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_failure_attribution(self, failure: dict[str, Any] | None) -> None:
        """写入失败归因；成功 run 也保留文件，方便测试和审计稳定读取。"""
        if failure is None:
            content = "# Failure Attribution\n\n未失败。\n"
            self._write_json("failure.json", {"status": "success", "failure": None})
        else:
            _validate_failure_category(str(failure.get("category")))
            content = (
                "# Failure Attribution\n\n"
                f"- stage: {failure.get('stage')}\n"
                f"- category: {failure.get('category')}\n"
                f"- evidence: {failure.get('evidence')}\n"
            )
            self._write_json(
                "failure.json",
                {
                    "status": "failed",
                    "failure": {
                        "category": failure.get("category"),
                        "stage": failure.get("stage"),
                        "evidence": failure.get("evidence"),
                    },
                },
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


def _validate_failure_category(category: str) -> None:
    if category not in {failure_category.value for failure_category in FailureCategory}:
        raise ValueError(f"unknown failure category: {category}")
