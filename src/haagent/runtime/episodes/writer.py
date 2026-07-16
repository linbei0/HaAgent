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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from haagent.models.types import ModelGatewayMetadata, ModelUsage
from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.sandbox.base import SandboxMetadata


EPISODE_VERSION = "1.0"


@dataclass(frozen=True)
class EpisodeWriter:
    path: Path
    task_path: Path
    _write_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    @classmethod
    def create(cls, runs_root: Path, task_path: Path) -> "EpisodeWriter":
        """创建新的 episode 目录，并初始化本阶段要求的核心文件。"""
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        episode_path = runs_root / run_id
        episode_path.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(task_path, episode_path / "task.yaml")
        attachments_dir = task_path.parent / "attachments"
        if attachments_dir.exists():
            shutil.copytree(attachments_dir, episode_path / "attachments")
        (episode_path / "transcript.jsonl").write_text("", encoding="utf-8")
        (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
        # 必需验证日志从 package 创建起存在，提前失败路径也必须可校验。
        verification_dir = episode_path / "verification"
        verification_dir.mkdir()
        (verification_dir / "commands.jsonl").write_text("", encoding="utf-8")
        (verification_dir / "files.jsonl").write_text("", encoding="utf-8")
        writer = cls(path=episode_path, task_path=task_path)
        writer.write_cost_metadata()
        return writer

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

    def write_performance(self, value: dict[str, Any]) -> None:
        """写入可选 performance.json；失败必须显式暴露 artifact 名称。"""

        try:
            self._write_json("performance.json", value)
        except OSError as error:
            raise RuntimeError(f"failed to write performance.json: {error}") from error

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

    def write_environment(
        self,
        workspace_root: Path | None = None,
        *,
        model_metadata: ModelGatewayMetadata | None = None,
        allowed_tools: list[str] | None = None,
        registry_tool_count: int | None = None,
        entrypoint: str = "unknown",
    ) -> None:
        allowed_tool_names = [str(name) for name in (allowed_tools or [])]
        environment = {
            "environment_schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "workspace_root": str(workspace_root) if workspace_root is not None else None,
            "python": sys.version,
            "platform": platform.platform(),
            "process": {
                "executable": sys.executable,
                "cwd": str(Path.cwd()),
            },
            "haagent": {
                "package_version": _package_version(),
                "entrypoint": entrypoint,
            },
            "model": _environment_model_metadata(model_metadata),
            "tools": {
                "allowed_tool_count": len(allowed_tool_names),
                "registry_tool_count": int(registry_tool_count or 0),
                "allowed_tools": allowed_tool_names,
            },
        }
        self._write_json("environment.json", environment)

    def write_cost_metadata(self) -> None:
        self._write_json("cost.json", _empty_cost_metadata())

    def append_model_usage(
        self,
        *,
        turn: int,
        attempt: int | None = None,
        provider: str,
        model: str | None,
        usage: ModelUsage | None,
    ) -> None:
        if usage is None:
            self.finalize_cost_metadata()
            return
        cost = _read_cost_metadata(self.path / "cost.json")
        cost["usage_available"] = True
        cost["pricing_available"] = False
        cost["currency"] = None
        cost["estimated_cost"] = None
        cost["pricing_source"] = None
        cost["reason"] = "pricing unavailable: no reliable catalog match"
        model_calls = cost.setdefault("model_calls", [])
        if not isinstance(model_calls, list):
            model_calls = []
            cost["model_calls"] = model_calls
        record = {
                "turn": turn,
                "provider": provider,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "raw_usage_source": usage.raw_source,
        }
        if attempt is not None:
            record["attempt"] = attempt
        model_calls.append(record)
        cost["totals"] = _usage_totals(model_calls)
        self._write_json("cost.json", cost)

    def finalize_cost_metadata(self) -> None:
        cost_path = self.path / "cost.json"
        if not cost_path.exists():
            self.write_cost_metadata()
            return
        cost = _read_cost_metadata(cost_path)
        if not cost.get("model_calls"):
            cost.update(_empty_cost_metadata())
        elif cost.get("usage_available") is True and not cost.get("pricing_available"):
            cost["reason"] = cost.get("reason") or "pricing unavailable: no reliable catalog match"
        self._write_json("cost.json", cost)

    def write_tool_artifact(self, tool_name: str, content: str, *, suffix: str = ".txt") -> str:
        artifact_dir = self.path / "artifacts" / "tool-results"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{_safe_artifact_name(tool_name)}-{uuid.uuid4().hex[:8]}{suffix}"
        artifact_path.write_text(content, encoding="utf-8")
        try:
            return artifact_path.relative_to(self.task_path.parent.resolve()).as_posix()
        except ValueError:
            return artifact_path.as_posix()

    def write_sandbox_metadata(self, metadata: SandboxMetadata) -> None:
        self._write_json("sandbox.json", metadata.to_dict())

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
        with self._write_lock:
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


def _empty_cost_metadata() -> dict[str, Any]:
    return {
        "cost_schema_version": "1.0",
        "usage_available": False,
        "pricing_available": False,
        "currency": None,
        "estimated_cost": None,
        "pricing_source": None,
        "reason": "model gateway did not provide usage metadata",
        "model_calls": [],
        "totals": {
            "model_call_count": 0,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }


def _read_cost_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_cost_metadata()
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_cost_metadata()
    return parsed if isinstance(parsed, dict) else _empty_cost_metadata()


def _usage_totals(model_calls: list[Any]) -> dict[str, int | None]:
    input_total = _sum_token_field(model_calls, "input_tokens")
    output_total = _sum_token_field(model_calls, "output_tokens")
    total_total = _sum_token_field(model_calls, "total_tokens")
    return {
        "model_call_count": len(model_calls),
        "input_tokens": input_total,
        "output_tokens": output_total,
        "total_tokens": total_total,
    }


def _sum_token_field(model_calls: list[Any], field_name: str) -> int | None:
    values = [
        call.get(field_name)
        for call in model_calls
        if isinstance(call, dict) and isinstance(call.get(field_name), int)
    ]
    if not values:
        return None
    return sum(values)


def _safe_artifact_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return safe or "tool"


def _environment_model_metadata(metadata: ModelGatewayMetadata | None) -> dict[str, object | None]:
    if metadata is None:
        return {
            "provider": "unknown",
            "model": None,
            "endpoint": None,
            "base_url": None,
            "profile_name": None,
            "request_config": None,
        }
    return {
        "provider": metadata.provider or "unknown",
        "model": metadata.model,
        "endpoint": _safe_url(metadata.endpoint),
        "base_url": _safe_url(metadata.base_url),
        "profile_name": metadata.profile_name,
        "request_config": metadata.request_config,
    }


def _safe_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return value.strip().split("?", 1)[0]
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _package_version() -> str:
    try:
        return package_metadata.version("haagent")
    except package_metadata.PackageNotFoundError:
        return "unknown"
