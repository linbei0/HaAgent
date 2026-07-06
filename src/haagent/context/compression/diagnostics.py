"""
src/haagent/context/compression/diagnostics.py - 压缩诊断记录

提供可写入 manifest、transcript 和 TUI 的统一压缩诊断结构。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionDiagnostic:
    stage: str
    subject: str
    decision: str
    reason: str
    original_chars: int = 0
    final_chars: int = 0
    original_tokens: int = 0
    final_tokens: int = 0
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stage": self.stage,
            "subject": self.subject,
            "decision": self.decision,
            "reason": self.reason,
            "original_chars": self.original_chars,
            "final_chars": self.final_chars,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
        }
        if self.artifact_path is not None:
            payload["artifact_path"] = self.artifact_path
        return payload
