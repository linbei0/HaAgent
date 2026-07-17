"""
tests/unit/runtime/test_episode_package_types.py - typed EpisodePackage codecs
"""

from __future__ import annotations

from haagent.runtime.episodes.package_types import (
    EpisodeMetadata,
    FailureRecord,
    ToolCallRecord,
    build_episode_package,
)


def test_failure_record_codec_roundtrip() -> None:
    success = FailureRecord.from_dict({"status": "success", "failure": None})
    assert success.is_success
    assert success.to_dict() == {"status": "success", "failure": None}

    failed = FailureRecord.from_dict(
        {
            "status": "failed",
            "failure": {"category": "tool_error", "stage": "executing", "evidence": "boom"},
        }
    )
    assert failed.category == "tool_error"
    assert failed.to_dict()["failure"]["evidence"] == "boom"


def test_tool_call_record_policy_and_argument_error() -> None:
    call = ToolCallRecord.from_dict(
        {
            "tool_name": "file_write",
            "status": "error",
            "error": {"type": "tool_argument_invalid", "message": "bad path"},
            "policy": {
                "tool_name": "file_write",
                "risk_level": "high",
                "action": "allow",
                "reason": "approved",
                "approval": {"required": True, "status": "granted", "reason": "user"},
            },
        }
    )
    assert call.error_type == "tool_argument_invalid"
    assert call.policy is not None
    assert call.policy.approval.status == "granted"


def test_build_episode_package_exposes_typed_helpers() -> None:
    package = build_episode_package(
        path=None,
        episode_metadata={
            "episode_version": "1.0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "task_path": "task.yaml",
            "status": "completed",
            "provider": "fake",
            "workspace_root": "E:/ws",
        },
        failure_record={"status": "success", "failure": None},
        context_manifest={"context_count": 0, "contexts": []},
        transcript=[{"event": "model_response", "content": "hello", "provider": "fake", "turn": 1}],
        tool_calls=[{"tool_name": "file_read", "status": "success", "policy": None}],
        verification_commands=[],
    )
    assert isinstance(package.metadata, EpisodeMetadata)
    assert package.tool_names_used() == ["file_read"]
    assert package.final_response_text() == "hello"


def test_approval_record_rejects_string_bool_coercion() -> None:
    """codec 自身必须拒绝非 bool；字符串 "false" 不得变成 True。"""
    import pytest

    from haagent.runtime.episodes.package_types import ApprovalRecord

    with pytest.raises(ValueError, match="required"):
        ApprovalRecord.from_dict({"required": "false", "status": "granted", "reason": "user"})

    with pytest.raises(ValueError, match="required"):
        ApprovalRecord.from_dict({"required": 1, "status": "granted", "reason": "user"})

    ok = ApprovalRecord.from_dict({"required": False, "status": "granted", "reason": "user"})
    assert ok.required is False
