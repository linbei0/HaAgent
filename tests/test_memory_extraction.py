"""
tests/test_memory_extraction.py - 长期记忆提取测试

验证 Memory Extraction 只生成候选队列记录，并且不会绕过治理或确定性落库。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haagent import cli
import haagent.memory.extraction as extraction
from haagent.memory import CandidateEvidence, CandidateQueue, MemoryStore
from haagent.memory.extraction import (
    MemoryExtractionRequest,
    MemoryExtractor,
)
from haagent.memory.retrieval import MemoryRetrievalRequest, MemoryRetriever
from haagent.models.gateway import ModelResponse
from haagent.runtime.chat_session import AgentSession


class RecordingGateway:
    provider_name = "recording"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, tool_schemas):
        self.calls.append({"messages": list(messages), "tool_schemas": list(tool_schemas)})
        return ModelResponse(self.response, [])


class SequentialGateway:
    provider_name = "recording"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, tool_schemas):
        self.calls.append({"messages": list(messages), "tool_schemas": list(tool_schemas)})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _request(
    tmp_path: Path,
    *,
    prompt: str,
    final_response: str = "done",
    gateway=None,
    verification_status: str = "not_run",
) -> MemoryExtractionRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    session_path = tmp_path / ".runs" / "sessions" / "session-test"
    return MemoryExtractionRequest(
        session_id="session-test",
        session_path=session_path,
        workspace_root=workspace,
        turn_index=1,
        user_prompt=prompt,
        final_response=final_response,
        status="completed",
        verification_status=verification_status,
        episode_path=tmp_path / ".runs" / "episode-test",
        working_state={"current_goal": prompt, "key_findings": [], "completed_actions": [], "next_steps": [], "last_updated_turn": 1},
        runtime_events=[],
        model_gateway=gateway,
        user_memory_root=tmp_path / "user-memory",
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _all_text(path: Path) -> str:
    chunks: list[str] = []
    if not path.exists():
        return ""
    for child in sorted(path.rglob("*")):
        if child.is_file():
            chunks.append(child.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_extraction_does_not_use_phrase_trigger_tables() -> None:
    assert not hasattr(extraction, "MEMORY_TRIGGER_PHRASES")
    assert not hasattr(extraction, "TEMPORARY_PHRASES")


def test_explicit_remember_creates_pending_candidate_not_memory(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Package manager",
                        "body": "HaAgent 使用 uv 管理依赖。",
                        "source_summary": "用户明确要求记住依赖管理事实。",
                        "basis": "用户说：记住这个：HaAgent 使用 uv 管理依赖。",
                        "category_rationale": "这是当前 workspace 的稳定项目事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(
        tmp_path,
        prompt="记住这个：HaAgent 使用 uv 管理依赖。",
        final_response="已记录为候选。",
        gateway=gateway,
    )

    result = MemoryExtractor().extract(request)

    queue = CandidateQueue(request.session_path)
    pending = queue.list(status="pending")
    store = MemoryStore(workspace_root=request.workspace_root, user_memory_root=tmp_path / "user-memory")
    assert result.created_count == 1
    assert len(pending) == 1
    assert pending[0].status == "pending"
    assert pending[0].source == "extraction"
    assert pending[0].title == "Package manager"
    assert pending[0].evidence.session_id == "session-test"
    assert pending[0].evidence.source_summary
    assert pending[0].evidence.basis
    assert pending[0].evidence.category_rationale
    assert store.list_records(scope="workspace", category="facts") == []


def test_stable_sop_or_decision_uses_gateway_and_creates_candidate(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "decisions",
                        "title": "Memory write path",
                        "body": "长期记忆必须先进入候选队列，用户确认后再确定性落库。",
                        "source_summary": "用户确认长期记忆写入路径。",
                        "basis": "用户说这是项目约定，助手最终答复确认该决策。",
                        "category_rationale": "这是关于长期记忆架构的明确决策。",
                        "tags": ["memory", "governance"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(
        tmp_path,
        prompt="这是项目约定：长期记忆必须先进入候选队列。",
        final_response="已按该项目约定处理。",
        gateway=gateway,
        verification_status="success",
    )

    result = MemoryExtractor().extract(request)

    pending = CandidateQueue(request.session_path).list(status="pending")
    assert result.created_count == 1
    assert pending[0].category == "decisions"
    assert pending[0].title == "Memory write path"
    assert gateway.calls
    assert gateway.calls[0]["tool_schemas"] == []
    assert "Memory Extraction" in gateway.calls[0]["messages"][0]["content"]


def test_ordinary_one_off_task_runs_bounded_extraction_but_queues_nothing(tmp_path: Path) -> None:
    gateway = RecordingGateway('{"candidates":[]}')
    request = _request(
        tmp_path,
        prompt="总结 README。",
        final_response="README 已总结。",
        gateway=gateway,
    )

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.status == "skipped"
    assert len(gateway.calls) == 1
    assert not CandidateQueue(request.session_path).path.exists()


def test_secret_output_does_not_enter_candidate_or_diagnostics(tmp_path: Path) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "API key",
                        "body": f"OpenAI key is {secret}",
                        "source_summary": "用户粘贴了 key。",
                        "basis": "模型输出包含 secret。",
                        "category_rationale": "事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="记住这个 API key", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1
    assert CandidateQueue(request.session_path).list(status="pending") == []
    assert secret not in _all_text(tmp_path)


def test_unverified_claim_is_rejected_before_queue(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Maybe package manager",
                        "body": "HaAgent 可能 uses uv.",
                        "source_summary": "模型猜测。",
                        "basis": "没有验证。",
                        "category_rationale": "事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="以后记住项目包管理器", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1
    assert CandidateQueue(request.session_path).list(status="pending") == []


def test_duplicate_confirmed_or_pending_memory_is_not_queued_again(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Package manager",
                        "body": "HaAgent 使用 uv 管理依赖。",
                        "source_summary": "用户明确要求记住依赖管理事实。",
                        "basis": "用户说：记住这个：HaAgent 使用 uv 管理依赖。",
                        "category_rationale": "这是当前 workspace 的稳定项目事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="记住这个：HaAgent 使用 uv 管理依赖。", gateway=gateway)
    store = MemoryStore(workspace_root=request.workspace_root, user_memory_root=tmp_path / "user-memory")
    queue = CandidateQueue(request.session_path)
    existing = store.create_candidate(
        queue,
        scope="workspace",
        category="facts",
        title="Package manager",
        body="HaAgent 使用 uv 管理依赖。",
        evidence=CandidateEvidence(source_type="episode", evidence_summary="用户确认。"),
        source="user_explicit",
    )
    store.confirm_candidate(queue, existing.candidate_id)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert len(queue.list(status="pending")) == 0

    pending = store.create_candidate(
        queue,
        scope="workspace",
        category="facts",
        title="Package manager pending",
        body="HaAgent 使用 uv 管理依赖。",
        evidence=CandidateEvidence(source_type="episode", evidence_summary="用户确认。"),
        source="user_explicit",
    )
    second = MemoryExtractor().extract(request)
    assert second.created_count == 0
    assert [item.candidate_id for item in queue.list(status="pending")] == [pending.candidate_id]


def test_missing_evidence_fields_are_rejected(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "sop",
                        "title": "Test command",
                        "body": "Run uv run pytest -q.",
                    }
                ]
            },
        ),
    )
    request = _request(tmp_path, prompt="以后按这个 SOP 做", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1
    assert CandidateQueue(request.session_path).list(status="pending") == []


def test_extracted_pending_candidate_is_not_retrieved(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Pending retrieval boundary",
                        "body": "Pending candidates must never enter retrieval.",
                        "source_summary": "用户要求记住 retrieval 边界。",
                        "basis": "用户说 pending candidates must never enter retrieval。",
                        "category_rationale": "这是当前 workspace 的稳定事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="记住这个：Pending candidates must never enter retrieval.", gateway=gateway)
    MemoryExtractor().extract(request)
    assert CandidateQueue(request.session_path).list(status="pending")

    result = MemoryRetriever().retrieve(
        MemoryRetrievalRequest(
            query="Pending candidates retrieval",
            workspace_root=request.workspace_root,
            user_memory_root=tmp_path / "user-memory",
        ),
    )

    assert result.memories == []
    assert "Pending candidates" not in result.to_model_block()


def test_extraction_writes_session_diagnostics(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Package manager",
                        "body": "HaAgent 使用 uv 管理依赖。",
                        "source_summary": "用户明确要求记住依赖管理事实。",
                        "basis": "用户说：记住这个：HaAgent 使用 uv 管理依赖。",
                        "category_rationale": "这是当前 workspace 的稳定项目事实。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="记住这个：HaAgent 使用 uv 管理依赖。", gateway=gateway)

    MemoryExtractor().extract(request)

    diagnostics = _jsonl(request.session_path / "memory_extraction.jsonl")
    assert diagnostics[0]["status"] == "created"
    assert diagnostics[0]["created_count"] == 1
    assert diagnostics[0]["session_id"] == "session-test"
    assert "source_chars" in diagnostics[0]


def test_agent_session_emits_candidate_notice(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway(
        [
            ModelResponse("已处理。", []),
            ModelResponse(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "scope": "workspace",
                                "category": "facts",
                                "title": "Package manager",
                                "body": "HaAgent 使用 uv 管理依赖。",
                                "source_summary": "用户明确要求记住依赖管理事实。",
                                "basis": "用户说：记住这个：HaAgent 使用 uv 管理依赖。",
                                "category_rationale": "这是当前 workspace 的稳定项目事实。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                [],
            ),
        ],
    )
    session = AgentSession(workspace_root=workspace, runs_root=tmp_path / ".runs", model_gateway=gateway)
    events = []

    result = session.run_prompt_events("记住这个：HaAgent 使用 uv 管理依赖。", event_sink=events.append)

    assert result.status == "completed"
    assert result.memory_candidates_created == 1
    assert "memory_candidates=1" in result.output_lines()
    assert any(event.event_type == "memory_candidates_created" for event in events)


def test_memory_cli_lists_confirms_and_rejects_candidates(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs_root = tmp_path / ".runs"
    session = AgentSession(
        workspace_root=workspace,
        runs_root=runs_root,
        model_gateway=SequentialGateway(
            [
                ModelResponse("done", []),
                ModelResponse(
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "scope": "workspace",
                                    "category": "facts",
                                    "title": "Package manager",
                                    "body": "HaAgent 使用 uv 管理依赖。",
                                    "source_summary": "用户明确要求记住依赖管理事实。",
                                    "basis": "用户说：记住这个：HaAgent 使用 uv 管理依赖。",
                                    "category_rationale": "这是当前 workspace 的稳定项目事实。",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    [],
                ),
                ModelResponse("done", []),
                ModelResponse(
                    json.dumps(
                        {
                            "candidates": [
                                {
                                    "scope": "workspace",
                                    "category": "facts",
                                    "title": "Memory review",
                                    "body": "HaAgent 的长期记忆候选需要人工审核。",
                                    "source_summary": "用户明确要求记住长期记忆审核要求。",
                                    "basis": "用户说：记住这个：HaAgent 的长期记忆候选需要人工审核。",
                                    "category_rationale": "这是当前 workspace 的稳定项目事实。",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    [],
                ),
            ],
        ),
    )
    session.run_prompt("记住这个：HaAgent 使用 uv 管理依赖。")
    candidate_id = CandidateQueue(session.session_path).list(status="pending")[0].candidate_id
    monkeypatch.chdir(workspace)

    list_code = cli.main(["memory", "list", "--runs-root", str(runs_root)])
    list_output = capsys.readouterr().out
    confirm_code = cli.main(["memory", "confirm", candidate_id, "--runs-root", str(runs_root)])
    confirm_output = capsys.readouterr().out

    assert list_code == 0
    assert candidate_id in list_output
    assert confirm_code == 0
    assert "memory_id=" in confirm_output
    assert CandidateQueue(session.session_path).get(candidate_id).status == "confirmed"

    session.run_prompt("记住这个：HaAgent 的长期记忆候选需要人工审核。")
    second_id = CandidateQueue(session.session_path).list(status="pending")[0].candidate_id
    reject_code = cli.main(["memory", "reject", second_id, "--runs-root", str(runs_root), "--reason", "not durable"])
    reject_output = capsys.readouterr().out

    assert reject_code == 0
    assert "status=rejected" in reject_output
    assert CandidateQueue(session.session_path).get(second_id).status == "rejected"
