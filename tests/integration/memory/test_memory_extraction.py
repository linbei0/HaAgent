"""
tests/integration/memory/test_memory_extraction.py - 长期记忆提取测试

验证 Memory Extraction 只生成候选队列记录，并且不会绕过治理或确定性落库。
"""

from __future__ import annotations

import json
import importlib.util
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
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.events import MemoryNoticeEvent, ToolActivityEvent
from haagent.runtime.execution.human_interaction import HumanInteractionResponse


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
    assert not (Path(extraction.__file__).with_name("intent.py")).exists()


def test_memory_runtime_does_not_ship_heuristic_memory_routing_modules() -> None:
    assert importlib.util.find_spec("haagent.memory.intent") is None
    assert importlib.util.find_spec("haagent.memory.artifact_guard") is None


def test_source_does_not_contain_legacy_memory_matching_contracts() -> None:
    forbidden = [
        "MEMORY_ARTIFACT_FILENAMES",
        "PROFILE_PATH_TOKENS",
        "MEMORY_INTERNAL_TARGETS",
        "MEMORY_INTERNAL_WRITE_OPERATIONS",
        "memory_artifact_denied",
        "memory_internal_denied",
    ]
    repo_root = Path(__file__).parents[1]
    offenders: list[str] = []
    for root in (repo_root / "src", repo_root / "tests"):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(repo_root)}: {token}")
    assert offenders == []


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
                        "evidence_source": "user_prompt",
                        "evidence_quote": "HaAgent 使用 uv 管理依赖",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "HaAgent 使用 uv 管理依赖",
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
                        "basis": "用户原文可定位。",
                        "category_rationale": "这是关于长期记忆架构的明确决策。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "长期记忆必须先进入候选队列",
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
    assert "Memory Settlement" in gateway.calls[0]["messages"][0]["content"]


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


def test_agent_session_does_not_extract_without_start_memory_update(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway([ModelResponse("你好，有什么我可以帮你？", [])])
    session = AgentSession(workspace_root=workspace, runs_root=tmp_path / ".runs", model_gateway=gateway)
    events = []

    result = session.run_prompt_events("你好", event_sink=events.append)

    assert result.status == "completed"
    assert result.memory_candidates_created == 0
    assert result.memory_extraction_status == "skipped"
    assert len(gateway.calls) == 1
    assert not CandidateQueue(session.session_path).path.exists()
    assert not any(isinstance(event, MemoryNoticeEvent) for event in events)


def test_agent_session_extracts_only_after_start_memory_update(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway(
        [
            ModelResponse(
                "我会把这个偏好放入候选记忆等待你确认。",
                [ToolCall(name="start_memory_update", args={"reason": "用户给出长期回答语言偏好"}, id="call_memory")],
            ),
            ModelResponse("已处理。", []),
            ModelResponse(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "scope": "user",
                                "category": "user_preferences",
                                "title": "回答语言偏好",
                                "body": "用户希望以后回答尽量使用中文。",
                                "source_summary": "用户明确表达长期回答语言偏好。",
                                "basis": "用户原文可定位。",
                                "category_rationale": "这是跨 workspace 可复用的用户偏好。",
                                "evidence_source": "user_prompt",
                                "evidence_quote": "以后回答我尽量用中文",
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

    result = session.run_prompt("以后回答我，尽量用中文。")

    assert result.status == "completed"
    assert result.memory_candidates_created == 1
    assert len(gateway.calls) == 3
    pending = CandidateQueue(session.session_path).list(status="pending")
    assert pending[0].evidence.source_type == "user_prompt"
    assert pending[0].evidence.evidence_quote == "以后回答我尽量用中文"
    assert pending[0].evidence.fingerprint


def test_final_response_cannot_be_evidence_source(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "user",
                        "category": "user_preferences",
                        "title": "饮食喜好",
                        "body": "用户喜欢吃饭。",
                        "source_summary": "候选只能从助手回答找到。",
                        "basis": "助手最终回答包含这句话。",
                        "category_rationale": "用户偏好。",
                        "evidence_source": "final_response",
                        "evidence_quote": "你喜欢吃饭",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="我的爱好是什么？", final_response="你喜欢吃饭。", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1
    assert CandidateQueue(request.session_path).list(status="pending") == []


def test_evidence_quote_must_be_locatable(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "user",
                        "category": "user_preferences",
                        "title": "饮食喜好",
                        "body": "用户喜欢吃饭。",
                        "source_summary": "模型改写了用户偏好。",
                        "basis": "quote 不在用户原文中。",
                        "category_rationale": "用户偏好。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "用户喜欢吃饭",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="我喜欢面条。", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1


def test_evidence_quote_allows_light_normalization(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "user",
                        "category": "user_preferences",
                        "title": "回答语言偏好",
                        "body": "用户希望以后回答尽量使用中文。",
                        "source_summary": "用户明确表达回答语言偏好。",
                        "basis": "用户原文可定位。",
                        "category_rationale": "这是跨 workspace 可复用的用户偏好。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "以后回答我尽量用中文",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="以后回答我，尽量用中文。", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 1
    assert result.rejected_count == 0


def test_sop_candidate_requires_execution_or_verification_evidence(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "sop",
                        "title": "测试命令",
                        "body": "修改记忆系统后运行 uv run pytest tests/integration/memory/test_memory_extraction.py -q。",
                        "source_summary": "助手建议了一个 SOP。",
                        "basis": "没有工具或验证证据。",
                        "category_rationale": "可复用流程。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "修复记忆系统问题",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="修复记忆系统问题", final_response="以后可以运行测试命令。", gateway=gateway)

    result = MemoryExtractor().extract(request)

    assert result.created_count == 0
    assert result.rejected_count == 1


def test_sop_candidate_can_use_verified_tool_result_evidence(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "sop",
                        "title": "记忆测试命令",
                        "body": "修改记忆系统后可运行 uv run pytest tests/integration/memory/test_memory_extraction.py -q 验证。",
                        "source_summary": "shell 验证命令执行成功。",
                        "basis": "工具结果中有可定位证据。",
                        "category_rationale": "这是经过验证的可复用 SOP。",
                        "evidence_source": "verification_result",
                        "evidence_quote": "17 passed",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(
        tmp_path,
        prompt="修复记忆系统问题",
        gateway=gateway,
        verification_status="success",
    )
    request = MemoryExtractionRequest(
        **{
            **request.__dict__,
            "runtime_events": [
                {
                    "event_type": "tool_finished",
                    "tool_name": "shell",
                    "result": {"status": "success", "exit_code": 0, "stdout_excerpt": "17 passed"},
                }
            ],
        }
    )

    result = MemoryExtractor().extract(request)

    assert result.created_count == 1
    assert result.rejected_count == 0


def test_rejected_duplicate_fingerprint_is_suppressed_without_candidate_audit(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "user",
                        "category": "user_preferences",
                        "title": "回答语言偏好",
                        "body": "用户希望以后回答尽量使用中文。",
                        "source_summary": "用户明确表达回答语言偏好。",
                        "basis": "用户原文可定位。",
                        "category_rationale": "这是跨 workspace 可复用的用户偏好。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "以后回答我尽量用中文",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="以后回答我，尽量用中文。", gateway=gateway)
    first = MemoryExtractor().extract(request)
    queue = CandidateQueue(request.session_path)
    store = MemoryStore(workspace_root=request.workspace_root, user_memory_root=tmp_path / "user-memory")
    store.reject_candidate(queue, first.created_candidates[0].candidate_id, reason="not durable")

    second = MemoryExtractor().extract(request)

    assert second.created_count == 0
    assert second.rejected_count == 1
    audit_path = tmp_path / "user-memory" / "audit.jsonl"
    audit_events = [json.loads(line)["event_type"] for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert audit_events == ["candidate_created", "memory_rejected"]


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


def test_uncertain_words_do_not_trigger_phrase_based_rejection(tmp_path: Path) -> None:
    gateway = RecordingGateway(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "workspace",
                        "category": "facts",
                        "title": "Maybe package manager",
                        "body": "HaAgent 可能 uses uv when running tests.",
                        "source_summary": "用户要求记录一句包含不确定语气的项目事实。",
                        "basis": "用户原话包含“可能”，但这是可审查候选，不应靠短语表直接拒绝。",
                        "category_rationale": "候选是否可信由 evidence 和用户确认决定，不由自然语言短语 gate 决定。",
                        "evidence_source": "user_prompt",
                        "evidence_quote": "以后记住项目包管理器",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    request = _request(tmp_path, prompt="以后记住项目包管理器", gateway=gateway)

    result = MemoryExtractor().extract(request)

    pending = CandidateQueue(request.session_path).list(status="pending")
    assert result.created_count == 1
    assert result.rejected_count == 0
    assert len(pending) == 1
    assert "unverified_claim" not in pending[0].risk_flags


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
                        "evidence_source": "user_prompt",
                        "evidence_quote": "HaAgent 使用 uv 管理依赖",
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
                        "evidence_source": "user_prompt",
                        "evidence_quote": "Pending candidates must never enter retrieval",
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
                        "evidence_source": "user_prompt",
                        "evidence_quote": "HaAgent 使用 uv 管理依赖",
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
            ModelResponse(
                "我会把这作为候选记忆等待你确认。",
                [ToolCall(name="start_memory_update", args={"reason": "用户给出长期项目事实"}, id="call_memory")],
            ),
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
                                "evidence_source": "user_prompt",
                                "evidence_quote": "HaAgent 使用 uv 管理依赖",
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
    assert any(isinstance(event, MemoryNoticeEvent) for event in events)


def test_memory_request_keeps_regular_tools_available_and_extracts_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway(
        [
            ModelResponse(
                "我会把这作为候选记忆等待你确认。",
                [ToolCall(name="start_memory_update", args={"reason": "用户给出长期身份偏好"}, id="call_memory")],
            ),
            ModelResponse("已处理。", []),
            ModelResponse(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "scope": "user",
                                "category": "user_preferences",
                                "title": "用户身份与爱好",
                                "body": "用户叫小明，爱好是唱跳rap篮球。",
                                "source_summary": "用户明确要求记住自己的名字和爱好。",
                                "basis": "用户说：我叫小明，爱好是唱跳rap篮球，记住我的爱好。",
                                "category_rationale": "这是跨 workspace 可复用的用户偏好和身份信息。",
                                "evidence_source": "user_prompt",
                                "evidence_quote": "我叫小明，爱好是唱跳rap篮球",
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

    result = session.run_prompt("我叫小明，爱好是唱跳rap篮球，记住我的爱好")

    assert result.status == "completed"
    assert result.memory_candidates_created == 1
    assert {tool["name"] for tool in gateway.calls[0]["tool_schemas"]}
    pending = CandidateQueue(session.session_path).list(status="pending")
    assert len(pending) == 1
    assert pending[0].scope == "user"
    assert pending[0].category == "user_preferences"
    records = MemoryStore(workspace_root=workspace).list_records(scope="user", category="user_preferences")
    assert all(record.source_candidate_id != pending[0].candidate_id for record in records)


def test_food_preference_memory_request_is_handled_by_extraction_not_phrase_routing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway(
        [
            ModelResponse(
                "我会把你的喜好作为候选记忆等待你确认。",
                [ToolCall(name="start_memory_update", args={"reason": "用户给出长期饮食偏好"}, id="call_memory")],
            ),
            ModelResponse("已处理。", []),
            ModelResponse(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "scope": "user",
                                "category": "user_preferences",
                                "title": "饮食喜好",
                                "body": "用户爱吃包子。",
                                "source_summary": "用户明确要求记住自己的饮食喜好。",
                                "basis": "用户说：我爱吃包子，记住我的喜好。",
                                "category_rationale": "这是跨 workspace 可复用的用户偏好。",
                                "evidence_source": "user_prompt",
                                "evidence_quote": "我爱吃包子",
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

    result = session.run_prompt("我爱吃包子，记住我的喜好")

    assert result.status == "completed"
    assert result.memory_candidates_created == 1
    assert {tool["name"] for tool in gateway.calls[0]["tool_schemas"]}
    assert CandidateQueue(session.session_path).list(status="pending")[0].title == "饮食喜好"


def test_agent_session_allows_profile_file_write_then_extracts_pending_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SequentialGateway(
        [
            ModelResponse(
                "Writing profile.",
                [
                    ToolCall(
                        name="file_write",
                        args={
                            "path": "user_profile.md",
                            "content": "# 用户档案\n\n- 名字：小明\n- 爱好：唱跳rap篮球\n",
                            "mode": "create",
                        },
                        id="call_profile",
                    ),
                    ToolCall(
                        name="start_memory_update",
                        args={"reason": "用户提供了长期个人资料"},
                        id="call_memory",
                    ),
                ],
            ),
            ModelResponse("我会把这些作为候选记忆等待你确认。", []),
            ModelResponse(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "scope": "user",
                                "category": "user_preferences",
                                "title": "用户身份与爱好",
                                "body": "用户叫小明，喜欢唱跳rap篮球。",
                                "source_summary": "用户提供了自己的名字和爱好。",
                                "basis": "用户说：请整理这个个人资料：我叫小明，喜欢唱跳rap篮球。",
                                "category_rationale": "这是跨 workspace 可复用的用户偏好和身份信息。",
                                "evidence_source": "user_prompt",
                                "evidence_quote": "我叫小明，喜欢唱跳rap篮球",
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

    result = session.run_prompt_events(
        "请整理这个个人资料：我叫小明，喜欢唱跳rap篮球",
        event_sink=events.append,
        interaction_handler=lambda request: HumanInteractionResponse(approved=True),
    )

    assert result.status == "completed"
    assert result.memory_candidates_created == 1
    assert (workspace / "user_profile.md").read_text(encoding="utf-8").startswith("# 用户档案")
    pending = CandidateQueue(session.session_path).list(status="pending")
    assert len(pending) == 1
    assert pending[0].scope == "user"
    assert pending[0].category == "user_preferences"
    store = MemoryStore(workspace_root=workspace)
    records = store.list_records(scope="user", category="user_preferences")
    assert all(record.source_candidate_id != pending[0].candidate_id for record in records)
    assert not [
        event
        for event in events
        if isinstance(event, ToolActivityEvent) and event.status == "failed"
    ]


def test_memory_cli_lists_confirms_and_rejects_candidates(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs_root = tmp_path / ".runs"
    session = AgentSession(
        workspace_root=workspace,
        runs_root=runs_root,
        model_gateway=SequentialGateway(
            [
                ModelResponse(
                    "done",
                    [ToolCall(name="start_memory_update", args={"reason": "用户给出长期项目事实"}, id="call_memory_1")],
                ),
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
                                    "evidence_source": "user_prompt",
                                    "evidence_quote": "HaAgent 使用 uv 管理依赖",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    [],
                ),
                ModelResponse(
                    "done",
                    [ToolCall(name="start_memory_update", args={"reason": "用户给出长期审核要求"}, id="call_memory_2")],
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
                                    "evidence_source": "user_prompt",
                                    "evidence_quote": "HaAgent 的长期记忆候选需要人工审核",
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

    assert list_code == 1
    assert confirm_code == 1
    assert "请运行 haagent 打开 TUI" in list_output
    assert "请运行 haagent 打开 TUI" in confirm_output
    assert CandidateQueue(session.session_path).get(candidate_id).status == "pending"

    session.run_prompt("记住这个：HaAgent 的长期记忆候选需要人工审核。")
    second_id = CandidateQueue(session.session_path).list(status="pending")[0].candidate_id
    reject_code = cli.main(["memory", "reject", second_id, "--runs-root", str(runs_root), "--reason", "not durable"])
    reject_output = capsys.readouterr().out

    assert reject_code == 1
    assert "请运行 haagent 打开 TUI" in reject_output
    assert CandidateQueue(session.session_path).get(second_id).status == "pending"
