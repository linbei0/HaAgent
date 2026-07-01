"""
haagent/app/memory_usecases.py - 记忆类应用用例

集中封装 AssistantService 的 memory candidate 查询与确认流程。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haagent.memory import CandidateQueueError, MemoryStoreError
from haagent.memory.governance import MemoryGovernanceError

if TYPE_CHECKING:
    from haagent.app.assistant_service import AssistantService


def list_memory_candidates(service: "AssistantService", status: str | None="pending"):
    queue = service._memory_queue()
    return queue.list(status=status)


def get_memory_candidate(service: "AssistantService", candidate_id: str):
    return service._memory_queue().get(candidate_id)


def confirm_memory_candidate(service: "AssistantService", candidate_id: str):
    try:
        return service._memory_store().confirm_candidate(
            service._memory_queue(),
            candidate_id,
            actor="user",
        )
    except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
        raise service.error_cls(str(error)) from error


def reject_memory_candidate(service: "AssistantService", candidate_id: str, reason: str):
    try:
        return service._memory_store().reject_candidate(
            service._memory_queue(),
            candidate_id,
            reason=reason,
            actor="user",
        )
    except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
        raise service.error_cls(str(error)) from error
