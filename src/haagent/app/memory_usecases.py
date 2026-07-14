"""
haagent/app/memory_usecases.py - Memory candidate 审查应用 Module

选择当前 workspace 的 candidate queue，并执行确定性的确认与拒绝流程。
"""

from __future__ import annotations

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import AssistantServiceError
from haagent.memory import CandidateQueue, CandidateQueueError, MemoryCandidate, MemoryRecord, MemoryStore, MemoryStoreError
from haagent.memory.governance import MemoryGovernanceError
from haagent.runtime.session.package import find_latest_session


class AssistantMemory:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context

    def list_candidates(self, status: str | None = "pending") -> list[MemoryCandidate]:
        return self._queue().list(status=status)

    def confirm_candidate(self, candidate_id: str) -> MemoryRecord:
        try:
            return self._store().confirm_candidate(self._queue(), candidate_id, actor="user")
        except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
            raise AssistantServiceError(str(error)) from error

    def reject_candidate(self, candidate_id: str, reason: str) -> MemoryCandidate:
        try:
            return self._store().reject_candidate(
                self._queue(),
                candidate_id,
                reason=reason,
                actor="user",
            )
        except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
            raise AssistantServiceError(str(error)) from error

    def _queue(self) -> CandidateQueue:
        if self._context.session is None:
            latest = find_latest_session(self._context.runs_root, self._context.workspace_root)
            if latest is None:
                raise AssistantServiceError("当前 workspace 没有可审查的 memory candidate session")
            session_path = latest.session_path
        else:
            session_path = self._context.session.session_path
        return CandidateQueue(session_path)

    def _store(self) -> MemoryStore:
        return MemoryStore(workspace_root=self._context.workspace_root)
