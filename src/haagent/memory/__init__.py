"""
haagent/memory/__init__.py - 长期记忆公共入口

导出 Memory System v1 第一阶段的候选队列、存储服务、schema 和治理异常。
"""

from haagent.memory.candidates import CandidateQueue, CandidateQueueError
from haagent.memory.governance import (
    MemoryConflictError,
    MemoryDuplicateError,
    MemoryGovernanceError,
)
from haagent.memory.schema import (
    CandidateEvidence,
    MemoryAuditEvent,
    MemoryCandidate,
    MemoryIndex,
    MemoryIndexItem,
    MemoryRecord,
    MemoryTombstone,
)
from haagent.memory.retrieval import (
    MemoryRetrievalBudget,
    MemoryRetrievalRequest,
    MemoryRetrievalResult,
    MemoryRetriever,
    RetrievedMemory,
)
from haagent.memory.store import MemoryStore, MemoryStoreError

__all__ = [
    "CandidateEvidence",
    "CandidateQueue",
    "CandidateQueueError",
    "MemoryAuditEvent",
    "MemoryCandidate",
    "MemoryConflictError",
    "MemoryDuplicateError",
    "MemoryGovernanceError",
    "MemoryIndex",
    "MemoryIndexItem",
    "MemoryRetrievalBudget",
    "MemoryRetrievalRequest",
    "MemoryRetrievalResult",
    "MemoryRetriever",
    "MemoryRecord",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryTombstone",
    "RetrievedMemory",
]
