"""
src/haagent/runtime/session/history.py - 当前会话历史证据检索

从 session package 的 turn 索引和 typed episode 中按需读取对话证据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from haagent.runtime.contracts.task import TaskLoadError, load_task
from haagent.runtime.episodes.validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.session.package import ChatSessionError, read_session_turns

DEFAULT_RESULT_LIMIT = 3
MAX_RESULT_LIMIT = 5
REQUEST_EVIDENCE_CHAR_LIMIT = 1_000
ASSISTANT_RESPONSE_CHAR_LIMIT = 4_000
_TRUNCATION_SUFFIX = "... [truncated]"
_TRUNCATED_BEFORE = "... [truncated before]\n"
_TRUNCATED_AFTER = "\n... [truncated after]"
_TRUNCATED_MIDDLE = "\n... [truncated middle] ...\n"
_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./\\-]+")
_CJK_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


class SessionHistoryError(RuntimeError):
    """当前 session 的历史证据无法安全读取时抛出。"""


@dataclass(frozen=True)
class SessionHistoryItem:
    """一个可回注模型的当前会话对话证据。"""

    turn_index: int
    request: str
    summary: str
    assistant_response: str
    status: str
    verification_status: str
    episode_ref: str

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "request": self.request,
            "summary": self.summary,
            "assistant_response": self.assistant_response,
            "status": self.status,
            "verification_status": self.verification_status,
            "episode_ref": self.episode_ref,
        }


@dataclass(frozen=True)
class SessionHistoryResult:
    """有界且可审计的检索返回。"""

    results: list[SessionHistoryItem]
    diagnostics: dict[str, object]

    def to_tool_result(self) -> dict[str, object]:
        return {
            "status": "success",
            "results": [item.to_dict() for item in self.results],
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class _SearchTerm:
    value: str
    weight: int
    kind: str


@dataclass(frozen=True)
class _TurnEvidence:
    turn: dict[str, object]
    request: str
    assistant_response: str | None


@dataclass(frozen=True)
class _TurnMatch:
    score: int
    fields: tuple[str, ...]
    terms: tuple[str, ...]
    request_terms: tuple[str, ...]
    assistant_terms: tuple[str, ...]


class SessionHistoryRetriever:
    """检索一个明确注入的 session path，绝不枚举其他会话。"""

    def __init__(self, session_path: Path, *, runs_root: Path | None = None) -> None:
        self._session_path = session_path.resolve()
        self._runs_root = runs_root.resolve() if runs_root is not None else None

    def search(self, query: str, *, limit: int = DEFAULT_RESULT_LIMIT) -> SessionHistoryResult:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise SessionHistoryError("session history query must be non-empty")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_RESULT_LIMIT:
            raise SessionHistoryError(f"session history limit must be between 1 and {MAX_RESULT_LIMIT}")
        try:
            turns = read_session_turns(self._session_path)
        except ChatSessionError as error:
            raise SessionHistoryError(str(error)) from error

        terms = _search_terms(normalized_query)
        episode_read_turns: set[int] = set()
        episode_failures: list[dict[str, object]] = []
        skipped_turns: list[int] = []
        ranked: list[tuple[int, int, _TurnEvidence, _TurnMatch]] = []
        for turn in turns:
            evidence = self._expand_turn_evidence(
                turn,
                episode_read_turns=episode_read_turns,
                episode_failures=episode_failures,
            )
            turn_index = int(turn["turn_index"])
            if evidence.assistant_response is None:
                skipped_turns.append(turn_index)
                continue
            match = _score_turn(terms, evidence)
            if match is not None:
                ranked.append((match.score, turn_index, evidence, match))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        selected = ranked[:limit]
        results = [self._item_from_evidence(evidence, match) for _score, _index, evidence, match in selected]
        selected_matches = {
            str(int(evidence.turn["turn_index"])): {
                "fields": list(match.fields),
                "terms": list(match.terms),
            }
            for _score, _index, evidence, match in selected
        }
        return SessionHistoryResult(
            results=results,
            diagnostics={
                "scope": "current_session",
                "query": normalized_query,
                "turn_count": len(turns),
                "matched_turn_count": len(ranked),
                "selected_turns": [item.turn_index for item in results],
                "result_limit": limit,
                "result_budget_chars": {
                    "request": REQUEST_EVIDENCE_CHAR_LIMIT,
                    "assistant_response": ASSISTANT_RESPONSE_CHAR_LIMIT,
                },
                "episode_read_turns": sorted(episode_read_turns),
                "failed_episode_turns": sorted(
                    {int(failure["turn_index"]) for failure in episode_failures}
                ),
                "episode_failures": episode_failures,
                "skipped_turns": skipped_turns,
                "match_reasons": selected_matches,
            },
        )

    def _expand_turn_evidence(
        self,
        turn: dict[str, object],
        *,
        episode_read_turns: set[int],
        episode_failures: list[dict[str, object]],
    ) -> _TurnEvidence:
        turn_index = int(turn["turn_index"])
        request = str(turn["request"])
        assistant_response = turn.get("assistant_display_text")
        needs_request = request.endswith(_TRUNCATION_SUFFIX)
        needs_response = not isinstance(assistant_response, str) or assistant_response.endswith(
            _TRUNCATION_SUFFIX,
        )
        if not needs_request and not needs_response:
            return _TurnEvidence(turn=turn, request=request, assistant_response=assistant_response)

        episode_read_turns.add(turn_index)
        try:
            episode_path = self._episode_path(str(turn["episode_path"]))
        except SessionHistoryError as error:
            self._record_episode_failure(episode_failures, turn_index, "episode_path", error)
            return _TurnEvidence(
                turn=turn,
                request=request,
                assistant_response=assistant_response if isinstance(assistant_response, str) else None,
            )

        if needs_request:
            try:
                request = load_task(episode_path / "task.yaml").goal
            except (OSError, TaskLoadError, yaml.YAMLError) as error:
                self._record_episode_failure(episode_failures, turn_index, "request", error)
        if needs_response:
            try:
                assistant_response = load_inspect_episode_package(episode_path).final_response_text()
            except (EpisodeValidationError, OSError) as error:
                self._record_episode_failure(
                    episode_failures,
                    turn_index,
                    "assistant_response",
                    error,
                )
        return _TurnEvidence(
            turn=turn,
            request=request,
            assistant_response=assistant_response if isinstance(assistant_response, str) else None,
        )

    def _episode_path(self, episode_ref: str) -> Path:
        episode_path = Path(episode_ref)
        if episode_path.is_absolute():
            return episode_path
        if self._runs_root is None:
            raise SessionHistoryError(
                f"relative history episode requires runs_root: {episode_ref}",
            )
        runs_candidate = (self._runs_root / episode_path).resolve()
        if runs_candidate.exists():
            return runs_candidate
        # 旧 package 可能保存相对启动 cwd 的完整 runs 路径；仅在该路径真实存在时兼容读取。
        cwd_candidate = episode_path.resolve()
        return cwd_candidate if cwd_candidate.exists() else runs_candidate

    @staticmethod
    def _record_episode_failure(
        failures: list[dict[str, object]],
        turn_index: int,
        field: str,
        error: Exception,
    ) -> None:
        # 单个旧 episode 损坏不能遮蔽其他有效历史；失败必须进入诊断而不是静默降级。
        failures.append(
            {
                "turn_index": turn_index,
                "field": field,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    @staticmethod
    def _item_from_evidence(evidence: _TurnEvidence, match: _TurnMatch) -> SessionHistoryItem:
        turn = evidence.turn
        assistant_response = evidence.assistant_response or ""
        return SessionHistoryItem(
            turn_index=int(turn["turn_index"]),
            request=_bounded_excerpt(
                evidence.request,
                matched_terms=match.request_terms,
                limit=REQUEST_EVIDENCE_CHAR_LIMIT,
            ),
            summary=str(turn["summary"]),
            assistant_response=_bounded_excerpt(
                assistant_response,
                matched_terms=match.assistant_terms,
                limit=ASSISTANT_RESPONSE_CHAR_LIMIT,
            ),
            status=str(turn["status"]),
            verification_status=str(turn["verification_status"]),
            episode_ref=str(turn["episode_path"]),
        )


def _search_terms(value: str) -> tuple[_SearchTerm, ...]:
    normalized = " ".join(value.lower().split())
    terms: dict[tuple[str, str], _SearchTerm] = {}

    def add(term: str, weight: int, kind: str) -> None:
        if term:
            terms[(kind, term)] = _SearchTerm(term, weight, kind)

    if len(normalized) >= 2:
        add(normalized, 12, "exact")
    for token in _ASCII_TOKEN_PATTERN.findall(normalized):
        if len(token) >= 2:
            add(token, 5, "ascii")
    for run in _CJK_RUN_PATTERN.findall(normalized):
        if len(run) <= 3:
            add(run, 6, "cjk_phrase")
            continue
        for size, weight, kind in ((3, 4, "cjk_trigram"), (2, 1, "cjk_bigram")):
            for index in range(len(run) - size + 1):
                add(run[index : index + size], weight, kind)
    return tuple(terms.values())


def _score_turn(terms: tuple[_SearchTerm, ...], evidence: _TurnEvidence) -> _TurnMatch | None:
    fields = {
        "request": evidence.request,
        "summary": str(evidence.turn["summary"]),
        "assistant_response": evidence.assistant_response or "",
    }
    field_weights = {"request": 3, "summary": 4, "assistant_response": 2}
    matched_by_field: dict[str, list[_SearchTerm]] = {}
    for field_name, value in fields.items():
        normalized = " ".join(value.lower().split())
        matched_by_field[field_name] = [term for term in terms if term.value in normalized]

    all_matches = [term for matches in matched_by_field.values() for term in matches]
    distinct_bigrams = {term.value for term in all_matches if term.kind == "cjk_bigram"}
    strong_match = any(term.kind != "cjk_bigram" for term in all_matches) or len(distinct_bigrams) >= 2
    if not all_matches or not strong_match:
        return None

    score = sum(
        term.weight * field_weights[field_name]
        for field_name, matches in matched_by_field.items()
        for term in matches
    )
    return _TurnMatch(
        score=score,
        fields=tuple(name for name, matches in matched_by_field.items() if matches),
        terms=tuple(dict.fromkeys(term.value for term in all_matches)),
        request_terms=tuple(term.value for term in matched_by_field["request"]),
        assistant_terms=tuple(term.value for term in matched_by_field["assistant_response"]),
    )


def _bounded_excerpt(value: str, *, matched_terms: tuple[str, ...], limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized

    lower = normalized.lower()
    anchors = [
        (len(term), lower.find(term))
        for term in matched_terms
        if term and lower.find(term) >= 0
    ]
    if not anchors:
        keep = limit - len(_TRUNCATED_MIDDLE)
        head = keep // 2
        tail = keep - head
        return f"{normalized[:head].rstrip()}{_TRUNCATED_MIDDLE}{normalized[-tail:].lstrip()}"

    _term_length, anchor = max(anchors)
    content_budget = limit - len(_TRUNCATED_BEFORE) - len(_TRUNCATED_AFTER)
    start = max(0, anchor - content_budget // 2)
    end = min(len(normalized), start + content_budget)
    start = max(0, end - content_budget)
    prefix = _TRUNCATED_BEFORE if start > 0 else ""
    suffix = _TRUNCATED_AFTER if end < len(normalized) else ""
    available = limit - len(prefix) - len(suffix)
    end = min(len(normalized), start + available)
    return f"{prefix}{normalized[start:end].strip()}{suffix}"[:limit]
