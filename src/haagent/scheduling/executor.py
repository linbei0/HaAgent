"""
haagent/scheduling/executor.py - 计划任务隔离执行器

为每次 run 创建独立 AssistantService，复用 session/runtime 链路；
将失败映射为稳定 failure_category，并只把路径与有界摘要写回 schedule DB。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from haagent.app.assistant_service import AssistantService
from haagent.models.model_connections import (
    ModelSelection,
    ProviderProfileError,
    load_model_selection_profile,
    user_config_dir,
)
from haagent.models.types import ModelCallError
from haagent.runtime.execution.command import redact_secret_like_text
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.turn_completion import ChatTurnResult
from haagent.scheduling.coordinator import (
    RETRYABLE_CATEGORIES,
    _compute_retry_delay,
)
from haagent.scheduling.interactions import (
    UnattendedInteractionHandler,
    UnattendedInteractionRequired,
)
from haagent.scheduling.models import (
    FailureCategory,
    RunClaim,
    RunStatus,
    ScheduleDefinition,
    merge_web_tools,
)
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError

ServiceFactory = Callable[..., AssistantService]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ScheduleRunResult:
    run_id: str
    status: RunStatus
    failure_category: FailureCategory | None = None
    summary: str = ""
    session_id: str | None = None
    session_path: str | None = None
    episode_path: str | None = None


def _bounded_summary(text: str, *, limit: int = 400) -> str:
    # 安全边界：写入 schedule DB 前脱敏，再截断
    redacted, _ = redact_secret_like_text(text or "")
    normalized = " ".join(redacted.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


# 结构化 runtime failure taxonomy → schedule category（禁止依赖自由文案匹配）
_TURN_CATEGORY_MAP: dict[str, tuple[RunStatus, FailureCategory]] = {
    "User Denied Failure": ("needs_attention", "policy_denied"),
    "Guardrail Failure": ("needs_attention", "policy_denied"),
    "Verification Failure": ("failed", "verification_failed"),
    "Model Failure": ("failed", "model_transient"),
    "Model Call Failure": ("failed", "model_transient"),
    "Tool Interface Failure": ("failed", "tool_failure"),
    "Tool Argument Failure": ("failed", "tool_failure"),
}


def _map_turn_failure(result: ChatTurnResult) -> tuple[RunStatus, FailureCategory | None, str]:
    """将 ChatTurnResult 映射为 schedule run 状态与分类。"""
    category_raw = (result.failure_category or "").strip()
    reason = _bounded_summary(result.reason or result.final_response or result.status)
    status_lower = (result.status or "").lower()

    if status_lower == "cancelled":
        return "cancelled", "cancelled", reason or "cancelled"

    if status_lower == "completed":
        if result.verification_status == "failed":
            return "failed", "verification_failed", reason or "verification failed"
        return "succeeded", None, _bounded_summary(result.final_response or "ok")

    mapped = _TURN_CATEGORY_MAP.get(category_raw)
    if mapped is not None:
        return mapped[0], mapped[1], reason or category_raw
    if category_raw and category_raw != "none":
        return "failed", "internal_error", reason or category_raw
    return "failed", "internal_error", reason or status_lower or "failed"


def _map_exception(error: BaseException) -> tuple[RunStatus, FailureCategory, str]:
    # 结构化类型优先；禁止用异常消息自然语言做 runtime 决策
    if isinstance(error, UnattendedInteractionRequired):
        return (
            "needs_attention",
            "interaction_required",
            _bounded_summary(f"{error.kind}: {error.summary}"),
        )
    if isinstance(error, ProviderProfileError):
        code = getattr(error, "code", None) or getattr(error, "error_code", None)
        if code in {"api_key_missing", "credential_unavailable", "not_available"}:
            return "needs_attention", "credential_unavailable", _bounded_summary(str(error))
        if code in {"profile_missing", "profile_unavailable", "connection_missing"}:
            return "needs_attention", "profile_unavailable", _bounded_summary(str(error))
        # 回退：ProviderProfileError 默认视为 profile/credential 配置问题
        return "needs_attention", "profile_unavailable", _bounded_summary(str(error))
    if isinstance(error, ModelCallError):
        details = getattr(error, "details", None)
        retryable = bool(getattr(details, "retryable", False))
        category = getattr(details, "category", None)
        if retryable or category in {"network", "timeout", "rate_limited", "server"}:
            return "failed", "model_transient", _bounded_summary(str(error))
        if category in {"auth", "quota_exhausted"}:
            return "needs_attention", "credential_unavailable", _bounded_summary(str(error))
        return "failed", "model_permanent", _bounded_summary(str(error))
    if isinstance(error, FileNotFoundError):
        return "needs_attention", "workspace_unavailable", _bounded_summary(str(error))
    if isinstance(error, NotADirectoryError):
        return "needs_attention", "workspace_unavailable", _bounded_summary(str(error))
    if isinstance(error, OSError):
        return "needs_attention", "workspace_unavailable", _bounded_summary(str(error))
    # AssistantServiceError：优先读结构化 code
    code = getattr(error, "code", None)
    if code in {"credential_unavailable", "api_key_missing"}:
        return "needs_attention", "credential_unavailable", _bounded_summary(str(error))
    if code in {"workspace_unavailable", "workspace_missing"}:
        return "needs_attention", "workspace_unavailable", _bounded_summary(str(error))
    if code in {"profile_unavailable", "connection_missing"}:
        return "needs_attention", "profile_unavailable", _bounded_summary(str(error))
    return "failed", "internal_error", _bounded_summary(str(error))


class ScheduledRunExecutor:
    """隔离执行计划 run；不复用 TUI 当前 session。"""

    def __init__(
        self,
        store: ScheduleStore,
        *,
        service_factory: ServiceFactory | None = None,
        runs_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Clock | None = None,
        config_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._service_factory = service_factory or self._default_service_factory
        self._runs_root = runs_root
        self._environ = environ
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._config_dir = config_dir
        self._active_sessions: dict[str, AgentSession] = {}
        self._cancel_requested: set[str] = set()

    def _default_service_factory(self, **kwargs: Any) -> AssistantService:
        return AssistantService(
            workspace_root=kwargs["workspace_root"],
            runs_root=kwargs.get("runs_root", Path(".runs")),
            environ=kwargs.get("environ"),
            gateway_factory=kwargs.get("gateway_factory"),
            session_cls=kwargs.get("session_cls", AgentSession),
            enable_web=kwargs.get("enable_web", False),
            max_turns=kwargs.get("max_turns"),
        )

    def request_cancel(self, run_id: str) -> None:
        self._cancel_requested.add(run_id)
        session = self._active_sessions.get(run_id)
        if session is not None and hasattr(session, "cancel_current_run"):
            session.cancel_current_run()

    def is_cancel_requested(self, run_id: str) -> bool:
        if run_id in self._cancel_requested:
            return True
        try:
            run = self._store.get_run(run_id)
        except Exception:
            return False
        return bool(run is not None and run.cancellation_requested)

    def execute(self, claim: RunClaim) -> ScheduleRunResult:
        now = self._clock()
        run_id = claim.run_id
        run = self._store.get_run(run_id)
        if run is None:
            raise ScheduleStoreError("not_found", f"run 不存在: {run_id}")

        # 入口同时校验 worker + attempt；禁止同 worker 旧 attempt 延迟启动冒充新 claim
        if (
            run.status != "running"
            or run.worker_id != claim.worker_id
            or int(run.attempt_count or 0) != int(claim.attempt)
        ):
            raise ScheduleStoreError(
                "stale_execute",
                (
                    f"run {run_id} claim token 失效 "
                    f"(status={run.status!r}, worker={run.worker_id!r}/"
                    f"{claim.worker_id!r}, attempt={run.attempt_count!r}/"
                    f"{claim.attempt!r})"
                ),
            )

        if run.cancellation_requested or run_id in self._cancel_requested:
            return self._finish(
                run_id,
                status="cancelled",
                now=now,
                summary="cancelled",
                failure_category="cancelled",
                claim=claim,
            )

        definition = self._store.get_revision(run.schedule_id, run.schedule_revision)
        if definition is None:
            definition = self._store.get(run.schedule_id)
        if definition is None:
            return self._finish(
                run_id,
                status="needs_attention",
                now=now,
                summary="schedule definition missing",
                failure_category="schedule_invalid",
                claim=claim,
            )

        # revision 漂移：禁止静默套用新 prompt
        current = self._store.get(run.schedule_id)
        if current is not None and current.revision != run.schedule_revision:
            if definition is None:
                return self._finish(
                    run_id,
                    status="needs_attention",
                    now=now,
                    summary="schedule revision mismatch",
                    failure_category="schedule_invalid",
                    claim=claim,
                )

        try:
            return self._execute_definition(
                run_id,
                definition,
                now=now,
                claim=claim,
            )
        except Exception as error:
            status, category, summary = _map_exception(error)
            return self._finish_with_retry(
                run_id,
                status=status,
                now=self._clock(),
                summary=summary,
                failure_category=category,
                needs_attention_reason=summary if status == "needs_attention" else None,
                claim=claim,
            )
        finally:
            self._active_sessions.pop(run_id, None)
            self._cancel_requested.discard(run_id)

    def _execute_definition(
        self,
        run_id: str,
        definition: ScheduleDefinition,
        *,
        now: datetime,
        claim: RunClaim,
    ) -> ScheduleRunResult:
        workspace = definition.workspace_root
        if not workspace.is_absolute():
            return self._finish(
                run_id,
                status="needs_attention",
                now=now,
                summary="workspace_root 必须是绝对路径",
                failure_category="workspace_unavailable",
                claim=claim,
            )
        resolved = workspace.resolve()
        if not resolved.exists() or not resolved.is_dir():
            # 失败边界：缺失 workspace 不自动创建
            return self._finish(
                run_id,
                status="needs_attention",
                now=now,
                summary=f"workspace 不可用: {resolved}",
                failure_category="workspace_unavailable",
                claim=claim,
            )

        environ = dict(self._environ) if self._environ is not None else dict(os.environ)
        runs_root = self._runs_root if self._runs_root is not None else resolved / ".runs"

        # 预检 connection/credential（在创建 session 前给出稳定分类）
        try:
            selection = ModelSelection(connection_id=definition.connection_id, model=definition.model)
            load_model_selection_profile(
                selection,
                environ=environ,
                config_dir=self._config_dir or user_config_dir(),
            )
        except ProviderProfileError as error:
            status, category, summary = _map_exception(error)
            return self._finish(
                run_id,
                status=status,
                now=now,
                summary=summary,
                failure_category=category,
                needs_attention_reason=summary if status == "needs_attention" else None,
                claim=claim,
            )

        service = self._service_factory(
            workspace_root=resolved,
            runs_root=runs_root,
            environ=environ,
            enable_web=definition.web_enabled,
        )
        # 计划指定模型连接，不继承 TUI 当前选择之外的偶然状态
        service._context.pending_model_selection = ModelSelection(
            connection_id=definition.connection_id,
            model=definition.model,
        )
        service._context.enable_web = definition.web_enabled
        if self._runs_root is not None:
            service._context.runs_root = Path(self._runs_root)

        session_status = self._open_destination(service, definition)
        _ = session_status
        session = service._context.session
        if session is None:
            return self._finish(
                run_id,
                status="failed",
                now=self._clock(),
                summary="session 创建失败",
                failure_category="internal_error",
                claim=claim,
            )

        # 工具快照：计划定义覆盖 session 默认工具集
        self._apply_tool_overrides(session, definition)
        self._active_sessions[run_id] = session

        if self.is_cancel_requested(run_id):
            return self._finish(
                run_id,
                status="cancelled",
                now=self._clock(),
                summary="cancelled",
                failure_category="cancelled",
                session_id=session.session_id,
                session_path=str(session.session_path.resolve()),
                claim=claim,
            )

        handler = UnattendedInteractionHandler()
        try:
            turn_result = service.sessions.run_prompt_events(
                definition.prompt,
                interaction_handler=handler,
                include_session_events=False,
            )
        except UnattendedInteractionRequired as error:
            status, category, summary = _map_exception(error)
            return self._finish(
                run_id,
                status=status,
                now=self._clock(),
                summary=summary,
                failure_category=category,
                needs_attention_reason=summary,
                session_id=session.session_id,
                session_path=str(session.session_path.resolve()),
                claim=claim,
            )
        except Exception as error:
            status, category, summary = _map_exception(error)
            return self._finish_with_retry(
                run_id,
                status=status,
                now=self._clock(),
                summary=summary,
                failure_category=category,
                needs_attention_reason=summary if status == "needs_attention" else None,
                session_id=session.session_id,
                session_path=str(session.session_path.resolve()),
                claim=claim,
            )

        if self.is_cancel_requested(run_id):
            return self._finish(
                run_id,
                status="cancelled",
                now=self._clock(),
                summary="cancelled",
                failure_category="cancelled",
                session_id=session.session_id,
                session_path=str(session.session_path.resolve()),
                episode_path=str(Path(turn_result.episode_path).resolve())
                if turn_result.episode_path
                else None,
                claim=claim,
            )

        run_status, category, summary = _map_turn_failure(turn_result)
        episode_path = str(Path(turn_result.episode_path).resolve())
        return self._finish_with_retry(
            run_id,
            status=run_status,
            now=self._clock(),
            summary=summary,
            failure_category=category,
            needs_attention_reason=summary if run_status == "needs_attention" else None,
            session_id=turn_result.session_id or session.session_id,
            session_path=str(session.session_path.resolve()),
            episode_path=episode_path,
            claim=claim,
        )

    def _open_destination(
        self, service: AssistantService, definition: ScheduleDefinition
    ):
        if definition.destination_kind == "resume_session":
            path = definition.destination_session_path
            if path is None:
                raise FileNotFoundError("resume_session 缺少 destination_session_path")
            if not Path(path).exists():
                raise FileNotFoundError(f"session 不存在: {path}")
            return service.sessions.resume(path)
        return service.sessions.create()

    def _apply_tool_overrides(
        self, session: AgentSession, definition: ScheduleDefinition
    ) -> None:
        # 安全边界：resume 也使用计划工具快照，不继承 session 更宽历史权限
        # web_enabled 必须真正放开联网工具；否则模型只能看到 file_* 只读集
        allowed = list(
            merge_web_tools(definition.allowed_tools, web_enabled=definition.web_enabled)
        )
        if hasattr(session, "enable_web"):
            session.enable_web = definition.web_enabled
        if hasattr(session, "_allowed_tools_override"):
            session._allowed_tools_override = allowed
        if hasattr(session, "_approval_allowed_tools_override"):
            session._approval_allowed_tools_override = list(definition.approval_allowed_tools)
        if hasattr(session, "_approved_tools_override"):
            session._approved_tools_override = list(definition.approved_tools)

    def _finish_with_retry(
        self,
        run_id: str,
        *,
        status: RunStatus,
        now: datetime,
        summary: str = "",
        failure_category: FailureCategory | None = None,
        needs_attention_reason: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
        episode_path: str | None = None,
        claim: RunClaim,
    ) -> ScheduleRunResult:
        # 生产链路接入 retry：可重试类别按 schedule retry_policy 进入 retry_wait
        if (
            status == "failed"
            and failure_category is not None
            and failure_category in RETRYABLE_CATEGORIES
        ):
            run = self._store.get_run(run_id)
            if run is not None and not run.cancellation_requested:
                definition = self._store.get_revision(
                    run.schedule_id, run.schedule_revision
                )
                if definition is None:
                    definition = self._store.get(run.schedule_id)
                # attempt 用 claim token，避免 live DB 已被 reclaimed 时算错
                attempt = int(claim.attempt)
                if (
                    definition is not None
                    and attempt < definition.retry_policy.max_attempts
                ):
                    delay = _compute_retry_delay(definition.retry_policy, attempt)
                    retry_at = now + timedelta(seconds=delay)
                    return self._finish(
                        run_id,
                        status="retry_wait",
                        now=now,
                        summary=summary,
                        failure_category=failure_category,
                        needs_attention_reason=None,
                        session_id=session_id,
                        session_path=session_path,
                        episode_path=episode_path,
                        retry_at_utc=retry_at,
                        claim=claim,
                    )
        return self._finish(
            run_id,
            status=status,
            now=now,
            summary=summary,
            failure_category=failure_category,
            needs_attention_reason=needs_attention_reason,
            session_id=session_id,
            session_path=session_path,
            episode_path=episode_path,
            claim=claim,
        )

    def _finish(
        self,
        run_id: str,
        *,
        status: RunStatus,
        now: datetime,
        summary: str = "",
        failure_category: FailureCategory | None = None,
        needs_attention_reason: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
        episode_path: str | None = None,
        retry_at_utc: datetime | None = None,
        claim: RunClaim,
    ) -> ScheduleRunResult:
        # fencing：必须用 execute 入口捕获的 claim token，禁止再读 live worker/attempt
        # （否则 stale worker 会读到 reclaimer 的 id 并成功覆盖）
        finished = self._store.finish_run(
            run_id,
            status=status,
            now=now,
            summary=_bounded_summary(summary),
            failure_category=failure_category,
            failure_reason=_bounded_summary(summary) if failure_category else None,
            needs_attention_reason=needs_attention_reason,
            session_id=session_id,
            session_path=session_path,
            episode_path=episode_path,
            retry_at_utc=retry_at_utc,
            expected_worker_id=claim.worker_id,
            expected_attempt=claim.attempt,
        )
        # 更新 last_run
        run = finished
        try:
            self._store.set_last_run(
                run.schedule_id, last_run_at_utc=now, now=now
            )
        except Exception:
            pass
        return ScheduleRunResult(
            run_id=run_id,
            status=finished.status,
            failure_category=finished.failure_category,
            summary=finished.summary,
            session_id=finished.session_id,
            session_path=finished.session_path,
            episode_path=finished.episode_path,
        )
