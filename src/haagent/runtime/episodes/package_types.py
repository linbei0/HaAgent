"""
src/haagent/runtime/episodes/package_types.py - typed EpisodePackage 与 record codecs

磁盘仍是多文件 package；本模块只提供校验后的 typed Interface，
字段变更集中在 codec，inspect/export/eval 不再散落字符串键。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


def _require_mapping(raw: object, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    return raw


def _require_str(raw: object, label: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a string")
    return raw


def _optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("expected string or null")
    return raw


def _require_bool(raw: object, label: str) -> bool:
    # 禁止 bool("false") / bool(1) 类宽松转换；仅接受真正的 bool。
    if not isinstance(raw, bool):
        raise ValueError(f"{label} must be a boolean")
    return raw


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_version: str
    created_at: str
    task_path: str
    status: str
    provider: str | None
    workspace_root: str | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EpisodeMetadata:
        return cls(
            episode_version=_require_str(raw.get("episode_version"), "episode_version"),
            created_at=_require_str(raw.get("created_at"), "created_at"),
            task_path=_require_str(raw.get("task_path"), "task_path"),
            status=_require_str(raw.get("status"), "status"),
            provider=_optional_str(raw.get("provider")),
            workspace_root=_optional_str(raw.get("workspace_root")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_version": self.episode_version,
            "created_at": self.created_at,
            "task_path": self.task_path,
            "status": self.status,
            "provider": self.provider,
            "workspace_root": self.workspace_root,
        }


@dataclass(frozen=True)
class FailureDetail:
    category: str
    stage: str
    evidence: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureDetail:
        return cls(
            category=_require_str(raw.get("category"), "failure.category"),
            stage=_require_str(raw.get("stage"), "failure.stage"),
            evidence=_require_str(raw.get("evidence"), "failure.evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "stage": self.stage,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FailureRecord:
    status: Literal["success", "failed"]
    failure: FailureDetail | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureRecord:
        status = _require_str(raw.get("status"), "failure.status")
        if status not in {"success", "failed"}:
            raise ValueError(f"failure.status is invalid: {status}")
        failure_raw = raw.get("failure")
        if status == "success":
            if failure_raw is not None:
                raise ValueError("success failure record must set failure=null")
            return cls(status="success", failure=None)
        if not isinstance(failure_raw, Mapping):
            raise ValueError("failed failure record requires failure object")
        return cls(status="failed", failure=FailureDetail.from_dict(failure_raw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    @property
    def category(self) -> str | None:
        return None if self.failure is None else self.failure.category

    @property
    def stage(self) -> str | None:
        return None if self.failure is None else self.failure.stage

    @property
    def evidence(self) -> str | None:
        return None if self.failure is None else self.failure.evidence


@dataclass(frozen=True)
class ApprovalRecord:
    required: bool
    status: str
    reason: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ApprovalRecord:
        return cls(
            required=_require_bool(raw.get("required"), "approval.required"),
            status=_require_str(raw.get("status"), "approval.status"),
            reason=_require_str(raw.get("reason"), "approval.reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyRecord:
    tool_name: str
    risk_level: str
    action: str
    reason: str
    approval: ApprovalRecord

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PolicyRecord:
        approval_raw = raw.get("approval")
        if not isinstance(approval_raw, Mapping):
            raise ValueError("policy.approval must be an object")
        return cls(
            tool_name=_require_str(raw.get("tool_name"), "policy.tool_name"),
            risk_level=_require_str(raw.get("risk_level"), "policy.risk_level"),
            action=_require_str(raw.get("action"), "policy.action"),
            reason=_require_str(raw.get("reason"), "policy.reason"),
            approval=ApprovalRecord.from_dict(approval_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "risk_level": self.risk_level,
            "action": self.action,
            "reason": self.reason,
            "approval": self.approval.to_dict(),
        }


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    status: str
    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    policy: PolicyRecord | None = None
    path_policy: dict[str, Any] | None = None
    guardrail: dict[str, Any] | None = None
    duration_seconds: float | None = None
    attempt_count: int = 1
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    recovered_after_retry: bool = False
    duplicate_suppressed: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ToolCallRecord:
        policy_raw = raw.get("policy")
        policy = PolicyRecord.from_dict(policy_raw) if isinstance(policy_raw, Mapping) else None
        error = raw.get("error")
        result = raw.get("result")
        args = raw.get("args")
        path_policy = raw.get("path_policy")
        guardrail = raw.get("guardrail")
        duration = raw.get("duration_seconds")
        retry_events = raw.get("retry_events")
        return cls(
            tool_name=_require_str(raw.get("tool_name"), "tool_name"),
            status=_require_str(raw.get("status"), "status"),
            args=dict(args) if isinstance(args, Mapping) else None,
            result=dict(result) if isinstance(result, Mapping) else None,
            error=dict(error) if isinstance(error, Mapping) else None,
            policy=policy,
            path_policy=dict(path_policy) if isinstance(path_policy, Mapping) else None,
            guardrail=dict(guardrail) if isinstance(guardrail, Mapping) else None,
            duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
            attempt_count=int(raw.get("attempt_count", 1)) if isinstance(raw.get("attempt_count", 1), int) else 1,
            retry_events=[dict(item) for item in retry_events if isinstance(item, Mapping)]
            if isinstance(retry_events, list)
            else [],
            recovered_after_retry=raw.get("recovered_after_retry") is True,
            duplicate_suppressed=raw.get("duplicate_suppressed") is True,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_name": self.tool_name,
            "status": self.status,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "policy": None if self.policy is None else self.policy.to_dict(),
            "path_policy": self.path_policy,
            "guardrail": self.guardrail,
            "duration_seconds": self.duration_seconds,
            "attempt_count": self.attempt_count,
            "retry_events": list(self.retry_events),
            "recovered_after_retry": self.recovered_after_retry,
            "duplicate_suppressed": self.duplicate_suppressed,
        }
        return payload

    @property
    def error_type(self) -> str | None:
        if self.error is None:
            return None
        value = self.error.get("type")
        return str(value) if value is not None else None

    @property
    def error_message(self) -> str:
        if self.error is None:
            return ""
        return str(self.error.get("message", ""))

    def policy_not_evaluated(self) -> bool:
        return (
            self.status == "error"
            and self.error_type in {"tool_not_allowed", "unknown_tool", "tool_call_skipped"}
        )


@dataclass(frozen=True)
class EnvironmentModel:
    provider: str | None = None
    model: str | None = None
    endpoint: str | None = None
    base_url: str | None = None
    profile_name: str | None = None
    request_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EnvironmentModel:
        if not isinstance(raw, Mapping):
            return cls()
        request_config = raw.get("request_config")
        return cls(
            provider=_optional_str(raw.get("provider")),
            model=_optional_str(raw.get("model")),
            endpoint=_optional_str(raw.get("endpoint")),
            base_url=_optional_str(raw.get("base_url")),
            profile_name=_optional_str(raw.get("profile_name")),
            request_config=dict(request_config) if isinstance(request_config, Mapping) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "profile_name": self.profile_name,
            "request_config": self.request_config,
        }


@dataclass(frozen=True)
class EnvironmentTools:
    allowed_tool_count: int | None = None
    registry_tool_count: int | None = None
    allowed_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EnvironmentTools:
        if not isinstance(raw, Mapping):
            return cls()
        tools = raw.get("allowed_tools")
        return cls(
            allowed_tool_count=raw.get("allowed_tool_count") if isinstance(raw.get("allowed_tool_count"), int) else None,
            registry_tool_count=(
                raw.get("registry_tool_count") if isinstance(raw.get("registry_tool_count"), int) else None
            ),
            allowed_tools=[str(item) for item in tools] if isinstance(tools, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tool_count": self.allowed_tool_count,
            "registry_tool_count": self.registry_tool_count,
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass(frozen=True)
class EnvironmentRecord:
    environment_schema_version: str | None = None
    created_at: str | None = None
    workspace_root: str | None = None
    python: str | None = None
    platform: str | None = None
    process: dict[str, Any] = field(default_factory=dict)
    haagent: dict[str, Any] = field(default_factory=dict)
    model: EnvironmentModel = field(default_factory=EnvironmentModel)
    tools: EnvironmentTools = field(default_factory=EnvironmentTools)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EnvironmentRecord:
        if not isinstance(raw, Mapping) or not raw:
            return cls()
        process = raw.get("process")
        haagent = raw.get("haagent")
        model_raw = raw.get("model")
        tools_raw = raw.get("tools")
        return cls(
            environment_schema_version=_optional_str(raw.get("environment_schema_version")),
            created_at=_optional_str(raw.get("created_at")),
            workspace_root=_optional_str(raw.get("workspace_root")),
            python=_optional_str(raw.get("python")),
            platform=_optional_str(raw.get("platform")),
            process=dict(process) if isinstance(process, Mapping) else {},
            haagent=dict(haagent) if isinstance(haagent, Mapping) else {},
            model=EnvironmentModel.from_dict(model_raw if isinstance(model_raw, Mapping) else None),
            tools=EnvironmentTools.from_dict(tools_raw if isinstance(tools_raw, Mapping) else None),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return {
            "environment_schema_version": self.environment_schema_version,
            "created_at": self.created_at,
            "workspace_root": self.workspace_root,
            "python": self.python,
            "platform": self.platform,
            "process": dict(self.process),
            "haagent": dict(self.haagent),
            "model": self.model.to_dict(),
            "tools": self.tools.to_dict(),
        }

    @property
    def haagent_version(self) -> str | None:
        value = self.haagent.get("package_version")
        return str(value) if value is not None else None


@dataclass(frozen=True)
class CostTotals:
    model_call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CostTotals:
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            model_call_count=raw.get("model_call_count") if isinstance(raw.get("model_call_count"), int) else None,
            input_tokens=raw.get("input_tokens") if isinstance(raw.get("input_tokens"), int) else None,
            output_tokens=raw.get("output_tokens") if isinstance(raw.get("output_tokens"), int) else None,
            total_tokens=raw.get("total_tokens") if isinstance(raw.get("total_tokens"), int) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_call_count": self.model_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class CostRecord:
    usage_available: bool | None = None
    pricing_available: bool | None = None
    currency: str | None = None
    estimated_cost: float | None = None
    reason: str | None = None
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    totals: CostTotals = field(default_factory=CostTotals)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CostRecord:
        if not isinstance(raw, Mapping) or not raw:
            return cls()
        model_calls = raw.get("model_calls")
        estimated = raw.get("estimated_cost")
        return cls(
            usage_available=raw.get("usage_available") if isinstance(raw.get("usage_available"), bool) else None,
            pricing_available=raw.get("pricing_available") if isinstance(raw.get("pricing_available"), bool) else None,
            currency=_optional_str(raw.get("currency")),
            estimated_cost=float(estimated) if isinstance(estimated, (int, float)) else None,
            reason=_optional_str(raw.get("reason")),
            model_calls=[dict(item) for item in model_calls if isinstance(item, Mapping)]
            if isinstance(model_calls, list)
            else [],
            totals=CostTotals.from_dict(raw.get("totals") if isinstance(raw.get("totals"), Mapping) else None),
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return {
            "usage_available": self.usage_available,
            "pricing_available": self.pricing_available,
            "currency": self.currency,
            "estimated_cost": self.estimated_cost,
            "reason": self.reason,
            "model_calls": list(self.model_calls),
            "totals": self.totals.to_dict(),
        }


@dataclass(frozen=True)
class VerificationCommandRecord:
    command: str
    status: str
    exit_code: int | None = None
    timeout: bool | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    stdout_truncated: bool | None = None
    stderr_truncated: bool | None = None
    stdout_original_length: int | None = None
    stderr_original_length: int | None = None
    redacted: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VerificationCommandRecord:
        return cls(
            command=_require_str(raw.get("command"), "command"),
            status=_require_str(raw.get("status"), "status"),
            exit_code=raw.get("exit_code") if isinstance(raw.get("exit_code"), int) else None,
            timeout=raw.get("timeout") if isinstance(raw.get("timeout"), bool) else None,
            stdout_excerpt=_optional_str(raw.get("stdout_excerpt")),
            stderr_excerpt=_optional_str(raw.get("stderr_excerpt")),
            stdout_truncated=raw.get("stdout_truncated") if isinstance(raw.get("stdout_truncated"), bool) else None,
            stderr_truncated=raw.get("stderr_truncated") if isinstance(raw.get("stderr_truncated"), bool) else None,
            stdout_original_length=(
                raw.get("stdout_original_length") if isinstance(raw.get("stdout_original_length"), int) else None
            ),
            stderr_original_length=(
                raw.get("stderr_original_length") if isinstance(raw.get("stderr_original_length"), int) else None
            ),
            redacted=raw.get("redacted") if isinstance(raw.get("redacted"), bool) else None,
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return {
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "timeout": self.timeout,
            "stdout_excerpt": self.stdout_excerpt,
            "stderr_excerpt": self.stderr_excerpt,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "stdout_original_length": self.stdout_original_length,
            "stderr_original_length": self.stderr_original_length,
            "redacted": self.redacted,
        }


@dataclass(frozen=True)
class ContextManifestRecord:
    context_count: int = 0
    contexts: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ContextManifestRecord:
        if not isinstance(raw, Mapping):
            return cls()
        contexts = raw.get("contexts")
        count = raw.get("context_count")
        return cls(
            context_count=int(count) if isinstance(count, int) else 0,
            contexts=[dict(item) for item in contexts if isinstance(item, Mapping)]
            if isinstance(contexts, list)
            else [],
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            return dict(self.raw)
        return {"context_count": self.context_count, "contexts": list(self.contexts)}


@dataclass(frozen=True)
class EpisodePackage:
    """校验后的 typed episode package；跨文件一致性仍由 validator 负责。"""

    path: Path | None
    metadata: EpisodeMetadata
    failure: FailureRecord
    context_manifest: ContextManifestRecord
    transcript: list[dict[str, Any]]
    tool_calls: list[ToolCallRecord]
    verification_commands: list[VerificationCommandRecord]
    plan: dict[str, Any] = field(default_factory=dict)
    environment: EnvironmentRecord = field(default_factory=EnvironmentRecord)
    cost: CostRecord = field(default_factory=CostRecord)
    sandbox: dict[str, Any] = field(default_factory=dict)
    workspace_preflight: dict[str, Any] = field(default_factory=dict)
    verification_reached: bool = True

    def last_model_response(self) -> dict[str, Any] | None:
        for record in reversed(self.transcript):
            if record.get("event") == "model_response":
                return record
        return None

    def final_response_text(self) -> str:
        response = self.last_model_response()
        if response is None:
            return "none"
        return str(response.get("content", ""))

    def tool_names_used(self) -> list[str]:
        return sorted({call.tool_name for call in self.tool_calls})

    def tool_argument_errors(self) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        for call in self.tool_calls:
            if call.error_type == "tool_argument_invalid":
                errors.append({"tool_name": call.tool_name, "message": call.error_message})
        return errors

    def approval_summaries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for call in self.tool_calls:
            if call.policy is None and call.policy_not_evaluated():
                rows.append(
                    {
                        "tool_name": call.tool_name,
                        "action": "not_evaluated",
                        "approval_required": False,
                        "approval_status": "not_evaluated",
                        "approval_reason": call.error_message,
                    }
                )
                continue
            if call.policy is None:
                continue
            rows.append(
                {
                    "tool_name": call.tool_name,
                    "action": call.policy.action,
                    "approval_required": call.policy.approval.required,
                    "approval_status": call.policy.approval.status,
                    "approval_reason": call.policy.approval.reason,
                }
            )
        return rows

    def tool_reliability_metrics(self) -> dict[str, int | float]:
        """汇总工具失败、恢复和重复调用。"""
        return summarize_tool_reliability(self.tool_calls)


def summarize_tool_reliability(
    tool_calls: list[ToolCallRecord],
) -> dict[str, int | float]:
    """从 typed tool trace 计算稳定、可导出的回归指标。"""
    call_count = len(tool_calls)
    failure_count = sum(call.status == "error" for call in tool_calls)
    argument_error_count = sum(call.error_type == "tool_argument_invalid" for call in tool_calls)
    retrying_calls = sum(bool(call.retry_events) for call in tool_calls)
    retry_recovered_count = sum(call.recovered_after_retry for call in tool_calls)
    duplicate_count = sum(call.duplicate_suppressed for call in tool_calls)

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "tool_call_count": call_count,
        "tool_failure_count": failure_count,
        "tool_argument_error_count": argument_error_count,
        "retry_recovered_count": retry_recovered_count,
        "duplicate_suppressed_count": duplicate_count,
        "tool_failure_rate": rate(failure_count, call_count),
        "tool_argument_error_rate": rate(argument_error_count, call_count),
        "retry_recovery_rate": rate(retry_recovered_count, retrying_calls),
        "duplicate_call_rate": rate(duplicate_count, call_count),
    }


def decode_tool_calls(raw_rows: list[dict[str, Any]]) -> list[ToolCallRecord]:
    return [ToolCallRecord.from_dict(row) for row in raw_rows]


def decode_verification_commands(raw_rows: list[dict[str, Any]]) -> list[VerificationCommandRecord]:
    return [VerificationCommandRecord.from_dict(row) for row in raw_rows]


def build_episode_package(
    *,
    path: Path | None,
    episode_metadata: Mapping[str, Any],
    failure_record: Mapping[str, Any],
    context_manifest: Mapping[str, Any] | None,
    transcript: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    verification_commands: list[dict[str, Any]],
    plan: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    sandbox: Mapping[str, Any] | None = None,
    workspace_preflight: Mapping[str, Any] | None = None,
    verification_reached: bool = True,
) -> EpisodePackage:
    """内部 decode 入口：仅在 validator 完成文件/字段/跨文件校验后调用。

    生产路径请使用 load_validated_episode_package / load_inspect_episode_package。
    单文件字段校验以 codec.from_dict 为边界；跨文件一致性仍由 validator 负责。
    """
    return EpisodePackage(
        path=path,
        metadata=EpisodeMetadata.from_dict(episode_metadata),
        failure=FailureRecord.from_dict(failure_record),
        context_manifest=ContextManifestRecord.from_dict(context_manifest),
        transcript=list(transcript),
        tool_calls=decode_tool_calls(tool_calls),
        verification_commands=decode_verification_commands(verification_commands),
        plan=dict(plan or {}),
        environment=EnvironmentRecord.from_dict(environment),
        cost=CostRecord.from_dict(cost),
        sandbox=dict(sandbox or {}),
        workspace_preflight=dict(workspace_preflight or {}),
        verification_reached=verification_reached,
    )
