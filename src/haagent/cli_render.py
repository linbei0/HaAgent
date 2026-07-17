"""
haagent/cli_render.py - CLI 用户可见输出渲染

集中渲染 run、smoke、eval 与 check 的短摘要输出。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.episodes.package_types import (
    CostRecord,
    EpisodeMetadata,
    EnvironmentRecord,
)
from haagent.runtime.episodes.validator import EpisodeValidationError, load_inspect_episode_package


def print_run_summary(result) -> None:
    """输出短 run 摘要；完整复盘仍交给 inspect。"""
    print(f"status={result.status.value}")
    print(f"episode_path={result.episode_path}")
    try:
        package = load_inspect_episode_package(result.episode_path)
    except EpisodeValidationError as error:
        print(f"summary_error={summary_value(str(error))}")
        return

    print(f"provider={summary_provider(package.metadata)}")
    print(f"model={summary_value(summary_model(package.environment))}")
    print(f"usage_available={summary_bool(package.cost.usage_available)}")
    print(f"total_tokens={summary_token_total(package.cost)}")
    print(f"estimated_cost={summary_estimated_cost(package.cost)}")
    if result.status.value == "completed":
        print(f"final_response={summary_value(package.final_response_text())}")
        return

    failure = package.failure.failure
    print(f"failed_stage={summary_value('unknown' if failure is None else failure.stage)}")
    print(f"failure_category={summary_value('unknown' if failure is None else failure.category)}")
    print(f"reason={summary_value('' if failure is None else failure.evidence)}")



def print_smoke_result(result) -> None:
    print(f"smoke={result.name}")
    print(f"status={result.status}")
    episode_path = "none" if result.episode_path is None else str(result.episode_path)
    print(f"episode_path={episode_path}")
    if result.status != "completed":
        print(f"failed_stage={summary_value(result.failed_stage or 'unknown')}")
        print(f"failure_category={summary_value(result.failure_category or 'unknown')}")
        print(f"reason={summary_value(result.reason or '')}")


def print_check_summary(report: dict[str, Any]) -> None:
    eval_report = report["eval_report"]
    print(f"status={report['status']}")
    print(f"eval_total={eval_report['total_count']}")
    print(f"eval_passed={eval_report['passed_count']}")
    print(f"eval_failed={eval_report['failed_count']}")
    print(f"eval_error={eval_report['error_count']}")
    for result in eval_report.get("results", []):
        if not isinstance(result, dict) or result.get("status") == "passed":
            continue
        print(f"failure_case={result.get('case_path')}")
        print(f"failure_reason={summary_value(str(result.get('failure_reason', '')))}")
    latency = report.get("interactive_latency")
    if isinstance(latency, dict):
        print(f"latency_total={latency.get('total', 0)}")
        print(f"latency_passed={latency.get('passed', 0)}")
        print(f"latency_failed={latency.get('failed', 0)}")
        for gate in latency.get("gates", []):
            if not isinstance(gate, dict) or gate.get("status") == "passed":
                continue
            # 失败输出：指标、阈值、实际值
            print(
                "latency_failure="
                f"{gate.get('name')} metric={gate.get('metric')} "
                f"threshold={gate.get('threshold')} actual={gate.get('actual')}"
            )
    pytest_report = report.get("pytest")
    if isinstance(pytest_report, dict):
        print(f"pytest_exit_code={pytest_report['exit_code']}")
        print(f"pytest_summary={summary_value(str(pytest_report.get('summary', 'none')))}")


def print_eval_summary(report: dict[str, Any]) -> None:
    status = "passed" if report["failed_count"] == 0 and report["error_count"] == 0 else "failed"
    print(f"status={status}")
    print(f"total={report['total_count']}")
    print(f"passed={report['passed_count']}")
    print(f"failed={report['failed_count']}")
    print(f"error={report['error_count']}")
    for result in report.get("results", []):
        if not isinstance(result, dict) or result.get("status") == "passed":
            continue
        print(f"failure_case={result.get('case_path')}")
        print(f"failure_reason={summary_value(str(result.get('failure_reason', '')))}")


def run_final_response(transcript: list[dict[str, Any]]) -> str:
    response = last_model_response(transcript)
    if response is None:
        return "none"
    return str(response.get("content", ""))


def summary_provider(episode_metadata: EpisodeMetadata) -> str:
    return str(episode_metadata.provider or "unknown")


def summary_model(environment: EnvironmentRecord) -> str:
    provider = str(environment.model.provider or "unknown")
    model_name = str(environment.model.model or "unknown")
    return f"{provider}/{model_name}"


def summary_bool(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def summary_token_total(cost: CostRecord) -> str:
    value = cost.totals.total_tokens
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "unavailable"


def summary_estimated_cost(cost: CostRecord) -> str:
    value = cost.estimated_cost
    if isinstance(value, int | float) and not isinstance(value, bool):
        currency = cost.currency
        return f"{value} {currency}" if currency else str(value)
    return summary_value(str(cost.reason or "unavailable"))



def last_model_response(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return record
    return None


def excerpt(content: str, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "... [truncated]"


def summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    return excerpt(normalized, limit)


def render_sandbox_status(status) -> str:
    return "\n".join(
        [
            f"backend={status.backend}",
            f"isolation_level={status.isolation_level}",
            f"network_policy={status.network_policy}",
            f"credential_policy={status.credential_policy}",
            f"degraded={str(status.degraded).lower()}",
            f"reason={status.reason}",
            f"config_path={status.config_path}",
            f"next_action={status.recommendation}",
        ],
    )


def render_sandbox_doctor(report) -> str:
    return "\n".join(
        [
            f"backend={report.backend}",
            f"ready={str(report.ready).lower()}",
            f"docker_cli={report.docker_cli}",
            f"docker_daemon={report.docker_daemon}",
            f"image={report.image}",
            f"auto_build_image={str(report.auto_build_image).lower()}",
            f"reason={report.reason}",
            f"next_action={report.next_action}",
        ],
    )
