"""
haagent/cli_render.py - CLI 用户可见输出渲染

集中渲染 run、smoke、eval 与 check 的短摘要输出。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.episodes.validator import EpisodeValidationError, load_inspect_episode_package


def print_run_summary(result) -> None:
    """输出短 run 摘要；完整复盘仍交给 inspect。"""
    print(f"status={result.status.value}")
    print(f"episode_path={result.episode_path}")
    try:
        package_view = load_inspect_episode_package(result.episode_path)
    except EpisodeValidationError as error:
        print(f"summary_error={summary_value(str(error))}")
        return

    print(f"provider={summary_provider(package_view.episode_metadata)}")
    print(f"model={summary_value(summary_model(package_view.environment))}")
    print(f"usage_available={summary_bool(package_view.cost.get('usage_available'))}")
    print(f"total_tokens={summary_token_total(package_view.cost)}")
    print(f"estimated_cost={summary_estimated_cost(package_view.cost)}")
    if result.status.value == "completed":
        print(f"final_response={summary_value(run_final_response(package_view.transcript))}")
        return

    failure = package_view.failure_record.get("failure")
    if not isinstance(failure, dict):
        failure = {}
    print(f"failed_stage={summary_value(str(failure.get('stage', 'unknown')))}")
    print(f"failure_category={summary_value(str(failure.get('category', 'unknown')))}")
    print(f"reason={summary_value(str(failure.get('evidence', '')))}")


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


def summary_provider(episode_metadata: dict[str, Any]) -> str:
    return str(episode_metadata.get("provider", "unknown"))


def summary_model(environment: dict[str, Any]) -> str:
    model = environment.get("model") if isinstance(environment, dict) else None
    if not isinstance(model, dict):
        return "unknown"
    provider = str(model.get("provider") or "unknown")
    model_name = str(model.get("model") or "unknown")
    return f"{provider}/{model_name}"


def summary_bool(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def summary_token_total(cost: dict[str, Any]) -> str:
    totals = cost.get("totals") if isinstance(cost, dict) else None
    if not isinstance(totals, dict):
        return "unavailable"
    value = totals.get("total_tokens")
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "unavailable"


def summary_estimated_cost(cost: dict[str, Any]) -> str:
    value = cost.get("estimated_cost") if isinstance(cost, dict) else None
    if isinstance(value, int | float) and not isinstance(value, bool):
        currency = cost.get("currency")
        return f"{value} {currency}" if currency else str(value)
    reason = cost.get("reason") if isinstance(cost, dict) else None
    return summary_value(str(reason or "unavailable"))


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
