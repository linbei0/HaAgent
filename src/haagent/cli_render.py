"""
haagent/cli_render.py - CLI 用户可见输出渲染

集中渲染 run、chat、smoke、eval 与 check 的短摘要输出。
"""

from __future__ import annotations

import json
from typing import Any

from haagent.runtime.chat_session import AgentSession, ChatEvent, ChatTurnResult
from haagent.runtime.episode_validator import EpisodeValidationError, load_inspect_episode_package
from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse


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
    if result.status.value == "completed":
        print(f"final_response={summary_value(run_final_response(package_view.transcript))}")
        return

    failure = package_view.failure_record.get("failure")
    if not isinstance(failure, dict):
        failure = {}
    print(f"failed_stage={summary_value(str(failure.get('stage', 'unknown')))}")
    print(f"failure_category={summary_value(str(failure.get('category', 'unknown')))}")
    print(f"reason={summary_value(str(failure.get('evidence', '')))}")


def run_chat_repl(session: AgentSession) -> int:
    print_chat_event(session.session_started_event())
    print_session_status(session)
    while True:
        try:
            raw_prompt = input("haagent> ")
        except EOFError:
            print_chat_event(session.session_finished_event())
            print("bye")
            return 0
        prompt = raw_prompt.strip()
        if not prompt:
            continue
        if prompt in {":quit", ":exit"}:
            print_chat_event(session.session_finished_event())
            print("bye")
            return 0
        if prompt == ":status":
            print_session_status(session)
            continue
        if prompt == ":new":
            session.new()
            print("session reset")
            continue

        result = session.run_prompt_events(
            prompt,
            event_sink=print_chat_event,
            interaction_handler=read_chat_interaction,
        )
        print_chat_turn_result(result)


def print_session_status(session: AgentSession) -> None:
    status = session.status()
    print(f"session_id={status['session_id']}")
    print(f"session_path={status['session_path']}")
    print(f"workspace_root={status['workspace_root']}")
    print(f"provider={status['provider']}")
    print(f"turn_count={status['turn_count']}")
    working_state = status.get("working_state") if isinstance(status.get("working_state"), dict) else {}
    working_state_exists = bool(working_state.get("exists"))
    print(f"working_state={'present' if working_state_exists else 'empty'}")
    if working_state_exists:
        print(f"working_state_goal={summary_value(str(working_state.get('current_goal', '')), 120)}")
        print(f"working_state_next_steps={working_state.get('next_steps_count', 0)}")


def print_chat_turn_result(result: ChatTurnResult) -> None:
    for line in result.output_lines():
        print(line)


def print_chat_event(event: ChatEvent) -> None:
    pieces = [f"event={event.event_type}"]
    if event.event_type in {
        "tool_started",
        "tool_finished",
        "tool_failed",
        "approval_requested",
        "approval_granted",
        "approval_denied",
    }:
        tool_name = event.payload.get("tool_name")
        if tool_name is not None:
            pieces.append(f"tool={tool_name}")
    if event.event_type == "tool_started":
        pieces.append(f"args={format_event_mapping(event.payload.get('args_summary'))}")
    elif event.event_type == "tool_finished":
        status = event.payload.get("status")
        if status is not None:
            pieces.append(f"status={status}")
        pieces.append(f"result={format_event_mapping(event.payload.get('result_summary'))}")
    elif event.event_type == "tool_failed":
        error_type = event.payload.get("error_type")
        if error_type is not None:
            pieces.append(f"error={error_type}")
        message = event.payload.get("message")
        if message:
            pieces.append(f"message={shell_token(str(message))}")
    elif event.event_type == "approval_requested":
        question = event.payload.get("question")
        if question:
            pieces.append(f"question={shell_token(str(question))}")
        pieces.append(f"args={format_event_mapping(event.payload.get('args_summary'))}")
    elif event.event_type in {"approval_granted", "approval_denied"}:
        approved = event.payload.get("approved")
        if approved is not None:
            pieces.append(f"approved={str(approved).lower()}")
    elif event.event_type == "user_input_requested":
        question = event.payload.get("question")
        if question:
            pieces.append(f"question={shell_token(str(question))}")
    elif event.event_type == "user_input_received":
        answer_chars = event.payload.get("answer_chars")
        if answer_chars is not None:
            pieces.append(f"answer_chars={answer_chars}")
    elif event.event_type == "assistant_message":
        content = event.payload.get("content")
        if content:
            pieces.append(f"message={shell_token(summary_value(str(content)))}")
    elif event.event_type == "memory_candidates_created":
        count = event.payload.get("count")
        if count is not None:
            pieces.append(f"count={count}")
        message = event.payload.get("message")
        if message:
            pieces.append(f"message={shell_token(str(message))}")
    elif event.event_type == "guardrail_triggered":
        for key in ["scope", "rule_id", "severity", "message"]:
            value = event.payload.get(key)
            if value:
                pieces.append(f"{key}={shell_token(str(value))}")
    elif event.event_type == "failure":
        for key in ["failed_stage", "failure_category", "reason"]:
            value = event.payload.get(key)
            if value:
                pieces.append(f"{key}={shell_token(str(value))}")
    elif event.event_type in {"turn_started", "turn_finished", "session_started", "session_finished"}:
        status = event.payload.get("status")
        if status is not None:
            pieces.append(f"status={status}")
    print(" ".join(pieces))


def read_chat_interaction(request: HumanInteractionRequest) -> HumanInteractionResponse:
    try:
        if request.interaction_type == "approval":
            raw_answer = input("approve [y/N]> ")
            approved = raw_answer.strip().lower() in {"y", "yes"}
            return HumanInteractionResponse(approved=approved, answer=raw_answer)
        answer = input("answer> ")
        return HumanInteractionResponse(approved=True, answer=answer)
    except EOFError:
        return HumanInteractionResponse(approved=False, answer="")


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


def format_event_mapping(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts = []
    for key in sorted(value):
        item = value[key]
        if item is None or item == "":
            continue
        parts.append(f"{key}={shell_token(str(item))}")
    return ",".join(parts) if parts else "none"


def shell_token(value: str) -> str:
    compact = summary_value(" ".join(value.split()), 160)
    if not compact:
        return "none"
    if any(character.isspace() or character in {",", "="} for character in compact):
        return json.dumps(compact, ensure_ascii=False)
    return compact


def run_final_response(transcript: list[dict[str, Any]]) -> str:
    response = last_model_response(transcript)
    if response is None:
        return "none"
    return str(response.get("content", ""))


def summary_provider(episode_metadata: dict[str, Any]) -> str:
    return str(episode_metadata.get("provider", "unknown"))


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
