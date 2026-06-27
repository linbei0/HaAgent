"""
haagent/tui/renderers.py - TUI 文本渲染逻辑

集中生成状态栏、侧栏、审批、失败和记忆候选文本，保持组件层轻量。
"""

from __future__ import annotations

from typing import Any

from haagent.app.assistant_service import AssistantWorkspaceStatus
from haagent.memory import MemoryCandidate
from haagent.runtime.human_interaction import HumanInteractionRequest
from haagent.tui.utils import safe_summary, short_session, truncate_end, truncate_status_line, workspace_label


def status_line(status: AssistantWorkspaceStatus, *, ui_state: str, width: int) -> str:
    width = max(1, width or 120)
    workspace_limit = 5 if width <= 80 else 16
    model_limit = 4 if width <= 80 else 14
    session_limit = 4 if width <= 80 else 12
    turn_count = status.current_turn_count if status.current_turn_count is not None else 0
    prefix = (
        f"ws:{workspace_label(status.workspace_root, workspace_limit)} "
        f"profile: {truncate_end(status.profile_name or 'missing', 12)} "
        f"{truncate_end(status.provider or '-', 14)}/{truncate_end(status.model or '-', model_limit)} "
        f"key: {compact_key_state(status)} "
        f"sid:{short_session(status.current_session_id or '-', session_limit)} "
        f"turn:{turn_count}"
    )
    state = f" state: {ui_state}"
    prefix_width = max(0, width - len(state))
    return f"{truncate_status_line(prefix, prefix_width)}{state}"


def side_bar(
    status: AssistantWorkspaceStatus,
    *,
    ui_state: str,
    tool_lines: list[str],
    last_failure: dict[str, str] | None,
    memory_text: str | None = None,
) -> str:
    if memory_text is not None:
        return memory_text
    tool_summary = "\n".join(f"  {line}" for line in tool_lines[-5:]) or "  none"
    turn_count = status.current_turn_count if status.current_turn_count is not None else 0
    return (
        "Profile\n"
        f"  name: {status.profile_name or 'missing'}\n"
        f"  provider: {status.provider or '-'}\n"
        f"  base_url: {status.base_url or '-'}\n"
        f"  model: {status.model or '-'}\n"
        f"  api_key_env: {status.api_key_env or '-'}\n"
        f"  key: {key_state(status)}\n"
        f"  keyring: {keyring_status(status)}\n\n"
        "Session\n"
        f"  id: {status.current_session_id or '-'}\n"
        f"  turns: {turn_count}\n"
        f"  state: {ui_state}\n\n"
        "Tools This Turn\n"
        f"{tool_summary}\n\n"
        "Last Failure\n"
        f"{format_last_failure(last_failure)}"
    )


def approval_body(request: HumanInteractionRequest) -> str:
    lines = [
        "工具请求需要确认",
        "",
        f"tool      {safe_summary(request.tool_name, 80)}",
        f"question  {safe_summary(request.question, 160)}",
    ]
    if request.reason:
        lines.append(f"reason    {safe_summary(request.reason, 160)}")
    if request.risk_level:
        lines.append(f"risk      {safe_summary(request.risk_level, 40)}")
    lines.extend(
        [
            f"args      {format_args_summary(request.args_summary)}",
            f"impact    {impact_summary(request.tool_name, request.args_summary)}",
            "",
            "高风险内容首版只展示摘要，不展示完整 patch、stdout 或 stderr。",
        ],
    )
    return "\n".join(lines)


def memory_panel_text(
    *,
    candidates: list[MemoryCandidate],
    selected_index: int,
    detail_mode: bool,
    notice: str | None,
    error: str | None,
) -> str:
    prefix = ["Memory", f"  {notice}", ""] if notice else []
    if error:
        return "\n".join([*prefix, "Memory Candidates", f"  Memory candidates unavailable: {error}"])
    if not candidates:
        return "\n".join([*prefix, "Memory Candidates", "  no pending candidates"])
    selected_index = min(max(selected_index, 0), len(candidates) - 1)
    if detail_mode:
        return "\n".join([*prefix, memory_candidate_detail(candidates[selected_index])])
    lines = [*prefix, "Memory Candidates"]
    for index, candidate in enumerate(candidates):
        marker = ">" if index == selected_index else " "
        lines.append(f"{marker} {candidate.candidate_id} [{candidate.scope}/{candidate.category}] {candidate.title}")
    lines.extend(["", "↑/↓ j/k 移动  g/G 首尾  Enter 详情  a/y 确认  r 拒绝  Esc 返回"])
    return "\n".join(lines)


def memory_candidate_detail(candidate: MemoryCandidate) -> str:
    evidence = candidate.evidence
    lines = [
        "Memory Candidate Detail",
        f"candidate_id: {candidate.candidate_id}",
        f"status: {candidate.status}",
        f"scope: {candidate.scope}",
        f"category: {candidate.category}",
        f"title: {candidate.title}",
        f"body: {candidate.body}",
        f"source: {candidate.source}",
        f"created_at: {candidate.created_at}",
        f"tags: {', '.join(candidate.tags) if candidate.tags else 'none'}",
        f"risk_flags: {', '.join(candidate.risk_flags) if candidate.risk_flags else 'none'}",
        "",
        "Evidence",
        f"source_type: {evidence.source_type}",
        f"source_summary: {evidence.source_summary or 'none'}",
        f"basis: {evidence.basis or 'none'}",
        f"category_rationale: {evidence.category_rationale or 'none'}",
        f"episode_path: {evidence.episode_path or 'none'}",
    ]
    return "\n".join(lines)


def format_args_summary(args_summary: dict[str, object]) -> str:
    if not args_summary:
        return "none"
    pieces = []
    for key, value in args_summary.items():
        if isinstance(value, list):
            safe_items = ", ".join(safe_summary(str(item), 80) for item in value[:3])
            if len(value) > 3:
                safe_items += ", ..."
            pieces.append(f"{key}=[{safe_items}]")
        else:
            pieces.append(f"{key}={safe_summary(str(value), 120)}")
    return "; ".join(pieces)


def impact_summary(tool_name: str, args_summary: dict[str, object]) -> str:
    if tool_name in {"file_write", "apply_patch"}:
        path = safe_summary(str(args_summary.get("path", "unknown")), 120)
        return f"会修改本地文件；path={path}"
    if tool_name == "apply_patch_set":
        paths = args_summary.get("paths")
        if isinstance(paths, list) and paths:
            return f"会修改本地文件；paths={safe_summary(', '.join(str(path) for path in paths[:3]), 160)}"
        return "会修改本地文件；paths=unknown"
    if tool_name == "shell":
        command = safe_summary(str(args_summary.get("command", "unknown")), 160)
        return f"会执行本地命令；是否修改本地文件取决于命令；command={command}"
    if tool_name == "code_run":
        return "会执行本地代码；可能读取或修改 workspace 内文件"
    return "影响范围以工具参数摘要为准"


def payload_text(payload: dict[str, object], key: str, default: str) -> str:
    value: Any = payload.get(key)
    if value is None:
        return default
    return str(value)


def key_state(status: AssistantWorkspaceStatus) -> str:
    if status.api_key_available and status.credential_source_used:
        return f"available via {status.credential_source_used}"
    return "missing"


def compact_key_state(status: AssistantWorkspaceStatus) -> str:
    return "ok" if status.api_key_available else "missing"


def keyring_status(status: AssistantWorkspaceStatus) -> str:
    if status.credential_store_available is False:
        reason = status.credential_store_error or "unknown"
        return f"keyring unavailable: {reason}"
    if status.credential_store_available is True:
        return "available"
    return "-"


def failure_body(failed_stage: str, category: str, reason: str, episode_path: str) -> str:
    lines: list[str] = []
    if category == "Loop Limit Failure":
        lines.append("本轮没有完成：模型连续调用工具但没有给出最终回答。")
    lines.extend(
        [
            f"stage={failed_stage}",
            f"category={category}",
            f"reason={reason}",
            f"episode_path={episode_path}",
        ],
    )
    return "\n".join(lines)


def format_last_failure(failure: dict[str, str] | None) -> str:
    if failure is None:
        return "  none"
    return (
        f"  category: {failure['failure_category']}\n"
        f"  stage: {failure['failed_stage']}\n"
        f"  reason: {failure['reason']}\n"
        f"  episode: {failure['episode_path']}"
    )
