"""
haagent/memory/prompts.py - 记忆结算提示词集中入口

集中维护长期记忆结算相关的工具说明、META-SOP 和抽取提示词。
"""

from __future__ import annotations


START_MEMORY_UPDATE_TOOL_DESCRIPTION = (
    "申请进入长期记忆结算流程。仅当当前任务中出现长期有价值的信息时调用；"
    "它不直接写正式记忆，不绕过用户确认，只通知 runtime 后续尝试抽取、校验、去重并生成候选。"
    "长期有价值的信息包括：用户稳定偏好、用户明确给出的长期事实、经工具/文件/验证确认的环境事实，"
    "以及经过执行验证的可复用 SOP、避坑经验或前置条件。不要因为普通寒暄、一次性问题、临时状态、"
    "模型推理、未验证计划或通用常识调用。没有长期价值时不要调用。"
)

MEMORY_META_SOP = [
    "只记长期有价值的信息。",
    "用户偏好必须来自用户明确表达。",
    "SOP 必须来自已执行或已验证结果。",
    "不记寒暄。",
    "不记临时状态。",
    "不记模型猜测。",
    "不记通用常识。",
    "不把助手回答当证据。",
    "没有有效 evidence 就不生成 candidate。",
]

EVIDENCE_RULES = [
    "每个候选必须包含 evidence_source 与 evidence_quote。",
    "允许的 evidence_source: user_prompt, tool_result, file_content, verification_result。",
    "禁止的 evidence_source: assistant_response, final_response, model_inference, memory_recall, unknown。",
    "evidence_quote 必须是对应来源原文中可定位的短引文，不允许改写。",
    "final_response 只可用于理解上下文，不能作为用户事实或 SOP 的证据。",
    "助手解决方案只有在工具、文件或验证结果可追溯时，才可整理为 SOP candidate。",
]


def build_memory_extraction_prompt(
    *,
    session_id: str,
    turn_index: int,
    verification_status: str,
    user_prompt: str,
    final_response: str,
    working_state: str,
    runtime_events: list[str],
) -> str:
    """构建长期记忆结算模型提示词；代码仍负责最终边界校验。"""
    lines = [
        "Memory Settlement: propose only durable long-term memory candidates.",
        "Return JSON only:",
        (
            '{"candidates":[{"scope":"workspace|user",'
            '"category":"facts|sop|glossary|decisions|user_preferences|habits|constraints",'
            '"title":"...","body":"...","source_summary":"...","basis":"...",'
            '"category_rationale":"...","evidence_source":"user_prompt|tool_result|file_content|verification_result",'
            '"evidence_quote":"exact quote from the selected source","tags":["..."]}]}'
        ),
        "",
        "META-SOP:",
        *[f"- {rule}" for rule in MEMORY_META_SOP],
        "",
        "Evidence rules:",
        *[f"- {rule}" for rule in EVIDENCE_RULES],
        "",
        f"session_id={session_id} turn_index={turn_index} verification={verification_status}",
        f"user_prompt={user_prompt}",
        (
            "final_response_for_context_only="
            f"{final_response} "
            "(not evidence; never extract user facts from assistant wording)"
        ),
        f"working_state={working_state}",
        "runtime_events:",
        *runtime_events,
    ]
    return "\n".join(lines)
