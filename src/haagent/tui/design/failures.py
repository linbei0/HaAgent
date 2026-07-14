"""
haagent/tui/failures.py - 失败摘要与保守下一步

为 TUI 展示失败阶段、来源和可安全执行的下一步建议。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.tui.design.utils import safe_summary


@dataclass(frozen=True)
class FailureView:
    failed_stage: str
    failure_category: str
    reason: str
    episode_path: str

    def block_text(self) -> str:
        lines: list[str] = []
        if self.failure_category == "Loop Limit Failure":
            lines.append("本轮没有完成：模型连续调用工具但没有给出最终回答。")
            lines.append("")
        lines.extend(
            [
            "本轮任务失败",
            "",
            f"阶段：{safe_summary(self.failed_stage, 80)}",
            f"来源：{safe_summary(self.failure_category, 100)}",
            f"错误：{safe_summary(self.reason, 240)}",
            f"记录：{safe_summary(self.episode_path, 240)}",
            "",
            f"stage={safe_summary(self.failed_stage, 80)}",
            f"category={safe_summary(self.failure_category, 100)}",
            f"reason={safe_summary(self.reason, 240)}",
            f"episode_path={safe_summary(self.episode_path, 240)}",
            "",
            "安全下一步：",
            ],
        )
        lines.extend(f"- {step}" for step in failure_next_steps(**self.__dict__))
        return "\n".join(lines)


def failure_next_steps(
    *,
    failed_stage: str,
    failure_category: str,
    reason: str,
    episode_path: str,
) -> list[str]:
    if "stream_interrupted" in reason or "stream interrupted" in reason:
        steps = [
            "已显示部分输出；为避免重复内容，本次未自动重试。",
            "请重新提交请求或调整范围后开始新的模型调用。",
        ]
        if episode_path and episode_path != "unknown":
            steps.append(f"需要复盘时查看 episode trace：{episode_path}。")
        return steps
    steps = [
        "查看对话中的失败摘要，确认失败发生在哪个阶段或服务边界。",
        "重试前先调整请求或补充更明确的文件、命令、范围。",
    ]
    if episode_path and episode_path != "unknown":
        steps.append(f"需要复盘时查看 episode trace：{episode_path}。")
    if failed_stage in {"executing", "verifying"}:
        steps.append("如果你信任当前 workspace，可手动运行相关检查命令确认状态。")
    return steps


def failure_from_payload(payload: dict[str, object], fallback_message: str = "") -> FailureView:
    status = _payload_text(payload.get("status"))
    fallback = _payload_text(fallback_message)
    failed_stage = _payload_text(payload.get("failed_stage"))
    failure_category = _payload_text(payload.get("failure_category"))
    reason = _payload_text(payload.get("reason"))
    if status == "cancelled":
        failed_stage = failed_stage or "cancelled"
        failure_category = failure_category or "Runtime Failure"
        reason = reason or fallback or "user cancelled current run"
    return FailureView(
        failed_stage=failed_stage or _missing_field("failed_stage"),
        failure_category=failure_category or _missing_field("failure_category"),
        reason=reason or fallback or _missing_field("reason"),
        episode_path=_payload_text(payload.get("episode_path")) or _missing_field("episode_path"),
    )


def _missing_field(field_name: str) -> str:
    return f"缺少字段: {field_name}"


def _payload_text(value: object) -> str:
    text = str(value or "").strip()
    if text in {"none", "unknown"}:
        return ""
    return text
