"""
haagent/tui/skills_flow.py - TUI skills 交互流程

封装 skills 列表、调用、信任、marketplace 搜索与安装确认等交互编排，降低主应用类的分支职责。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haagent.tui.overlays.modals import ConfirmModal
from haagent.tui.overlays.skill_picker import SkillPickerOverlay
from haagent.tui.widgets import PromptInput

if TYPE_CHECKING:
    from haagent.tui.application.app import HaAgentTuiApp


def handle_skills_command(app: "HaAgentTuiApp", argument: str) -> None:
    raw_value = argument.strip()
    value = raw_value.lower()
    try:
        if value == "trust":
            summary = app.service.trust_project_skills()
            app._conversation.append_block("Skills", "已信任当前 workspace 的项目 skills。\n" + skills_summary_text(summary))
        elif value == "untrust":
            summary = app.service.untrust_project_skills()
            app._conversation.append_block("Skills", "已取消信任当前 workspace 的项目 skills。\n" + skills_summary_text(summary))
        elif value.startswith("search "):
            query = raw_value.split(" ", 1)[1].strip()
            if not query:
                app._conversation.append_block("Command", skills_usage_text())
            else:
                result = app.service.search_skill_marketplace(query, limit=10)
                app._conversation.append_block("Skills marketplace", skill_marketplace_summary_text(result))
        elif value.startswith("install "):
            result_id = raw_value.split(" ", 1)[1].strip()
            if not result_id:
                app._conversation.append_block("Command", skills_usage_text())
            else:
                app.push_screen(
                    ConfirmModal(
                        "安装远端 skill",
                        (
                            f"将安装 marketplace 搜索结果：{result_id}\n"
                            "远端内容会作为外部引用写入用户级 skills；使用前仍需审阅来源页面。确认？"
                        ),
                    ),
                    lambda confirmed, result_id=result_id: app._handle_skill_marketplace_install_confirmed(
                        result_id,
                        confirmed,
                    ),
                )
        elif not value:
            open_skill_picker(app, mode="manage")
        else:
            app._conversation.append_block("Command", skills_usage_text())
    except Exception as error:
        app._conversation.append_block("Skills warning", f"skills 操作失败：{error}")
    app._refresh()


def handle_skill_marketplace_install_confirmed(
    app: "HaAgentTuiApp",
    result_id: str,
    confirmed: bool | None,
) -> None:
    if not confirmed:
        app._conversation.append_block("Skills marketplace", f"已取消安装 marketplace skill：{result_id}")
        app._refresh()
        return
    try:
        installed = app.service.install_marketplace_skill(result_id)
    except Exception as error:
        app._conversation.append_block("Skills warning", f"skills 操作失败：{error}")
    else:
        app._conversation.append_block("Skills marketplace", skill_marketplace_install_text(installed))
    app._refresh()


def handle_skill_command(app: "HaAgentTuiApp", argument: str) -> None:
    text = argument.strip()
    if not text:
        open_skill_picker(app, mode="use")
        return
    skill_name, _, request = text.partition(" ")
    try:
        skill = app.service.read_skill_for_user(skill_name)
    except Exception as error:
        app._conversation.append_block("Skills warning", f"读取 skill 失败：{error}")
        app._refresh()
        return
    prompt = "\n".join(
        [
            f"Use skill {skill.command_name} explicitly.",
            "",
            "Skill content:",
            skill.content,
            "",
            "User request:",
            request.strip() or f"Follow the {skill.command_name} skill for this task.",
        ],
    )
    app._conversation.append_block("Skills", f"已加载 skill：{skill.name}")
    app._start_prompt(prompt)


def open_skill_picker(app: "HaAgentTuiApp", *, mode: str) -> None:
    try:
        summary = app.service.list_skills()
    except Exception as error:
        app._conversation.append_block("Skills warning", f"读取 skills 失败：{error}")
        app._refresh()
        return
    skills = list(getattr(summary, "skills", []) or [])
    if not skills:
        app._conversation.append_block("Skills", "暂无可用 skills。")
        app._refresh()
        return
    blocked_roots = list(getattr(summary, "blocked_project_skill_roots", []) or [])
    if mode == "manage":
        app.push_screen(
            SkillPickerOverlay(
                skills,
                blocked_project_skill_roots=blocked_roots,
                title="Skills",
                instruction="输入过滤，↑/↓ 浏览，Esc 关闭",
                footer="使用 /skill 选择并调用；/skills search 远端搜索；/skills trust 信任项目 skills。",
                select_on_enter=False,
            ),
            app._handle_skill_picker_result,
        )
        return
    app.push_screen(
        SkillPickerOverlay(skills, blocked_project_skill_roots=blocked_roots),
        app._handle_skill_picker_result,
    )


def handle_skill_picker_result(app: "HaAgentTuiApp", skill: dict[str, object] | None) -> None:
    if skill is None:
        app.set_timer(0.01, app._restore_prompt_focus)
        return
    command_name = str(skill.get("command_name") or skill.get("name") or "").strip()
    if not command_name:
        app._conversation.append_block("Skills warning", "选择的 skill 缺少命令名。")
        app._refresh()
        app.set_timer(0.01, app._restore_prompt_focus)
        return
    prompt_input = app.query_one("#prompt-input", PromptInput)
    app._set_prompt_value(prompt_input, f"/skill {command_name} ")
    app.set_timer(0.01, app._restore_prompt_focus)


def skills_summary_text(summary) -> str:
    lines: list[str] = []
    skills = list(getattr(summary, "skills", []) or [])
    if skills:
        for item in skills:
            name = str(item.get("name", "unknown"))
            source = str(item.get("source", "unknown"))
            description = str(item.get("description", ""))
            suffix = " user-only" if item.get("disable_model_invocation") else ""
            lines.append(f"- {name} [{source}]: {description}{suffix}".rstrip())
    else:
        lines.append("暂无可用 skills。")
    blocked = list(getattr(summary, "blocked_project_skill_roots", []) or [])
    if blocked:
        lines.append("")
        lines.append("项目 skills 未信任：")
        for path in blocked:
            lines.append(f"- {path}")
        lines.append("输入 /skills trust 信任当前 workspace 的项目 skills。")
    return "\n".join(lines)


def skills_usage_text() -> str:
    return "\n".join(
        [
            "用法：",
            "- /skills",
            "- /skills trust",
            "- /skills untrust",
            "- /skills search <query>",
            "- /skills install <result-id>",
        ],
    )


def skill_marketplace_summary_text(result) -> str:
    lines = [
        f"查询：{getattr(result, 'query', '')}",
        f"状态：{getattr(result, 'status', 'unknown')}",
    ]
    results = list(getattr(result, "results", []) or [])
    if results:
        lines.append("")
        lines.append("结果：")
        for item in results:
            lines.append(skill_marketplace_result_line(item))
    else:
        lines.append("")
        lines.append("未找到 marketplace skills。")
    warnings = list(getattr(result, "warnings", []) or [])
    if warnings:
        lines.append("")
        lines.append("警告：")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def skill_marketplace_result_line(item) -> str:
    result_id = str(getattr(item, "result_id", "unknown"))
    provider = str(getattr(item, "provider", "unknown"))
    name = str(getattr(item, "name", "unknown"))
    source = str(getattr(item, "source", ""))
    summary = str(getattr(item, "summary", ""))
    detail_url = str(getattr(item, "detail_url", ""))
    install_state = "可安装" if bool(getattr(item, "installable", False)) else "暂不支持直接安装"
    quality = skill_marketplace_quality_text(getattr(item, "quality", {}) or {})
    pieces = [f"- {result_id} [{provider}] {name}"]
    if source:
        pieces.append(f"by {source}")
    pieces.append(f"({install_state})")
    if quality:
        pieces.append(quality)
    if summary:
        pieces.append(f"- {summary}")
    if detail_url:
        pieces.append(f"- {detail_url}")
    return " ".join(pieces)


def skill_marketplace_quality_text(quality) -> str:
    if not isinstance(quality, dict) or not quality:
        return ""
    pairs = [f"{key}={value}" for key, value in sorted(quality.items())]
    return "[" + ", ".join(pairs) + "]"


def skill_marketplace_install_text(installed) -> str:
    lines = [
        f"已安装 marketplace skill：{getattr(installed, 'name', 'unknown')}",
        f"命令：${getattr(installed, 'command_name', 'unknown')}",
        f"目录：{getattr(installed, 'skill_dir', '')}",
        f"来源：{getattr(installed, 'source_url', '')}",
        "远端内容已作为外部引用写入；使用前请审阅来源页面。",
    ]
    return "\n".join(lines)
