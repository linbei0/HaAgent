"""
src/haagent/runtime/evaluation/dogfood.py - 真实模型 dogfood 运行器

用临时 fixture workspace 驱动 AgentSession，验证真实模型能否通过现有 runtime 完成小型编辑任务。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from haagent.models.types import ModelGateway
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.runtime.settings import DEFAULT_DOGFOOD_MAX_TURNS


@dataclass(frozen=True)
class DogfoodTaskResult:
    name: str
    status: str
    tools: list[str]
    failure_reason: str
    episode_path: Path
    workspace_path: Path
    most_needed_improvement: str


@dataclass(frozen=True)
class DogfoodReport:
    status: str
    runs_root: Path
    tasks: list[DogfoodTaskResult]
    most_needed_improvement: str
    skipped_reason: str = "none"


@dataclass(frozen=True)
class _DogfoodTask:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    verify: Callable[[Path, list[dict[str, Any]]], tuple[bool, str]]


def run_dogfood_tasks(
    model_gateway: ModelGateway,
    *,
    runs_root: Path | None = None,
    max_turns: int = DEFAULT_DOGFOOD_MAX_TURNS,
    auto_approve: bool = True,
) -> DogfoodReport:
    root = runs_root if runs_root is not None else Path(tempfile.mkdtemp(prefix="haagent-dogfood-runs-"))
    root.mkdir(parents=True, exist_ok=True)
    results: list[DogfoodTaskResult] = []
    for task in _dogfood_tasks():
        workspace = Path(tempfile.mkdtemp(prefix=f"haagent-dogfood-{task.name}-"))
        task.setup(workspace)
        session = AgentSession(
            workspace_root=workspace,
            runs_root=root,
            model_gateway=model_gateway,
            max_turns=max_turns,
            memory_extraction_enabled=False,
        )
        turn = session.run_prompt_events(
            task.prompt,
            interaction_handler=_approval_handler(auto_approve),
        )
        tool_calls = _read_tool_calls(turn.episode_path)
        verified, verification_reason = task.verify(workspace, tool_calls)
        status = "completed" if turn.status == "completed" and verified else "failed"
        failure_reason = "none" if status == "completed" else _failure_reason(turn.reason, verification_reason)
        improvement = _task_improvement(task.name, status, tool_calls, failure_reason)
        results.append(
            DogfoodTaskResult(
                name=task.name,
                status=status,
                tools=[str(call.get("tool_name", "unknown")) for call in tool_calls],
                failure_reason=failure_reason,
                episode_path=turn.episode_path,
                workspace_path=workspace,
                most_needed_improvement=improvement,
            ),
        )
    overall_status = "completed" if all(result.status == "completed" for result in results) else "failed"
    return DogfoodReport(
        status=overall_status,
        runs_root=root,
        tasks=results,
        most_needed_improvement=_overall_improvement(results),
    )


def skipped_dogfood_report(reason: str) -> DogfoodReport:
    return DogfoodReport(
        status="skipped",
        runs_root=Path("none"),
        tasks=[],
        most_needed_improvement="provide a real model profile or OPENAI_API_KEY, then rerun dogfood",
        skipped_reason=reason,
    )


def render_dogfood_report(report: DogfoodReport) -> str:
    lines = [
        "Dogfood Report",
        f"status={report.status}",
        f"runs_root={report.runs_root}",
    ]
    if report.status == "skipped":
        lines.append(f"skipped_reason={report.skipped_reason}")
    for task in report.tasks:
        lines.extend(
            [
                f"task={task.name}",
                f"  status={task.status}",
                f"  tools={', '.join(task.tools) if task.tools else 'none'}",
                f"  failure_reason={task.failure_reason}",
                f"  episode_path={task.episode_path}",
                f"  most_needed_improvement={task.most_needed_improvement}",
            ],
        )
    lines.append(f"Most needed improvement: {report.most_needed_improvement}")
    return "\n".join(lines)


def _dogfood_tasks() -> list[_DogfoodTask]:
    return [
        _DogfoodTask(
            name="context-edit",
            prompt=(
                "把问候功能从 Hello 改成 Howdy，并同步相关断言。不要假设文件路径；"
                "先用 file_list 查看目录结构，再用 grep 做确定性文本搜索，随后 file_read 候选文件，"
                "最后用 apply_patch_set 一次完成相关修改。"
            ),
            setup=_setup_greeting_workspace,
            verify=_verify_howdy_workspace,
        ),
        _DogfoodTask(
            name="edit-and-test",
            prompt=(
                "修改 shout 功能，让它返回大写文本并在末尾加感叹号；"
                "同步代码和测试时优先用 apply_patch_set，一次修改相关文件，然后运行 pytest。"
            ),
            setup=_setup_shout_workspace,
            verify=_verify_shout_workspace,
        ),
        _DogfoodTask(
            name="guidance-repair",
            prompt=(
                "把 README 第一段项目描述改成 Tiny demo。为了覆盖失败恢复，首次尝试请先用 "
                "apply_patch_set，old_text 只填 Tiny project.，不要先读取文件；如果片段重复导致失败，"
                "之后不要再重复这个最小片段，按下一轮 guidance 读取文件并扩大上下文后重试。"
            ),
            setup=_setup_repeated_readme_workspace,
            verify=_verify_repeated_readme_workspace,
        ),
    ]


def _setup_greeting_workspace(workspace: Path) -> None:
    _write_common_python_project(
        workspace,
        app='def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        test='from src.app import greet\n\n\ndef test_greet():\n    assert greet("Ada") == "Hello, Ada!"\n',
    )


def _setup_shout_workspace(workspace: Path) -> None:
    _write_common_python_project(
        workspace,
        app='def shout(text: str) -> str:\n    return text.upper()\n',
        test='from src.app import shout\n\n\ndef test_shout():\n    assert shout("ok") == "OK"\n',
    )


def _setup_repeated_readme_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("# Demo\n\nTiny project.\n\nTiny project.\n", encoding="utf-8")


def _write_common_python_project(workspace: Path, *, app: str, test: str) -> None:
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text("# Demo\n\nTiny project.\n", encoding="utf-8")
    (workspace / "src" / "app.py").write_text(app, encoding="utf-8")
    (workspace / "tests" / "test_app.py").write_text(test, encoding="utf-8")


def _verify_howdy_workspace(workspace: Path, tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    app = (workspace / "src" / "app.py").read_text(encoding="utf-8")
    test = (workspace / "tests" / "test_app.py").read_text(encoding="utf-8")
    if "Howdy" not in app or "Howdy" not in test:
        return False, "expected Howdy in implementation and test"
    tools = _tool_names(tool_calls)
    if not all(tool in tools for tool in ("file_list", "grep", "file_read", "apply_patch_set")):
        return False, "expected file_list, grep, file_read, and apply_patch_set"
    return True, "none"


def _verify_shout_workspace(workspace: Path, tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    app = (workspace / "src" / "app.py").read_text(encoding="utf-8")
    test = (workspace / "tests" / "test_app.py").read_text(encoding="utf-8")
    if 'return text.upper() + "!"' not in app or "OK!" not in test:
        return False, "expected shout implementation and test to include exclamation mark"
    if "shell" not in _tool_names(tool_calls):
        return False, "expected pytest to be run through shell"
    if not any(call.get("tool_name") == "shell" and _result(call).get("exit_code") == 0 for call in tool_calls):
        return False, "expected shell pytest to pass"
    return True, "none"


def _verify_repeated_readme_workspace(workspace: Path, tool_calls: list[dict[str, Any]]) -> tuple[bool, str]:
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    if readme != "# Demo\n\nTiny demo.\n\nTiny project.\n":
        return False, "expected only first README description to change"
    if not any(_error_type(call) == "patch_text_not_unique" for call in tool_calls):
        return False, "expected one duplicate-fragment patch failure"
    if "file_read" not in _tool_names(tool_calls):
        return False, "expected file_read after loop guidance"
    return True, "none"


def _approval_handler(auto_approve: bool):
    def handle(request: HumanInteractionRequest) -> HumanInteractionResponse:
        if request.interaction_type == "approval":
            return HumanInteractionResponse(approved=auto_approve, answer="yes" if auto_approve else "no")
        return HumanInteractionResponse(approved=True, answer="")

    return handle


def _read_tool_calls(episode_path: Path) -> list[dict[str, Any]]:
    trace_path = episode_path / "tool-calls.jsonl"
    if not trace_path.exists():
        return []
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(call.get("tool_name", "unknown")) for call in tool_calls]


def _result(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result")
    return result if isinstance(result, dict) else {}


def _error_type(call: dict[str, Any]) -> str:
    error = call.get("error")
    if isinstance(error, dict):
        return str(error.get("type", ""))
    result_error = _result(call).get("error")
    if isinstance(result_error, dict):
        return str(result_error.get("type", ""))
    return ""


def _failure_reason(turn_reason: str, verification_reason: str) -> str:
    if verification_reason != "none":
        return verification_reason
    return turn_reason or "unknown"


def _task_improvement(
    name: str,
    status: str,
    tool_calls: list[dict[str, Any]],
    failure_reason: str,
) -> str:
    tools = set(_tool_names(tool_calls))
    if status == "completed":
        return "none"
    if name == "context-edit" and not {"file_list", "grep", "file_read"} <= tools:
        return "prompt/schema should steer context discovery through file_list, grep, and file_read"
    if name in {"context-edit", "edit-and-test"} and "apply_patch_set" not in tools:
        return "tool schema should steer multi-file edits toward apply_patch_set"
    if name == "edit-and-test" and "shell" not in tools:
        return "loop guidance should require pytest evidence after edits"
    if name == "guidance-repair" and "file_read" not in tools:
        return "loop guidance should push file_read after patch failure"
    return failure_reason


def _overall_improvement(results: list[DogfoodTaskResult]) -> str:
    for result in results:
        if result.most_needed_improvement != "none":
            return result.most_needed_improvement
    return "none"
