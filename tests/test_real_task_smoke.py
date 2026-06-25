"""
tests/test_real_task_smoke.py - 真实任务 smoke pack v1

用 AgentSession 驱动真实工具，在临时工作区验证自然语言任务的端到端能力。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haagent.cli_inspect import render_episode_summary
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.chat_session import AgentSession
from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse


class ScriptedGateway:
    provider_name = "real-task-smoke"

    def __init__(self, steps: list[ModelResponse]) -> None:
        self._steps = steps
        self.calls: list[dict[str, Any]] = []

    def generate(self, task, model_input, tool_schemas, observations):
        self.calls.append(
            {
                "task": task,
                "model_input": model_input,
                "tool_schemas": list(tool_schemas),
                "observations": list(observations),
            },
        )
        index = min(len(self.calls) - 1, len(self._steps) - 1)
        return self._steps[index]


def test_real_task_smoke_understands_small_project_and_summarizes_structure(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse("list files", [ToolCall("file_list", {})]),
            ModelResponse("read package", [ToolCall("file_read", {"path": "src/app.py"})]),
            ModelResponse("Project has src/app.py, README.md, and tests/test_app.py.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "理解这个小项目并总结结构")

    assert result.status == "completed"
    assert "src/app.py" in result.final_response
    assert _tool_names(result.episode_path) == ["file_list", "file_read"]
    assert _transcript_events(result.episode_path).count("model_call") == 3


def test_real_task_smoke_modifies_markdown_file_with_approval(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "update README",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "README.md",
                            "old_text": "# Demo\n\nTiny project.\n",
                            "new_text": "# Demo\n\nTiny project.\n\n## Usage\n\nRun pytest.\n",
                        },
                    ),
                ],
            ),
            ModelResponse("README updated.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "给 README 增加使用说明", approvals=True)

    assert result.status == "completed"
    assert "## Usage" in (workspace / "README.md").read_text(encoding="utf-8")
    assert _tool_names(result.episode_path) == ["apply_patch"]
    assert "approval_requested" in _transcript_events(result.episode_path)
    assert "approval_granted" in _transcript_events(result.episode_path)


def test_real_task_smoke_modifies_python_file_and_runs_tests(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "change greeting",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "src/app.py",
                            "old_text": 'return f"Hello, {name}!"',
                            "new_text": 'return f"Hi, {name}!"',
                        },
                    ),
                ],
            ),
            ModelResponse(
                "update test",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "tests/test_app.py",
                            "old_text": 'assert greet("Ada") == "Hello, Ada!"',
                            "new_text": 'assert greet("Ada") == "Hi, Ada!"',
                        },
                    ),
                ],
            ),
            ModelResponse("run tests", [ToolCall("shell", {"command": "python -m pytest -q", "timeout_seconds": 20})]),
            ModelResponse("Greeting updated and tests pass.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "把 greet 的文案改成 Hi 并跑测试", approvals=True)

    assert result.status == "completed"
    assert 'return f"Hi, {name}!"' in (workspace / "src" / "app.py").read_text(encoding="utf-8")
    shell_call = _tool_calls(result.episode_path)[-1]
    assert shell_call["tool_name"] == "shell"
    assert shell_call["result"]["exit_code"] == 0


def test_real_task_smoke_context_find_read_patch_set_and_runs_tests(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "find greeting feature",
                [ToolCall("context_find", {"query": "greeting function and its test"})],
            ),
            ModelResponse("read app", [ToolCall("file_read", {"path": "src/app.py", "keyword": "greet", "limit": 20})]),
            ModelResponse("read test", [ToolCall("file_read", {"path": "tests/test_app.py", "keyword": "test_greet", "limit": 20})]),
            ModelResponse(
                "edit implementation and test",
                [
                    ToolCall(
                        "apply_patch_set",
                        {
                            "replacements": [
                                {
                                    "path": "src/app.py",
                                    "old_text": 'return f"Hello, {name}!"',
                                    "new_text": 'return f"Hi, {name}!"',
                                },
                                {
                                    "path": "tests/test_app.py",
                                    "old_text": 'assert greet("Ada") == "Hello, Ada!"',
                                    "new_text": 'assert greet("Ada") == "Hi, Ada!"',
                                },
                            ],
                        },
                    ),
                ],
            ),
            ModelResponse("run tests", [ToolCall("shell", {"command": "python -m pytest -q", "timeout_seconds": 20})]),
            ModelResponse("Greeting feature updated and tests pass.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "把问候功能改成 Hi，并同步测试后运行 pytest", approvals=True)

    assert result.status == "completed"
    assert [call["tool_name"] for call in _tool_calls(result.episode_path)] == [
        "context_find",
        "file_read",
        "file_read",
        "apply_patch_set",
        "shell",
    ]
    assert 'return f"Hi, {name}!"' in (workspace / "src" / "app.py").read_text(encoding="utf-8")
    assert 'assert greet("Ada") == "Hi, Ada!"' in (workspace / "tests" / "test_app.py").read_text(encoding="utf-8")
    assert _tool_calls(result.episode_path)[-1]["result"]["exit_code"] == 0


def test_real_task_smoke_runs_script_validation(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "run script",
                [
                    ToolCall(
                        "shell",
                        {"command": "python scripts/check.py", "timeout_seconds": 20},
                    ),
                ],
            ),
            ModelResponse("Script validation passed.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "运行项目检查脚本", approvals=True)

    assert result.status == "completed"
    call = _tool_calls(result.episode_path)[0]
    assert call["tool_name"] == "shell"
    assert call["result"]["stdout_excerpt"].strip() == "check ok"


def test_real_task_smoke_request_user_input_then_writes_requested_file(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "need filename",
                [ToolCall("request_user_input", {"question": "Which note file?", "reason": "Need target"})],
            ),
            ModelResponse(
                "write note",
                [
                    ToolCall(
                        "file_write",
                        {"path": "notes/today.md", "content": "Answered by user.\n", "mode": "create"},
                    ),
                ],
            ),
            ModelResponse("Created requested note.", []),
        ],
    )

    result = _run_chat(
        workspace,
        gateway,
        "创建用户指定的笔记文件",
        answers={"Which note file?": "notes/today.md"},
        approvals=True,
    )

    assert result.status == "completed"
    assert (workspace / "notes" / "today.md").read_text(encoding="utf-8") == "Answered by user.\n"
    events = _transcript_events(result.episode_path)
    assert "user_input_requested" in events
    assert "user_input_received" in events


def test_real_task_smoke_high_risk_write_requires_approval(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "write generated file",
                [ToolCall("file_write", {"path": "generated.txt", "content": "approved\n", "mode": "create"})],
            ),
            ModelResponse("Generated file created.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "创建 generated.txt", approvals=True)

    assert result.status == "completed"
    assert (workspace / "generated.txt").read_text(encoding="utf-8") == "approved\n"
    assert ["approval_requested", "approval_granted"] == [
        event for event in _transcript_events(result.episode_path) if event.startswith("approval_")
    ]


def test_real_task_smoke_denied_approval_does_not_modify_file(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    original = (workspace / "README.md").read_text(encoding="utf-8")
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "try denied patch",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "README.md",
                            "old_text": "Tiny project.",
                            "new_text": "Should not be written.",
                        },
                    ),
                ],
            ),
        ],
    )

    result = _run_chat(workspace, gateway, "尝试修改 README 但用户拒绝", approvals=False)

    assert result.status == "failed"
    assert result.failure_category == "User Denied Failure"
    assert (workspace / "README.md").read_text(encoding="utf-8") == original
    assert _tool_calls(result.episode_path)[0]["error"]["type"] == "approval_denied"


def test_real_task_smoke_failed_task_is_clear_in_inspect(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "bad read",
                [ToolCall("file_read", {"path": "missing.py"})],
            ),
        ],
    )

    result = _run_chat(workspace, gateway, "读取不存在的文件")
    summary = render_episode_summary(result.episode_path)

    assert result.status == "failed"
    assert result.failure_category == "Tool Argument Failure"
    assert "Structured Failure" in summary
    assert "Tool Argument Failure" in summary
    assert "path does not exist: missing.py" in summary
    assert "file_read: path does not exist: missing.py" in summary


def test_real_task_smoke_recovers_from_wrong_file_path_with_suggestions(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse("guess path", [ToolCall("file_read", {"path": "app.py"})]),
            ModelResponse("use suggested path", [ToolCall("file_read", {"path": "src/app.py"})]),
            ModelResponse("Read the app after suggestion.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "读取 app.py")

    assert result.status == "completed"
    calls = _tool_calls(result.episode_path)
    assert [call["tool_name"] for call in calls] == ["file_read", "file_read"]
    assert calls[0]["error"]["type"] == "tool_argument_invalid"
    assert calls[0]["result"] is None
    assert Path(calls[1]["result"]["path"]).parts[-2:] == ("src", "app.py")
    assert "loop_suggestion_added" in _transcript_events(result.episode_path)


def test_real_task_smoke_reads_file_after_patch_miss_then_repairs(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "bad patch",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "README.md",
                            "old_text": "Tiny old project.",
                            "new_text": "Tiny repaired project.",
                        },
                    ),
                ],
            ),
            ModelResponse("read current file", [ToolCall("file_read", {"path": "README.md"})]),
            ModelResponse(
                "patch exact text",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "README.md",
                            "old_text": "Tiny project.",
                            "new_text": "Tiny repaired project.",
                        },
                    ),
                ],
            ),
            ModelResponse("Patch repaired.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "修正 README 文案", approvals=True)

    assert result.status == "completed"
    assert "Tiny repaired project." in (workspace / "README.md").read_text(encoding="utf-8")
    assert [call["tool_name"] for call in _tool_calls(result.episode_path)] == [
        "apply_patch",
        "file_read",
        "apply_patch",
    ]
    assert "loop_suggestion_added" in _transcript_events(result.episode_path)


def test_real_task_smoke_patch_set_failure_does_not_partially_write(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    original_app = (workspace / "src" / "app.py").read_text(encoding="utf-8")
    original_test = (workspace / "tests" / "test_app.py").read_text(encoding="utf-8")
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "try multi edit",
                [
                    ToolCall(
                        "apply_patch_set",
                        {
                            "replacements": [
                                {
                                    "path": "src/app.py",
                                    "old_text": 'return f"Hello, {name}!"',
                                    "new_text": 'return f"Hi, {name}!"',
                                },
                                {
                                    "path": "tests/test_app.py",
                                    "old_text": "missing assertion",
                                    "new_text": 'assert greet("Ada") == "Hi, Ada!"',
                                },
                            ],
                        },
                    ),
                ],
            ),
            ModelResponse("read app after failure", [ToolCall("file_read", {"path": "src/app.py", "keyword": "greet", "limit": 20})]),
            ModelResponse(
                "read test after failure",
                [ToolCall("file_read", {"path": "tests/test_app.py", "keyword": "test_greet", "limit": 20})],
            ),
            ModelResponse("No partial write occurred after the failed patch set.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "确认 apply_patch_set 失败不会部分写入", approvals=True)

    assert result.status == "completed"
    calls = _tool_calls(result.episode_path)
    call = calls[0]
    assert call["tool_name"] == "apply_patch_set"
    assert call["error"]["type"] == "patch_text_not_found"
    assert [item["tool_name"] for item in calls] == ["apply_patch_set", "file_read", "file_read"]
    assert (workspace / "src" / "app.py").read_text(encoding="utf-8") == original_app
    assert (workspace / "tests" / "test_app.py").read_text(encoding="utf-8") == original_test


def test_real_task_smoke_repeated_patch_set_fragment_reads_then_uses_longer_context(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    (workspace / "README.md").write_text("# Demo\n\nTiny project.\n\nTiny project.\n", encoding="utf-8")
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "ambiguous edit",
                [
                    ToolCall(
                        "apply_patch_set",
                        {"replacements": [{"path": "README.md", "old_text": "Tiny project.", "new_text": "Tiny demo."}]},
                    ),
                ],
            ),
            ModelResponse("read current file", [ToolCall("file_read", {"path": "README.md"})]),
            ModelResponse(
                "edit with larger context",
                [
                    ToolCall(
                        "apply_patch_set",
                        {
                            "replacements": [
                                {
                                    "path": "README.md",
                                    "old_text": "# Demo\n\nTiny project.\n\n",
                                    "new_text": "# Demo\n\nTiny demo.\n\n",
                                },
                            ],
                        },
                    ),
                ],
            ),
            ModelResponse("Repeated fragment repaired with longer context.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "把 README 第一段项目描述改成 Tiny demo", approvals=True)

    assert result.status == "completed"
    assert [call["tool_name"] for call in _tool_calls(result.episode_path)] == [
        "apply_patch_set",
        "file_read",
        "apply_patch_set",
    ]
    assert _tool_calls(result.episode_path)[0]["error"]["type"] == "patch_text_not_unique"
    assert (workspace / "README.md").read_text(encoding="utf-8") == "# Demo\n\nTiny demo.\n\nTiny project.\n"


def test_real_task_smoke_agent_runs_validation_then_completes(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse("run validation", [ToolCall("shell", {"command": "python -m pytest -q", "timeout_seconds": 20})]),
            ModelResponse("Validated with pytest.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "运行测试验证项目", approvals=True)

    assert result.status == "completed"
    calls = _tool_calls(result.episode_path)
    assert calls[0]["tool_name"] == "shell"
    assert calls[0]["result"]["exit_code"] == 0


def test_real_task_smoke_finds_context_before_edit_without_path(tmp_path: Path) -> None:
    workspace = _make_project_workspace(tmp_path)
    gateway = ScriptedGateway(
        [
            ModelResponse(
                "find greeting implementation",
                [ToolCall("context_find", {"query": "greeting function implementation"})],
            ),
            ModelResponse("read candidate", [ToolCall("file_read", {"path": "src/app.py", "keyword": "greet", "limit": 20})]),
            ModelResponse(
                "patch found file",
                [
                    ToolCall(
                        "apply_patch",
                        {
                            "path": "src/app.py",
                            "old_text": 'return f"Hello, {name}!"',
                            "new_text": 'return f"Howdy, {name}!"',
                        },
                    ),
                ],
            ),
            ModelResponse("Greeting implementation updated.", []),
        ],
    )

    result = _run_chat(workspace, gateway, "把问候函数的输出改成 Howdy", approvals=True)

    assert result.status == "completed"
    assert 'return f"Howdy, {name}!"' in (workspace / "src" / "app.py").read_text(encoding="utf-8")
    assert [call["tool_name"] for call in _tool_calls(result.episode_path)] == [
        "context_find",
        "file_read",
        "apply_patch",
    ]
    assert "loop_suggestion_added" in _transcript_events(result.episode_path)


def _run_chat(
    workspace: Path,
    gateway: ScriptedGateway,
    prompt: str,
    *,
    answers: dict[str, str] | None = None,
    approvals: bool | None = None,
):
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=gateway,
        max_turns=12,
    )
    return session.run_prompt_events(
        prompt,
        interaction_handler=_interaction_handler(answers or {}, approvals),
    )


def _interaction_handler(answers: dict[str, str], approvals: bool | None):
    def handle(request: HumanInteractionRequest) -> HumanInteractionResponse:
        if request.interaction_type == "approval":
            return HumanInteractionResponse(approved=bool(approvals), answer="yes" if approvals else "no")
        return HumanInteractionResponse(approved=True, answer=answers.get(request.question, ""))

    return handle


def _make_project_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "scripts").mkdir()
    (workspace / "notes").mkdir()
    (workspace / "README.md").write_text("# Demo\n\nTiny project.\n", encoding="utf-8")
    (workspace / "src" / "app.py").write_text(
        'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        encoding="utf-8",
    )
    (workspace / "tests" / "test_app.py").write_text(
        "from src.app import greet\n\n\ndef test_greet():\n    assert greet(\"Ada\") == \"Hello, Ada!\"\n",
        encoding="utf-8",
    )
    (workspace / "scripts" / "check.py").write_text("print('check ok')\n", encoding="utf-8")
    return workspace


def _tool_calls(episode_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _tool_names(episode_path: Path) -> list[str]:
    return [str(call["tool_name"]) for call in _tool_calls(episode_path)]


def _transcript_events(episode_path: Path) -> list[str]:
    return [
        str(json.loads(line)["event"])
        for line in (episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
