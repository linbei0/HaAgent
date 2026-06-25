"""
tests/test_dogfood.py - Real Model Dogfood 入口测试

验证 dogfood runner 使用真实 runtime 路径，并在无真实模型配置时显式跳过。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haagent import cli
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.dogfood import render_dogfood_report, run_dogfood_tasks


class ScriptedGateway:
    provider_name = "scripted-dogfood"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, tool_schemas):
        model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        self.calls.append(
            {
                "model_input": model_input,
                "tool_schemas": list(tool_schemas),
                "messages": list(messages),
            },
        )
        index = len(self.calls) - 1
        return self._responses[index]


def test_dogfood_runner_uses_runtime_tools_and_records_granted_approval(tmp_path: Path) -> None:
    gateway = ScriptedGateway(
        [
            ModelResponse("find greet", [ToolCall("context_find", {"query": "greeting behavior"})]),
            ModelResponse("read app", [ToolCall("file_read", {"path": "src/app.py", "keyword": "greet", "limit": 20})]),
            ModelResponse(
                "patch app and test",
                [
                    ToolCall(
                        "apply_patch_set",
                        {
                            "replacements": [
                                {
                                    "path": "src/app.py",
                                    "old_text": 'return f"Hello, {name}!"',
                                    "new_text": 'return f"Howdy, {name}!"',
                                },
                                {
                                    "path": "tests/test_app.py",
                                    "old_text": 'assert greet("Ada") == "Hello, Ada!"',
                                    "new_text": 'assert greet("Ada") == "Howdy, Ada!"',
                                },
                            ],
                        },
                    ),
                ],
            ),
            ModelResponse("done task 1", []),
            ModelResponse(
                "patch shout",
                [
                    ToolCall(
                        "apply_patch_set",
                        {
                            "replacements": [
                                {
                                    "path": "src/app.py",
                                    "old_text": "return text.upper()",
                                    "new_text": 'return text.upper() + "!"',
                                },
                                {
                                    "path": "tests/test_app.py",
                                    "old_text": 'assert shout("ok") == "OK"',
                                    "new_text": 'assert shout("ok") == "OK!"',
                                },
                            ],
                        },
                    ),
                ],
            ),
            ModelResponse("run pytest", [ToolCall("shell", {"command": "python -m pytest -q", "timeout_seconds": 20})]),
            ModelResponse("done task 2", []),
            ModelResponse(
                "ambiguous patch",
                [
                    ToolCall(
                        "apply_patch_set",
                        {"replacements": [{"path": "README.md", "old_text": "Tiny project.", "new_text": "Tiny demo."}]},
                    ),
                ],
            ),
            ModelResponse("read after guidance", [ToolCall("file_read", {"path": "README.md"})]),
            ModelResponse(
                "retry with context",
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
            ModelResponse("done task 3", []),
        ],
    )

    report = run_dogfood_tasks(gateway, runs_root=tmp_path / "runs", max_turns=12, auto_approve=True)

    assert report.status == "completed"
    assert [task.status for task in report.tasks] == ["completed", "completed", "completed"]
    assert "context_find" in report.tasks[0].tools
    assert "apply_patch_set" in report.tasks[0].tools
    assert "shell" in report.tasks[1].tools
    assert report.tasks[2].failure_reason == "none"
    assert "Patch text is not unique" in _transcript_text(report.tasks[2].episode_path)
    first_tool_call = _tool_calls(report.tasks[0].episode_path)[2]
    assert first_tool_call["tool_name"] == "apply_patch_set"
    assert first_tool_call["policy"]["approval"]["status"] == "granted"
    assert (report.tasks[0].episode_path / "contexts").exists()
    assert "Prefer context_find before file_search" in gateway.calls[0]["model_input"]
    assert "Prefer this over repeated apply_patch calls" in json.dumps(gateway.calls[0]["tool_schemas"])
    assert "Most needed improvement: none" in render_dogfood_report(report)


def test_cli_dogfood_without_real_model_config_skips_explicitly(capsys) -> None:
    exit_code = cli.main(["dogfood"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status=skipped" in captured.out
    assert "provide --profile or --provider" in captured.out


def _tool_calls(episode_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _context_text(episode_path: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((episode_path / "contexts").glob("*.json"))
        if not path.name.endswith("-manifest.json")
    )


def _transcript_text(episode_path: Path) -> str:
    transcript_path = episode_path / "transcript.jsonl"
    if not transcript_path.exists():
        return ""
    return transcript_path.read_text(encoding="utf-8")
