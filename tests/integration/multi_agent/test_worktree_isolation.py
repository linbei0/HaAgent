"""
tests/integration/multi_agent/test_worktree_isolation.py - worker worktree 隔离集成测试

验证代码类 worker 在自己的 Git worktree 中修改文件，不污染主工作区。
"""

import subprocess
from pathlib import Path

from haagent.models.fake import FakeModelGateway
from haagent.models.types import ModelResponse, ToolCall
from haagent.multi_agent.profiles import WorkerProfileRuntime
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


def test_code_worker_writes_only_inside_its_worktree(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")

    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=repo,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(
            ModelResponse(
                content="write README",
                tool_calls=[
                    ToolCall(
                        name="file_write",
                        args={"path": "README.md", "content": "changed in worktree\n", "mode": "overwrite"},
                        id="call-write-readme",
                    ),
                    ToolCall(
                        name="file_read",
                        args={"path": "README.md"},
                        id="call-read-readme",
                    ),
                ],
            ),
        ),
        path_policy=default_path_policy(repo),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "fake_tool", "file_read", "file_write"],
        inherited_approval_allowed_tools=["file_write"],
        inherited_approved_tools=["file_write"],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
    )
    runtime._profile_resolver = lambda *args, **kwargs: WorkerProfileRuntime(
        name="code-worker",
        subagent_type="worker",
        system_prompt="你是代码实现助手。",
        model_profile=None,
        allowed_tools=["fake_tool", "file_write", "file_read"],
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
        max_turns=None,
        enable_web=None,
        backend="subprocess",
        worktree=True,
    )

    worker = runtime.spawn_worker(
        description="edit code",
        prompt="modify README.md",
        subagent_type="worker",
    )

    assert worker["worktree_path"] != str(repo)
    finished = runtime.wait_for_task(worker["task_id"], timeout=15)
    assert finished["status"] == "completed"
    worktree_path = Path(worker["worktree_path"])
    assert (repo / "README.md").read_text(encoding="utf-8") == "original\n"
    assert (worktree_path / "README.md").read_text(encoding="utf-8") == "changed in worktree\n"


def _init_git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)
    return path
