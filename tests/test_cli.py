"""
tests/test_cli.py - AgentFoundry CLI 测试

验证 run 子命令参数解析、runs-root 传递和 stdout 输出。
"""

from pathlib import Path

from agentfoundry import cli
from agentfoundry.runtime.state import RunStatus


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


def test_cli_run_uses_default_runs_root_and_prints_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path)])

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}
    output = capsys.readouterr().out
    assert "status=completed" in output
    assert f"episode_path={tmp_path / '.runs' / 'episode-1'}" in output


def test_cli_run_accepts_custom_runs_root(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    custom_runs = tmp_path / "custom-runs"
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(custom_runs / "episode-1")

    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path), "--runs-root", str(custom_runs)])

    assert exit_code == 0
    assert calls == {"runs_root": custom_runs, "task_path": task_path}
