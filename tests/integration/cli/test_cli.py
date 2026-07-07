"""
tests/integration/cli/test_cli.py - HaAgent CLI 聚合入口测试

验证用户文档中的 tests/test_cli.py 验收入口至少覆盖 run 子命令的新 authoring 参数。
"""

import json
from pathlib import Path

from haagent import cli


def test_cli_sandbox_status_outputs_default_local_status(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    exit_code = cli.main(["sandbox", "status"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "backend=local_subprocess" in output
    assert "degraded=true" in output
    assert "haagent sandbox enable docker" in output


def test_cli_sandbox_enable_and_disable_write_settings(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    settings_path = config_dir / "settings.json"
    active_model = {"connection_id": "local", "model": "deepseek-chat"}
    settings_path.write_text(
        json.dumps({"interactive_max_turns": 80, "active_model": active_model}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: home)

    enable_code = cli.main(["sandbox", "enable", "docker"])
    enable_output = capsys.readouterr().out
    enabled = json.loads(settings_path.read_text(encoding="utf-8"))
    disable_code = cli.main(["sandbox", "disable"])
    disable_output = capsys.readouterr().out
    disabled = json.loads(settings_path.read_text(encoding="utf-8"))

    assert enable_code == 0
    assert "backend=docker" in enable_output
    assert enabled["active_model"] == active_model
    assert enabled["interactive_max_turns"] == 80
    assert enabled["sandbox"]["enabled"] is True
    assert enabled["sandbox"]["fail_if_unavailable"] is True
    assert disable_code == 0
    assert "backend=local_subprocess" in disable_output
    assert disabled["active_model"] == active_model
    assert disabled["sandbox"]["enabled"] is False


def test_cli_sandbox_parser_accepts_status_doctor_enable_disable() -> None:
    parser = cli.build_parser()

    status = parser.parse_args(["sandbox", "status"])
    doctor = parser.parse_args(["sandbox", "doctor"])
    enable = parser.parse_args(["sandbox", "enable", "docker", "--allow-fallback"])
    disable = parser.parse_args(["sandbox", "disable"])

    assert status.command == "sandbox"
    assert status.sandbox_action == "status"
    assert doctor.sandbox_action == "doctor"
    assert enable.sandbox_action == "enable"
    assert enable.backend == "docker"
    assert enable.fail_if_unavailable is False
    assert disable.sandbox_action == "disable"


def test_cli_run_parser_accepts_goal_authoring_arguments() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run",
            "--goal",
            "Fix a small bug.",
            "--workspace-root",
            "examples/workspaces/hello",
            "--verify",
            "uv run pytest",
            "--provider",
            "fake",
        ],
    )

    assert args.task_yaml is None
    assert args.goal == "Fix a small bug."
    assert args.workspace_root == Path("examples/workspaces/hello")
    assert args.verify == "uv run pytest"
    assert args.provider == "fake"


def test_cli_chat_parser_accepts_explicit_web_flag() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["chat", "--web", "Search current docs", "--provider", "fake"])

    assert args.command == "chat"
    assert args.request == "Search current docs"
    assert args.enable_web is True
    assert args.handler.__name__ == "handle_tui_migration"


def test_default_parser_starts_tui_with_explicit_web_flag() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["--web"])

    assert args.command is None
    assert args.enable_web is True


def test_cli_tui_subcommand_reports_migration() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["tui", "--web"])

    assert args.command == "tui"
    assert args.enable_web is True
    assert args.handler.__name__ == "handle_tui_migration"


def test_root_help_only_shows_tui_entry() -> None:
    parser = cli.build_parser()

    help_text = parser.format_help()

    assert "open the Textual TUI" in help_text
    assert "/model" in help_text
    assert "==SUPPRESS==" not in help_text
    assert "run a task.yaml file" not in help_text
    assert "haagent chat" not in help_text
