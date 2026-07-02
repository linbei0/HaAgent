"""
tests/integration/cli/test_cli.py - HaAgent CLI 聚合入口测试

验证用户文档中的 tests/test_cli.py 验收入口至少覆盖 run 子命令的新 authoring 参数。
"""

from pathlib import Path

from haagent import cli


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
