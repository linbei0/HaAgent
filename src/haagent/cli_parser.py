"""
haagent/cli_parser.py - CLI 参数解析器构建

集中定义 HaAgent 子命令参数，并将解析结果绑定到对应 command handler。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from haagent.cli_commands import (
    handle_check,
    handle_chat,
    handle_dogfood,
    handle_eval,
    handle_export_eval,
    handle_inspect,
    handle_memory,
    handle_run,
    handle_sessions,
    handle_setup,
    handle_smoke,
    handle_tui,
)
from haagent.cli_runtime import CliRuntime


def build_cli_parser(runtime: CliRuntime) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haagent", description="HaAgent local personal assistant")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for the assistant session (default: current directory)",
    )
    parser.add_argument("--resume", help="resume a chat session by session id or session package path")
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_session",
        help="resume the latest chat session for the current workspace",
    )
    _add_runs_root(parser, help_text="directory for assistant session packages (default: .runs)")
    _add_model_provider(
        parser,
        default=None,
        provider_help="model provider override; omit to use the active profile from haagent setup",
    )
    parser.set_defaults(command="chat", request=None, handler=lambda args: handle_chat(args, runtime))
    subparsers = parser.add_subparsers(dest="command", required=False)

    setup_parser = subparsers.add_parser("setup", help="configure the default model profile")
    setup_parser.set_defaults(handler=handle_setup)

    sessions_parser = subparsers.add_parser("sessions", help="list recent sessions for this workspace")
    sessions_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root to list sessions for (default: current directory)",
    )
    _add_runs_root(sessions_parser, help_text="directory for assistant session packages (default: .runs)")
    sessions_parser.set_defaults(handler=handle_sessions)

    memory_parser = subparsers.add_parser("memory", help="review long-term memory candidates")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_action", required=True)
    memory_list = memory_subparsers.add_parser("list", help="list pending memory candidates")
    _add_memory_common_args(memory_list)
    memory_list.add_argument("--all", action="store_true", help="include confirmed and rejected candidates")
    memory_list.set_defaults(handler=handle_memory)
    memory_confirm = memory_subparsers.add_parser("confirm", help="confirm a pending memory candidate")
    memory_confirm.add_argument("candidate_id", help="candidate id to confirm")
    _add_memory_common_args(memory_confirm)
    memory_confirm.add_argument("--title", help="edited title to commit")
    memory_confirm.add_argument("--body", help="edited body to commit")
    memory_confirm.add_argument("--tag", action="append", help="edited tag; repeat for multiple tags")
    memory_confirm.set_defaults(handler=handle_memory)
    memory_reject = memory_subparsers.add_parser("reject", help="reject a pending memory candidate")
    memory_reject.add_argument("candidate_id", help="candidate id to reject")
    _add_memory_common_args(memory_reject)
    memory_reject.add_argument("--reason", default="rejected by user", help="rejection reason")
    memory_reject.set_defaults(handler=handle_memory)

    tui_parser = subparsers.add_parser("tui", help="open the HaAgent terminal UI")
    tui_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for the TUI session (default: current directory)",
    )
    _add_runs_root(tui_parser, help_text="directory for assistant session packages (default: .runs)")
    tui_parser.set_defaults(handler=handle_tui)

    run_parser = subparsers.add_parser("run", help="run a task.yaml file")
    run_parser.add_argument("task_yaml", nargs="?", type=Path, help="path to task.yaml")
    run_parser.add_argument("--goal", help="task goal used when task_yaml is omitted")
    run_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root used when task_yaml is omitted",
    )
    run_parser.add_argument("--verify", help="verification command used when task_yaml is omitted")
    _add_runs_root(run_parser, help_text="directory for episode packages (default: .runs)")
    _add_model_provider(run_parser)
    _add_max_turns(
        run_parser,
        default=3,
        help_text="maximum model/tool turns before failing the run (default: 3)",
    )
    run_parser.set_defaults(handler=lambda args: handle_run(args, runtime))

    chat_parser = subparsers.add_parser("chat", help="run a natural language request")
    chat_parser.add_argument(
        "request",
        nargs="?",
        help="natural language request to run in the workspace; omit to enter REPL",
    )
    chat_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for the chat request (default: current directory)",
    )
    chat_parser.add_argument("--resume", help="resume a chat session by session id or session package path")
    chat_parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_session",
        help="resume the latest chat session for the current workspace",
    )
    _add_runs_root(chat_parser, help_text="directory for assistant session packages (default: .runs)")
    _add_model_provider(
        chat_parser,
        default=None,
        provider_help="model provider override; omit to use the active profile from haagent setup",
    )
    chat_parser.set_defaults(handler=lambda args: handle_chat(args, runtime))

    smoke_parser = subparsers.add_parser("smoke", help="run the minimal HaAgent smoke suite")
    _add_runs_root(smoke_parser, help_text="directory for episode packages (default: .runs)")
    smoke_parser.add_argument("--profile", help="real provider profile name from .haagent/providers.json")
    _add_max_turns(
        smoke_parser,
        default=12,
        help_text="maximum model/tool turns per smoke task (default: 12)",
    )
    smoke_parser.set_defaults(handler=lambda args: handle_smoke(args, runtime))

    dogfood_parser = subparsers.add_parser(
        "dogfood",
        help="run manual real-model dogfood tasks outside default CI",
    )
    dogfood_parser.add_argument(
        "--runs-root",
        type=Path,
        help="directory for dogfood episode packages; defaults to a temporary directory",
    )
    dogfood_parser.add_argument("--profile", help="real provider profile name from .haagent/providers.json")
    _add_model_provider(
        dogfood_parser,
        choices=["openai", "openai-chat"],
        default=None,
        provider_help="real provider to use when --profile is omitted",
        model_help="model name for --provider dogfood runs",
        base_url_help="OpenAI-compatible base URL for --provider dogfood runs",
        include_profile=False,
    )
    _add_max_turns(
        dogfood_parser,
        default=16,
        help_text="maximum model/tool turns per dogfood task (default: 16)",
    )
    dogfood_parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="deny high-risk tool approvals instead of auto-granting them",
    )
    dogfood_parser.set_defaults(handler=lambda args: handle_dogfood(args, runtime))

    inspect_parser = subparsers.add_parser("inspect", help="inspect an episode package")
    inspect_parser.add_argument("episode_path", type=Path, help="path to an episode directory")
    inspect_parser.set_defaults(handler=lambda args: handle_inspect(args))

    export_eval_parser = subparsers.add_parser("export-eval", help="export an eval case JSON")
    export_eval_parser.add_argument(
        "episode_paths",
        nargs="+",
        type=Path,
        help="path to one or more episode directories",
    )
    export_eval_parser.add_argument(
        "--output",
        type=Path,
        help="write eval case JSON to this file instead of stdout",
    )
    export_eval_parser.add_argument(
        "--output-dir",
        type=Path,
        help="write one eval case JSON file per episode into this existing directory",
    )
    export_eval_parser.set_defaults(handler=lambda args: handle_export_eval(args))

    eval_parser = subparsers.add_parser("eval", help="run exported eval case JSON locally")
    eval_parser.add_argument("eval_path", type=Path, help="eval case JSON, directory, or batch manifest")
    eval_parser.add_argument(
        "--output",
        type=Path,
        help="write eval report JSON to this file instead of only printing a summary",
    )
    _add_runs_root(eval_parser, help_text="directory for eval run episode packages (default: .runs)")
    _add_model_provider(eval_parser)
    eval_parser.set_defaults(handler=lambda args: handle_eval(args, runtime))

    check_parser = subparsers.add_parser("check", help="run the local HaAgent quality gate")
    check_parser.add_argument(
        "--eval-path",
        type=Path,
        default=runtime.project_root / "examples" / "evals",
        help="eval suite path to run (default: examples/evals)",
    )
    check_parser.add_argument("--output", type=Path, help="write check report JSON to this file")
    _add_runs_root(check_parser, help_text="directory for check episode packages (default: .runs)")
    check_parser.add_argument(
        "--pytest",
        action="store_true",
        help="also run uv run pytest -q after the eval suite",
    )
    _add_model_provider(
        check_parser,
        provider_help="model provider for eval replay (default: fake)",
    )
    check_parser.set_defaults(handler=lambda args: handle_check(args, runtime))
    return parser


def _add_runs_root(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(".runs"),
        help=help_text,
    )


def _add_memory_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for memory review (default: current directory)",
    )
    parser.add_argument("--session", help="session id or session package path")
    _add_runs_root(parser, help_text="directory for assistant session packages (default: .runs)")


def _add_model_provider(
    parser: argparse.ArgumentParser,
    *,
    choices: list[str] | None = None,
    default: str | None = "fake",
    provider_help: str = "model provider to use (default: fake)",
    model_help: str = "OpenAI model name; only used when --provider openai",
    base_url_help: str = "OpenAI-compatible Responses API base URL; only used when --provider openai",
    include_profile: bool = True,
) -> None:
    parser.add_argument(
        "--provider",
        choices=choices or ["fake", "openai", "openai-chat"],
        default=default,
        help=provider_help,
    )
    if include_profile:
        parser.add_argument("--profile", help="provider profile name from .haagent/providers.json")
    parser.add_argument("--model", help=model_help)
    parser.add_argument("--base-url", help=base_url_help)


def _add_max_turns(parser: argparse.ArgumentParser, *, default: int, help_text: str) -> None:
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=default,
        help=help_text,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-turns must be a positive integer")
    return parsed
