"""
haagent/cli_parser.py - CLI 参数解析器构建

集中定义 HaAgent 子命令参数，并将解析结果绑定到对应 command handler。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from haagent.cli_commands import (
    handle_check,
    handle_dogfood,
    handle_eval,
    handle_export_eval,
    handle_gateway,
    handle_inspect,
    handle_run,
    handle_sandbox,
    handle_schedule_worker,
    handle_smoke,
    handle_tui_entry,
    handle_tui_migration,
)
from haagent.cli_runtime import CliRuntime
from haagent.models.config.connections import user_runs_dir
from haagent.runtime.settings import (
    DEFAULT_DOGFOOD_MAX_TURNS,
    DEFAULT_RUN_MAX_TURNS,
    DEFAULT_SMOKE_MAX_TURNS,
)


class _RootHelpParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return (
            "usage: haagent [--workspace-root PATH] [--runs-root PATH] "
            "[--resume SESSION | --continue] [--web]\n\n"
            "HaAgent local personal assistant\n\n"
            "ordinary interactive entry:\n"
            "  haagent              open the Textual TUI in the current directory\n\n"
            "options:\n"
            "  -h, --help           show this help message and exit\n"
            "  --workspace-root PATH\n"
            "                       workspace root for the assistant session\n"
            "  --runs-root PATH     directory for assistant session packages (default: ~/.haagent/runs)\n"
            "  --resume SESSION     resume a session by id or session package path\n"
            "  --continue           resume the latest session for the current workspace\n"
            "  --web                enable read-only web tools for the TUI session\n\n"
            "inside the TUI:\n"
            "  use /model, /sessions, /memory, /web, /new, /resume, /cancel, /help\n"
        )


def build_cli_parser(runtime: CliRuntime) -> argparse.ArgumentParser:
    parser = _RootHelpParser(prog="haagent", description="HaAgent local personal assistant")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root for the assistant session (default: current directory)",
    )
    parser.add_argument("--resume", help="resume a session by session id or session package path")
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_session",
        help="resume the latest session for the current workspace",
    )
    _add_runs_root(
        parser,
        help_text="directory for assistant session packages (default: ~/.haagent/runs)",
        default=user_runs_dir(),
    )
    _add_web_flag(parser)
    parser.set_defaults(command="tui", handler=handle_tui_entry)
    subparsers = parser.add_subparsers(dest="command", required=False, parser_class=argparse.ArgumentParser)

    for legacy_command in ("setup", "chat", "sessions", "memory", "tui"):
        _add_migration_command(subparsers, legacy_command)

    sandbox_parser = subparsers.add_parser("sandbox", help=argparse.SUPPRESS)
    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_action", required=True)
    sandbox_status = sandbox_subparsers.add_parser("status", help="show sandbox status")
    sandbox_status.set_defaults(handler=handle_sandbox)
    sandbox_doctor = sandbox_subparsers.add_parser("doctor", help="diagnose Docker sandbox readiness")
    sandbox_doctor.set_defaults(handler=handle_sandbox)
    sandbox_enable = sandbox_subparsers.add_parser("enable", help="enable a sandbox backend")
    sandbox_enable.add_argument("backend", choices=["docker"], help="sandbox backend to enable")
    fallback_group = sandbox_enable.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--fail-if-unavailable",
        action="store_true",
        dest="fail_if_unavailable",
        default=True,
        help="fail runs when Docker is unavailable (default)",
    )
    fallback_group.add_argument(
        "--allow-fallback",
        action="store_false",
        dest="fail_if_unavailable",
        help="allow fallback to local_subprocess when Docker is unavailable",
    )
    sandbox_enable.set_defaults(handler=handle_sandbox)
    sandbox_disable = sandbox_subparsers.add_parser("disable", help="disable Docker sandbox")
    sandbox_disable.set_defaults(handler=handle_sandbox)

    run_parser = subparsers.add_parser("run", help="run a task.yaml file")
    run_parser.add_argument("task_yaml", nargs="?", type=Path, help="path to task.yaml")
    run_parser.add_argument("--goal", help="task goal used when task_yaml is omitted")
    run_parser.add_argument(
        "--workspace-root",
        type=Path,
        help="workspace root used when task_yaml is omitted",
    )
    run_parser.add_argument("--verify", help="verification command used when task_yaml is omitted")
    _add_runs_root(
        run_parser,
        help_text="directory for episode packages (default: ~/.haagent/runs)",
        default=argparse.SUPPRESS,
    )
    _add_model_provider(run_parser)
    _add_max_turns(
        run_parser,
        default=DEFAULT_RUN_MAX_TURNS,
        help_text=f"maximum model/tool turns before failing the run (default: {DEFAULT_RUN_MAX_TURNS})",
    )
    run_parser.set_defaults(handler=lambda args: handle_run(args, runtime))

    smoke_parser = subparsers.add_parser("smoke", help="run the minimal HaAgent smoke suite")
    _add_runs_root(
        smoke_parser,
        help_text="directory for episode packages (default: ~/.haagent/runs)",
        default=argparse.SUPPRESS,
    )
    smoke_parser.add_argument("--profile", help="real provider profile name from .haagent/providers.json")
    _add_max_turns(
        smoke_parser,
        default=DEFAULT_SMOKE_MAX_TURNS,
        help_text=f"maximum model/tool turns per smoke task (default: {DEFAULT_SMOKE_MAX_TURNS})",
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
        default=DEFAULT_DOGFOOD_MAX_TURNS,
        help_text=f"maximum model/tool turns per dogfood task (default: {DEFAULT_DOGFOOD_MAX_TURNS})",
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
    _add_runs_root(
        eval_parser,
        help_text="directory for eval run episode packages (default: ~/.haagent/runs)",
        default=argparse.SUPPRESS,
    )
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
    _add_runs_root(
        check_parser,
        help_text="directory for check episode packages (default: ~/.haagent/runs)",
        default=argparse.SUPPRESS,
    )
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

    # 高级运维入口：前台运行聊天渠道网关，不替代普通 TUI。
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="advanced channel gateway (run/status); not the ordinary TUI entry",
    )
    gateway_sub = gateway_parser.add_subparsers(dest="gateway_action", required=True)
    gateway_run = gateway_sub.add_parser("run", help="run channel gateway in foreground")
    gateway_run.add_argument(
        "--workspace-root",
        type=Path,
        help="default workspace root for channel sessions (default: current directory)",
    )
    gateway_run.set_defaults(handler=handle_gateway)
    gateway_status = gateway_sub.add_parser("status", help="show configured channel instances")
    gateway_status.set_defaults(handler=handle_gateway)
    gateway_pair = gateway_sub.add_parser("pair", help="re-issue one-time pairing code for a channel instance")
    gateway_pair.add_argument(
        "--instance-id",
        dest="instance_id",
        default=None,
        help="channel instance id (default: sole enabled instance)",
    )
    gateway_pair.set_defaults(handler=handle_gateway)

    # 高级内部入口：系统后台 worker；不进入普通根帮助主线。
    schedule_worker_parser = subparsers.add_parser(
        "schedule-worker",
        help=argparse.SUPPRESS,
    )
    schedule_worker_parser.add_argument(
        "--once",
        action="store_true",
        help="expand due work, claim/execute claimable runs, then exit",
    )
    schedule_worker_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="path to schedules SQLite database (default: ~/.haagent/schedules.sqlite3)",
    )
    schedule_worker_parser.set_defaults(handler=handle_schedule_worker)

    return parser


def _add_runs_root(parser: argparse.ArgumentParser, *, help_text: str, default) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=default,
        help=help_text,
    )


def _add_migration_command(subparsers, command: str) -> None:
    parser = subparsers.add_parser(
        command,
        help=argparse.SUPPRESS,
        add_help=False,
        prefix_chars="+",
    )
    # 旧命令只负责给迁移提示，不再维护已失效的参数/子命令模型。
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER)
    parser.set_defaults(handler=handle_tui_migration)


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


def _add_web_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--web",
        action="store_true",
        dest="enable_web",
        help="explicitly enable read-only web_search and web_fetch tools for the TUI session",
    )


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
