"""
haagent/tools/shell_paths.py - shell 命令文件路径预检

对常见 Bash/PowerShell 文件命令做 best-effort 静态扫描；它不是进程级 sandbox。
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Mapping


FILE_COMMANDS = frozenset(
    {
        "add-content",
        "cat",
        "cd",
        "chdir",
        "chmod",
        "chown",
        "copy",
        "copy-item",
        "cp",
        "del",
        "dir",
        "erase",
        "get-childitem",
        "get-content",
        "ls",
        "md",
        "mkdir",
        "move",
        "move-item",
        "mv",
        "new-item",
        "popd",
        "push-location",
        "pushd",
        "rd",
        "remove-item",
        "rm",
        "rmdir",
        "select-string",
        "set-content",
        "set-location",
        "test-path",
        "touch",
        "type",
    },
)

_POWERSHELL_ENV = re.compile(
    r"\$(?:\{env:([A-Za-z_][A-Za-z0-9_]*)\}|env:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))",
    re.IGNORECASE,
)
_REDIRECTION = re.compile(r"(?:^|\s)(?:>>?|<)\s*(\"[^\"]+\"|'[^']+'|[^\s]+)")


def collect_shell_paths(
    command: str,
    *,
    cwd: Path,
    environ: Mapping[str, str] | None = None,
) -> list[Path]:
    """返回已知文件命令中可确定的路径，保留首次出现顺序。"""
    env = os.environ if environ is None else environ
    found: list[Path] = []
    seen: set[str] = set()
    for statement in _split_statements(command):
        tokens = _tokens(statement)
        command_name = _command_name(tokens)
        candidates: list[str] = []
        if command_name in FILE_COMMANDS:
            candidates.extend(token for token in tokens[1:] if not token.startswith("-"))
        candidates.extend(match.group(1) for match in _REDIRECTION.finditer(statement))
        for token in candidates:
            path = _resolve_path_token(token, cwd, env)
            if path is None:
                continue
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found


def _split_statements(command: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            buffer.append(char)
            if char == quote and (index == 0 or command[index - 1] != "\\"):
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            buffer.append(char)
            index += 1
            continue
        if char in {";", "\n", "|"} or command[index : index + 2] == "&&":
            value = "".join(buffer).strip()
            if value:
                statements.append(value)
            buffer = []
            index += 2 if command[index : index + 2] in {"&&", "||"} else 1
            continue
        buffer.append(char)
        index += 1
    value = "".join(buffer).strip()
    if value:
        statements.append(value)
    return statements


def _tokens(statement: str) -> list[str]:
    try:
        return shlex.split(statement, posix=False)
    except ValueError:
        return []


def _command_name(tokens: list[str]) -> str:
    if not tokens:
        return ""
    index = 1 if tokens[0] in {"&", "."} and len(tokens) > 1 else 0
    return Path(_strip_quotes(tokens[index])).stem.casefold()


def _resolve_path_token(
    token: str,
    cwd: Path,
    environ: Mapping[str, str],
) -> Path | None:
    value = _strip_quotes(token.strip().rstrip(","))
    if not value or "://" in value:
        return None
    expanded = _expand_environment(value, environ)
    if expanded is None:
        return None
    wildcard = min((expanded.find(mark) for mark in "*?[" if mark in expanded), default=-1)
    if wildcard >= 0:
        prefix = expanded[:wildcard].rstrip("/\\")
        expanded = str(Path(prefix).parent if Path(prefix).suffix else Path(prefix))
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve()


def _expand_environment(value: str, environ: Mapping[str, str]) -> str | None:
    unresolved = False

    def replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = next(group for group in match.groups() if group is not None)
        replacement = _environment_value(environ, name)
        if replacement is None:
            unresolved = True
            return match.group(0)
        return replacement

    expanded = _POWERSHELL_ENV.sub(replace, value)
    if unresolved:
        return None
    for name, replacement in environ.items():
        expanded = expanded.replace(f"%{name}%", replacement)
    return expanded


def _environment_value(environ: Mapping[str, str], name: str) -> str | None:
    target = name.casefold()
    for key, value in environ.items():
        if key.casefold() == target:
            return value
    return None


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
