from __future__ import annotations

import shutil
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Callable

from agent_foundry.episode import EpisodeWriter


class ToolRoutingError(RuntimeError):
    """Raised when orchestration wants to fail a run on tool errors."""


class ToolRouter:
    def __init__(
        self,
        allowed_tools: list[str],
        episode_writer: EpisodeWriter,
        workspace_root: Path,
    ) -> None:
        self._allowed_tools = set(allowed_tools)
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root.resolve()
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "fake_tool": self._fake_tool,
            "file_search": self._file_search,
            "file_read": self._file_read,
            "apply_patch": self._apply_patch,
            "shell": self._shell,
        }

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if tool_name not in self._allowed_tools:
                result = _error("tool_not_allowed", f"tool is not allowed: {tool_name}")
            elif tool_name not in self._handlers:
                result = _error("unknown_tool", f"unknown tool: {tool_name}")
            else:
                result = self._handlers[tool_name](args)
        except Exception as error:
            result = _error(type(error).__name__, str(error))

        self._write_trace(tool_name, args, result, started)
        return result

    def raise_for_error(self, result: dict[str, Any]) -> None:
        if result.get("status") == "error":
            error = result.get("error") or {}
            raise ToolRoutingError(str(error.get("message", "tool failed")))

    def _fake_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "args": args}

    def _file_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            return _error("invalid_arguments", "query must be a non-empty string")

        root_arg = args.get("root", ".")
        if not isinstance(root_arg, str):
            return _error("invalid_arguments", "root must be a string")
        root = self._resolve_workspace_path(root_arg)
        if root is None:
            return _error("path_outside_workspace", "root must be inside workspace")

        rg = shutil.which("rg")
        if rg:
            command = [rg, "--json", "--", query, str(root)]
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            if completed.returncode not in (0, 1):
                return _error("search_failed", completed.stderr.strip() or "ripgrep failed")
            return {"status": "success", "matches": _parse_rg_json(completed.stdout)}

        matches = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if query in line:
                        matches.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "column": line.find(query) + 1,
                                "text": line,
                            },
                        )
            except UnicodeDecodeError:
                continue
        return {"status": "success", "matches": matches}

    def _file_read(self, args: dict[str, Any]) -> dict[str, Any]:
        path_arg = args.get("path")
        if not isinstance(path_arg, str):
            return _error("invalid_arguments", "path must be a string")
        path = self._resolve_workspace_path(path_arg)
        if path is None:
            return _error("path_outside_workspace", "path must be inside workspace")

        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 200))
        if offset < 0 or limit < 0:
            return _error("invalid_arguments", "offset and limit must be non-negative")

        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        return {
            "status": "success",
            "path": str(path),
            "offset": offset,
            "limit": limit,
            "content": "".join(selected),
        }

    def _apply_patch(self, args: dict[str, Any]) -> dict[str, Any]:
        path_arg = args.get("path")
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        if not all(isinstance(value, str) for value in (path_arg, old_text, new_text)):
            return _error("invalid_arguments", "path, old_text, and new_text must be strings")

        path = self._resolve_workspace_path(path_arg)
        if path is None:
            return _error("path_outside_workspace", "path must be inside workspace")

        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count == 0:
            return _error("patch_text_not_found", "old_text was not found")
        if count > 1:
            return _error("patch_text_not_unique", "old_text must match exactly once")

        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return {"status": "success", "path": str(path), "replacements": 1}

    def _shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command:
            return _error("invalid_arguments", "command must be a non-empty string")

        cwd_arg = args.get("cwd", ".")
        if not isinstance(cwd_arg, str):
            return _error("invalid_arguments", "cwd must be a string")
        cwd = self._resolve_workspace_path(cwd_arg)
        if cwd is None:
            return _error("path_outside_workspace", "cwd must be inside workspace")

        timeout_seconds = float(args.get("timeout_seconds", 60))
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "status": "error",
                "exit_code": None,
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
                "error": {
                    "type": "timeout",
                    "message": f"command timed out after {timeout_seconds} seconds",
                },
            }

        result = {
            "status": "success" if completed.returncode == 0 else "error",
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            result["error"] = {
                "type": "command_failed",
                "message": f"command exited with code {completed.returncode}",
            }
        return result

    def _resolve_workspace_path(self, path: str) -> Path | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        resolved = candidate.resolve()
        if resolved == self._workspace_root or self._workspace_root in resolved.parents:
            return resolved
        return None

    def _write_trace(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        started: float,
    ) -> None:
        self._episode_writer.append_tool_call(
            {
                "tool_name": tool_name,
                "args": args,
                "status": result["status"],
                "result": result if result["status"] == "success" else None,
                "error": result.get("error"),
                "duration_seconds": time.perf_counter() - started,
            },
        )


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"type": error_type, "message": message}}


def _parse_rg_json(output: str) -> list[dict[str, Any]]:
    matches = []
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        submatches = data.get("submatches") or [{"start": 0}]
        matches.append(
            {
                "path": data["path"]["text"],
                "line": data["line_number"],
                "column": submatches[0]["start"] + 1,
                "text": data["lines"]["text"].rstrip("\n"),
            },
        )
    return matches
