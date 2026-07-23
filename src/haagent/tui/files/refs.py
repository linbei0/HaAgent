"""
haagent/tui/files/refs.py - workspace 内文件引用检索

基于当前 workspace 做模糊文件匹配，并生成稳定的 @file 引用 token。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class FileReferenceMatch:
    path: Path
    display_path: str


@dataclass(frozen=True)
class FileReferenceIndex:
    root: Path
    files: tuple[FileReferenceMatch, ...]
    _ranked_files: tuple[FileReferenceMatch, ...] = field(init=False, repr=False, compare=False)
    _ranked_folded_paths: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # 排序成本只在后台索引构建时支付一次；逐键查询按全局排名扫描并在 limit 处停止。
        ranked = tuple(sorted(self.files, key=lambda item: (len(item.display_path), item.display_path.casefold())))
        object.__setattr__(self, "_ranked_files", ranked)
        object.__setattr__(self, "_ranked_folded_paths", tuple(item.display_path.casefold() for item in ranked))

    def matches(self, query: str, *, limit: int = 20) -> list[FileReferenceMatch]:
        needle = query.strip().casefold()
        if limit <= 0:
            return []
        matches: list[FileReferenceMatch] = []
        for item, folded_path in zip(self._ranked_files, self._ranked_folded_paths, strict=True):
            if needle and not _fuzzy_contains(folded_path, needle):
                continue
            matches.append(item)
            if len(matches) >= limit:
                break
        return matches


def build_file_reference_index(workspace_root: Path) -> FileReferenceIndex:
    root = workspace_root.resolve()
    if not root.exists():
        return FileReferenceIndex(root=root, files=())
    files = list(_iter_file_reference_candidates(root))
    return FileReferenceIndex(root=root, files=tuple(files))


def fuzzy_file_matches(workspace_root: Path, query: str, *, limit: int = 20) -> list[FileReferenceMatch]:
    return build_file_reference_index(workspace_root).matches(query, limit=limit)


def path_reference_token(workspace_root: Path, path: Path) -> str:
    root = workspace_root.resolve()
    resolved = path.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError("文件引用必须位于 workspace root 内")
    display = resolved.relative_to(root).as_posix()
    escaped = display.replace("\\", "\\\\").replace('"', '\\"')
    return f'@file("{escaped}")'


def query_after_at(text: str) -> str | None:
    at_index = _reference_at_index(text)
    if at_index < 0:
        return None
    suffix = text[at_index + 1 :]
    if any(char.isspace() for char in suffix):
        return None
    return suffix


def replace_at_query(text: str, token: str) -> str:
    at_index = _reference_at_index(text)
    if at_index < 0:
        return text
    end = at_index + 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return f"{text[:at_index]}{token}{text[end:]}"


def _reference_at_index(text: str) -> int:
    stripped = text[:-1] if text.endswith("@") and text.count("@") > 1 else text
    return stripped.rfind("@")


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    position = 0
    for char in needle:
        found = haystack.find(char, position)
        if found < 0:
            return False
        position = found + 1
    return True


def _iter_file_reference_candidates(root: Path):
    # ripgrep 优先：尊重 .gitignore、速度快。rg 缺失时回退到纯 Python 遍历，
    # 这是刻意保留的环境兼容路径（未装 ripgrep 的机器仍能用 @ 文件引用），
    # 不属于兜底屎山；两条路径都由 test_file_reference_index_* 覆盖。
    rg_path = shutil.which("rg")
    if rg_path is not None:
        yield from _iter_rg_file_candidates(root, rg_path)
        return
    yield from _iter_python_file_candidates(root)


def _iter_rg_file_candidates(root: Path, rg_path: str):
    command = [rg_path, "--files", "--color", "never", "--no-messages"]
    if _looks_like_git_repo(root):
        command.append("--hidden")
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        if process.stdout is None:
            return
        for raw_line in process.stdout:
            display = raw_line.strip().replace("\\", "/")
            if not display:
                continue
            path = (root / display).resolve()
            if not _is_relative_to(path, root) or _is_hidden_run_artifact(root, path):
                continue
            yield FileReferenceMatch(path=path, display_path=display)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _iter_python_file_candidates(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or _is_hidden_run_artifact(root, path):
            continue
        resolved = path.resolve()
        if not _is_relative_to(resolved, root):
            continue
        display = resolved.relative_to(root).as_posix()
        yield FileReferenceMatch(path=resolved, display_path=display)


def _looks_like_git_repo(path: Path) -> bool:
    current = path
    for _ in range(6):
        if (current / ".git").exists():
            return True
        if current.parent == current:
            break
        current = current.parent
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_hidden_run_artifact(root: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in {".git", ".runs", "__pycache__"} for part in parts)
