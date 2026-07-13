"""
src/haagent/context/instruction_cache.py - 工作区 AGENTS.md 指令缓存

仅缓存 workspace_root/AGENTS.md；按 path/exists/mtime_ns/size 命中，
不遍历父目录，不静默返回陈旧正文。
"""

from __future__ import annotations

import hashlib
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CacheDiagnosticsSink = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class LoadedInstructions:
    content: str | None
    source_path: str | None
    fingerprint: str


@dataclass(frozen=True)
class _InstructionKey:
    workspace_root: str
    path: str
    exists: bool
    mtime_ns: int | None
    size: int | None

    def fingerprint(self) -> str:
        raw = f"{self.workspace_root}|{self.path}|{self.exists}|{self.mtime_ns}|{self.size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class InstructionCache:
    """进程内 AGENTS.md 缓存；高级入口未注入时由 ContextBuilder 自建私有实例。"""

    def __init__(self) -> None:
        # workspace_root -> (key fingerprint, loaded)
        self._by_workspace: dict[str, tuple[str, LoadedInstructions]] = {}

    def load(
        self,
        workspace_root: Path,
        *,
        diagnostics_sink: CacheDiagnosticsSink | None = None,
    ) -> LoadedInstructions:
        root = Path(workspace_root).expanduser().resolve()
        path = root / "AGENTS.md"
        key = self._key_for(root, path)
        cache_id = key.fingerprint()
        workspace_key = str(root)
        cached = self._by_workspace.get(workspace_key)
        if cached is not None and cached[0] == cache_id:
            self._publish(diagnostics_sink, "hit", cached[1])
            return cached[1]
        status = "reload" if cached is not None else "miss"
        if not key.exists:
            loaded = LoadedInstructions(content=None, source_path=None, fingerprint=cache_id)
            self._by_workspace[workspace_key] = (cache_id, loaded)
            self._publish(diagnostics_sink, status, loaded)
            return loaded
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as error:
            # 延迟导入避免与 ContextBuilder 循环依赖；读取失败不得回退旧正文。
            from haagent.context.builder import ContextBuildError

            raise ContextBuildError(f"failed to read AGENTS.md: {error}") from error
        loaded = LoadedInstructions(
            content=content,
            source_path=str(path.resolve()),
            fingerprint=cache_id,
        )
        self._by_workspace[workspace_key] = (cache_id, loaded)
        self._publish(diagnostics_sink, status, loaded)
        return loaded

    @staticmethod
    def _publish(
        sink: CacheDiagnosticsSink | None,
        status: str,
        loaded: LoadedInstructions,
    ) -> None:
        if sink is None:
            return
        sink(
            {
                "status": status,
                "count": 1 if loaded.content is not None else 0,
                "chars": len(loaded.content or ""),
                "fingerprint": f"sha256:{loaded.fingerprint}",
            },
        )

    @staticmethod
    def _key_for(root: Path, path: Path) -> _InstructionKey:
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            return _InstructionKey(
                workspace_root=str(root),
                path=str(path),
                exists=False,
                mtime_ns=None,
                size=None,
            )
        except OSError as error:
            from haagent.context.builder import ContextBuildError

            raise ContextBuildError(f"failed to read AGENTS.md metadata: {error}") from error
        exists = stat_module.S_ISREG(file_stat.st_mode)
        return _InstructionKey(
            workspace_root=str(root),
            path=str(path),
            exists=exists,
            mtime_ns=file_stat.st_mtime_ns,
            size=file_stat.st_size,
        )
