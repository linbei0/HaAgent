"""
src/haagent/context/soul.py - Soul 文件加载

只加载固定的用户级与受信任工作区 Soul，并返回上下文选择所需的审计信息。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from haagent.models.model_connections import user_config_dir
from haagent.runtime.settings import SoulSettings


SOUL_FILE_NAME = "SOUL.md"
MAX_SOUL_FILE_BYTES = 2 * 1024 * 1024


class SoulLoadError(RuntimeError):
    """Soul 文件违反读取、边界或 UTF-8 合同。"""


@dataclass(frozen=True)
class SoulLoadResult:
    content: str | None
    metadata: dict[str, object] = field(default_factory=dict)
    skip_reason: str | None = None


def load_soul(workspace_root: Path, settings: SoulSettings) -> SoulLoadResult:
    global_path = user_config_dir() / SOUL_FILE_NAME
    workspace_path = workspace_root.expanduser().resolve() / SOUL_FILE_NAME
    blocks: list[str] = []
    sources: list[dict[str, object]] = []
    skip_reason: str | None = None
    loaded_global_resolved: Path | None = None

    if _soul_path_is_file(global_path):
        content = _read_soul_file(global_path)
        sources.append(
            {
                "scope": "global",
                "path": str(global_path),
                "status": "loaded" if content else "empty",
                "chars": len(content),
            },
        )
        if content:
            blocks.append(f"Global Soul (baseline):\n{content}")
        # 成功读过全局 Soul 后记录 resolved 路径，供工作区同源去重。
        loaded_global_resolved = global_path.resolve()

    if _soul_path_is_file(workspace_path):
        if not is_workspace_soul_trusted(workspace_root, settings):
            # 工作区 Soul 是 prompt 来源；未显式信任时绝不读取正文。
            sources.append(
                {
                    "scope": "workspace",
                    "path": str(workspace_path),
                    "status": "skipped_untrusted",
                },
            )
            if not blocks:
                skip_reason = "workspace_untrusted"
        elif (
            loaded_global_resolved is not None
            and _same_soul_target(workspace_path, loaded_global_resolved)
        ):
            # workspace 与全局 Soul 指向同一文件时不重复读、不重复注入。
            pass
        else:
            content = _read_soul_file(
                workspace_path,
                allowed_root=workspace_root,
            )
            sources.append(
                {
                    "scope": "workspace",
                    "path": str(workspace_path),
                    "status": "loaded" if content else "empty",
                    "chars": len(content),
                },
            )
            if content:
                blocks.append(
                    "Workspace Soul (takes precedence for identity and tone):\n"
                    f"{content}",
                )

    return SoulLoadResult(
        content="\n\n".join(blocks) or None,
        metadata={"sources": sources},
        skip_reason=skip_reason,
    )


def _soul_path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError as error:
        # 存在性探测也属于加载边界；权限/IO 失败不得泄漏裸 OSError。
        raise SoulLoadError(f"cannot read Soul file as UTF-8: {path}") from error


def _same_soul_target(path: Path, resolved_other: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SoulLoadError(f"cannot read Soul file as UTF-8: {path}") from error
    return os.path.normcase(str(resolved)) == os.path.normcase(str(resolved_other))


def is_workspace_soul_trusted(workspace_root: Path, settings: SoulSettings) -> bool:
    normalized_root = os.path.normcase(str(workspace_root.expanduser().resolve()))
    trusted_roots = {
        os.path.normcase(str(Path(root).expanduser().resolve()))
        for root in settings.trusted_workspace_roots
    }
    return normalized_root in trusted_roots


def _read_soul_file(path: Path, *, allowed_root: Path | None = None) -> str:
    try:
        resolved_target = path.resolve(strict=True)
        if allowed_root is not None:
            resolved_root = allowed_root.expanduser().resolve()
            try:
                resolved_target.relative_to(resolved_root)
            except ValueError as error:
                # 可信 workspace 仍不能通过 symlink 读取 workspace 外部文件。
                raise SoulLoadError(
                    f"workspace Soul target escapes workspace: {path}",
                ) from error
        # 同一文件描述符完成大小检查和有界读取，避免 check/use 间更换文件。
        with resolved_target.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > MAX_SOUL_FILE_BYTES:
                raise SoulLoadError(
                    f"Soul file exceeds {MAX_SOUL_FILE_BYTES} bytes: {path}",
                )
            raw = handle.read(MAX_SOUL_FILE_BYTES + 1)
        if len(raw) > MAX_SOUL_FILE_BYTES:
            # 文件在 fstat 后增长时仍只读取上限加一字节，并显式拒绝。
            raise SoulLoadError(
                f"Soul file exceeds {MAX_SOUL_FILE_BYTES} bytes: {path}",
            )
        return raw.decode("utf-8").strip()
    except SoulLoadError:
        raise
    except (OSError, UnicodeError) as error:
        # 已决定加载的 Soul 失败时显式中止，禁止使用旧人格或静默忽略。
        raise SoulLoadError(f"cannot read Soul file as UTF-8: {path}") from error
