"""
src/haagent/tui/files/__init__.py - TUI 文件引用包

集中导出文件引用索引、匹配和 overlay。
"""

from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import (
    FileReferenceIndex,
    FileReferenceMatch,
    build_file_reference_index,
    fuzzy_file_matches,
    path_reference_token,
    query_after_at,
    replace_at_query,
)

__all__ = [
    "FileReferenceIndex",
    "FileReferenceMatch",
    "FileReferenceOverlay",
    "build_file_reference_index",
    "fuzzy_file_matches",
    "path_reference_token",
    "query_after_at",
    "replace_at_query",
]

