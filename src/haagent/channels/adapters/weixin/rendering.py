"""
haagent/channels/adapters/weixin/rendering.py - 微信文本规范化与安全切分
"""

from __future__ import annotations


def normalize_weixin_text(text: str) -> str:
    """第一阶段：不支持的 Markdown 降级为纯文本空白规范化。"""
    return (text or "").replace("\r\n", "\n").strip()


def split_weixin_text(text: str, *, limit: int = 3000) -> list[str]:
    """按 limit 切分，优先在段落/换行边界断开。"""
    cleaned = normalize_weixin_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= limit:
        return [cleaned]
    chunks: list[str] = []
    remaining = cleaned
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        window = remaining[:limit]
        # 优先段落，其次换行，再次空格。
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if cut < limit // 2:
            cut = limit
        piece = remaining[:cut].rstrip()
        if not piece:
            piece = remaining[:limit]
            cut = limit
        chunks.append(piece)
        remaining = remaining[cut:].lstrip()
    return chunks
