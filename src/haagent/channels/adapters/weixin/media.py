"""
haagent/channels/adapters/weixin/media.py - 微信媒体边界（M2）

文本阶段：入站媒体由 Adapter 过滤丢弃并推进 cursor；本模块禁止下载/上传。
媒体阶段依赖：可选 extra `channels-weixin-media`（cryptography）。
"""

from __future__ import annotations


class WeixinMediaNotImplemented(RuntimeError):
    """文本阶段不支持媒体；需安装 channels-weixin-media 并实现 M2。"""


def download_media(*_args, **_kwargs) -> None:
    # 显式失败，禁止 silent no-op 或文本路径协议。
    raise WeixinMediaNotImplemented(
        "weixin media is M2 only; install optional extra channels-weixin-media when implementing"
    )
