"""
tests/e2e/channels/test_weixin_smoke.py - 微信真实沙箱 smoke（显式开关）

默认跳过；仅在 --run-weixin-e2e 且环境变量齐全时访问真实 iLink。
禁止把 token、用户 ID、消息正文写入 artifact。
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "weixin_e2e: real Weixin iLink sandbox; requires --run-weixin-e2e",
    )


def _weixin_e2e_enabled(config: pytest.Config | None = None) -> bool:
    if config is not None and config.getoption("--run-weixin-e2e", default=False):
        return True
    return os.environ.get("HAAGENT_RUN_WEIXIN_E2E", "").strip() in {"1", "true", "yes"}


@pytest.fixture(scope="module")
def weixin_e2e_env(pytestconfig: pytest.Config) -> dict[str, str]:
    if not _weixin_e2e_enabled(pytestconfig):
        pytest.skip("weixin e2e disabled; pass --run-weixin-e2e")
    token = os.environ.get("HAAGENT_WEIXIN_BOT_TOKEN", "").strip()
    if not token:
        pytest.skip("HAAGENT_WEIXIN_BOT_TOKEN not set")
    # 不把 token 暴露给断言消息；仅供协议客户端使用。
    return {"bot_token": token}


@pytest.mark.weixin_e2e
def test_weixin_protocol_get_updates_smoke(weixin_e2e_env: dict[str, str]) -> None:
    """真实 getupdates 一轮：成功即通过；失败只报告 errcode 类别。"""
    import asyncio

    from haagent.channels.adapters.weixin.protocol import WeixinProtocolClient
    from haagent.channels.adapters.weixin.types import WeixinAuthenticationExpired, WeixinProtocolError

    async def _run() -> None:
        client = WeixinProtocolClient(bot_token=weixin_e2e_env["bot_token"])
        try:
            updates = await client.get_updates(cursor="")
            assert updates.cursor is not None
            # 不断言消息正文，避免 secret/隐私进入日志。
            assert isinstance(updates.messages, list)
        except WeixinAuthenticationExpired:
            pytest.fail("weixin auth expired; re-login via TUI /channels")
        except WeixinProtocolError as error:
            pytest.fail(f"weixin protocol error errcode={error.errcode}")
        finally:
            await client.aclose()

    asyncio.run(_run())
