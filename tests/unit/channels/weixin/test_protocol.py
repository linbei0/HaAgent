"""
tests/unit/channels/weixin/test_protocol.py - 微信 iLink 协议客户端测试
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from haagent.channels.adapters.weixin.protocol import WeixinProtocolClient
from haagent.channels.adapters.weixin.types import (
    WeixinAuthenticationExpired,
    WeixinProtocolError,
    WeixinRateLimited,
    WeixinUnsupportedBaseUrl,
)


OFFICIAL_BASE = "https://ilinkai.weixin.qq.com"


def _json_response(data: dict[str, Any], status: int = 200) -> httpx.Response:
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return httpx.Response(status, content=body, headers={"content-type": "application/json"})


def test_base_url_rejects_non_official() -> None:
    for bad in (
        "http://ilinkai.weixin.qq.com",
        "https://evil.example.com",
        "https://ilinkai.weixin.qq.com/path",
        "https://user:pass@ilinkai.weixin.qq.com",
        "https://ilinkai.weixin.qq.com?x=1",
        "https://127.0.0.1",
    ):
        with pytest.raises(WeixinUnsupportedBaseUrl):
            WeixinProtocolClient(base_url=bad, bot_token="tok")


def test_qr_and_four_state_poll() -> None:
    """对齐 wechatbot/iLink：qrcode 为轮询 token，qrcode_img_content 为展示 URL；状态用 GET。"""
    states = iter(
        [
            {"errcode": 0, "status": "wait"},
            {"errcode": 0, "status": "scaned"},
            {
                "errcode": 0,
                "status": "confirmed",
                "bot_token": "secret-bot-token",
                "ilink_bot_id": "bot-1",
                "ilink_user_id": "user-meta",
                "baseurl": OFFICIAL_BASE,
            },
        ]
    )
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/get_qrcode_status") or "get_qrcode_status" in path:
            assert request.method == "GET"
            assert request.url.params.get("qrcode") == "qr-token-1"
            return _json_response(next(states))
        if "get_bot_qrcode" in path:
            assert request.method == "POST"
            assert request.url.params.get("bot_type") == "3"
            body = json.loads(request.content.decode("utf-8") or "{}")
            assert "local_token_list" in body
            # 官方字段：qrcode=轮询 token，qrcode_img_content=可展示 URL
            return _json_response(
                {
                    "errcode": 0,
                    "qrcode": "qr-token-1",
                    "qrcode_img_content": "https://login.example/qr-img",
                }
            )
        return _json_response({"errcode": 1, "errmsg": "unknown path"})

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="",
                http_client=client,
            )
            qr = await proto.get_qrcode()
            assert qr.qrcode_url == "https://login.example/qr-img"
            assert qr.qrcode_id == "qr-token-1"
            s1 = await proto.poll_qrcode_status(qr.qrcode_id)
            assert s1.status == "wait"
            s2 = await proto.poll_qrcode_status(qr.qrcode_id)
            assert s2.status == "scaned"
            s3 = await proto.poll_qrcode_status(qr.qrcode_id)
            assert s3.status == "confirmed"
            assert s3.bot_token == "secret-bot-token"
            # 测试输出不得泄露 token：repr 检查
            assert "secret-bot-token" not in repr(s3)
        assert any(m == "POST" and "get_bot_qrcode" in p for m, p in calls)
        assert any(m == "GET" and "get_qrcode_status" in p for m, p in calls)

    asyncio.run(_run())


def test_headers_and_content_length_without_leaking_token() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _json_response({"errcode": 0, "msgs": [], "get_updates_buf": "c2"})

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="super-secret-token",
                http_client=client,
            )
            await proto.get_updates(cursor="c1")
        req = captured[0]
        assert req.headers.get("AuthorizationType") == "ilink_bot_token" or "Authorization" in req.headers
        auth = req.headers.get("Authorization", "")
        assert "super-secret-token" in auth
        assert req.headers.get("Content-Type", "").startswith("application/json")
        assert "X-WECHAT-UIN" in req.headers or "X-Wechat-Uin" in {k.title() for k in req.headers.keys()}
        # Content-Length 与 body 字节一致
        body = req.content
        cl = req.headers.get("Content-Length")
        if cl is not None:
            assert int(cl) == len(body)
        # 客户端 repr 不泄露 token
        assert "super-secret-token" not in repr(proto)

    asyncio.run(_run())


def test_lifecycle_notifications_use_authenticated_protocol_endpoints() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _json_response({"ret": 0})

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="lifecycle-secret-token",
                http_client=client,
            )
            await proto.notify_start()
            await proto.notify_stop()

    asyncio.run(_run())
    assert [request.url.path for request in captured] == [
        "/ilink/bot/msg/notifystart",
        "/ilink/bot/msg/notifystop",
    ]
    for request in captured:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["base_info"]["bot_agent"].startswith("HaAgent/")
        assert request.headers["AuthorizationType"] == "ilink_bot_token"
        assert "lifecycle-secret-token" in request.headers["Authorization"]
        assert "lifecycle-secret-token" not in repr(request.url)


def test_get_qrcode_omits_empty_bearer_authorization() -> None:
    """登录取码时无 bot_token，禁止发送非法 Authorization: Bearer 。"""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _json_response(
            {
                "errcode": 0,
                "qrcode": "qr-token-1",
                "qrcode_img_content": "https://login.example/qr-img",
            }
        )

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="",
                http_client=client,
            )
            qr = await proto.get_qrcode()
            assert qr.qrcode_id == "qr-token-1"
            assert qr.qrcode_url.startswith("https://")
        req = captured[0]
        auth = req.headers.get("Authorization")
        # httpx 拒绝尾随空格的 Bearer；无 token 时必须省略或非空合法值。
        assert auth is None or (auth.startswith("Bearer ") and len(auth) > len("Bearer "))
        assert "Bearer " != (auth or "")
        # 登录公共头（对齐 wechatbot）
        assert req.headers.get("iLink-App-Id") == "bot" or req.headers.get("ilink-app-id") == "bot"

    asyncio.run(_run())


def test_long_poll_timeout_is_exposed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="tok",
                http_client=client,
            )
            with pytest.raises(httpx.ReadTimeout):
                await proto.get_updates(cursor="keep-me")

    asyncio.run(_run())


def test_successful_updates_return_messages_and_cursor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            {
                "errcode": 0,
                "get_updates_buf": "new-cursor",
                "msgs": [
                    {
                        "msg_id": "m1",
                        "from_user_id": "u1",
                        "context_token": "ctx-secret",
                        "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
                        "create_time_ms": 1,
                    }
                ],
            }
        )

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="tok",
                http_client=client,
            )
            result = await proto.get_updates(cursor="")
            assert result.cursor == "new-cursor"
            assert len(result.messages) == 1
            assert result.messages[0].message_id == "m1"
            assert result.messages[0].from_user_id == "u1"
            assert result.messages[0].text == "hello"
            assert "ctx-secret" not in repr(result.messages[0])

    asyncio.run(_run())


def test_errcode_minus_14_is_auth_expired() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"errcode": -14, "errmsg": "auth expired"})

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="tok",
                http_client=client,
            )
            with pytest.raises(WeixinAuthenticationExpired):
                await proto.get_updates(cursor="")

    asyncio.run(_run())


def test_rate_limit_and_unknown_error_separated() -> None:
    def rate_handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"errcode": -1, "errmsg": "rate limit", "ret": 1001})

    def unknown_handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"errcode": -999, "errmsg": "weird"})

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(rate_handler)) as client:
            proto = WeixinProtocolClient(base_url=OFFICIAL_BASE, bot_token="tok", http_client=client)
            # 限流：errcode 特殊或 errmsg 含 rate；我们用 WeixinRateLimited 对已知限流码
            # 协议层把 ret=1001 或 errmsg 含 rate 判为限流
            with pytest.raises((WeixinRateLimited, WeixinProtocolError)):
                await proto.get_updates(cursor="")
        async with httpx.AsyncClient(transport=httpx.MockTransport(unknown_handler)) as client:
            proto = WeixinProtocolClient(base_url=OFFICIAL_BASE, bot_token="tok", http_client=client)
            with pytest.raises(WeixinProtocolError) as exc:
                await proto.get_updates(cursor="")
            assert not isinstance(exc.value, WeixinRateLimited)
            assert not isinstance(exc.value, WeixinAuthenticationExpired)

    asyncio.run(_run())


def test_sendmessage_requires_context_token_and_unique_client_id() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        return _json_response({"errcode": 0})

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(
                base_url=OFFICIAL_BASE,
                bot_token="tok",
                http_client=client,
            )
            with pytest.raises(WeixinProtocolError):
                await proto.send_text(to_user_id="u1", text="hi", context_token="")
            await proto.send_text(to_user_id="u1", text="hi", context_token="ctx-1")
            await proto.send_text(to_user_id="u1", text="hi2", context_token="ctx-1")
        assert len(captured) == 2
        assert captured[0].get("context_token") == "ctx-1" or "context_token" in json.dumps(captured[0])
        ids = []
        for p in captured:
            # client_id 可能嵌套
            cid = p.get("client_id") or p.get("msg", {}).get("client_id")
            if cid is None:
                # 搜索任意 client_id 字段
                blob = json.dumps(p)
                assert "client_id" in blob
                # 提取简单
                import re

                m = re.search(r'"client_id"\s*:\s*"([^"]+)"', blob)
                assert m
                cid = m.group(1)
            ids.append(cid)
        assert ids[0] != ids[1]

    asyncio.run(_run())


def test_owned_client_survives_start_then_poll_in_one_event_loop() -> None:
    """登录 start+poll 必须在同一 event loop 内完成，避免 httpx 绑到已关闭 loop。"""
    poll_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "get_bot_qrcode" in path:
            return _json_response(
                {
                    "errcode": 0,
                    "qrcode": "qr-token-loop",
                    "qrcode_img_content": "https://login.example/qr-img",
                }
            )
        if "get_qrcode_status" in path:
            poll_calls["n"] += 1
            if poll_calls["n"] < 2:
                return _json_response({"errcode": 0, "status": "wait"})
            return _json_response(
                {
                    "errcode": 0,
                    "status": "confirmed",
                    "bot_token": "tok-loop",
                    "ilink_bot_id": "b",
                    "ilink_user_id": "u",
                    "baseurl": OFFICIAL_BASE,
                }
            )
        return _json_response({"errcode": 1, "errmsg": "unknown"})

    transport = httpx.MockTransport(handler)

    async def _with_mock() -> str:
        # 与 TUI worker 对齐：单次 run 内 start 后连续 poll。
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(base_url=OFFICIAL_BASE, bot_token="", http_client=client)
            qr = await proto.get_qrcode()
            status = await proto.poll_qrcode_status(qr.qrcode_id)
            assert status.status == "wait"
            status = await proto.poll_qrcode_status(qr.qrcode_id)
            assert status.status == "confirmed"
            return status.bot_token or ""

    token = asyncio.run(_with_mock())
    assert token == "tok-loop"


def test_confirmed_qr_rejects_non_official_baseurl_override() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "get_qrcode_status" in path:
            return _json_response(
                {
                    "errcode": 0,
                    "status": "confirmed",
                    "bot_token": "t",
                    "ilink_bot_id": "b",
                    "ilink_user_id": "u",
                    "baseurl": "https://evil.example.com",
                }
            )
        if "get_bot_qrcode" in path:
            return _json_response(
                {
                    "errcode": 0,
                    "qrcode": "qr-token-1",
                    "qrcode_img_content": "https://login.example/qr-img",
                }
            )
        return _json_response({"errcode": 1, "errmsg": "unknown"})

    transport = httpx.MockTransport(handler)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            proto = WeixinProtocolClient(base_url=OFFICIAL_BASE, bot_token="", http_client=client)
            qr = await proto.get_qrcode()
            with pytest.raises(WeixinUnsupportedBaseUrl):
                await proto.poll_qrcode_status(qr.qrcode_id)

    asyncio.run(_run())
