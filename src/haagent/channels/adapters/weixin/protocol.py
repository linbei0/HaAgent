"""
haagent/channels/adapters/weixin/protocol.py - 微信 iLink HTTP 协议客户端

仅实现官方 base URL 上的 QR、长轮询、发送与 typing；token 不入 repr。
"""

from __future__ import annotations

import base64
import json
import secrets
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from haagent.channels.adapters.weixin.types import (
    WeixinAuthenticationExpired,
    WeixinInboundMessage,
    WeixinProtocolError,
    WeixinQrCode,
    WeixinQrStatus,
    WeixinRateLimited,
    WeixinSendResult,
    WeixinUnsupportedBaseUrl,
    WeixinUpdates,
)

OFFICIAL_BASE_URL = "https://ilinkai.weixin.qq.com"
CLIENT_VERSION = "1.0.0"
# 业务路径（对齐 corespeed-io/wechatbot docs/protocol.md 与腾讯 openclaw-weixin）
PATH_GET_QRCODE = "/ilink/bot/get_bot_qrcode"
PATH_QR_STATUS = "/ilink/bot/get_qrcode_status"
PATH_GET_UPDATES = "/ilink/bot/getupdates"
PATH_NOTIFY_START = "/ilink/bot/msg/notifystart"
PATH_NOTIFY_STOP = "/ilink/bot/msg/notifystop"
PATH_SEND_MESSAGE = "/ilink/bot/sendmessage"
PATH_GET_CONFIG = "/ilink/bot/getconfig"
PATH_SEND_TYPING = "/ilink/bot/sendtyping"
ILINK_APP_ID = "bot"
DEFAULT_BOT_AGENT = f"HaAgent/{CLIENT_VERSION}"


def validate_weixin_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme != "https":
        raise WeixinUnsupportedBaseUrl(f"unsupported_weixin_base_url: scheme {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise WeixinUnsupportedBaseUrl("unsupported_weixin_base_url: userinfo not allowed")
    if parsed.query or parsed.fragment:
        raise WeixinUnsupportedBaseUrl("unsupported_weixin_base_url: query/fragment not allowed")
    if parsed.path not in {"", "/"}:
        raise WeixinUnsupportedBaseUrl("unsupported_weixin_base_url: path not allowed")
    host = (parsed.hostname or "").lower()
    if host != "ilinkai.weixin.qq.com":
        raise WeixinUnsupportedBaseUrl(f"unsupported_weixin_base_url: host {host!r}")
    # 拒绝 IP literal
    if host.replace(".", "").isdigit():
        raise WeixinUnsupportedBaseUrl("unsupported_weixin_base_url: ip literal")
    return f"https://{host}"


class WeixinProtocolClient:
    def __init__(
        self,
        *,
        base_url: str = OFFICIAL_BASE_URL,
        bot_token: str = "",
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._base_url = validate_weixin_base_url(base_url)
        self._bot_token = bot_token
        self._client = http_client
        self._owns_client = http_client is None
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return f"WeixinProtocolClient(base_url={self._base_url!r}, has_token={bool(self._bot_token)})"

    def set_bot_token(self, token: str) -> None:
        self._bot_token = token

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
            self._owns_client = True
        return self._client

    def _common_headers(self) -> dict[str, str]:
        # 登录 GET/POST 与业务 POST 共用 iLink App 标识。
        return {
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": "1",
        }

    def _auth_headers(self) -> dict[str, str]:
        # 无 bot_token 时禁止发送 "Bearer "：httpx 拒绝尾随空格的非法 header。
        # X-WECHAT-UIN 需 base64(str(uint32))，与 wechatbot 一致。
        uin = str(secrets.randbelow(2**32))
        headers = {
            "Content-Type": "application/json",
            "X-WECHAT-UIN": base64.b64encode(uin.encode("utf-8")).decode("ascii"),
            **self._common_headers(),
        }
        token = (self._bot_token or "").strip()
        if token:
            headers["AuthorizationType"] = "ilink_bot_token"
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _base_info(self) -> dict[str, str]:
        return {"channel_version": CLIENT_VERSION, "bot_agent": DEFAULT_BOT_AGENT}

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        url = f"{self._base_url}{path}"
        hdrs = dict(headers or self._common_headers())
        content: bytes | None = None
        if payload is not None:
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
            hdrs["Content-Length"] = str(len(content))
        try:
            response = await client.request(
                method,
                url,
                params=params,
                content=content,
                headers=hdrs,
            )
        except httpx.TimeoutException:
            raise
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise WeixinProtocolError("invalid json response") from error
        if not isinstance(data, dict):
            raise WeixinProtocolError("response must be object")
        self._raise_for_errcode(data)
        return data

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            path,
            payload=payload,
            headers=self._auth_headers(),
        )

    def _raise_for_errcode(self, data: dict[str, Any]) -> None:
        errcode = data.get("errcode", 0)
        ret = data.get("ret")
        try:
            code = int(errcode) if errcode is not None else 0
        except (TypeError, ValueError):
            code = 0
        try:
            ret_code = int(ret) if ret is not None else 0
        except (TypeError, ValueError):
            ret_code = 0
        if code == 0 and ret_code == 0:
            return
        errmsg = str(data.get("errmsg") or data.get("msg") or "")
        effective = code if code != 0 else ret_code
        if effective == -14:
            raise WeixinAuthenticationExpired(errmsg or "authentication expired", errcode=effective)
        if (
            effective in {-1, 1001}
            or ret_code == 1001
            or "rate" in errmsg.lower()
            or "limit" in errmsg.lower()
        ):
            # 限流与未知错误分开：命中限流信号时抛 WeixinRateLimited。
            if "rate" in errmsg.lower() or "limit" in errmsg.lower() or ret_code == 1001:
                raise WeixinRateLimited(errmsg or "rate limited", errcode=effective)
        raise WeixinProtocolError(errmsg or f"protocol error {effective}", errcode=effective)

    async def get_qrcode(self, *, local_token_list: list[str] | None = None) -> WeixinQrCode:
        # POST /ilink/bot/get_bot_qrcode?bot_type=3
        # 响应：qrcode=轮询 token，qrcode_img_content=展示 URL（非我们旧假设的 qrcode_id/url）。
        data = await self._request_json(
            "POST",
            PATH_GET_QRCODE,
            params={"bot_type": "3"},
            payload={"local_token_list": list(local_token_list or [])},
            headers=self._common_headers(),
        )
        qid = str(data.get("qrcode") or data.get("qrcode_id") or "")
        url = str(
            data.get("qrcode_img_content")
            or data.get("qrcode_url")
            or data.get("qrcode_img")
            or ""
        )
        if not url or not qid:
            raise WeixinProtocolError("qrcode response missing fields")
        return WeixinQrCode(qrcode_url=url, qrcode_id=qid)

    async def poll_qrcode_status(
        self,
        qrcode_id: str,
        *,
        verify_code: str | None = None,
        poll_base_url: str | None = None,
    ) -> WeixinQrStatus:
        # GET /ilink/bot/get_qrcode_status?qrcode=<token>[&verify_code=...]
        base = (poll_base_url or self._base_url).rstrip("/")
        if poll_base_url:
            # 仅允许 https 主机；IDC redirect 时 host 由服务端给出。
            parsed = urlparse(base)
            if parsed.scheme != "https" or not parsed.hostname:
                raise WeixinUnsupportedBaseUrl(f"unsupported poll base: {base!r}")
            base = f"https://{parsed.hostname}"
        params: dict[str, str] = {"qrcode": qrcode_id}
        if verify_code:
            params["verify_code"] = verify_code
        client = await self._ensure_client()
        url = f"{base}{PATH_QR_STATUS}"
        try:
            response = await client.get(
                url,
                params=params,
                headers=self._common_headers(),
            )
        except httpx.TimeoutException:
            raise
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            raise WeixinProtocolError("invalid json response") from error
        if not isinstance(data, dict):
            raise WeixinProtocolError("response must be object")
        self._raise_for_errcode(data)
        status_raw = str(data.get("status") or "unknown").lower()
        if status_raw in {"scanned", "scaned"}:
            status = "scaned"
        elif status_raw in {"wait", "waiting"}:
            status = "wait"
        elif status_raw in {"confirmed", "confirm"}:
            status = "confirmed"
        elif status_raw in {"expired", "expire"}:
            status = "expired"
        elif status_raw in {
            "scaned_but_redirect",
            "binded_redirect",
            "need_verifycode",
            "verify_code_blocked",
        }:
            status = status_raw  # type: ignore[assignment]
        else:
            status = "unknown"
        base_url = data.get("baseurl") or data.get("base_url")
        if status == "confirmed" and base_url:
            validate_weixin_base_url(str(base_url))
        return WeixinQrStatus(
            status=status,  # type: ignore[arg-type]
            bot_token=str(data["bot_token"]) if data.get("bot_token") else None,
            ilink_bot_id=str(data["ilink_bot_id"]) if data.get("ilink_bot_id") else None,
            ilink_user_id=str(data["ilink_user_id"]) if data.get("ilink_user_id") else None,
            base_url=str(base_url) if base_url else None,
        )

    async def get_updates(self, *, cursor: str = "") -> WeixinUpdates:
        payload = {"get_updates_buf": cursor or "", "base_info": self._base_info()}
        try:
            data = await self._post(PATH_GET_UPDATES, payload)
        except httpx.TimeoutException:
            # 网络超时：空更新但保留原 cursor，不伪装协议成功。
            return WeixinUpdates(messages=[], cursor=cursor)
        new_cursor = str(data.get("get_updates_buf") or cursor or "")
        messages: list[WeixinInboundMessage] = []
        for item in data.get("msgs") or data.get("messages") or []:
            if not isinstance(item, dict):
                continue
            parsed = _parse_inbound(item)
            if parsed is not None:
                messages.append(parsed)
        return WeixinUpdates(messages=messages, cursor=new_cursor)

    async def notify_start(self) -> None:
        """通知微信后端当前 bot 实例上线，重建长轮询服务端会话。"""
        await self._post(PATH_NOTIFY_START, {"base_info": self._base_info()})

    async def notify_stop(self) -> None:
        """通知微信后端当前 bot 实例下线，释放长轮询服务端会话。"""
        await self._post(PATH_NOTIFY_STOP, {"base_info": self._base_info()})

    async def send_text(
        self,
        *,
        to_user_id: str,
        text: str,
        context_token: str,
    ) -> WeixinSendResult:
        if not context_token:
            # 未拿到 context_token 时显式失败。
            raise WeixinProtocolError("context_token required for send")
        if not text:
            return WeixinSendResult(ok=False, error="empty_text")
        client_id = str(uuid.uuid4())
        payload = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": client_id,
                "message_type": 2,
                "message_state": 2,
                "context_token": context_token,
                "item_list": [{"type": 1, "text_item": {"text": text}}],
            },
            "base_info": self._base_info(),
        }
        await self._post(PATH_SEND_MESSAGE, payload)
        return WeixinSendResult(ok=True)

    async def get_typing_ticket(self, *, context_token: str, ilink_user_id: str = "") -> str:
        payload: dict[str, Any] = {
            "context_token": context_token,
            "base_info": self._base_info(),
        }
        if ilink_user_id:
            payload["ilink_user_id"] = ilink_user_id
        data = await self._post(PATH_GET_CONFIG, payload)
        ticket = str(data.get("typing_ticket") or data.get("ticket") or "")
        if not ticket:
            raise WeixinProtocolError("typing_ticket missing")
        return ticket

    async def send_typing(
        self,
        *,
        typing_ticket: str,
        active: bool,
        context_token: str,
        ilink_user_id: str = "",
    ) -> None:
        payload: dict[str, Any] = {
            "typing_ticket": typing_ticket,
            "status": 1 if active else 2,
            "base_info": self._base_info(),
        }
        if ilink_user_id:
            payload["ilink_user_id"] = ilink_user_id
        if context_token:
            payload["context_token"] = context_token
        await self._post(PATH_SEND_TYPING, payload)


def _parse_inbound(item: dict[str, Any]) -> WeixinInboundMessage | None:
    message_id = str(item.get("msg_id") or item.get("message_id") or "")
    from_user_id = str(item.get("from_user_id") or item.get("from_user") or "")
    context_token = str(item.get("context_token") or "")
    if not message_id or not from_user_id:
        return None
    text = ""
    for part in item.get("item_list") or []:
        if not isinstance(part, dict):
            continue
        if int(part.get("type") or 0) == 1:
            text_item = part.get("text_item") or {}
            if isinstance(text_item, dict):
                text = str(text_item.get("text") or "")
                break
    create_time_ms = int(item.get("create_time_ms") or item.get("create_time") or 0)
    return WeixinInboundMessage(
        message_id=message_id,
        from_user_id=from_user_id,
        text=text,
        context_token=context_token,
        create_time_ms=create_time_ms,
        raw=item,
    )
