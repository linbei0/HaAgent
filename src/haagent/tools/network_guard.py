"""
haagent/tools/network_guard.py - 联网目标安全校验

为只读联网工具校验 HTTP(S) URL、公网解析结果、代理配置和重定向链路。
支持 direct（本地解析目标）与 proxy（代理解析目标）两种解析模式。
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Callable
from enum import Enum
from urllib.parse import ParseResult, urljoin, urlparse

import httpx


IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], set[IPAddress]]
DEFAULT_PORTS = {"http": 80, "https": 443}
# 连接/读取分离，避免笼统 “timed out”；常量集中，规模克制。
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 15.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 10.0
DEFAULT_POOL_TIMEOUT_SECONDS = 10.0
LOCAL_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
}
LOCAL_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".cluster.local",
)


class ResolutionMode(str, Enum):
    """目标主机解析方式：无代理直连，有显式代理则交给代理端解析。"""

    DIRECT = "direct"
    PROXY = "proxy"


class NetworkGuardError(ValueError):
    """联网目标违反安全策略时抛出。"""


def default_http_timeout(
    *,
    connect: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read: float = DEFAULT_READ_TIMEOUT_SECONDS,
    write: float = DEFAULT_WRITE_TIMEOUT_SECONDS,
    pool: float = DEFAULT_POOL_TIMEOUT_SECONDS,
) -> httpx.Timeout:
    """构建分离 connect/read/write/pool 的超时配置。"""
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def validate_http_url(url: str) -> None:
    """校验基本 HTTP(S) URL 语法，不允许内嵌凭据。"""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise NetworkGuardError("only http and https URLs are allowed")
    if not parsed.netloc or not parsed.hostname:
        raise NetworkGuardError("URL must include a host")
    if parsed.username or parsed.password:
        raise NetworkGuardError("URLs with embedded credentials are not allowed")


def get_resolution_mode(*, proxy: str | None = None) -> ResolutionMode:
    """有显式代理时自动进入 proxy，否则 direct。"""
    return ResolutionMode.PROXY if proxy else ResolutionMode.DIRECT


def ensure_public_http_url(url: str, *, resolver: Resolver | None = None) -> None:
    """拒绝 loopback、私网、metadata 和其它非公网 HTTP 目标（direct 模式）。"""
    ensure_http_url_allowed(url, mode=ResolutionMode.DIRECT, resolver=resolver)


def ensure_http_url_allowed(
    url: str,
    *,
    mode: ResolutionMode,
    resolver: Resolver | None = None,
) -> None:
    """按解析模式校验单个出站 URL。"""
    if mode is ResolutionMode.PROXY:
        _ensure_proxy_safe_http_url(url)
        return
    parsed = _validated_parsed_http_url(url)
    hostname = _normalized_hostname(parsed.hostname)
    literal = _parse_ip_literal(hostname)
    if literal is not None:
        _ensure_global_literal_ip(literal)
        return
    _ensure_not_local_hostname(hostname)
    port = parsed.port or DEFAULT_PORTS[parsed.scheme]
    addresses = (resolver or _resolve_host_addresses)(hostname, port)
    if not addresses:
        raise NetworkGuardError(f"target host did not resolve: {hostname}")
    blocked = sorted({str(address) for address in addresses if not address.is_global})
    if blocked:
        raise NetworkGuardError(_format_blocked_addresses(blocked))


def fetch_public_http_response(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
    json_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout = DEFAULT_READ_TIMEOUT_SECONDS,
    max_redirects: int = 5,
    resolver: Resolver | None = None,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Response:
    """请求 HTTP 资源，并在每个重定向 hop 前按当前模式重新校验目标。"""
    current_url = url
    current_params = params
    proxy = _configured_proxy()
    mode = get_resolution_mode(proxy=proxy)
    request_timeout = timeout if isinstance(timeout, httpx.Timeout) else default_http_timeout(read=float(timeout))

    client_kwargs: dict[str, object] = {
        "follow_redirects": False,
        "timeout": request_timeout,
        "trust_env": False,
        "transport": transport,
    }
    # 显式代理且无自定义 transport 时交给 httpx；MockTransport 测试路径不注入 proxy。
    if proxy is not None and transport is None:
        client_kwargs["proxy"] = proxy

    with httpx.Client(**client_kwargs) as client:
        for redirect_count in range(max_redirects + 1):
            ensure_http_url_allowed(current_url, mode=mode, resolver=resolver)
            response = client.request(
                method,
                current_url,
                params=current_params,
                headers=headers,
                json=json_body,
            )
            if not response.has_redirect_location:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            if redirect_count >= max_redirects:
                raise NetworkGuardError(f"too many redirects (>{max_redirects})")
            current_url = urljoin(str(response.url), location)
            current_params = None
            json_body = None

    raise NetworkGuardError("request failed before receiving a response")


def _configured_proxy() -> str | None:
    # 唯一显式代理入口；不继承 HTTP(S)_PROXY / ALL_PROXY。
    proxy = os.environ.get("HAAGENT_WEB_PROXY")
    if not proxy:
        return None
    validate_http_url(proxy)
    return proxy


def _ensure_proxy_safe_http_url(url: str) -> None:
    """代理模式：普通域名由代理端解析，本地只拦主机名与 IP 字面量。"""
    parsed = _validated_parsed_http_url(url)
    hostname = _normalized_hostname(parsed.hostname)
    literal = _parse_ip_literal(hostname)
    if literal is not None:
        _ensure_global_literal_ip(literal)
        return
    _ensure_not_local_hostname(hostname)


def _resolve_host_addresses(host: str, port: int) -> set[IPAddress]:
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError as error:
        raise NetworkGuardError(f"could not resolve target host {host}: {error}") from error
    addresses: set[IPAddress] = set()
    for family, _, _, _, sockaddr in infos:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        candidate = sockaddr[0]
        if not isinstance(candidate, str):
            continue
        parsed = _parse_ip_literal(candidate)
        if parsed is not None:
            addresses.add(parsed)
    return addresses


def _validated_parsed_http_url(url: str) -> ParseResult:
    validate_http_url(url)
    parsed = urlparse(url)
    assert parsed.hostname is not None
    return parsed


def _normalized_hostname(hostname: str | None) -> str:
    assert hostname is not None
    return hostname.rstrip(".").lower()


def _parse_ip_literal(value: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _ensure_global_literal_ip(address: IPAddress) -> None:
    if not address.is_global:
        raise NetworkGuardError(f"target resolves to non-public address(es): {address}")


def _ensure_not_local_hostname(hostname: str) -> None:
    if hostname in LOCAL_HOSTNAMES or any(hostname.endswith(suffix) for suffix in LOCAL_HOST_SUFFIXES):
        raise NetworkGuardError(f"local hostnames are not allowed: {hostname}")
    if "." not in hostname:
        raise NetworkGuardError(f"single-label hostnames are not allowed: {hostname}")


def _format_blocked_addresses(blocked: list[str]) -> str:
    rendered = ", ".join(blocked[:3])
    if len(blocked) > 3:
        rendered += ", ..."
    return f"target resolves to non-public address(es): {rendered}"
