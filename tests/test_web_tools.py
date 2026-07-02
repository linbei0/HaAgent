"""
tests/test_web_tools.py - 联网工具测试

验证 web_search/web_fetch 的 provider 映射、外部内容清洗和公网 URL 防护。
"""

from __future__ import annotations

import ipaddress
import json

import httpx
import pytest

from haagent.tools.network_guard import (
    NetworkGuardError,
    ensure_public_http_url,
    fetch_public_http_response,
    validate_http_url,
)
from haagent.tools.web import EXTERNAL_CONTENT_BANNER, web_fetch, web_search


def _public_resolver(host: str, port: int):
    del host, port
    return {ipaddress.ip_address("93.184.216.34")}


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("ftp://example.com/file", "only http and https URLs are allowed"),
        ("https://user:pass@example.com/", "embedded credentials"),
        ("http://127.0.0.1:8000/", "non-public"),
        ("http://localhost:8000/", "local hostnames"),
        ("http://metadata.google.internal/", "local hostnames"),
        ("http://intranet/", "single-label hostnames"),
    ],
)
def test_network_guard_rejects_unsafe_targets(url: str, message: str) -> None:
    with pytest.raises(NetworkGuardError, match=message):
        if url.startswith("ftp") or "user:pass" in url:
            validate_http_url(url)
        else:
            ensure_public_http_url(url, resolver=_public_resolver)


def test_fetch_public_http_response_revalidates_redirect_hops() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/start":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}, request=request)
        return httpx.Response(200, text="unexpected", request=request)

    with pytest.raises(NetworkGuardError, match="non-public"):
        fetch_public_http_response(
            "https://example.com/start",
            resolver=_public_resolver,
            transport=httpx.MockTransport(handler),
        )


def test_web_fetch_cleans_html_and_marks_external_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><style>.x{color:red}</style><script>bad()</script></head>"
                "<body><h1>Title</h1><p>Readable text</p></body></html>"
            ),
            request=request,
        )

    result = web_fetch(
        {"url": "https://example.com/page", "max_chars": 500},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["status"] == "success"
    assert result["final_url"] == "https://example.com/page"
    assert result["content"].startswith(EXTERNAL_CONTENT_BANNER)
    assert "bad()" in result["content"]
    visible_content = result["model_visible"]["content"]
    assert result["model_visible"]["content_format"] == "simplified_html"
    assert "Readable text" in visible_content
    assert "bad()" not in visible_content
    assert ".x{color:red}" not in visible_content
    assert result["truncated"] is False


def test_web_fetch_truncates_content_after_external_banner() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 1000, request=request)

    result = web_fetch(
        {"url": "https://example.com/long", "max_chars": 500},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["content"].endswith("\n...[truncated]")


def test_web_fetch_returns_simplified_html_in_model_visible_and_preserves_raw_trace_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>Readable Page</title><script>track()</script></head>"
                "<body><nav>Top nav</nav><main><h1>Readable Page</h1>"
                "<p>Important body</p><a href=\"/next\">Next link</a></main>"
                "<footer>Footer links</footer></body></html>"
            ),
            request=request,
        )

    result = web_fetch(
        {"url": "https://example.com/page", "max_chars": 1200},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["status"] == "success"
    assert "track()" in result["content"]
    visible = result["model_visible"]
    assert visible["final_url"] == "https://example.com/page"
    assert visible["content_format"] == "simplified_html"
    assert "<main>" in visible["content"]
    assert "Important body" in visible["content"]
    assert "href=\"https://example.com/next\"" in visible["content"]
    assert "track()" not in visible["content"]
    assert "Top nav" not in visible["content"]
    assert "Footer links" not in visible["content"]


def test_web_fetch_simplified_html_removes_low_value_page_chrome() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=(
                "<article><header>Article header</header><h1>Title</h1>"
                "<aside>Related posts</aside><p>Useful paragraph</p>"
                "<form><input value='noise'></form></article>"
            ),
            request=request,
        )

    result = web_fetch(
        {"url": "https://example.com/article", "max_chars": 1200},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    visible_content = result["model_visible"]["content"]
    assert "Useful paragraph" in visible_content
    assert "Related posts" not in visible_content
    assert "value=" not in visible_content


def test_tavily_web_search_maps_results_without_leaking_api_key() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "query": "haagent docs",
                "results": [
                    {
                        "title": "HaAgent Docs",
                        "url": "https://example.com/docs",
                        "content": "Docs snippet",
                        "score": 0.9,
                    },
                ],
            },
            request=request,
        )

    result = web_search(
        {"query": "haagent docs", "max_results": 3, "topic": "news", "freshness": "week"},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
        environ={"TAVILY_API_KEY": "tvly-secret"},
    )

    assert result["status"] == "success"
    assert result["provider"] == "tavily"
    assert result["results"] == [
        {
            "title": "HaAgent Docs",
            "url": "https://example.com/docs",
            "snippet": "Docs snippet",
            "score": 0.9,
        },
    ]
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["authorization"] == "Bearer tvly-secret"
    assert seen["payload"] == {
        "query": "haagent docs",
        "max_results": 3,
        "topic": "news",
        "time_range": "week",
        "search_depth": "basic",
    }
    assert "tvly-secret" not in json.dumps(result, ensure_ascii=False)


def test_brave_web_search_maps_results_and_freshness_without_leaking_api_key() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("x-subscription-token")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave Result",
                            "url": "https://example.com/brave",
                            "description": "Brave snippet",
                            "age": "2 days ago",
                        },
                    ],
                },
            },
            request=request,
        )

    result = web_search(
        {"query": "haagent", "provider": "brave", "max_results": 2, "freshness": "week"},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
        environ={"BRAVE_SEARCH_API_KEY": "brave-secret"},
    )

    assert result["status"] == "success"
    assert result["provider"] == "brave"
    assert result["results"][0] == {
        "title": "Brave Result",
        "url": "https://example.com/brave",
        "snippet": "Brave snippet",
        "published_at": "2 days ago",
    }
    assert "q=haagent" in str(seen["url"])
    assert "count=2" in str(seen["url"])
    assert "freshness=pw" in str(seen["url"])
    assert seen["token"] == "brave-secret"
    assert "brave-secret" not in json.dumps(result, ensure_ascii=False)


def test_web_search_missing_api_key_returns_explicit_error() -> None:
    result = web_search({"query": "haagent"}, environ={})

    assert result == {
        "status": "error",
        "error": {
            "type": "web_search_configuration_error",
            "message": "TAVILY_API_KEY is required for tavily web search",
        },
    }
