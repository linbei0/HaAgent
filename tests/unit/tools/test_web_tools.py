"""
tests/unit/tools/test_web_tools.py - 联网工具测试

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


def test_web_fetch_offloads_long_simplified_content_to_artifact() -> None:
    saved: dict[str, str] = {}
    long_body = "start " + ("middle " * 2200) + "important tail"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f"<html><body><main><p>{long_body}</p></main></body></html>",
            request=request,
        )

    def artifact_writer(tool_name: str, content: str) -> str:
        saved["tool_name"] = tool_name
        saved["content"] = content
        return ".runs/episode/artifacts/tool-results/web_fetch-test.txt"

    result = web_fetch(
        {"url": "https://example.com/long", "max_chars": 12000},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
        artifact_writer=artifact_writer,
    )

    visible = result["model_visible"]
    assert visible["kind"] == "tool_result_view"
    assert visible["truncated"] is True
    assert visible["artifact"]["path"] == ".runs/episode/artifacts/tool-results/web_fetch-test.txt"
    assert visible["artifact"]["original_chars"] == len(saved["content"])
    assert "file_read" in visible["continuation_hint"]
    assert visible["artifact"]["path"] in visible["continuation_hint"]
    assert len(visible["content"]) < len(saved["content"])
    assert "start" in visible["content"]
    assert "important tail" in visible["content"]
    assert saved["tool_name"] == "web_fetch"
    assert saved["content"].startswith("<main>")
    assert "important tail" in saved["content"]


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
    assert result["model_visible"] == {
        "provider": "tavily",
        "query": "haagent docs",
        "returned_count": 1,
        "results": [
            {
                "title": "HaAgent Docs",
                "url": "https://example.com/docs",
                "snippet": "Docs snippet",
            },
        ],
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

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "web_search_configuration_error",
        "category": "provider",
        "message": "TAVILY_API_KEY is required for tavily web search",
        "retryable": False,
    }


def test_web_search_falls_back_to_configured_provider_when_default_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.search.brave.com"
        return httpx.Response(
            200,
            json={"web": {"results": [{"title": "Fallback", "url": "https://example.com/fallback", "description": "usable summary"}]}},
            request=request,
        )

    result = web_search(
        {"query": "haagent"},
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
        environ={"BRAVE_SEARCH_API_KEY": "brave-secret"},
    )

    assert result["status"] == "success"
    assert result["provider"] == "brave"
    assert result["provider_fallback"] == {
        "from": "tavily",
        "to": "brave",
        "reason": "web_search_configuration_error",
    }


def test_web_search_explicit_provider_never_falls_back() -> None:
    result = web_search(
        {"query": "haagent", "provider": "tavily"},
        environ={"BRAVE_SEARCH_API_KEY": "brave-secret"},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "web_search_configuration_error"
    assert "provider_fallback" not in result


def test_direct_mode_still_resolves_and_rejects_non_public_dns() -> None:
    def private_resolver(host: str, port: int):
        del host, port
        return {ipaddress.ip_address("10.0.0.8")}

    with pytest.raises(NetworkGuardError, match="non-public"):
        ensure_public_http_url("https://example.com/", resolver=private_resolver)


def test_proxy_env_is_passed_to_httpx_client_with_trust_env_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class CapturingClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        def __enter__(self) -> CapturingClient:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            del method, kwargs
            request = httpx.Request("GET", url)
            return httpx.Response(200, text="ok", request=request)

    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://proxy.example.com:7890")
    monkeypatch.setattr(httpx, "Client", CapturingClient)

    response = fetch_public_http_response("https://example.com/page", resolver=_public_resolver)

    assert response.status_code == 200
    assert seen["trust_env"] is False
    assert seen["proxy"] == "http://proxy.example.com:7890"
    assert seen["follow_redirects"] is False


def test_proxy_mode_does_not_call_local_resolver_for_ordinary_hostnames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolver(host: str, port: int):
        raise AssertionError(f"proxy mode must not resolve {host}:{port} locally")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok", request=request)

    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://proxy.example.com:7890")
    response = fetch_public_http_response(
        "https://www.reuters.com/",
        resolver=fail_resolver,
        transport=httpx.MockTransport(handler),
    )
    assert response.status_code == 200


def test_proxy_url_with_embedded_credentials_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://user:secret@proxy.example.com:7890")
    with pytest.raises(NetworkGuardError, match="embedded credentials"):
        fetch_public_http_response(
            "https://example.com/",
            resolver=_public_resolver,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://localhost.localdomain/",
        "http://metadata.google.internal/",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
    ],
)
def test_proxy_mode_rejects_local_hostnames_and_private_ip_literals(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://proxy.example.com:7890")
    with pytest.raises(NetworkGuardError, match="local hostnames|non-public"):
        fetch_public_http_response(
            url,
            resolver=_public_resolver,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        )


def test_proxy_mode_revalidates_dangerous_redirect_hops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/start":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}, request=request)
        return httpx.Response(200, text="unexpected", request=request)

    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://proxy.example.com:7890")
    with pytest.raises(NetworkGuardError, match="non-public"):
        fetch_public_http_response(
            "https://example.com/start",
            resolver=_public_resolver,
            transport=httpx.MockTransport(handler),
        )


def test_web_fetch_maps_connect_timeout_to_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        request = httpx.Request("GET", "https://www.reuters.com/")
        raise httpx.ConnectTimeout("timed out", request=request)

    monkeypatch.setattr("haagent.tools.web.fetch_public_http_response", boom)
    result = web_fetch({"url": "https://www.reuters.com/world/"})

    assert result["status"] == "error"
    error = result["error"]
    assert error["type"] == "web_connect_timeout"
    assert error["target_host"] == "www.reuters.com"
    assert error["failure_stage"] == "connect"
    assert error["proxy_configured"] is False
    assert error["resolution_mode"] == "direct"
    assert error["retryable"] is True
    assert "timeout_seconds" in error
    assert "timed out" not in error["message"].lower() or "connect" in error["message"].lower()
    assert result["recovery"]["action"] == "retry_same_call"


@pytest.mark.parametrize(
    ("exc", "expected_type", "stage", "proxy_env", "url"),
    [
        (
            httpx.ReadTimeout("read timed out", request=httpx.Request("GET", "https://example.com/")),
            "web_read_timeout",
            "read",
            None,
            "https://example.com/page",
        ),
        (
            httpx.ProxyError("proxy rejected", request=httpx.Request("GET", "https://example.com/")),
            "web_proxy_failed",
            "proxy",
            "http://proxy.example.com:7890",
            "https://example.com/page",
        ),
        (
            httpx.ConnectError(
                "[Errno 11001] getaddrinfo failed",
                request=httpx.Request("GET", "https://missing.example/"),
            ),
            "web_dns_failed",
            "dns",
            None,
            "https://missing.example/",
        ),
    ],
)
def test_web_fetch_maps_read_timeout_proxy_error_and_dns_failures(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
    expected_type: str,
    stage: str,
    proxy_env: str | None,
    url: str,
) -> None:
    monkeypatch.setattr(
        "haagent.tools.web.fetch_public_http_response",
        lambda *a, e=exc, **k: (_ for _ in ()).throw(e),
    )
    if proxy_env:
        monkeypatch.setenv("HAAGENT_WEB_PROXY", proxy_env)
    else:
        monkeypatch.delenv("HAAGENT_WEB_PROXY", raising=False)

    result = web_fetch({"url": url})
    error = result["error"]
    assert error["type"] == expected_type
    assert error["failure_stage"] == stage
    dumped = json.dumps(result, ensure_ascii=False)
    assert "proxy.example.com" not in dumped
    assert "user:secret" not in dumped


def test_web_fetch_error_does_not_leak_proxy_url_or_sensitive_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        request = httpx.Request("GET", "https://example.com/path?token=super-secret")
        raise httpx.ProxyError(
            "Unable to connect to proxy http://user:pass@proxy.example.com:7890",
            request=request,
        )

    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://user:pass@proxy.example.com:7890")
    monkeypatch.setattr("haagent.tools.web.fetch_public_http_response", boom)
    # 代理凭据在配置阶段就会拒绝；此处模拟 fetch 层已收到 ProxyError 时的脱敏。
    monkeypatch.delenv("HAAGENT_WEB_PROXY", raising=False)
    monkeypatch.setenv("HAAGENT_WEB_PROXY", "http://proxy.example.com:7890")

    result = web_fetch({"url": "https://example.com/path?token=super-secret"})
    dumped = json.dumps(result, ensure_ascii=False)
    assert result["error"]["type"] == "web_proxy_failed"
    assert "proxy.example.com" not in dumped
    assert "user:pass" not in dumped
    assert "super-secret" not in dumped
    assert result["error"]["target_host"] == "example.com"
    assert result["error"]["proxy_configured"] is True


def test_web_fetch_maps_http_status_and_target_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    def status_error(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        request = httpx.Request("GET", "https://example.com/missing")
        response = httpx.Response(404, text="missing", request=request)
        response.raise_for_status()
        return response

    monkeypatch.setattr("haagent.tools.web.fetch_public_http_response", status_error)
    result = web_fetch({"url": "https://example.com/missing"})
    assert result["error"]["type"] == "web_http_error"
    assert result["error"]["failure_stage"] == "http"
    assert result["error"]["retryable"] is False
    assert result["recovery"]["action"] == "use_alternate_source"

    def denied(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        raise NetworkGuardError("target resolves to non-public address(es): 10.0.0.1")

    monkeypatch.setattr("haagent.tools.web.fetch_public_http_response", denied)
    denied_result = web_fetch({"url": "https://example.com/"})
    assert denied_result["error"]["type"] == "web_target_denied"
    assert denied_result["error"]["retryable"] is False


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_web_fetch_marks_only_transient_http_statuses_retryable(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    def status_error(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        request = httpx.Request("GET", "https://example.com/transient")
        response = httpx.Response(status_code, headers={"Retry-After": "3"}, request=request)
        response.raise_for_status()
        return response

    monkeypatch.setattr("haagent.tools.web.fetch_public_http_response", status_error)

    result = web_fetch({"url": "https://example.com/transient"})

    assert result["status"] == "error"
    assert result["error"]["retryable"] is True
    assert result["error"]["retry_after_seconds"] == 3
    assert result["recovery"]["action"] == "retry_same_call"
