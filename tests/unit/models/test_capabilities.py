"""
tests/unit/models/test_capabilities.py - 模型能力需求与匹配测试

验证工具、图像、流式和上下文需求能够被确定性提取并显式比较。
"""

from haagent.models.capabilities import (
    ModelCapabilities,
    apply_context_window_limit,
    build_model_requirements,
    effective_input_window_tokens,
    missing_capabilities,
)


def test_build_model_requirements_detects_tools_vision_streaming_and_tokens() -> None:
    requirements = build_model_requirements(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_attachment", "path": "attachments/image.png"},
                ],
            },
        ],
        tool_schemas=[{"name": "file_read", "input_schema": {"type": "object"}}],
        streaming=True,
    )

    assert requirements.tools is True
    assert requirements.vision is True
    assert requirements.streaming is True
    assert requirements.estimated_input_tokens > 0


def test_missing_capabilities_ignores_unknown_but_reports_explicit_unsupported() -> None:
    requirements = build_model_requirements(
        messages=[{"role": "user", "content": "Read the workspace"}],
        tool_schemas=[{"name": "file_read", "input_schema": {"type": "object"}}],
        streaming=True,
    )
    capabilities = ModelCapabilities(
        tools="unsupported",
        streaming="unknown",
        vision="supported",
        reasoning="unknown",
        tools_mode="none",
        context_window_tokens=1,
        protocols=frozenset({"chat_completions"}),
    )

    assert missing_capabilities(requirements, capabilities) == ("tools", "context_window")


def test_compatible_tool_mode_counts_as_supported() -> None:
    requirements = build_model_requirements(
        messages=[{"role": "user", "content": "Use a tool"}],
        tool_schemas=[{"name": "shell", "input_schema": {"type": "object"}}],
        streaming=False,
    )
    capabilities = ModelCapabilities(
        tools="supported",
        streaming="supported",
        vision="unsupported",
        reasoning="unknown",
        tools_mode="compat",
        context_window_tokens=32_768,
        protocols=frozenset({"responses", "chat_completions"}),
    )

    assert missing_capabilities(requirements, capabilities) == ()


def test_effective_input_window_prefers_input_limit() -> None:
    assert effective_input_window_tokens(
        ModelCapabilities(context_window_tokens=1_000_000, input_window_tokens=400_000),
    ) == 400_000


def test_user_context_limit_never_expands_provider_window() -> None:
    cases = [
        (ModelCapabilities(context_window_tokens=1_000_000), (400_000, 400_000)),
        (ModelCapabilities(context_window_tokens=128_000), (128_000, 128_000)),
        (ModelCapabilities(input_window_tokens=128_000), (128_000, 128_000)),
        (
            ModelCapabilities(context_window_tokens=1_000_000, input_window_tokens=180_000),
            (400_000, 180_000),
        ),
        (None, (400_000, 400_000)),
    ]

    for capabilities, expected_windows in cases:
        limited = apply_context_window_limit(capabilities, 400_000)
        assert limited is not None
        assert (limited.context_window_tokens, limited.input_window_tokens) == expected_windows
