"""
tests/e2e/test_real_model_dogfood.py - 真实模型手动 dogfood

仅在传入 --real-llm 时调用真实模型；默认 pytest 必须显式跳过。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from haagent.models.gateway import OpenAIChatCompletionsGateway, OpenAIResponsesGateway
from haagent.models.provider_profile import ProviderProfileError, load_provider_profile
from haagent.runtime.evaluation.dogfood import render_dogfood_report, run_dogfood_tasks


pytestmark = pytest.mark.real_llm


def test_real_model_dogfood(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    if not pytestconfig.getoption("--real-llm"):
        pytest.skip("real model dogfood skipped; rerun with --real-llm")

    gateway = _real_gateway_or_skip()
    report = run_dogfood_tasks(
        gateway,
        runs_root=tmp_path / "dogfood-runs",
        max_turns=int(os.environ.get("HAAGENT_DOGFOOD_MAX_TURNS", "16")),
        auto_approve=True,
    )
    print("\n" + render_dogfood_report(report))

    assert report.status == "completed"


def _real_gateway_or_skip():
    profile_name = os.environ.get("HAAGENT_DOGFOOD_PROFILE")
    if profile_name:
        try:
            profile = load_provider_profile(profile_name)
        except ProviderProfileError as error:
            pytest.skip(f"real model dogfood skipped: {error}")
        gateway_kwargs = {
            "api_key": profile.api_key,
            "model": profile.model,
            "base_url": profile.base_url,
        }
        if profile.provider == "openai":
            return OpenAIResponsesGateway(**gateway_kwargs)
        if profile.provider == "openai-chat":
            return OpenAIChatCompletionsGateway(**gateway_kwargs)
        pytest.skip(f"real model dogfood skipped: unsupported provider {profile.provider}")

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("real model dogfood skipped: OPENAI_API_KEY is not set")

    provider = os.environ.get("HAAGENT_DOGFOOD_PROVIDER", "openai")
    model = os.environ.get("HAAGENT_DOGFOOD_MODEL", "gpt-4.1-mini")
    base_url = os.environ.get("HAAGENT_DOGFOOD_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if provider == "openai":
        return OpenAIResponsesGateway(model=model, base_url=base_url)
    if provider == "openai-chat":
        return OpenAIChatCompletionsGateway(model=model, base_url=base_url)
    pytest.skip(f"real model dogfood skipped: unsupported provider {provider}")
