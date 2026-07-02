"""
tests/integration/web/test_skill_marketplace.py - 远端 skill marketplace 客户端测试

验证 skills.sh 与 SkillsMP 搜索结果归一化、错误显式化，以及用户级引用 skill 安装边界。
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from haagent.skills.marketplace import (
    MarketplaceError,
    MarketplaceProvider,
    MarketplaceSkillCard,
    install_marketplace_skill_card,
    search_marketplace,
)


def test_search_marketplace_normalizes_skills_sh_and_skillsmp_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "skills.sh":
            return httpx.Response(
                200,
                json={
                    "skills": [
                        {
                            "id": "cowork-os/cowork-os/analyze-csv",
                            "skillId": "analyze-csv",
                            "name": "analyze-csv",
                            "installs": 48,
                            "source": "cowork-os/cowork-os",
                        },
                    ],
                },
            )
        if request.url.host == "skillsmp.com":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "skills": [
                            {
                                "id": "openai-csv-workbench",
                                "name": "csv-workbench",
                                "author": "openai",
                                "description": "Analyze CSV files and return concise summaries.",
                                "githubUrl": "https://github.com/openai/openai-agents-python/tree/main/examples/tools/skills/csv-workbench",
                                "skillUrl": "https://skillsmp.com/creators/openai/csv-workbench",
                                "stars": 27327,
                            },
                        ],
                    },
                },
            )
        raise AssertionError(f"unexpected host: {request.url.host}")

    result = search_marketplace("csv", limit=5, transport=httpx.MockTransport(handler))

    assert result.status == "success"
    assert [card.provider for card in result.cards] == [
        MarketplaceProvider.SKILLS_SH,
        MarketplaceProvider.SKILLSMP,
    ]
    assert result.cards[0].installable is True
    assert result.cards[0].detail_url == "https://skills.sh/cowork-os/cowork-os/analyze-csv"
    assert result.cards[0].quality["installs"] == 48
    assert result.cards[1].installable is False
    assert result.cards[1].summary == "Analyze CSV files and return concise summaries."
    assert result.cards[1].quality["stars"] == 27327


def test_search_marketplace_keeps_successful_provider_when_other_provider_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "skills.sh":
            return httpx.Response(
                200,
                json={
                    "skills": [
                        {
                            "id": "vercel-labs/bash-tool/csv",
                            "skillId": "csv",
                            "name": "csv",
                            "source": "vercel-labs/bash-tool",
                        },
                    ],
                },
            )
        return httpx.Response(503, json={"error": "busy"})

    result = search_marketplace("csv", transport=httpx.MockTransport(handler))

    assert result.status == "partial"
    assert len(result.cards) == 1
    assert result.cards[0].provider == MarketplaceProvider.SKILLS_SH
    assert result.warnings == ["skillsmp search failed: HTTP 503"]


def test_search_marketplace_rejects_unknown_provider() -> None:
    with pytest.raises(MarketplaceError, match="unsupported marketplace provider"):
        search_marketplace("csv", providers=["generic_agent"], transport=httpx.MockTransport(lambda request: httpx.Response(200)))


def test_install_marketplace_skill_card_writes_reference_skill_without_overwriting(tmp_path: Path) -> None:
    card = MarketplaceSkillCard(
        provider=MarketplaceProvider.SKILLS_SH,
        result_id="skills-sh-1",
        remote_id="cowork-os/cowork-os/analyze-csv",
        name="analyze-csv",
        source="cowork-os/cowork-os",
        summary="Analyze CSV files.",
        detail_url="https://skills.sh/cowork-os/cowork-os/analyze-csv",
        installable=True,
        quality={"installs": 48},
    )

    installed = install_marketplace_skill_card(card, config_dir=tmp_path / ".haagent")

    skill_file = installed.skill_dir / "SKILL.md"
    assert installed.command_name == "analyze-csv"
    assert skill_file.exists()
    content = skill_file.read_text(encoding="utf-8")
    assert "source: marketplace" in content
    assert "provider: skills_sh" in content
    assert "https://skills.sh/cowork-os/cowork-os/analyze-csv" in content
    assert "External marketplace reference" in content
    with pytest.raises(MarketplaceError, match="already exists"):
        install_marketplace_skill_card(card, config_dir=tmp_path / ".haagent")


def test_install_marketplace_skill_card_rejects_non_installable_provider(tmp_path: Path) -> None:
    card = MarketplaceSkillCard(
        provider=MarketplaceProvider.SKILLSMP,
        result_id="skillsmp-1",
        remote_id="openai-csv-workbench",
        name="csv-workbench",
        source="openai",
        summary="Analyze CSV files.",
        detail_url="https://skillsmp.com/creators/openai/csv-workbench",
        installable=False,
    )

    with pytest.raises(MarketplaceError, match="only skills_sh results are installable"):
        install_marketplace_skill_card(card, config_dir=tmp_path / ".haagent")
