"""
haagent/app/skill_usecases.py - Skills 与 marketplace 应用 Module

管理本地 skill 信任、读取、marketplace 查询缓存和安装。
"""

from __future__ import annotations

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantMarketplaceInstall,
    AssistantMarketplaceSearch,
    AssistantMarketplaceSkill,
    AssistantServiceError,
    AssistantSkillContent,
    AssistantSkillList,
)
from haagent.skills import trust_project_root, untrust_project_root
from haagent.skills.marketplace import (
    MarketplaceError,
    MarketplaceSkillCard,
    install_marketplace_skill_card,
    search_marketplace,
)
from haagent.tools.skills import skill_list, skill_read


class AssistantSkills:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context
        self._marketplace_results: dict[str, MarketplaceSkillCard] = {}

    def list(self) -> AssistantSkillList:
        result = skill_list(
            {},
            self._context.workspace_root,
            skill_catalog=self._context.skill_catalog,
        )
        if result.get("status") != "success":
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            raise AssistantServiceError(str(error.get("message", "failed to list skills")))
        return AssistantSkillList(
            skills=list(result.get("skills", [])),
            blocked_project_skill_roots=[
                str(path) for path in result.get("blocked_project_skill_roots", [])
            ],
        )

    def trust_project(self) -> AssistantSkillList:
        trust_project_root(self._context.workspace_root)
        self._invalidate_skill_catalog()
        return self.list()

    def untrust_project(self) -> AssistantSkillList:
        untrust_project_root(self._context.workspace_root)
        self._invalidate_skill_catalog()
        return self.list()

    def read_for_user(self, name: str) -> AssistantSkillContent:
        result = skill_read(
            {"name": name},
            self._context.workspace_root,
            skill_catalog=self._context.skill_catalog,
            user_invoked=True,
        )
        if result.get("status") != "success":
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            raise AssistantServiceError(str(error.get("message", f"skill not found: {name}")))
        return AssistantSkillContent(
            name=str(result["name"]),
            command_name=str(result.get("command_name") or result["name"]),
            content=str(result["content"]),
        )

    def search_marketplace(
        self,
        query: str,
        *,
        providers: list[str] | None = None,
        limit: int = 10,
    ) -> AssistantMarketplaceSearch:
        try:
            result = search_marketplace(query, providers=providers, limit=limit)
        except MarketplaceError as error:
            raise AssistantServiceError(str(error)) from error
        self._marketplace_results = {card.result_id: card for card in result.cards}
        return AssistantMarketplaceSearch(
            status=result.status,
            query=result.query,
            results=[_marketplace_skill(card) for card in result.cards],
            warnings=list(result.warnings),
        )

    def install_marketplace(self, result_id: str) -> AssistantMarketplaceInstall:
        card = self._marketplace_results.get(result_id)
        if card is None:
            raise AssistantServiceError(f"unknown marketplace result id: {result_id}")
        try:
            installed = install_marketplace_skill_card(card)
        except MarketplaceError as error:
            raise AssistantServiceError(str(error)) from error
        self._invalidate_skill_catalog()
        return AssistantMarketplaceInstall(
            name=installed.name,
            command_name=installed.command_name,
            skill_dir=installed.skill_dir,
            skill_file=installed.skill_file,
            source_url=installed.source_url,
        )

    def _invalidate_skill_catalog(self) -> None:
        self._context.skill_catalog.invalidate_workspace(self._context.workspace_root)


def _marketplace_skill(card: MarketplaceSkillCard) -> AssistantMarketplaceSkill:
    return AssistantMarketplaceSkill(
        result_id=card.result_id,
        provider=card.provider.value,
        name=card.name,
        source=card.source,
        summary=card.summary,
        detail_url=card.detail_url,
        installable=card.installable,
        quality=dict(card.quality),
    )
