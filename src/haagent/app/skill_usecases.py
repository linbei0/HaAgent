"""
haagent/app/skill_usecases.py - 技能类应用用例

集中封装 AssistantService 的本地 skills 与 marketplace 相关操作。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haagent.skills import trust_project_root, untrust_project_root
from haagent.skills.marketplace import MarketplaceError, install_marketplace_skill_card, search_marketplace
from haagent.tools.skills import skill_list, skill_read

if TYPE_CHECKING:
    from haagent.app.assistant_service import (
        AssistantMarketplaceInstall,
        AssistantMarketplaceSearch,
        AssistantSkillContent,
        AssistantSkillList,
        AssistantService,
    )


def list_skills_for_user(service: "AssistantService") -> "AssistantSkillList":
    result = service.skill_list_fn({}, service.workspace_root)
    if result.get("status") != "success":
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        raise service.error_cls(str(error.get("message", "failed to list skills")))
    return service.skill_list_cls(
        skills=list(result.get("skills", [])),
        blocked_project_skill_roots=[
            str(path) for path in result.get("blocked_project_skill_roots", [])
        ],
    )


def trust_project_skills(service: "AssistantService") -> "AssistantSkillList":
    service.trust_project_root_fn(service.workspace_root)
    return list_skills_for_user(service)


def untrust_project_skills(service: "AssistantService") -> "AssistantSkillList":
    service.untrust_project_root_fn(service.workspace_root)
    return list_skills_for_user(service)


def read_skill_for_user(service: "AssistantService", name: str) -> "AssistantSkillContent":
    result = service.skill_read_fn({"name": name}, service.workspace_root, user_invoked=True)
    if result.get("status") != "success":
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        raise service.error_cls(str(error.get("message", f"skill not found: {name}")))
    return service.skill_content_cls(
        name=str(result["name"]),
        command_name=str(result.get("command_name") or result["name"]),
        content=str(result["content"]),
    )


def search_skill_marketplace(
    service: "AssistantService",
    query: str,
    *,
    providers: list[str] | None=None,
    limit: int=10,
) -> "AssistantMarketplaceSearch":
    try:
        result = service.search_marketplace_fn(query, providers=providers, limit=limit)
    except MarketplaceError as error:
        raise service.error_cls(str(error)) from error
    service._marketplace_results = {card.result_id: card for card in result.cards}
    return service.marketplace_search_cls(
        status=result.status,
        query=result.query,
        results=[service.marketplace_skill_mapper(card) for card in result.cards],
        warnings=list(result.warnings),
    )


def install_marketplace_skill(
    service: "AssistantService",
    result_id: str,
) -> "AssistantMarketplaceInstall":
    card = service._marketplace_results.get(result_id)
    if card is None:
        raise service.error_cls(f"unknown marketplace result id: {result_id}")
    try:
        installed = service.install_marketplace_skill_card_fn(card)
    except MarketplaceError as error:
        raise service.error_cls(str(error)) from error
    return service.marketplace_install_cls(
        name=installed.name,
        command_name=installed.command_name,
        skill_dir=installed.skill_dir,
        skill_file=installed.skill_file,
        source_url=installed.source_url,
    )
