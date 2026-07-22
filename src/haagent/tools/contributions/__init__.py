"""
haagent/tools/contributions/__init__.py - 静态工具 contribution 聚合入口

新增静态工具只在对应 contribution 模块登记一次，不在全局列表重复枚举。
"""

from __future__ import annotations

from haagent.tools.catalog import ToolContribution
from haagent.tools.contributions.agent import AGENT_CONTRIBUTIONS
from haagent.tools.contributions.core import CORE_CONTRIBUTIONS
from haagent.tools.contributions.files import FILE_CONTRIBUTIONS
from haagent.tools.contributions.jobs import JOB_CONTRIBUTIONS
from haagent.tools.contributions.mcp import MCP_CONTRIBUTIONS
from haagent.tools.contributions.shell import SHELL_CONTRIBUTIONS
from haagent.tools.contributions.skills import SKILL_CONTRIBUTIONS
from haagent.tools.contributions.web import WEB_CONTRIBUTIONS


def all_static_contributions() -> list[ToolContribution]:
    return [
        *CORE_CONTRIBUTIONS,
        *AGENT_CONTRIBUTIONS,
        *FILE_CONTRIBUTIONS,
        *SKILL_CONTRIBUTIONS,
        *WEB_CONTRIBUTIONS,
        *MCP_CONTRIBUTIONS,
        *SHELL_CONTRIBUTIONS,
        *JOB_CONTRIBUTIONS,
    ]
