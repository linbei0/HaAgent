"""
haagent/tools/contributions/agent.py - 后台智能体静态工具 contribution
"""

from __future__ import annotations

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.catalog import ToolContribution

_AGENT_CONTROL = frozenset({"chat_default", "agent_control"})

AGENT_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="agent",
        description="spawn a background worker agent for delegated research, implementation, or verification",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
                "subagent_type": {
                    "type": "string",
                    "enum": ["explorer", "worker", "verification"],
                },
                "team": {"type": "string"},
                "model_profile": {"type": "string"},
                "profile": {
                    "type": "string",
                    "description": "agent profile name; defaults to subagent_type when omitted",
                },
            },
            "required": ["description", "prompt", "subagent_type"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
    ToolContribution(
        name="send_message",
        description="send a follow-up message to an existing worker agent",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["to", "message"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
    ToolContribution(
        name="task_stop",
        description="request a running worker task to stop",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "force": {"type": "boolean"},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
    ToolContribution(
        name="task_get",
        description="get status and metadata for one background worker task",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
    ToolContribution(
        name="task_list",
        description="list background worker tasks for the current session",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["queued", "running", "idle", "completed", "failed", "stopped"],
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
    ToolContribution(
        name="task_output",
        description="read bounded output from a background worker task episode",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "description": "maximum output characters to return; capped at 50000",
                },
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=_AGENT_CONTROL,
        router_owned=True,
    ),
]
