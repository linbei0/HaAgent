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
        description=(
            "Delegate a complex, self-contained multi-step research, implementation, or verification task to a "
            "background agent. Do not delegate a single file read, one grep, or a short direct operation. The "
            "prompt must state the scope, whether edits are allowed, constraints, and expected evidence. Start "
            "independent tasks concurrently, then use task_get/task_output to collect results instead of repeating "
            "the delegated work in the main agent."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "short task label shown in worker status and task lists",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "self-contained instructions including scope, edit permission, constraints, deliverable, "
                        "and the evidence or verification the worker must return"
                    ),
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["explorer", "worker", "verification"],
                    "description": (
                        "explorer for read-only investigation, worker for implementation, verification for "
                        "independent checks"
                    ),
                },
                "team": {
                    "type": "string",
                    "description": "optional team label for grouping related workers",
                },
                "model_profile": {
                    "type": "string",
                    "description": "optional configured model profile for this worker",
                },
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
        description=(
            "Send new constraints, discovered evidence, or a correction to an existing worker. Use only when the "
            "worker can continue from its current context; start a new agent when the task itself is different."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "worker or task identifier returned by agent/task_list",
                },
                "message": {
                    "type": "string",
                    "description": "specific follow-up information or correction; do not resend the full task",
                },
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
        description=(
            "Stop a worker only when its result is no longer needed, its scope is wrong, or it is unsafe to let "
            "the current operation continue. Prefer a normal stop; force only when it does not stop cooperatively."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "task identifier returned by agent/task_list"},
                "force": {
                    "type": "boolean",
                    "description": "force termination only after a normal stop is insufficient",
                },
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
        description=(
            "Inspect one worker's current status and metadata. Use this to decide whether to wait, send a "
            "follow-up, read completed output, or stop the task; it does not return the full task result."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "task identifier returned by agent/task_list"},
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
        description=(
            "List worker tasks in the current session, optionally filtered by status. Use when task identifiers "
            "or overall worker progress are unknown; use task_get for one known task."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["queued", "running", "idle", "completed", "failed", "stopped"],
                    "description": "optional exact status filter",
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
        description=(
            "Read the bounded result and evidence produced by a worker after task_get reports useful progress or "
            "completion. If output is truncated, request a larger max_chars within the cap instead of rerunning "
            "the delegated task."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "task identifier returned by agent/task_list"},
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
