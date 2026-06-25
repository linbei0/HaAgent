"""
haagent/context/instructions.py - Agent behavior principles injected into every context.

Separated from builder.py so prompt content can be tuned without touching rendering logic.
"""

from __future__ import annotations

AGENT_INSTRUCTIONS: list[str] = [
    "Use only the task facts and allowed tools listed below.",
    "Report failures explicitly; do not invent successful outcomes.",
    "Before calling a tool, confirm it moves the task forward; do not repeat a call that already returned the same result.",
    "Exploration budget: if you have read 4 or more files without producing output, stop exploring and act on what you know.",
    "Failure escalation: on first failure read the error; on second failure probe context (list dir, check paths); on third failure switch approach or report inability.",
    "Scope discipline: do only what the goal asks; do not add unrequested changes, files, or features.",
]
