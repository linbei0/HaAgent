# AGENTS.md

## Project Overview

AgentFoundry is a harness-first Agent Runtime MVP written in Python.
The current runtime loads `task.yaml`, runs a small orchestrator state machine, routes local tools, records model/tool traces, and writes episode packages.

## Setup Commands

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a focused test file: `uv run pytest tests/test_tool_router.py -q`

## Development Workflow

- Use `uv` for virtual environment and dependency management.
- Keep the package in `src/agent_foundry`.
- Keep tests in `tests`.
- Prefer `apply_patch` for file edits to avoid PowerShell encoding issues.
- Do not add UI, browser automation, multi-agent behavior, or long-term memory unless explicitly requested.

## Testing Instructions

- Add or update pytest coverage for every behavior change.
- For bug fixes and new behavior, write the failing test first, then implement the smallest code that passes.
- Run `uv run pytest` before claiming completion.

## Code Style

- Responses and code comments should use Simplified Chinese when explaining project-specific behavior.
- Every Python file must start with a module docstring in this style:

  ```python
  """
  path/to/file.py - 简短职责说明

  说明该文件在 AgentFoundry 中负责什么。
  """
  ```

- Add concise comments for complex workflows, failure boundaries, provider/tool behavior, or security-sensitive checks.
- Do not comment obvious assignments or one-line boilerplate.
- Keep comments current when changing behavior.

## Runtime Rules

- Model calls must go through the `ModelGateway` interface.
- Tool calls must go through `ToolRouter`.
- Every tool call must append a record to `tool-calls.jsonl`.
- Model calls and responses must append records to `transcript.jsonl`.
- Failures must be explicit and structured; do not add silent fallbacks or simulated success paths.
- Path-mutating tools must stay inside the configured workspace root.
