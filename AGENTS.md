# AGENTS.md

## Project Overview

HaAgent is a local personal AI assistant written in Python. Its product target is: configure a model once, enter any local directory, run `haagent`, and talk naturally while HaAgent reads files, organizes material, edits documents, analyzes local projects, runs commands, and continues multi-turn tasks inside that directory.

HaAgent is not a Codex clone, not an IDE, and not only a code repository assistant. Code development remains one task type, but ordinary product language and default CLI flow must cover personal assistant work across local files and folders.

Harness remains important, but it stays behind the scenes. The runtime should constrain tools, record model/tool traces, write episode packages, and support inspect/eval without forcing ordinary users to understand `task.yaml`, episode internals, dogfood, or eval export before using the assistant.

## Project Reference Documents

Before making non-trivial changes, consult the relevant project documents:

- `docs/harness-requirements.md` defines the product direction, current stage, non-goals, and the two baseline constraints:
  - do not increase user mental burden;
  - do not increase model input token usage.
- `docs/unresolved-risks-and-roadmap.md` defines the current unresolved risks and near-term roadmap. It must stay aligned with `docs/harness-requirements.md`; if they conflict, the requirements document wins.
- `docs/code-governance.md` defines code ownership boundaries, unique runtime entry points, change categories, verification expectations, and refactoring guardrails.

Use these documents as decision inputs, not as permission to expand scope. For small mechanical edits, read only the directly relevant document. For feature, contract, runtime, context, episode, tool, provider, or CLI behavior changes, read the relevant sections before editing.

The current priority is the personal assistant startup experience:

- Prefer `haagent setup` followed by plain `haagent` as the primary user path:

  ```powershell
  uv run haagent setup
  cd E:\some-folder
  uv run haagent
  ```

- Keep `haagent chat` as an explicit/advanced natural-language entry point, not the only ordinary path.
- Keep `task.yaml` for advanced reproducibility, batch tasks, smoke cases, and eval construction; do not treat it as the ordinary user entry point.
- Do not block real task execution on harness completeness. Build the direct Agent experience first, keep harness constraints and traces intact, and fill in missing harness engineering after the experience proves useful.
- Plain `haagent` and chat should default to the current working directory as workspace root, allow explicit `--workspace-root`, and keep file/shell tools bounded by that root.
- The real task tool pack includes `file_read`, `file_write`, `apply_patch`, `shell`, and `code_run`; keep these tools atomic and workspace-bound.
- Tasks may be file organization, document summarization, CSV inspection, draft editing, project analysis, code-changing, or verification-oriented. Do not assume every task must modify code or have a verification command.

## Document Precedence

- `AGENTS.md` defines the active working rules for coding agents.
- `docs/harness-requirements.md` defines product and engineering direction.
- `docs/code-governance.md` defines code organization and change discipline.
- `docs/unresolved-risks-and-roadmap.md` defines current priorities and known risks.

If documents disagree, prefer the narrower and more current rule. Do not silently choose one; mention the conflict and update the stale document when the task scope includes documentation.

## Setup Commands

- Install dependencies: `uv sync`
- Run a focused test file during development: `uv run pytest tests/test_tool_router.py -q`
- Run all tests when needed for final regression: `uv run pytest -q`
- Run the fast local quality gate: `uv run haagent check`

## Development Workflow

- Use `uv` for virtual environment and dependency management.
- Keep the package in `src/haagent`.
- Keep tests in `tests`.
- Prefer `apply_patch` for file edits to avoid PowerShell encoding issues.
- Do not add UI, browser automation, multi-agent behavior, or long-term memory unless explicitly requested.
- For CLI work, prioritize the direct personal assistant experience: `haagent setup`, then interactive `haagent` from any directory. Keep `haagent chat "<request>"` and interactive `haagent chat` as explicit entries.
- Interactive `haagent` / `haagent chat` is backed by `AgentSession`; keep session state and bounded summaries in runtime code, not in `cli.py`.
- Keep `run`, `inspect`, and `export-eval` functional, but do not optimize them ahead of the chat experience unless the task explicitly asks.

## Compatibility Policy

- HaAgent is currently a pre-user, pre-1.0 development project.
- Do not preserve compatibility for historical `.runs`, old episode schemas, old context manifests, old eval cases, or old internal test interfaces unless explicitly requested.
- Do not add legacy paths, fallback behavior, old-field support, old-status support, or silent degradation just to keep development artifacts readable.
- Schema and trace format changes may break old local run artifacts; new runs must remain explicit, validated, inspectable, and covered by tests.
- Compatibility is allowed only for current real needs:
  - external provider differences, such as OpenAI Responses and OpenAI-compatible Chat Completions;
  - task authoring ergonomics, such as omitted `policy` or `workspace_root`;
  - real partial-failure states, such as a run failing before verification files are written.
- If compatibility seems necessary, state who depends on it, what real failure it prevents, and why fail-fast behavior is not better.

## Boundary and Matching Policy

- Do not use brittle hard-coded matching as a runtime, tool, memory, UI, provider, or context-state boundary. In particular, do not route or block behavior by matching user-language phrase lists, AI-output wording, profile/memory filename token lists, content vocabulary, or shell/code strings. User language habits vary, model outputs are stochastic, and these checks create false confidence while missing easy bypasses.
- Prefer explicit capability and protocol boundaries: structured tool schemas, typed events, task/session state, approval policy, normalized workspace paths, exact storage roots, service methods, and validated metadata. If the system needs a durable decision, make that decision a field, state transition, API call, or path/capability check rather than a guess over free text.
- Matching is acceptable only when it operates on a stable technical surface and is not interpreting intent: schema enum values, exact tool names, known protocol fields, normalized path containment, parser selection by file extension, or security-focused secret detection. Keep these rules narrow, named, tested, and documented.
- Before adding any new string/regex/table-based rule, state what stable contract it represents, why a structured boundary is not better, what false positives/false negatives are acceptable, and which tests prove that ordinary user phrasing or model randomness cannot change the outcome.

## Testing Instructions

- Add or update pytest coverage for every behavior change.
- For bug fixes and new behavior, write the failing test first, then implement the smallest code that passes.
- During the TDD inner loop, run the smallest relevant pytest target first: a single test, a single test file, or the directly affected test group.
- Do not automatically run full `uv run pytest` after every small edit.
- Before claiming completion, run the tests directly relevant to the changed behavior.
- Run full `uv run pytest -q` when a change crosses multiple core modules, changes shared runtime contracts, touches `ToolRouter`, `ModelGateway`, context, episode, CLI entry points, workspace boundaries, or secret handling, or when preparing a commit, merge, release, or user-facing handoff.
- Run `uv run haagent check` before user-facing handoff when the change affects harness, eval, smoke behavior, CLI quality gates, or runtime task execution.
- Prefer extending or parameterizing existing tests over adding near-duplicate tests. TDD scaffolding tests may be removed before completion when they no longer provide independent behavioral signal.

## Code Style

- Responses and code comments should use Simplified Chinese when explaining project-specific behavior.
- Every Python file must start with a module docstring in this style:

  ```python
  """
  path/to/file.py - 简短职责说明

  说明该文件在 HaAgent 中负责什么。
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
- Path-mutating and execution tools must stay inside the configured workspace root and must not bypass ToolRouter policy or approval decisions.
- Chat/natural-language entry points must not bypass the runtime contracts. If they generate temporary task contracts internally, those contracts must be recorded in the episode for later inspection.
- REPL chat may carry only bounded session summaries into the next model input; it must not copy full history, full episode traces, or full tool outputs.
- Harness audit data should not be copied wholesale into model input. Use compact observations and bounded source budgets.

## Context and Prompt Engineering

- Do not fix runtime, tool, session, UI, provider, or context-state bugs by adding symptom-specific prompt instructions to model input.
- Before adding any model-visible instruction, first determine whether the behavior should be enforced by code, state machines, schemas, tool contracts, validation, or deterministic context facts.
- Model input should contain durable task facts, bounded observations, compact state, and reusable workflow rules. It should not accumulate one-off corrective instructions for individual failures.
- If a new prompt/context line is necessary, it must be general, reusable across task types, token-conscious, and backed by tests that prove why code-level enforcement is not the right boundary.
- Prefer neutral structured facts over imperative prompt patches. For example, expose a compact state record only when the model needs that fact; do not add instructions that merely tell the model not to repeat a previously observed mistake.
