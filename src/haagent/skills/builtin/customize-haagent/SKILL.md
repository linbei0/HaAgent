---
name: haagent-config
description: Use when configuring, diagnosing, or changing HaAgent itself, including model connections, active models, runtime settings, sessions, skills, permissions, or user configuration files.
user-invocable: true
---

# Configuring HaAgent

This skill describes HaAgent's own configuration surfaces. Treat the owning
service and schema in the source tree as the source of truth; this document is
a navigation guide, not a second schema.

## Configuration layers

- User model connections live in `~/.haagent/providers.json`. It is version 4
  and contains connection records, model options, variants, and optional
  per-model `max_context_tokens`.
- User model routing lives in `~/.haagent/settings.json`. It stores the active
  model, optional fallback model, and cloud fallback consent.
- Runtime user settings also live under `~/.haagent`; use the runtime settings
  service rather than editing its defaults in source code.
- Workspace and session state live under the current workspace's `.runs`
  directory. Episode packages are evidence for one prompt; session state is a
  bounded resume summary and is not a replacement for episode evidence.
- Skills are loaded from HaAgent's built-in package skills, user skills, and
  trusted project skill directories. Project skill loading requires trust.

## Authoritative code

Before changing a field, inspect the owner instead of guessing its shape:

- `haagent.models.config.connections` owns provider connection records and
  `providers.json` paths.
- `haagent.models.model_options` owns per-model options and validation.
- `haagent.models.config.selection_store` owns active/fallback model routing.
- `haagent.runtime.settings` owns non-model runtime settings.
- `haagent.skills.loader` and `haagent.skills.catalog` own skill discovery,
  trust, and cache behavior.

Preserve fields the user did not ask to change. Never change a source default
to satisfy a user-level configuration request.

## Normal user entrypoints

The normal interactive entrypoint is `haagent` in the target workspace. Use
the TUI's `/connect` flow to configure a provider and `/model` to select a
model. Use `/sessions`, `--continue`, or `--resume` for session continuity.
Advanced run, inspect, eval, and smoke commands are development or validation
paths, not the normal configuration workflow.

## Credentials and permissions

API keys are resolved from the current environment, the system credential
store, or an explicitly selected insecure user file. Never copy a real key into
project files, episode packages, transcripts, logs, skill content, or UI text.

The workspace root and approved external roots are separate boundaries. A user
configuration file outside the workspace may be read or changed only through
the normal external-root and permission flow. A successful tool call does not
grant access to other paths. "Allow once" applies only to the current tool
call; "always allow" records a bounded session permission pattern that is
re-evaluated for later calls.

## Reload semantics

- A `/connect` save updates the user profile and the service status; it does
  not rewrite a turn that is already running. Finish or cancel that turn first.
- A model or active-selection change can be applied to the live
  `AgentSession` through the service reload path. `AgentSession.reload` keeps
  the bounded session state and can reuse MCP/tool resources while replacing a
  changed model gateway.
- `/sessions`, `--continue`, and `--resume` load the persisted session package;
  they do not replay an old transcript or tool trace into the prompt.
- Changes to package code, installed package-owned skills, or process-level
  runtime defaults require restarting HaAgent. Editing a source default is not
  a user configuration change.
- After any reload, read the active connection/model and run a small
  non-destructive check before reporting that the new setting is effective.

## Configuration change procedure

1. Identify whether the target is user configuration, workspace/session state,
   runtime settings, or a skill.
2. Locate the owning service/schema and read the current value.
3. Resolve the exact connection, model, or setting. Ask the user when the
   target is ambiguous.
4. Obtain the required permission for paths outside the workspace.
5. Apply the smallest requested change through the owning service or an
   approved structured file tool.
6. Reload the affected session/model through the application flow when needed.
7. Run an explicit verification command when the task or configuration change
   requires one. A read-back is optional confirmation, not a completion gate.

Use native structured tool calls. Do not print provider-specific tool markup as
ordinary assistant text.
