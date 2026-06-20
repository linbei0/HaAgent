# Real LLM tool-call smoke

This manual smoke task verifies that `agentfoundry run --provider openai` can send tool schemas to a real model, receive a provider tool call, execute `fake_tool`, and preserve the episode trace for inspection.

It is intentionally not part of pytest because it requires a real OpenAI API key and network access.

## Minimal commands

Set `OPENAI_API_KEY` in your shell first.

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Run the smoke task with the OpenAI provider:

```powershell
uv run agentfoundry run examples/tasks/openai_tool_call_smoke.yaml --provider openai --model gpt-4.1-mini
```

Inspect the generated episode path printed by the run command:

```powershell
uv run agentfoundry inspect <episode_path>
```

The expected trace should show `provider=openai`, a successful `fake_tool` call, and a tool observation before the final model response.
