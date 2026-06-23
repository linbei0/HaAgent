# Real LLM tool-call smoke

This manual smoke task verifies that the run-level runtime can send tool schemas to a real model, receive a provider tool call, execute `fake_tool`, and preserve the episode trace for inspection.

This is not the primary user experience. HaAgent is a local personal AI assistant: ordinary users should run `haagent setup`, enter a directory, then run `haagent`. `task.yaml` and `haagent run` are kept for reproducible smoke, batch, and eval-oriented workflows.

It is intentionally not part of pytest because it requires a real OpenAI API key and network access.

## Minimal commands

Set `OPENAI_API_KEY` in your shell first.

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Run the smoke task with the OpenAI provider:

```powershell
uv run haagent run examples/tasks/openai_tool_call_smoke.yaml --provider openai --model gpt-4.1-mini
```

For the normal assistant path, prefer configuring a profile and starting from a natural-language request:

```powershell
uv run haagent setup
cd E:\some-folder
uv run haagent
```

Inspect the generated episode path printed by the run command:

```powershell
uv run haagent inspect <episode_path>
```

The expected trace should show `provider=openai`, a successful `fake_tool` call, and a tool observation before the final model response.
