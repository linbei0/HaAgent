# Real Model Dogfood v1

Real Model Dogfood 用真实模型在临时 fixture workspace 中跑小型端到端任务，用来检查 HaAgent 是否能独立使用 `context_find`、`file_read`、`apply_patch_set`、`shell`、loop guidance 和 human approval。

Dogfood 是开发/验证能力，不是普通用户入口。普通用户路径是先运行 `uv run haagent setup` 配置模型连接，然后进入任意目录运行 `uv run haagent`。

## 手动入口

CLI：

```powershell
uv run haagent dogfood --profile local-openai
```

或直接用环境变量：

```powershell
$env:OPENAI_API_KEY="..."
uv run haagent dogfood --provider openai --model gpt-4.1-mini
```

pytest：

```powershell
uv run pytest tests/test_real_model_dogfood.py -q --real-llm
```

## 跳过规则

- 默认 `uv run pytest -q` 不调用真实模型。
- `tests/test_real_model_dogfood.py` 未传 `--real-llm` 时显式 `skip`。
- 未配置 `OPENAI_API_KEY` 或可用 `HAAGENT_DOGFOOD_PROFILE` 时显式 `skip`，不会回退到 fake model。
- `uv run haagent dogfood` 未提供 `--profile` 或 `--provider` 时输出 `status=skipped`。

## 报告内容

Dogfood report 会列出每个任务的完成状态、使用工具、失败原因、episode 路径和最需要改进的点。每个任务都写入标准 episode package，包括 `tool-calls.jsonl`、`transcript.jsonl` 和 `contexts/`。

## 覆盖任务

1. 用户不提供路径，只描述问候功能变更，模型需要定位上下文、读取文件并用 `apply_patch_set` 修改。
2. 修改代码和测试后，模型需要通过 `shell` 运行 `pytest`。
3. 先用重复片段触发 patch 失败，再根据 loop guidance 读取文件并扩大 `old_text` 上下文修复。
