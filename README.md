# HaAgent

HaAgent 是一个本地个人 AI 助手。你配置一次模型后，可以进入任意目录运行 `haagent`，让它围绕当前目录读取文件、整理资料、修改文档、分析项目、运行命令，并延续多轮任务。

HaAgent 不是 Codex clone，不是 IDE，也不是只面向代码仓库的工具。代码开发只是它能处理的一类任务；同样重要的任务包括总结目录里的文档、整理本地文件、读取 CSV 并解释内容、把草稿改成正式说明、检查项目结构并建议下一步、运行脚本并解释结果。

## 快速开始

```powershell
uv run haagent setup
cd E:\some-folder
uv run haagent
```

`haagent setup` 会写入用户级模型连接配置：

- `~/.haagent/providers.json`: profile 列表，保存 `name`、`provider`、`base_url`、`model`、`api_key_env`。
- `~/.haagent/settings.json`: 当前默认 profile。

Profile 是“模型连接配置”。`provider` 支持 OpenAI Responses-compatible 的 `openai`，以及 OpenAI Chat Completions-compatible 的 `openai-chat`；`base_url` 填对应兼容 endpoint 的基础地址。

真实 API key 不写入 HaAgent 配置、项目目录、episode、transcript 或会话摘要。请把真实密钥放在 `api_key_env` 指向的环境变量中。

## 日常使用

```powershell
uv run haagent
```

无子命令时，HaAgent 默认进入个人助手聊天模式，workspace root 是当前目录。文件和命令工具都受当前 workspace root 限制。

显式入口仍然可用：

```powershell
uv run haagent chat "总结这个目录里的文档"
uv run haagent sessions
uv run haagent --continue
uv run haagent chat --resume <session-id>
```

## 高级入口

`task.yaml`、`haagent run`、`inspect`、`export-eval`、`eval`、`dogfood` 和 episode package 是开发、复现、验证和评估能力。它们保留可用，但不是普通用户入门路径。
