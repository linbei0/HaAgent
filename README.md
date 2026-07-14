<div align="center">

# HaAgent

*在任意目录启动的本地个人 AI 助手*

![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776ab?style=flat-square&logo=python&logoColor=white)
![Textual](https://img.shields.io/badge/TUI-Textual-5b5bd6?style=flat-square)
![uv](https://img.shields.io/badge/uv-managed-2b6cb0?style=flat-square)
![pytest](https://img.shields.io/badge/tests-pytest-0a7f64?style=flat-square)

[功能亮点](#功能亮点) • [快速开始](#快速开始) • [TUI 常用操作](#tui-常用操作) • [配置模型](#配置模型) • [开发与验证](#开发与验证)

</div>

HaAgent 是一个本地优先的个人 AI 助手。用户配置一次模型后，可以在任意目录运行 `haagent`，通过 Textual TUI 围绕当前目录完成文件阅读、资料整理、文档修改、项目分析、命令执行和多轮任务延续。

> [!IMPORTANT]
> 普通交互入口是无子命令的 `haagent`。`task.yaml`、`run`、`inspect`、`eval`、`dogfood` 等能力仍然存在，但它们面向开发、复现、验证和 CI，不是普通用户的主路径。

## 功能亮点

- **当前目录即工作区**：默认把启动目录作为 workspace root，也可以用 `--workspace-root` 显式指定。
- **Textual TUI 会话工作台**：在一个终端界面里管理对话、状态、工具审批、失败信息和上下文帮助。
- **多轮会话恢复**：支持继续最新 session，或按 session id / package 路径恢复历史工作。
- **本地/混合模型中心**：管理云端 OpenAI-compatible 连接，发现 Ollama / LM Studio，并可配置一个显式备用模型。
- **工具与权限边界**：文件、命令、联网和 MCP 工具都通过 runtime 统一路由，并受 workspace 与 policy 约束。
- **记忆候选审查**：长期记忆先进入候选队列，再由用户或确定性策略确认。
- **可观测 runtime**：每轮任务写 episode package，支持 inspect、eval、export 和 smoke/dogfood 验证工作流。

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- PowerShell 7+（Windows 推荐，项目命令示例默认使用 `pwsh` 语义）

### 安装依赖

```powershell
git clone <your-repo-url>
cd HaAgent
uv sync
```

### 启动助手

在你想让 HaAgent 处理的目录中运行：

```powershell
uv run haagent
```

指定另一个工作目录：

```powershell
uv run haagent --workspace-root E:\some-folder
```

恢复会话：

```powershell
uv run haagent --continue
uv run haagent --resume <session>
```

启用只读联网工具：

```powershell
uv run haagent --web
```

## TUI 常用操作

HaAgent 的日常操作都在 TUI 内完成。常用 slash commands：

| 命令 | 用途 |
| --- | --- |
| `/connect` | 配置供应商连接和凭据 |
| `/model` | 切换模型、扫描本机运行时或设置备用模型 |
| `/sessions` | 打开会话列表 |
| `/memory` | 审查记忆候选 |
| `/channels` | 配置微信等外部聊天渠道 |
| `/schedules` | 管理计划任务与运行收件箱 |
| `/web` | 切换联网工具 |
| `/new` | 新建 session |
| `/resume` | 继续当前 workspace 的最新 session |
| `/cancel` | 取消当前任务 |
| `/help` | 打开上下文帮助 |

> [!NOTE]
> `haagent setup`、`haagent chat`、`haagent sessions`、`haagent memory`、`haagent tui` 是迁移提示入口。新的普通流程是在 TUI 内完成模型配置、会话恢复、记忆审查和任务输入。

## 配置模型

HaAgent 使用用户级 profile 管理模型连接配置：

- profile 文件：`~/.haagent/providers.json`
- 当前激活 profile：`~/.haagent/settings.json`
- 支持 gateway：`openai`、`openai-chat`，以及模型目录明确识别的 Anthropic 与 Google 原生 gateway；运行时类型包括 `remote`、`ollama` 与 `lm_studio`
- API key 解析顺序：当前环境变量、系统凭据库、显式 opt-in 的明文用户文件
- Ollama 默认无需凭据；LM Studio 开启认证时使用 `LM_STUDIO_API_KEY` 或已配置凭据

推荐在 TUI 的 `/connect` 中新增供应商连接，在 `/model` 中切换模型。模型中心按 `l` 只扫描 `127.0.0.1:11434`（Ollama）和 `127.0.0.1:1234`（LM Studio），选择模型后才保存本地连接；`b` 设置本地备用，`c` 设置已明确同意的云端备用。远端连接的 API key 通过 masked 输入直接写入系统凭据库；TUI 不回显、复制或写入明文配置。

`providers.json` 当前为 version 3。一个无需凭据的 Ollama 连接如下；本地能力来自实时发现，不写入配置：

```json
{
  "version": 3,
  "connections": [
    {
      "id": "local-ollama",
      "name": "Ollama",
      "provider_id": "ollama",
      "provider_name": "Ollama",
      "gateway_provider": "openai",
      "base_url": "http://127.0.0.1:11434/v1",
      "api_key_env": "",
      "credential_source": "none",
      "runtime_kind": "ollama"
    }
  ],
  "custom_models": []
}
```

普通 route 会先协商模型能力和协议。Responses 仅在能力元数据不支持，或首个有效输出前收到 404/405/501 时降级到 Chat Completions；备用模型仅在能力明确不足，或内部重试耗尽后的 network、timeout、可重试 429/5xx 且尚无有效 delta 时启用。认证、无效请求、取消及已有输出后的失败会直接暴露，不会静默重放。

## 开发与验证

常用开发命令：

```powershell
uv sync
uv run pytest -q
uv run pytest tests/tui -q
uv run haagent check
```

高级 runtime 与验证入口：

| 命令 | 定位 |
| --- | --- |
| `uv run haagent run <task.yaml>` | 运行结构化任务或复现实验任务 |
| `uv run haagent inspect <episode>` | 查看 episode package 摘要与诊断 |
| `uv run haagent eval <path>` | 运行评测用例 |
| `uv run haagent export-eval <episode>` | 从 episode 导出评测样本 |
| `uv run haagent smoke` | 运行最小 smoke 套件 |
| `uv run haagent dogfood` | 手动运行真实模型 dogfood 任务 |

项目使用 `src/haagent` 存放源码，`tests` 存放测试。行为变更应配套 pytest 覆盖；触及 runtime 合同、工具路由、模型网关、上下文、episode、CLI 入口、workspace 边界或 secret 处理时，交付前运行完整测试。
