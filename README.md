<div align="center">

# HaAgent

*在任意目录启动的本地个人 AI 助手*

![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776ab?style=flat-square\&logo=python\&logoColor=white)
![Textual](https://img.shields.io/badge/TUI-Textual-5b5bd6?style=flat-square)
![uv](https://img.shields.io/badge/uv-managed-2b6cb0?style=flat-square)
![pytest](https://img.shields.io/badge/tests-pytest-0a7f64?style=flat-square)

[功能亮点](#功能亮点) • [安装与启动](#安装与启动) • [TUI 常用操作](#tui-常用操作) • [升级与卸载](#升级与卸载) • [开发与发布](#开发与发布)

</div>

HaAgent 是一个本地优先的个人 AI 助手。用户配置一次模型后，可以在任意目录运行 `haagent`，通过 Textual TUI 围绕当前目录完成文件阅读、资料整理、文档修改、项目分析、命令执行和多轮任务延续。

## 功能亮点

- **当前目录即工作区**：默认把启动目录作为 workspace root，也可以用 `--workspace-root` 显式指定。
- **Textual TUI 会话工作台**：在一个终端界面里管理对话、状态、工具审批、失败信息和上下文帮助。
- **多轮会话恢复**：支持继续最新 session，或按 session id / package 路径恢复历史工作。
- **本地/混合模型中心**：管理云端 OpenAI-compatible 连接，发现 Ollama / LM Studio，并可配置一个显式备用模型。
- **工具与权限边界**：文件、命令、联网和 MCP 工具都通过 runtime 统一路由，并受 workspace 与 policy 约束。
- **记忆候选审查**：长期记忆先进入候选队列，再由用户或确定性策略确认。
- **可观测 runtime**：每轮任务写 episode package，支持 inspect、eval、export 和 smoke/dogfood 验证工作流。

## 安装与启动

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- PowerShell 7+（Windows 推荐，项目命令示例默认使用 `pwsh` 语义）

### 普通用户安装

当前正式分发源是 GitHub 仓库。使用 `uv tool install` 安装后，`haagent` 会成为用户级命令，可以在任意目录直接运行：

```powershell
uv tool install "git+https://github.com/linbei0/HaAgent.git"
```

确认安装成功：

```powershell
haagent --help
```

进入希望助手处理的目录并启动：

```powershell
cd E:\some-folder
haagent
```

也可以从任意位置显式指定工作区：

```powershell
haagent --workspace-root E:\some-folder
```

恢复会话：

```powershell
haagent --continue
haagent --resume <session>
```

启用只读联网工具：

```powershell
haagent --web
```

### 微信二维码可选依赖

`/channels` 的微信扫码登录可以使用 `channels-weixin` extra 在终端渲染 ASCII 二维码。首次安装时启用：

```powershell
uv tool install "haagent[channels-weixin] @ git+https://github.com/linbei0/HaAgent.git"
```

如果已经安装基础版本，使用同一命令加 `--force` 重装：

```powershell
uv tool install --force "haagent[channels-weixin] @ git+https://github.com/linbei0/HaAgent.git"
```

不安装该 extra 时，微信登录仍可继续，但 TUI 只显示二维码 URL 文本。

### 从源码开发运行

需要修改代码或运行测试时再克隆仓库：

```powershell
git clone https://github.com/linbei0/HaAgent.git
cd HaAgent
uv sync
uv run haagent
```

希望本地源码改动立即反映到全局 `haagent` 命令时，可以在仓库根目录安装 editable tool：

```powershell
uv tool install --force --editable .
```

## TUI 常用操作

HaAgent 的日常操作都在 TUI 内完成。常用 slash commands：

| 命令           | 用途                         |
| ------------ | -------------------------- |
| `/connect`   | 配置供应商连接和凭据                 |
| `/model`     | 切换模型、扫描本机运行时或设置备用模型        |
| `/sessions`  | 打开会话列表                     |
| `/memory`    | 审查记忆候选                     |
| `/channels`  | 配置微信等外部聊天渠道                |
| `/schedules` | 管理计划任务与运行收件箱               |
| `/web`       | 切换联网工具                     |
| `/new`       | 新建 session                 |
| `/resume`    | 继续当前 workspace 的最新 session |
| `/cancel`    | 取消当前任务                     |
| `/help`      | 打开上下文帮助                    |

## 升级与卸载

升级通过 GitHub 安装的版本：

```powershell
uv tool upgrade haagent
```

如果使用 editable 源码安装，先在仓库中更新代码和锁定依赖：

```powershell
git pull --ff-only
uv sync --locked
```

卸载不会删除用户级 `~/.haagent/` 配置，也不会删除各工作区中的 `.runs/` 会话数据：

```powershell
uv tool uninstall haagent
```

如需彻底清理数据，请先自行备份，再分别处理用户配置目录和工作区 `.runs/`；HaAgent 不提供自动删除用户数据的命令。

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

## 开发与发布

### 开发与验证

常用开发命令：

```powershell
uv sync
uv run pytest -q
uv run pytest tests/tui -q -n 0
uv run pytest tests/extended -q -n 0
uv run haagent check
```

高级 runtime 与验证入口：

| 命令                                     | 定位                       |
| -------------------------------------- | ------------------------ |
| `uv run haagent run <task.yaml>`       | 运行结构化任务或复现实验任务           |
| `uv run haagent inspect <episode>`     | 查看 episode package 摘要与诊断 |
| `uv run haagent eval <path>`           | 运行评测用例                   |
| `uv run haagent export-eval <episode>` | 从 episode 导出评测样本         |
| `uv run haagent smoke`                 | 运行最小 smoke 套件            |
| `uv run haagent dogfood`               | 手动运行真实模型 dogfood 任务      |

项目使用 `src/haagent` 存放源码，`tests` 存放测试。行为变更应配套 pytest 覆盖；触及 runtime 合同、工具路由、模型网关、上下文、episode、CLI 入口、workspace 边界或 secret 处理时，交付前运行完整测试。

### 维护者发布流程

1. 在 `pyproject.toml` 更新版本号，并确认 GitHub Release 中准备了对应变更说明。
2. 从干净工作树同步锁定依赖并运行发布门禁：
   ```powershell
   uv sync --locked
   uv run pytest -q
   uv run pytest tests/tui -q -n 0
   uv run pytest tests/extended -q -n 0
   uv run haagent check --pytest
   ```
3. 构建 wheel 和源码包，并在本机安装 wheel 验证普通命令入口：
   ```powershell
   uv build
   $wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
   uv tool install --force $wheel
   haagent --help
   ```
4. 创建并推送与版本一致的标签，例如：
   ```powershell
   git tag v0.1.0
   git push origin v0.1.0
   ```
5. 在 GitHub 创建 Release 并上传 `dist/` 中的 wheel 与源码包。只有在项目完成 PyPI 账户、token 或 Trusted Publishing 配置后，才执行 `uv publish`；不要把发布凭据写入仓库、命令历史、日志或 episode。

发布失败时不要复用同一个版本号覆盖已有制品；修复后递增版本并重新执行完整门禁。
