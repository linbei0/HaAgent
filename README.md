# HaAgent

HaAgent 是一个本地个人 AI 助手。你配置一次模型后，可以进入任意目录运行 `haagent`，让它围绕当前目录读取文件、整理资料、修改文档、分析项目、运行命令，并延续多轮任务。

HaAgent 不是 Codex clone，不是 IDE，也不是只面向代码仓库的工具。代码开发只是它能处理的一类任务；同样重要的任务包括总结目录里的文档、整理本地文件、读取 CSV 并解释内容、把草稿改成正式说明、检查项目结构并建议下一步、运行脚本并解释结果。

## 快速开始

```powershell
cd E:\some-folder
uv run haagent
```

无子命令时，HaAgent 会打开 Textual TUI。首次使用或需要切换模型时，在 TUI 中输入 `/model` 打开模型中心；可以从模型目录选择，也可以手动配置 OpenAI-compatible / OpenAI Chat-compatible profile。

模型配置会写入用户级文件：

- `~/.haagent/providers.json`: profile 列表，保存 `name`、`provider`、`base_url`、`model`、`api_key_env`、`credential_source`。
- `~/.haagent/settings.json`: 当前默认 profile。

Profile 是“模型连接配置”。`provider` 支持 OpenAI Responses-compatible 的 `openai`，以及 OpenAI Chat Completions-compatible 的 `openai-chat`；`base_url` 填对应兼容 endpoint 的基础地址。

真实 API key 默认保存到系统凭据库（Windows Credential Manager、macOS Keychain、Linux Secret Service/keyring），跨终端可用。`api_key_env` 指向的环境变量始终优先，可用于 CI 或临时覆盖；明文用户文件只在 TUI 中显式选择 `insecure_file` 并确认后启用。真实 API key 不写入 HaAgent profile、settings、项目目录、episode、transcript 或会话摘要。

## 日常使用

```powershell
uv run haagent
```

TUI 是唯一普通交互入口，workspace root 是当前目录。文件和命令工具都受当前 workspace root 限制。

常用入口参数：

```powershell
uv run haagent --workspace-root E:\some-folder
uv run haagent --continue
uv run haagent --resume <session-id>
uv run haagent --web
```

TUI 内常用命令包括 `/model`、`/sessions`、`/memory`、`/web`、`/new`、`/resume`、`/cancel` 和 `/help`。

## 联网搜索

普通聊天默认不会联网。需要联网搜索或读取公网网页时，显式加 `--web`：

```powershell
$env:TAVILY_API_KEY = "tvly-..."
uv run haagent --web
```

TUI 内也可以随时显式切换联网能力：

```text
/web
```

第一版联网能力是 HaAgent 原生只读工具：

- `web_search`: 默认使用 Tavily，返回标题、URL 和摘要；可用 `HAAGENT_WEB_SEARCH_PROVIDER=brave` 切换 Brave。
- `web_fetch`: 读取单个公网 HTTP(S) URL，返回清洗后的紧凑文本。

可选环境变量：

- `TAVILY_API_KEY`: Tavily 搜索 API key，默认搜索后端需要。
- `BRAVE_SEARCH_API_KEY`: Brave Search API key，`HAAGENT_WEB_SEARCH_PROVIDER=brave` 时需要。
- `HAAGENT_WEB_SEARCH_PROVIDER`: `tavily` 或 `brave`，未设置时为 `tavily`。
- `HAAGENT_WEB_PROXY`: 联网工具使用的 HTTP(S) 代理；不允许在代理 URL 中嵌入用户名或密码。

联网工具只读、可审计，并且所有调用仍经过 `ToolRouter` 写入 episode trace。`web_fetch` 会拒绝 localhost、私网、metadata、单标签 hostname 和带凭据的 URL，并把网页内容标记为外部不可信数据。

第一版不包含 OpenAI Hosted Web Search、DuckDuckGo HTML 解析或真实浏览器自动化；这些能力后续只会作为显式高级方案考虑。

## 高级入口

`task.yaml`、`haagent run`、`inspect`、`export-eval`、`eval`、`dogfood` 和 episode package 是开发、复现、验证和评估能力。它们保留可用，但不是普通用户入门路径。

旧的 `haagent setup`、`haagent chat`、`haagent sessions`、`haagent memory` 和 `haagent tui` 交互入口已迁移到无子命令 `haagent` 的 TUI。
