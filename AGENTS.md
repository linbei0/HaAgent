# AGENTS.md

## 当前事实源

本文件只做进入项目后的短入口地图。当前 `docs` 目录中唯一仍有效的深层规则文档是：

- `docs/active-rules-summary.md`

进行非平凡变更前，先阅读该文件中与任务相关的章节。不要再引用已经移除或过时的旧深层规则文档。

如果本文件与 `docs/active-rules-summary.md` 冲突，以 `docs/active-rules-summary.md` 为准，并在本次任务包含文档维护时同步修正冲突。

## 项目定位

HaAgent 是本地个人 AI 助手。用户配置一次模型后，在任意目录运行 `haagent`，通过 Textual TUI 围绕当前目录完成文件阅读、资料整理、文档修改、项目分析、命令执行和多轮任务延续。

HaAgent 不是 Codex clone、不是 IDE，也不是纯代码仓库助手。代码开发只是支持的一类任务；普通产品语言和默认 CLI 流程必须覆盖本地文件夹中的个人助手工作。

Harness 能力仍然重要，但应留在后台：runtime 约束工具、记录模型和工具 trace、写 episode package，并支持 inspect/eval，不把 `task.yaml`、episode、dogfood、eval export 暴露成普通用户主路径。

## 普通入口

普通交互入口是无子命令 `haagent`：

```powershell
cd E:\some-folder
uv run haagent
```

关键规则：

- 无子命令 `haagent` 默认打开 Textual TUI，是唯一普通交互入口。
- 模型配置、会话恢复、自然语言任务、联网开关、工具审批、记忆候选和失败状态都应通过 TUI 管理。
- `haagent setup`、`haagent chat`、`haagent sessions`、`haagent memory`、`haagent tui` 只保留迁移提示，不作为真实普通交互入口。
- `task.yaml`、`run`、`inspect`、`eval`、`export-eval`、`dogfood`、`check`、`smoke` 属于高级、开发、复现、验证或 CI 能力。
- 默认 workspace root 是当前目录；允许通过 `--workspace-root` 显式指定。
- `haagent --continue` 和 `haagent --resume <session>` 作为启动参数进入 TUI 后恢复会话。

## 核心架构边界

- CLI 只负责解析启动参数、打开 TUI，并保留非交互开发/CI 命令的短输出。
- TUI 是普通交互前端，只负责交互和展示，不实现 Agent loop，不解析 CLI 文本输出，不绕过 runtime。
- `AssistantService` 是 CLI 与 TUI 共享的应用服务层，负责读取 profile、检查非敏感凭据状态、管理 session，并转发事件流。
- `AgentSession` 负责多轮会话、bounded summary、working state、session package 和前端无关事件流。
- `RunOrchestrator` 负责 task contract、模型调用、工具执行、episode trace 和 verification。
- 所有模型调用必须经过 `ModelGateway`。
- 所有工具调用必须经过 `ToolRouter`。
- 文件和命令工具必须受 workspace root 限制。
- 每条用户 prompt 仍写独立 episode；session package 只保存索引、摘要和有界工作状态，不复制 episode 证据。

## 工具、权限与上下文

- 真实任务工具包包括 `file_read`、`file_write`、`apply_patch`、`shell` 和 `code_run` 等 workspace-bound 原子工具。
- 自然语言入口不要求用户写 `task.yaml`，但 runtime 仍应生成结构化临时 `TaskSpec`，并写入 episode 供 inspect。
- `policy` 只影响 Policy Engine 对工具调用的决策，不改变工具自身执行方式，也不自动引入交互式审批。
- 高风险工具缺少允许或批准时必须被 policy 拒绝，handler 不执行，并记录 `policy_denied` 与 `approval.status=missing`。
- 模型输入默认保持薄，只放本轮必需的高信号内容。
- 完整历史、完整 audit、完整 episode、完整 transcript、完整 tool trace、完整工具输出、完整候选记忆池、长文件和大表格应留在磁盘、工具、执行环境或检索索引里。
- 上下文按需加载必须由结构化信号触发，不靠用户话术表或“复杂度判断”猜测。
- 工具注册可以完整，但模型可见工具集应是当前任务需要的最小集合。
- 不用 prompt 规则修 runtime、工具、记忆或上下文状态 bug；优先用代码、状态机、schema、工具契约、验证或确定性上下文事实解决。

## Secret 与凭据

- Profile 是模型连接配置，支持 OpenAI Responses-compatible endpoint（`openai`）和 OpenAI Chat Completions-compatible endpoint（`openai-chat`）。
- 默认 profile 存放在用户级 `~/.haagent/providers.json`；active profile 存放在 `~/.haagent/settings.json`。
- 真实 API key 解析优先级是：当前环境变量、系统凭据库、显式 opt-in 的明文用户文件。
- TUI 模型配置默认使用系统凭据库；环境变量适合 CI 或临时覆盖；明文用户文件必须显式选择并标记为 insecure。
- 真实 API key 不写入项目配置、episode、transcript、日志、session summary、UI snapshot 或 tool-calls。
- UI 只能展示环境变量名、凭据来源和 key 是否可用等非敏感状态，不能输入、保存、复制或显示真实 API key。

## 开发工作流

- 使用 `uv` 管理依赖和虚拟环境。
- 代码放在 `src/haagent`。
- 测试放在 `tests`。
- 优先用 `apply_patch` 编辑文件，避免 PowerShell 编码问题。
- 优先做小而明确的改动，避免把个人助手体验改造成 IDE、多 Agent 系统或平台化产品。
- 不为了旧实验 artifact 增加复杂兼容逻辑。
- 不靠自然语言匹配实现 slash commands、安全边界、上下文选择或 runtime 决策；命令、工具、session、workspace 都应走结构化 service 方法和明确状态字段。
- 不把完整 stdout、patch、episode trace 或工具详情默认塞进主对话；默认展示摘要，详情按需打开。

## 常用命令

```powershell
uv sync
uv run haagent
uv run haagent --workspace-root E:\some-folder
uv run pytest tests/integration/tools/test_tool_router.py -q
uv run pytest -q
uv run pytest -q -n 0
uv run pytest tests/tui -q
uv run pytest tests/extended -q
uv run pytest tests/e2e -q --run-e2e
uv run haagent check
```

测试选择规则：

- 行为变更必须有 pytest 覆盖。
- Bug 修复和新行为优先写失败测试，再实现最小代码通过。
- TDD 内循环优先运行最小相关测试。
- 改动共享 runtime 合同、`ToolRouter`、`ModelGateway`、context、episode、CLI 入口、workspace 边界或 secret 处理时，运行完整 `uv run pytest -q`。
- 改动 harness、eval、smoke、CLI 质量门禁或 runtime 任务执行时，交付前运行 `uv run haagent check`。
- `tests/tui/`、`tests/e2e/`、`tests/extended/` 默认不进入快测；需要时显式运行对应路径和 flags。

## 代码风格

- 面向用户的回复使用简体中文。
- 项目内解释性注释使用简体中文。
- 每个 Python 文件必须以模块 docstring 开头，格式如下：

  ```python
  """
  path/to/file.py - 简短职责说明

  说明该文件在 HaAgent 中负责什么。
  """
  ```

- 以下场景必须写简洁注释，不能只靠代码自解释：
  - 失败边界、拒绝执行、显式抛错、取消静默 fallback、或保留 fallback 的地方。
  - secret、凭据、workspace/path 边界、权限审批、工具执行、模型调用等安全敏感逻辑。
  - Textual TUI 的焦点恢复、键盘事件拦截、modal/overlay 生命周期、worker/thread 与 `call_from_thread` 边界。
  - 性能相关逻辑，如批处理刷新、避免全量重绘、缓存/索引复用、长输出截断或摘要预算。
  - runtime event 到 UI 展示、context selection、memory 候选、episode/session 状态等跨层契约映射。
  - 为了产品规则保留或删除兼容路径时，必须说明原因和适用边界。
- 不注释显而易见的赋值或一行样板。
- 改动行为时保持注释同步。

## 详细规则

产品、架构、运行时、上下文、记忆、TUI、测试和非目标的完整当前规则见：

- `docs/active-rules-summary.md`
