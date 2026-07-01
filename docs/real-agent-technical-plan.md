# HaAgent 本地个人 AI 助手技术方案

更新时间：2026-06-25\
状态：路线草案 v0.6

更新说明：本文包含早期 CLI-first / chat-first 路线的历史记录。当前实现已切换为 TUI-first：无子命令 `haagent` 默认进入 Textual TUI，模型配置在 TUI 内 `/model` 完成，旧 `haagent setup/chat/sessions/memory/tui` 交互入口只提示迁移。后台 `task.yaml`、run、inspect、eval、dogfood、check 仍服务复现、验证和 CI。

## 1. 背景与修正

上一版路线已经确认自然语言会话应成为主入口，但理解仍偏开发者 CLI。新的目标需要更明确：HaAgent 是本地个人 AI 助手，用户配置一次模型后，进入任意目录运行 `haagent`，围绕当前目录完成文件阅读、资料整理、文档修改、项目分析、命令执行和多轮任务延续。当前代码中，`haagent` 默认进入 TUI，旧文本 chat/REPL 入口已经迁移。

HaAgent 不是 Codex clone，不是 IDE，也不是纯代码仓库助手。代码开发只是支持的任务类型之一。Harness 不放弃，但它应该在后台提供约束、记录、复盘和 eval 能力；真实用户面对的应是一个能直接对话、能操作本地目录、能解释进度和结果的个人助手。

## 2. 目标用户与体验目标

近期目标用户分两层：

- 第一层：会打开终端的普通开发者或高阶用户。他们不想写 `task.yaml`，希望进入任意目录后直接输入自然语言处理本地文件和任务。
- 第二层：更普通的用户。他们最终需要 TUI、桌面或聊天式界面，但核心能力必须先在可复用的会话内核里跑通。

因此，终端个人助手入口不是最终体验，只是最小前端。真正要沉淀的是一个可复用的 Agent 会话核心。

目标体验：

```powershell
cd E:\some-folder
uv run haagent
```

进入 TUI 后直接对话：

```text
> 帮我理解这个文件夹在做什么
> 检查当前项目有什么明显问题
> 把这份文档整理成面向用户的说明
> 修复失败的测试，必要时运行命令验证
```

也支持启动时恢复会话或显式启用联网：

```powershell
uv run haagent --workspace-root E:\some-project
uv run haagent --resume session-1234abcd
uv run haagent --continue
uv run haagent --web
```

## 3. 外部资料与 GenericAgent 启发

### 3.1 GenericAgent 本地笔记结论

从 `E:\md-note\GenericAgent` 的笔记看，GenericAgent 的强能力主要来自这些设计：

- 统一 Agent Loop：构建上下文 -> LLM 决策 -> 工具执行 -> 结构化反馈 -> 继续循环。
- 任务队列和流式输出：前端只需要 `put_task()`，再从 `display_queue` 读取 `next`/`done`。
- 最小原子工具集：少量无重叠工具降低工具 schema 成本和动作选择复杂度。
- 文件/命令工具直接：`file_read` 支持范围、关键词和路径建议；`file_patch` 要求唯一匹配；`code_run` 负责真实执行。
- 长任务工作记忆：每轮保留目标、关键发现、进度和下一步，而不是把全部历史塞给模型。
- 多前端桥接：CLI、TUI、桌面、聊天平台和协议桥都复用同一个 Agent 核心。
- 组合涌现：CLI、文件协议、Reflect Mode 等简单原语可以组合出更高级能力，但核心 loop 不膨胀。

这些结论来自本地笔记：

- `E:\md-note\GenericAgent\02-agent-main-loop.md`
- `E:\md-note\GenericAgent\05-local-tools-file-and-code.md`
- `E:\md-note\GenericAgent\07-memory-context-and-long-tasks.md`
- `E:\md-note\GenericAgent\08-frontends-and-bridges.md`
- `E:\md-note\GenericAgent\chapter09-最小原子工具集.md`
- `E:\md-note\GenericAgent\chapter10-分层记忆架构.md`
- `E:\md-note\GenericAgent\chapter11-上下文截断与压缩.md`
- `E:\md-note\GenericAgent\chapter13-涌现能力.md`

### 3.2 联网资料结论

OpenAI Agents SDK 文档强调，Agent 应由应用负责 orchestration、tool execution、approvals 和 state；tracing/eval 是调试和优化手段，不应替代先跑通一个工作流。参考：[OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)、[OpenAI Agents SDK Python](https://openai.github.io/openai-agents-python/)。

OpenAI 的 agent 定义文档还强调，conversation history 是模型看到的内容，run context 是应用代码看到的内容。这与 HaAgent 的 B2 一致：harness 内部状态不应无条件进入模型输入。参考：[Agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents)。

MCP 把 Resources、Prompts、Tools 分成不同能力，并引入 roots、elicitation 等客户端边界概念。这提示 HaAgent 不应只做工具执行，还要明确工作区边界、用户补充信息和工具能力的分离。参考：[MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25/index)。

Agent Client Protocol 把会话建模为 `session/new`、`session/prompt`、`session/update`、`session/cancel`。这说明面向普通用户的 Agent 需要会话、进度更新和中断能力，而不是只有一次性命令。参考：[ACP Overview](https://agentclientprotocol.com/protocol/v1/overview)、[ACP Prompt Turn](https://agentclientprotocol.com/protocol/prompt-turn)、[ACP Session Setup](https://agentclientprotocol.com/protocol/session-setup)。

Datawhale 的 GenericAgent 教程强调上下文信息密度、最小原子工具集、分层记忆和上下文压缩。这与本地笔记一致，也说明 HaAgent 不应靠堆更多工具或更长 prompt 对齐能力。参考：[hello-generic-agent](https://datawhalechina.github.io/hello-generic-agent/)。

## 4. 核心判断

HaAgent 当前偏差不是 harness 太多，而是缺少面向普通用户的个人助手启动体验和会话层。

目前的 `run` 更像“执行一个任务包”。个人助手体验更像“打开一个目录里的 Agent 会话，持续交代事情，Agent 在本地环境里推进”。因此，下一阶段核心不是先做 GUI，也不是补 eval，而是抽出并巩固 `AgentSession`：

```text
Frontend/CLI/TUI/Desktop
        |
        v
AgentSession
        |
        +-- ModelGateway
        +-- ContextBuilder
        +-- ToolRouter
        +-- EpisodeWriter
        +-- SessionMemory / WorkingState
```

`haagent` 是普通用户的第一个前端；旧 `haagent chat` 只保留迁移提示，不再作为真实交互入口。

## 5. 技术架构方案

### 5.1 AgentSession

新增会话核心，负责：

- 保存 session id、workspace root、provider、turn limit。
- 接收用户自然语言 prompt。
- 维护短期会话状态和有界 working_state。
- 把用户 prompt 转成内部 task contract。
- 调用现有 orchestrator 或等价 runtime loop。
- 产生统一事件流。
- 写 episode 或 session package。

不要把这些逻辑塞进 `cli.py`。CLI 只负责读输入、展示事件和退出。

### 5.2 TUI 入口

普通交互只支持一个入口：

```powershell
uv run haagent
```

TUI 支持最小 slash commands：

- `/model`：配置、测试和切换模型 profile。
- `/sessions`：搜索、恢复、继续最新或新建会话。
- `/memory`：审查记忆候选。
- `/web`：显式切换只读联网工具。
- `/new`、`/resume`、`/cancel`、`/help`：管理当前会话、运行和帮助。

### 5.3 内部任务契约

自然语言入口不要求用户写 `task.yaml`，但 runtime 仍需要一个结构化任务对象。

建议新增两层：

- `TaskSpec` 继续作为 runtime 消费的结构。
- `TaskAuthoring` 负责从自然语言入口生成 `TaskSpec`。

默认规则：

- `goal` = 用户自然语言请求。
- `workspace_root` = 当前目录或 `--workspace-root`。
- `allowed_tools` = 文件工具 + shell。
- `verification_commands` = 用户明确提供时才有。
- `policy` = 个人助手入口默认允许申请 shell 等高风险工具，但仍受 workspace root、timeout、审批和结构化错误约束。

生成出的临时 task contract 必须写入 episode，便于 inspect。

### 5.4 事件流

为普通用户体验和未来前端统一输出，runtime 应产生事件，而不是只在 CLI 里 print。

当前 `AgentSession` 已输出前端无关的 `ChatEvent`，结构固定为：

- `event_type`
- `session_id`
- `turn_index`
- `message`
- `payload`

当前稳定事件类型：

| 事件 | 用途 |
| --- | --- |
| `session_started` | 告诉前端 session/workspace/provider |
| `turn_started` | 第几轮开始 |
| `tool_started` | 工具名和关键参数 |
| `tool_finished` | status、exit_code、短摘要或结果摘要 |
| `tool_failed` | 工具失败类型和短错误信息 |
| `assistant_message` | 最终或阶段性回复 |
| `approval_requested` | 高风险工具审批请求 |
| `approval_granted` | 用户批准审批请求 |
| `approval_denied` | 用户拒绝审批请求 |
| `user_input_requested` | 模型请求用户补充信息 |
| `user_input_received` | 用户补充信息已收到，只暴露长度等摘要 |
| `turn_finished` | 单轮任务结束 |
| `failure` | 结构化失败 |
| `session_finished` | 单次任务结束 |

payload 只放前端展示和状态判断需要的摘要，不放完整工具输出、完整文件内容、完整用户答案、完整 episode trace 或 secret。TUI 和未来前端只消费 `ChatEvent` 并渲染，不实现 Agent loop、工具执行或 episode 读取逻辑。

当前事件契约已稳定，无子命令 `haagent` 已进入 TUI；本地 Web、ACP bridge 和其他前端暂不实现。

### 5.5 工具能力对齐

HaAgent 已有 `file_list`、`file_search`、`file_read`、`apply_patch`、`shell`，能覆盖第一阶段本地任务。

与 GenericAgent 对齐时，近期缺口不是浏览器，而是：

- `file_read` 更适合普通 Agent 使用：关键词定位、相似路径提示、行号/范围读取体验。
- `apply_patch` 继续保持 fail-fast，但需要让模型更容易从失败中恢复。
- `shell` 承担 GenericAgent `code_run` 的一部分能力，但应考虑后续新增 `code_run` 或“临时脚本执行”工具，降低多行 Python/PowerShell 转义成本。
- 需要 `ask_user` 或等价机制，让模型在不确定、高风险、信息不足时请求用户补充，而不是硬猜。
- 需要 `update_working_checkpoint` 或等价工作记忆机制，服务长任务。

第一阶段不做浏览器、GUI、移动端。

### 5.6 上下文和记忆

短期必须做：

- 最近用户消息和最终 assistant 摘要进入 bounded session history。
- 工具 observation 继续 compact。
- 每轮维护 `working_state`：当前目标、关键发现、已做动作、下一步。
- 不把完整 episode、完整工具日志、完整历史塞进 `model_input`。

当前已做：

- 轻量 session package 保存 session metadata 和 turn 摘要索引。
- `haagent --resume <session_id-or-path>` 可在启动 TUI 时恢复 bounded session summary、turn_count 和 workspace_root。
- 恢复只读取 `turns.jsonl` 的摘要，不读取完整 episode transcript、tool-calls 或 verification 输出。
- `working_state.json` 保存短期工作状态：当前目标、关键发现、已完成动作、下一步和最近更新 turn。
- working_state 由 runtime 根据 turn 结果和事件摘要确定性更新，不要求模型调用工具写 memory。
- working_state 进入下一轮 `model_input` 时有固定预算和 context source，不复制完整工具输出、完整 transcript 或完整 episode trace。
- `examples/evals/` 已提供最小内置回归评测集，可用 `haagent eval examples/evals` 本地检查文件读取、文件修改、命令执行、guardrail 和 session/working_state 关键链路是否退化。
- `haagent check` 已提供最小一键质量门禁，默认运行内置 eval suite，并可通过 `--pytest` 显式追加 `uv run pytest -q`。

中期再做：

- 按 `docs/superpowers/specs/2026-06-25-memory-system-v1-design.md` 实现 MemorySystem v1：Session Memory、Workspace Memory、User Memory 物理分开。
- Workspace Memory 分为 facts、sop、glossary、decisions；长期记忆必须先进入候选队列，用户确认后由确定性服务落库。
- 长任务文件协议。

记忆写入要遵守 No Execution, No Memory：没有执行验证过的经验，不进入长期记忆。

### 5.7 Episode 与 Session 的关系

每条用户 prompt 仍写一个独立 episode。Session package 只保存会话索引和摘要，不改变 episode schema。

当前结构：

```text
.runs/
  sessions/
    session-xxxx/
      session.json
      turns.jsonl
      working_state.json
  <episode-id>/
    episode.json
    transcript.jsonl
    tool-calls.jsonl
    ...
```

`session.json` 保存 `session_id`、`workspace_root`、`provider`、`created_at`、`updated_at`、`turn_count`。`turns.jsonl` 保存每轮请求、turn 摘要、status、episode_path 和 verification_status。`working_state.json` 保存有界短期工作状态，用于下一轮 context，不改变 episode schema。

后续如果需要更强的会话复盘，再考虑扩展 session package：

```text
.runs/sessions/session-xxxx/
  session.json
      turns.jsonl
      working_state.json
      events.jsonl
```

不要为 session resume 引入长期记忆、向量检索或跨会话知识库。

## 6. 分阶段路线

### Phase 1：个人助手启动体验

状态：TUI-first 版本已完成。

目标：用户配置一次模型后，可以进入任意目录直接启动并使用。

任务：

- 无子命令 `haagent` 默认进入 TUI。
- TUI 内 `/model` 写入用户级 profile 和 active profile。
- 旧 `haagent setup/chat/sessions/memory/tui` 只提示迁移。
- 支持 TUI 输入多条任务。
- 默认 workspace root 为当前目录。
- 默认模型连接来自 active profile；`--provider fake` 仅保留给测试和开发。
- 默认 turn limit 提高到 20。
- 内部生成临时 `TaskSpec`。
- 输出简洁工具进度和最终结果。
- 每条用户任务写 episode。
- TUI 内 `/sessions` 和启动参数 `--continue` / `--resume` 服务目录会话恢复。

验收：

- 能在任意文件夹运行 `uv run haagent`。
- 首次使用时通过 TUI `/model` 配置默认模型连接。
- 不需要 `task.yaml`。
- 能处理不改代码的任务。
- 能处理文档修改任务。
- 能处理代码修改 + 命令验证任务。

### Phase 2：会话体验

状态：初版已完成；轻量 session package、`--resume` 和最小 working_state 已完成。

目标：用户感觉是在和同一个 Agent 持续协作。

任务：

- 引入 `AgentSession`。
- TUI 会话内保存 session history summary。
- 支持 `/sessions`、`/new`、`/cancel`、`/help`。
- 输出 turn 级进度。
- 维护 bounded working state。
- 没有验证时明确说明验证状态。
- 退出后可通过轻量 session package 恢复必要会话事实。

验收：

- 用户能先问“这个项目是什么”，再说“那帮我改 README”，Agent 能利用前一轮摘要。
- 多轮任务不会线性撑大 `model_input`。

当前实现：

- `src/haagent/runtime/chat_session.py` 定义 `AgentSession`，管理 session id、workspace root、provider、runs root、turn limit 和 bounded session summary。
- `src/haagent/runtime/working_state.py` 定义最小 working_state，字段为 current_goal、key_findings、completed_actions、next_steps 和 last_updated_turn，均有固定条目数或字符限制。
- `haagent` 无子命令时进入 TUI，支持 `/sessions`、`/new`、`/cancel`、`/help` 等 slash commands。
- 每条用户输入仍生成临时 task contract，并通过 `RunOrchestrator` 写独立 episode。
- 下一轮 context 通过 `session_summary` source 接收固定预算摘要，不复制完整 transcript、tool-calls 或 episode 内容。
- 下一轮 context 也通过 `working_state` source 接收固定预算的当前目标、关键发现、已完成动作和下一步。
- `.runs/sessions/<session_id>/` 保存 `session.json`、`turns.jsonl` 和 `working_state.json`。
- `haagent --continue` 可恢复当前目录最近会话。
- `haagent --resume <session_id-or-path>` 可恢复 turn_count、workspace_root、bounded session summary 和 working_state。

### Phase 3：能力补齐

目标：提升真实任务成功率。

状态：Real Task Tool Pack v1、事件流、human interaction、最小确定性 guardrails、本地 eval runner、最小内置回归评测集、最小一键质量门禁和最小 working_state 已完成。

任务：

- 已强化 `file_read`：关键词定位、路径建议、行号范围。
- 已新增 `file_write`，用于 workspace 内新建、覆盖或追加文本文件。
- 已新增 `code_run`，用于把多行 Python 脚本写入 `.haagent-tmp` 后执行。
- 已新增 Execution Boundary v1：`shell`/`code_run` 的 cwd 必须留在 workspace root 内，timeout 默认 60 秒且上限 120 秒，输出只向工具结果和 context 暴露 excerpt、timeout、truncated 等摘要，并对 secret-like 输出做脱敏。
- 已稳定 ChatEvent 事件流，并支持 approval、user input 和 guardrail 触发摘要。
- 已新增最小 deterministic guardrails，覆盖 input、tool input 和 output 三个边界。
- 已新增本地 deterministic eval runner，形成 `episode -> export-eval -> haagent eval -> report` 的基础回归闭环。
- 已新增 `examples/evals/` 最小确定性回归 suite，覆盖 `file_read`、`file_write`、`code_run`、guardrail 拒绝和 session resume 后的 bounded working_state 注入。
- 已新增 `haagent check` 最小质量门禁，默认快速运行内置 eval suite；`--pytest` 才执行完整 pytest，避免默认自检变慢。
- 后续继续策略化、配置化 guardrails，但不把当前实现描述成完整安全沙箱。
- 当前 eval suite 是本地确定性回归，不依赖真实 provider；它不是 LLM-as-judge、不是生产监控，也不提供 dashboard 或 HTML 报告。
- 当前 `haagent check` 是本地开发快速自检入口，不是 CI，也不上传报告或连接外部平台。
- 当前 Execution Boundary v1 是进程级和 runtime 层边界，不是 Docker、容器、虚拟机或生产安全沙箱；`code_run` 保留 workspace 内 `.haagent-tmp/` 脚本路径用于失败复盘。
- 后续继续扩展真实任务和 chat 主路径 eval 覆盖，但不做 dashboard、LLM-as-judge 或生产监控平台。
- 暂不新增让模型直接写 memory 的工具；working_state 由 runtime 确定性维护。
- 改善工具失败后的 next-step guidance。

验收：

- 模型猜错路径时能自我恢复。
- 多行脚本不需要模型和 shell 转义搏斗。
- 高风险/不确定任务能主动问用户。
- 明显泄密、workspace 绕过和高风险工具参数会在 runtime 层显式失败或拒绝。
- 已导出的 eval case 可以被本地批量重跑，并报告状态、工具轨迹和最终输出的确定性匹配结果。
- 内置最小 suite 可以通过 `uv run haagent eval examples/evals` 运行，并输出 total/passed/failed/error 汇总和失败 case 原因。
- 本地开发可通过 `uv run haagent check` 一键看到 status、eval_total/eval_passed/eval_failed/eval_error；需要完整测试时显式运行 `uv run haagent check --pytest`。

### Phase 4：普通用户前端

目标：TUI 作为唯一普通终端交互入口继续打磨，其他前端后续再评估。

状态：事件契约、轻量 session package 和无子命令 `haagent` TUI 已稳定；本地 Web、ACP bridge 暂不实现。

任务：

- 基于已稳定的前端无关事件流继续收敛最小前端适配层。
- 打磨 TUI；本地 Web/Desktop bridge 后续再评估。
- 评估 ACP bridge，让外部客户端通过 session/prompt 驱动 HaAgent。
- 支持取消当前任务。

验收：

- CLI、TUI 或 bridge 都复用同一个 `AgentSession`。
- 前端可以看到工具进度、最终结果和失败原因。

### Phase 5：长期能力

目标：让 HaAgent 越用越顺手。

任务：

- 分层记忆。
- SOP 沉淀。
- 会话恢复。
- 文件协议长任务。
- Reflect/monitor 模式。

这些不进入近期 P0，避免再次走向平台化。

## 7. 非目标

近期不做：

- 浏览器自动化。
- GUI/mobile automation。
- 多 Agent。
- 自动 PR。
- Dashboard。
- 长期后台任务。
- 完整 IDE。
- 大规模记忆系统。

这些能力可以之后通过 `AgentSession`、事件流、文件协议或 bridge 组合出来，但不应阻塞直接对话能力。

## 8. 关键设计取舍

### 8.1 为什么不先做桌面 UI？

桌面 UI 会让问题看起来像前端问题。当前真正缺的是可复用 Agent 会话核心和低门槛启动体验。先做终端个人助手入口，是为了验证核心能力；之后 TUI/桌面只是前端适配。

### 8.2 为什么不继续强化 `run`？

`run` 适合可复现任务，但普通用户想要的是“我说一句，它开始做”。继续强化 `run` 会把产品体验锁在 task contract 上。

### 8.3 为什么 session package 只保存摘要？

当前 episode 已经能记录模型、工具、context、failure。Session package 的职责只是让会话可恢复、可索引，而不是复制 episode 证据。这样可以恢复必要上下文，同时避免把完整历史、工具日志或审计文件塞进下一轮 `model_input`。

### 8.4 为什么不马上做长期记忆？

长期记忆只有在真实任务反复执行后才知道该沉淀什么。过早做记忆，容易把未验证假设写进系统。先做工作记忆和 session summary。

## 9. 实施建议

第一条实现任务建议（已完成）：

> 历史实现任务：实现 CLI-first 个人助手入口。当前已被 TUI-first 入口取代：无子命令 `haagent` 默认进入 TUI，旧交互命令只提示迁移。

后续类似任务按当前分层验证策略执行：TDD 内循环运行最小相关 pytest，交付或触及共享 runtime 合同时再运行完整 `uv run pytest -q`；涉及 harness/eval/smoke 或质量门禁时运行 `uv run haagent check`。

第二条实现任务（已完成）：

> 历史实现任务：在文本 REPL 中维护 bounded session summary。当前文本 REPL 已移除，对应能力由 TUI slash commands 和 `AssistantService` 承担。

第三条实现任务（已完成）：

> 抽出 `AgentSession` 和事件流，让 CLI 不直接承载会话逻辑，为后续 TUI/桌面/ACP bridge 做准备。

说明：`AgentSession` 已抽出；`ChatEvent` 契约已稳定为前端无关事件流；无子命令 `haagent` 的 TUI 已接入同一服务层。后续可在不改变 Agent loop 的前提下继续打磨 TUI，或接本地 Web / ACP bridge。

## 10. 待确认问题

1. 是否需要首次启动提示工作区边界和高风险工具审批语义？
2. TUI 取消是否需要覆盖正在执行的外部命令，还是只支持轮次间退出？
3. TUI 是否继续打磨，还是先评估本地 Web / ACP bridge？
4. 中期长期记忆应如何与当前 bounded working_state 分层，避免把未验证信息写入长期状态？
