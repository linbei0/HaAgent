# HaAgent 未过时规则汇总

本文汇总 `docs` 目录中当前仍有效的产品、架构、运行时、上下文、记忆、TUI 和测试规则。整理时已排除专项计划/规格子目录及普通文档中的相关交叉引用；已标记为历史、迁移、旧入口或被新产品决策取代的内容不纳入本文。

## 1. 产品定位与普通入口

- HaAgent 的产品定位是本地个人 AI 助手：用户配置一次模型后，在任意目录运行 `haagent`，围绕当前目录完成文件阅读、资料整理、文档修改、项目分析、命令执行和多轮任务延续。
- HaAgent 不是 Codex clone、不是 IDE、也不是纯代码仓库助手；代码开发只是支持的一类任务。
- 无子命令 `haagent` 是唯一普通交互入口，默认打开 Textual TUI。
- 模型配置、会话恢复、自然语言任务、联网开关、工具审批、记忆候选和失败状态都应通过 TUI 管理。
- `haagent setup`、`haagent chat`、`haagent sessions`、`haagent memory`、`haagent tui` 只保留迁移提示，不作为真实普通交互入口。
- `task.yaml`、`run`、`inspect`、`eval`、`export-eval`、`dogfood`、`check`、`smoke` 属于高级、开发、复现、验证或 CI 能力；保持可用，但不要写成普通用户价值主线。
- 默认 workspace root 是当前目录；允许通过 `--workspace-root` 显式指定。
- `haagent --continue` 和 `haagent --resume <session>` 作为启动参数进入 TUI 后恢复会话。

## 2. 核心架构边界

- CLI 只负责解析启动参数、打开 TUI，并保留非交互开发/CI 命令的短输出。
- TUI 是普通交互前端，只负责交互和展示，不实现 Agent loop，不解析 CLI 文本输出，不绕过 runtime。
- `AssistantService` 是 CLI 与 TUI 共享的应用服务层；它读取 profile、检查非敏感凭据状态、管理 session，并转发 `RuntimeUiEvent` 流。
- `AgentSession` 负责多轮会话、bounded summary、working state、session package 和前端无关事件流。
- `RunOrchestrator` 负责 task contract、模型调用、工具执行、episode trace 和 verification。
- 所有模型调用必须经过 `ModelGateway`。
- 所有工具调用必须经过 `ToolRouter`。
- 文件和命令工具必须受 workspace root 限制。
- 每条用户 prompt 仍写独立 episode；session package 只保存索引、摘要和有界工作状态，不复制 episode 证据。

## 3. Profile、凭据与 Secret

- Profile 是模型连接配置，支持云端、本地 Ollama、LM Studio 及显式混合 fallback；OpenAI-compatible endpoint 使用 `openai` / `openai-chat` gateway，模型目录明确识别的 Anthropic 与 Google provider 使用对应原生 gateway。
- 本地发现只探测 `127.0.0.1:11434` 和 `127.0.0.1:1234`，不扫描局域网；发现失败必须区分不可达、未授权和无效响应。
- `providers.json` 只接受 version 4，不维护旧版本迁移路径。本地连接可以使用 `credential_source=none`，能力快照不持久化。
- `settings.json` 可保存一个 `fallback_model` 和 `cloud_fallback_consent`。本地到云端 fallback 必须有明确 consent，本地到本地不需要；fallback 不得在已有输出后重放。
- 默认 profile 存放在用户级 `~/.haagent/providers.json`；active profile 存放在 `~/.haagent/settings.json`。
- Workspace 和 session 是目录相关运行状态，默认写入当前目录的 `.runs/sessions`。
- 真实 API key 解析优先级是：当前环境变量、系统凭据库、显式 opt-in 的明文用户文件。
- TUI 模型配置默认使用系统凭据库；环境变量适合 CI 或临时覆盖；明文用户文件必须显式选择并标记为 insecure。
- 真实 API key 不写入项目配置、episode、transcript、日志、session summary、UI snapshot 或 tool-calls。
- TUI 可以通过 masked 输入临时接收真实 API key，并直接写入系统凭据库；除该受控写入流程外，UI 只能展示环境变量名、凭据来源和 key 是否可用等非敏感状态，不能回显、复制或写入明文配置。

## 4. Task Contract 与 Policy

- 自然语言入口不要求用户写 `task.yaml`，但 runtime 仍应生成结构化临时 `TaskSpec`，并写入 episode 供 inspect。
- 默认 `goal` 来自用户自然语言请求，默认 `workspace_root` 来自当前目录或 `--workspace-root`。
- `policy` 字段只影响 Policy Engine 对工具调用的决策，不改变工具自身执行方式，也不自动引入交互式审批。
- `policy.approval_allowed_tools` 表示高风险工具可以申请审批，不代表已经获准执行。
- `policy.approved_tools` 表示高风险工具在本次任务中已被显式批准执行。
- `policy` 缺失时两个列表都默认为空；两个字段都必须是 `list[str]`。
- 列表中的工具名必须存在于 Tool Registry；`approved_tools` 中的工具必须同时出现在 `approval_allowed_tools` 中。
- 高风险工具缺少允许或批准时必须被 policy 拒绝，handler 不执行，并记录 `policy_denied` 与 `approval.status=missing`。
- 低风险和中风险工具不需要审批，仍按原规则执行，并记录 `approval.status=not_required`。

## 5. 上下文与模型输入

- 模型输入默认保持薄，只放本轮必需的高信号内容。
- 完整历史、完整 audit、完整 episode、完整 transcript、完整 tool trace、完整工具输出、完整候选记忆池、长文件和大表格应留在磁盘、工具、执行环境或检索索引里。
- 只有被明确选中的摘要、事实、文件片段、工具结果摘录或结构化观察才能进入 prompt。
- 每轮调用模型前应有明确的 context assembly / context selection 阶段，输出结构化 `model_input`，不要在调用点到处拼字符串。
- Prompt 变厚必须有 source、reason、预算和 diagnostics。
- `diagnostics`、selected/skipped 决策和预算报告默认只写入 episode / manifest / trace，不进入模型输入。
- 上下文按需加载必须由结构化信号触发，不靠用户话术表或“复杂度判断”猜测。
- 项目规则由 workspace 和入口要求触发；session summary 由历史或恢复状态触发；working state 由持续任务状态触发；长期记忆由检索命中、scope、可信来源和预算触发；工具说明由当前允许且相关的工具能力触发；文件内容由显式引用或检索命中触发；工具结果由最近且必要的压缩观察触发。
- 工具注册可以完整，但模型可见工具集应是当前任务需要的最小集合。
- 大文件、大表格和搜索结果不应原样进入 prompt；`shell` / `code_run` 可作为数据处理隔离层，模型只接收统计、样例、错误和必要摘录。
- Context selection 的当前方向是本地、同步、确定性、可测试的选择层；不要引入 embedding、向量数据库、后台索引服务、复杂插件生态或普通用户可配置的上下文策略。

## 6. Session、Episode 与事件流

- `AgentSession` 应维护 bounded session summary 和 bounded working state；多轮任务不得线性撑大 `model_input`。
- 会话恢复只读取 `turns.jsonl` 摘要、turn\_count、workspace\_root 和 working\_state，不读取完整 episode transcript、tool-calls 或 verification 输出。
- `working_state.json` 保存当前目标、关键发现、已完成动作、下一步和最近更新 turn，字段必须有固定条目数或字符限制。
- `RuntimeUiEvent` 是前端无关的强类型事件契约，字段只放展示和状态判断需要的摘要。
- `RuntimeUiEvent` 不放完整工具输出、完整文件内容、完整用户答案、完整 episode trace 或 secret。
- 稳定事件类型包括 session/turn 开始结束、工具开始/完成/失败、assistant 消息、审批请求/批准/拒绝、用户补充输入请求/接收、failure 和 session finished。
- 模型路由还记录 `model_protocol_fallback` 与 `model_fallback`，包含脱敏连接、模型、协议、原因和能力缺失；顶部状态和活动流应展示实际使用模型。

## 7. 记忆系统

- 长期记忆写入必须满足证据边界：允许用户直接声明、成功工具结果、明确文件内容；不允许助手回答、模型推理、猜测、未验证计划、memory recall 或 unknown 来源作为用户事实证据。
- 正式长期记忆必须先进入候选队列，再由确定性服务确认和落库；不得由模型工具直接写正式记忆。
- 候选到正式记忆必须经过 canonical fingerprint、去重、冲突检查、rejected tombstone 抑制、scope/category 校验，以及用户确认或明确策略授权。
- SOP 类候选必须有成功工具结果、明确文件内容或成功验证结果作为证据，不能只凭助手最终回答或用户泛泛要求生成。
- 候选和正式记忆分离；被拒绝、过期或替代的记忆要参与后续抑制，避免重复候选刷屏。
- 用户偏好、工作区事实、会话进度、工具观察和操作流程不应混在一个文件里；User / Workspace / Session 记忆必须物理或逻辑分开。
- 记忆读取必须满足 scope 匹配、来源可信、命中可解释、达到最低相关阈值、有 token 预算且不与更高优先级事实冲突。
- confirmed memory 优先；candidate 默认不进 prompt。
- 每次记忆注入或跳过都应记录 query、score、命中字段、source、预算和 skip reason。
- 中文单字检索误命中是已知风险，但不要用脆弱停用字表抢修；后续应从结构化命中原因、短语级匹配、阈值和 rerank 入手。
- 不用 prompt 规则修 runtime、工具、记忆或上下文状态 bug；证据边界、去重、候选状态、检索阈值应由代码和测试保证。

## 8. 工具执行与运行边界

- 真实任务工具包包括 `file_read`、`file_write`、`apply_patch`、`shell` 和 `code_run` 等 workspace-bound 原子工具。
- `file_read` 应支持范围读取、关键词定位和路径建议，服务普通 Agent 使用。
- `apply_patch` 继续保持 fail-fast；失败应帮助模型从结构化错误中恢复，而不是吞掉错误。
- `shell` / `code_run` 的 cwd 必须留在 workspace root 内。
- `shell` / `code_run` timeout 默认 60 秒，上限 120 秒。
- `code_run` 用于降低多行脚本和 shell 转义成本；临时脚本保留在 workspace 内 `.haagent-tmp/` 以便失败复盘。
- 工具输出向工具结果和 context 暴露摘要、excerpt、timeout、truncated 等字段，并对 secret-like 输出做脱敏。
- 明显泄密、workspace 绕过和高风险工具参数必须在 runtime 层显式失败或拒绝。
- 高风险或信息不足场景应通过审批或用户补充输入机制处理，不要让模型硬猜。

## 9. TUI 规则

- TUI 的目标是个人助手会话工作台，不是 IDE、代码编辑器、文件树主界面或复杂多标签工作台。
- TUI 应通过 `AssistantService` 驱动会话，复用 `AgentSession`、`ModelGateway`、`ToolRouter`、workspace root 和 episode trace。
- TUI 信息架构优先级是：当前上下文、对话流、可恢复状态、配置健康度、低频操作。
- 当前布局以顶部状态栏、主对话 timeline、输入区和上下文 footer 为核心；低频能力通过 overlay/modal 打开，80x24 应可完成输入、阅读、审批、查看失败和退出。
- 顶部状态栏只展示工作区、当前模型、联网开关和当前工作状态。profile、provider、API key、权限、sandbox、session、turn 和原始 state 等诊断字段不得进入常驻状态栏；异常在对应配置、审批、权限或失败界面展示。
- 状态栏按终端 cell 宽度截断：80–119 列优先保留当前目录名，模型可以截断，联网和工作状态必须完整；只给联网与工作状态片段使用语义色，不整行随状态变色。
- 主对话区采用非对称安静结构：用户消息使用低对比表面和细左边线，不显示角色标签；assistant 回答直接落在主背景上，不使用卡片、边框或标题；系统、命令、操作和提示使用紧凑行内通知，失败保留明确符号与错误语义。
- 工具过程运行时显示中文动作摘要，过程标题按秒只刷新当前块的步骤数与耗时，不触发 timeline 全量重绘；最终回答完成后连同本轮工具失败、任务受阻诊断一起折叠为“已完成 N 步 · 本轮耗时 ›”并冻结耗时。没有最终回答的失败、待审批和待补充输入保持可见。普通界面使用中文工具名，详情同时显示中文名与原始标识；完整 transcript、tool output、stdout、stderr、patch 和 episode trace 仍按需打开。
- 输入区使用多行 `TextArea`：`Enter` 提交，`Ctrl+Enter` 换行，空输入不提交，`Esc` 关闭当前 modal/overlay 或返回上层交互。焦点只使用一格细左边线、轻微背景和光标变化，不显示高亮粗边框。
- 聊天 footer 固定为 `/ 命令 · Ctrl+F 搜索 · ? 帮助 · Ctrl+Q 退出`，不显示 `Enter 发送`；运行时用 `Ctrl+X 取消任务` 替换 `/ 命令`。其他上下文 footer 最多四组操作，完整键位进入 `?` 帮助。
- 计划任务未读数不进入顶部状态栏；数量增加时只发一次非持久通知，完整数量留在计划任务界面。
- 运行时请求用户补充信息时，输入区进入回答状态，提交后继续同一个 turn，不能变成新 prompt。
- 工具审批 modal 必须展示工具名、影响范围和关键参数摘要；文件修改和命令执行等高影响操作默认焦点应放在 Deny。
- 审批 modal 是 focus trap；审批结果必须回到同一个 turn。
- 帮助应以 modal、overlay 或上下文化帮助呈现，不应污染对话流。
- 记忆候选审查必须支持多候选导航和确认/拒绝目标项。
- 所有功能必须键盘可达，鼠标只作为增强；颜色不能作为唯一语义，`NO_COLOR` 下仍应可读。
- 普通用户路径统一使用简体中文；HaAgent、OpenAI、DeepSeek、模型 ID、环境变量、Slash 命令和高级诊断原始值保持原名。
- Failure 展示必须包含 failed\_stage、failure\_category、reason 和 episode\_path；不要静默 fallback，也不要过度推断。

## 10. 测试与质量门禁

- 所有行为变更必须有 pytest 覆盖。
- Bug 修复和新行为优先写失败测试，再实现最小代码通过。
- TDD 内循环优先运行最小相关测试，完成前至少运行与改动直接相关的测试。
- 跨多个核心模块、改动共享 runtime 合同、触及 `ToolRouter`、`ModelGateway`、context、episode、CLI 入口、workspace 边界或 secret 处理，或准备提交、合并、发布、交付时，运行完整 `uv run pytest -q`。
- 改动 harness、eval、smoke、CLI 质量门禁或 runtime 任务执行时，交付前运行 `uv run haagent check`。
- 默认快测只保留高信号、高风险、低成本用例。
- 测试价值判定：保留 workspace/path 边界、secret redaction、approval policy、ToolRouter、ModelGateway、episode/transcript schema、CLI/TUI 主路径和历史 bug 回归。
- 同一行为的多文案、多状态标签、多枚举错误矩阵应合并为代表场景或结构性断言。
- 只锁定中文措辞、视觉层级、内部实现细节且上层已有同等行为保护的测试，应删除或降级。
- 真实模型、长 dogfood、完整 TUI 键盘漫游、慢 smoke、inspect/eval/export 高级 harness 回归应迁到显式入口。
- `tests/tui/`、`tests/e2e/`、`tests/extended/` 默认不进入快测；需要时显式运行对应路径和 flags。

## 11. 变更治理与非目标

- 代码和文档术语都应服务“在目标目录直接运行 `haagent` 并进入 TUI”的体验。
- 优先做小而明确的改动，避免把个人助手体验改造成 IDE、多 Agent 系统或平台化产品。
- 不为了旧实验 artifact 增加复杂兼容逻辑。
- 普通用户文档优先说明无子命令 `haagent`、TUI 内用 `/connect` 配置连接并用 `/model` 切换模型、当前目录 workspace、多轮会话和 `/sessions` / `--continue` / `--resume`。
- 不要把 harness、eval、dogfood、inspect 暴露成普通用户主路径。
- 已有轻量 worker 与显式计划任务应保持隐藏、受控和可恢复；不把它们扩展成复杂多 Agent 编排平台或通用长期后台任务平台。浏览器自动化、GUI/mobile automation、自动 PR、Dashboard、完整 IDE 或大规模记忆系统仍不在当前范围，除非后续有明确产品决策。
- 不做 Web UI、Electron / 桌面 App、复杂插件系统、复杂主题市场或自动安装依赖作为 TUI 首版能力。
- 不靠自然语言匹配实现 slash commands、安全边界、上下文选择或 runtime 决策；命令、工具、session、workspace 都应走结构化 service 方法和明确状态字段。
- 不把完整 stdout、patch、episode trace 或工具详情默认塞进主对话；默认展示摘要，详情按需打开。

