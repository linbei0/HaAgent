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
- workspace root 是文件和命令工具的默认信任根；外部目录访问必须由工具执行器发现并在同一工具调用内获得用户授权。
- 每条用户 prompt 仍写独立 episode；session package 只保存索引、摘要和有界工作状态，不复制 episode 证据。
- 模型流中断恢复分层：首个 delta 前透明 retry；首个 delta 后在有界预算内 attempt reset 并重放同一 `ModelInvocation`；仅官方 OpenAI Responses + 显式 `background` 才可 retrieve 同一 Response。
- 已经向 UI 提交模型输出后，不得切换协议或备用模型；恢复只允许同一模型、同一协议、同一逻辑 turn。
- 只有 `network`、`timeout`、明确 retryable 的 408/409/429/5xx 与 provider 流内 transient error 可自动恢复；`auth`、`quota_exhausted`、validation、response parse、content filter、用户取消与未知非 retryable 不得自动恢复。

## 3. Profile、凭据与 Secret

- Profile 是模型连接配置，支持云端、本地 Ollama、LM Studio 及显式混合 fallback；OpenAI-compatible endpoint 使用 `openai` / `openai-chat` gateway，模型目录明确识别的 Anthropic 与 Google provider 使用对应原生 gateway。
- 本地发现只探测 `127.0.0.1:11434` 和 `127.0.0.1:1234`，不扫描局域网；发现失败必须区分不可达、未授权和无效响应。
- `providers.json` 只接受 version 4，不维护旧版本迁移路径。本地连接可以使用 `credential_source=none`，能力快照不持久化。
- `providers.json` 的每个模型可选配置 `max_context_tokens` 作为 HaAgent 本地有效上限；它与 provider/本地发现窗口取较小值，不作为 provider 请求参数发送。
- `settings.json` 可保存一个 `fallback_model` 和 `cloud_fallback_consent`。本地到云端 fallback 必须有明确 consent，本地到本地不需要；fallback 不得在已有输出后重放。
- 默认 profile 存放在用户级 `~/.haagent/providers.json`；active profile 存放在 `~/.haagent/settings.json`。
- Workspace 和 session 是目录相关运行状态，默认写入用户级 `~/.haagent/runs`：session 位于 `sessions/YYYY/MM/DD/<session-id>`，其每条 prompt 的 episode 位于 `episodes/YYYY/MM/DD/<session-id>/<episode-id>`；无 session 的高级入口写入 `episodes/YYYY/MM/DD/runs/<episode-id>`。`--runs-root` 可为当前命令显式覆盖该位置。
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
- 静态工具经 `ToolContribution` / `ToolCatalog` 同源登记 definition、handler binder、展示/observation 投影、guardrail 与 chat tags；新增静态工具只改一个 contribution。静态执行统一走 catalog 绑定的 handler，并经 `ToolExecutionContext` 注入逐次 `interaction_handler`；Router 不对 `file_write`/`apply_patch*`/`request_user_input` 按名旁路。动态 `mcp__*` 不进静态 binder，仍由 ToolRouter 安全 Seam 处理。
- 高风险工具缺少允许或批准时必须被 policy 拒绝，handler 不执行，并记录 `policy_denied` 与 `approval.status=missing`。
- 低风险和中风险工具不需要审批，仍按原规则执行，并记录 `approval.status=not_required`。
- `request_user_input` 只接受 `questions + reason` 结构化输入：每次 1–3 题、ID 唯一；选项题 2–4 项，`multiple` 仅用于选项题，`custom` 默认允许自定义。不得保留旧顶层 `question` 双契约，也不支持 Secret、自动默认答案或跨会话复用回答。
- 用户补充输入返回 `answered`、`dismissed` 或 `timed_out`。关闭和超时都是成功工具结果，由模型调整方案或解释阻塞；缺少交互 handler 才返回明确的 `user_input_unavailable` 工具错误。
- 聊天频道按问题顺序逐题发送并为每题生成独立 nonce；`/answer <nonce> 1` 与多选 `1,3` 映射为选项标签，非纯数字内容仅在允许自定义时接收，`/dismiss <nonce>` 返回 `dismissed`。非法编号或禁止的自定义答案必须保留 pending，超时单独返回 `timed_out`。
- `todo_update`、`submit_plan` 与 `task_wait` 是主 Agent 专用的静态工具：普通执行模式默认可见 `todo_update`，`submit_plan` 只在 Plan Mode 可见；Worker 不得读取或推进主任务 Todo，也不得替 leader 阻塞等待其他 Worker。
- `todo_update` 以完整列表原子替换 `TaskLedger`，最多 20 项、最多一个 `in_progress`；未完成项不得静默消失，blocker、evidence 和 checkpoint 不得隐式改变 Todo 状态。
- `submit_plan` 只提交完整 Plan revision 并等待用户确认。反馈必须产生新 revision；批准结束规划 turn，由 session 层幂等初始化 Todo 并自动进入执行，旧 plan id 或旧 revision 必须明确拒绝。

## 5. 上下文与模型输入

- 模型输入默认保持薄，只放本轮必需的高信号内容。
- 完整历史、完整 audit、完整 episode、完整 transcript、完整 tool trace、完整工具输出、完整候选记忆池、长文件和大表格应留在磁盘、工具、执行环境或检索索引里。
- 只有被明确选中的摘要、事实、文件片段、工具结果摘录或结构化观察才能进入 prompt。
- 每轮调用模型前应有明确的 context assembly / context selection 阶段，输出结构化 `model_input`，不要在调用点到处拼字符串。
- Prompt 变厚必须有 source、reason、预算和 diagnostics。
- `diagnostics`、selected/skipped 决策和预算报告默认只写入 episode / manifest / trace，不进入模型输入。
- 上下文按需加载必须由结构化信号触发，不靠用户话术表或“复杂度判断”猜测。
- 项目规则由 workspace 和入口要求触发；session summary 由历史或恢复状态触发；working state 由持续任务状态触发；长期记忆由检索命中、scope、可信来源和预算触发；工具说明由当前允许且相关的工具能力触发；文件内容由显式引用或检索命中触发；工具结果由最近且必要的压缩观察触发。
- Plan Mode 必须注入严格的只读规划约束、当前任务和最新 Plan revision；普通执行模式只在存在活动 Todo 时注入完整 Todo 列表，全终态 Todo 不进入模型输入。
- 工具注册可以完整，但模型可见工具集应是当前任务需要的最小集合。
- 大文件、大表格和搜索结果不应原样进入 prompt；`shell` / `code_run` 可作为数据处理隔离层，模型只接收统计、样例、错误和必要摘录。
- Context selection 的当前方向是本地、同步、确定性、可测试的选择层；不要引入 embedding、向量数据库、后台索引服务、复杂插件生态或普通用户可配置的上下文策略。

## 6. Session、Episode 与事件流

- `AgentSession` 应维护 bounded session summary、只包含关键发现的 bounded working state，以及独立的 planning state 与 task ledger；多轮任务不得线性撑大 `model_input`。
- 会话恢复读取 `session.json`、`turns.jsonl`、`working_state.json`、`task-ledger.json` 和 `planning-state.json`，不读取完整 episode transcript、tool-calls 或 verification 输出。
- `AgentSession` 持有 `SessionSnapshot`（可序列化 package 状态）、`SessionResources`（gateway/MCP/callback 等 live 资源）和独立的 session-owned `ModelContextRuntime`；`apply_state` 只绑定三者，不再逐字段镜像。gateway/MCP/callback 不进磁盘 schema。
- `session.json` 必须写入 `session_snapshot_schema_version`；当前 snapshot schema 为 v6，resume 必须严格校验，缺失、旧版本和未来版本都明确拒绝，不维护宽松兼容路径。
- `ModelContextRuntime` 独占模型消息链、snapshot、epoch/revision、delta、rebuild、checkpoint、恢复校验和 diagnostics。`model-context.json` v2 以 `messages + snapshot + rebuild_required` 作为一次校验、一次原子替换的完整聚合；`session.json` 不重复保存 context 状态。
- Episode 消费经 typed `EpisodePackage` / record codecs（metadata、failure、tool-call、environment、cost 等）；inspect/export/eval 只走 typed 字段，不保留裸 dict 双契约。跨文件 validator 保留；`build_episode_package` 仅在 validator 之后内部 decode，codec 自身拒绝宽松 bool 转换。
- `working_state.json` 只保存有界 `key_findings` 和 `last_updated_turn`；任务目标、进度和下一步的唯一事实源分别是 PlanningState 与 TaskLedger，不得恢复旧双契约。
- `planning-state.json` 保存完整 Plan proposal、revision、runtime id 和 `planning` / `awaiting_confirmation` / `approved_pending_execution` / `execution_started` / `cancelled` 状态；`task-ledger.json` 保存 Todo 四态 `pending` / `in_progress` / `completed` / `cancelled`。
- `RuntimeUiEvent` 是前端无关的强类型事件契约，字段只放展示和状态判断需要的摘要。
- `RuntimeUiEvent` 不放完整工具输出、完整文件内容、完整用户答案、完整 episode trace 或 secret。
- 用户补充输入的 event、transcript 和 working state 只保存短标题、问题数、outcome、回答数和字符数；完整问题正文仅停留在当前 live interaction，完整答案只返回当前模型工具结果。
- 稳定事件类型包括 session/turn 开始结束、工具开始/完成/失败、assistant 消息、审批请求/批准/拒绝、用户补充输入请求/接收、failure 和 session finished。
- 模型路由还记录 `model_protocol_fallback` 与 `model_fallback`，包含脱敏连接、模型、协议、原因和能力缺失；顶部状态和活动流应展示实际使用模型。
- 模型流恢复额外发出 `assistant_attempt_reset`、`model_retry_scheduled` / `model_retry_exhausted`；reset 事件只携带 turn/attempt/category 等标识，不携带失败 attempt 全文或 provider raw payload。
- 恢复中的 retry/retrieve 状态不是 failure；预算耗尽或不可恢复错误后才发 failure。episode 可记录 attempt/reset/retrieve 证据；session package 与 UI snapshot 不复制失败 attempt 全文。

## 7. 记忆系统

- 未信任候选只经 `MemoryCandidateIntake.submit` 进入 queue；模型/用户/runtime draft 共享 `MemoryIdentity` 与治理结果。提取器只产 draft，不直写 store/queue。`MemoryStore` 仅保留内部 `_persist_candidate`，禁止公开旁路入队。
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

- 真实任务工具包包括 `file_read`、`file_write`、`apply_patch`、`shell`、`code_run` 以及 `job_start` / `job_status` / `job_logs` / `job_cancel` 等默认 workspace-bound 原子工具。
- 多智能体 `agent` 调用只确认 Worker 已启动并立即返回完整 `task_id`，不得在主 Agent loop 中隐式等待。当前动作硬依赖 Worker 结果时显式调用 `task_wait`；仅查询状态用 `task_get` / `task_list`，读取完整产物用 `task_output`。`task_wait` 使用有界条件等待，至少一个目标进入 completed / failed / stopped / interrupted / awaiting_approval 即返回，超时是成功状态响应，不得转成高频模型轮询。
- Worker 终态写入 TeamStore 的持久通知收件箱；model 与 UI 使用独立消费者游标，互不吞通知。下一模型 turn 只注入一次有界未读摘要，TUI 可独立提示，但完成通知不自动唤醒模型、不创建协调 turn，也不冒充 assistant 最终回答。
- Worker 生命周期只保证跨 turn、当前 HaAgent 进程内持续运行。新建或切换 session、关闭 session、退出 TUI 时，仍运行的 Worker 必须有限等待清理并明确记录为 `interrupted`；进程重启后无法恢复的非终态记录也应对账为 `interrupted`，不得长期保留假 `running`。
- 文件工具接受绝对路径或 workspace 相对路径；未授权外部路径触发 `external_directory` 审批，允许一次只作用于当前调用，始终允许写入有界 session 权限规则。
- `file_read` 应支持范围读取、关键词定位和路径建议，服务普通 Agent 使用。
- `apply_patch` 继续保持 fail-fast；失败应帮助模型从结构化错误中恢复，而不是吞掉错误。
- `shell` / `code_run` 的 cwd 默认位于 workspace root；显式外部 cwd 经 `external_directory` 批准后可用于当前调用。
- `shell` 在执行前对常见 Bash/PowerShell 文件命令做 best-effort 路径扫描并合并申请外部目录权限；该扫描不是进程级 sandbox，未知命令和动态脚本仍以高风险工具审批及实际 sandbox 为准。
- `code_run` 无法可靠静态分析任意 Python 路径，访问外部目录时必须通过 `external_directories` 显式声明并在执行前审批。
- `shell` / `code_run` timeout 默认 60 秒，上限 120 秒；适合短命令。长任务使用后台 job 工具：`job_start` 立即返回 `job_id`，`job_status` 默认在工具层等待最多 30 秒并在终态附带近期日志摘要，仍运行时再调用；`job_logs` 只用于实时诊断或补充输出，必要时用 `job_cancel`。等待不得由模型高频轮询驱动；job 日志与元数据写在用户级 `~/.haagent/jobs/`，不污染 workspace。
- 后台 job 的 wall-clock timeout 默认 3600 秒，上限 7200 秒；超时或取消会终止进程树。
- `code_run` 用于降低多行脚本和 shell 转义成本；临时脚本写入系统 temp 并在执行后删除，不污染 workspace；复盘依赖 episode 中的 `code` 参数与工具结果。
- 工具输出向工具结果和 context 暴露摘要、excerpt、timeout、truncated 等字段，并对 secret-like 输出做脱敏。
- 明显泄密、workspace 绕过和高风险工具参数必须在 runtime 层显式失败或拒绝。
- 高风险或信息不足场景应通过审批或用户补充输入机制处理，不要让模型硬猜。
- Plan Mode 采用双层硬边界：模型只看见只读文件、只读联网、session history、skills、图片、结构化询问和 `submit_plan` 白名单；`ToolRouter` 同时拒绝写文件、shell、code、job、worker、Todo、memory 与所有动态 MCP 调用。

## 9. TUI 规则

- TUI 的目标是个人助手会话工作台，不是 IDE、代码编辑器、文件树主界面或复杂多标签工作台。
- TUI 应通过 `AssistantService` 驱动会话，复用 `AgentSession`、`ModelGateway`、`ToolRouter`、workspace root 和 episode trace。
- TUI 信息架构优先级是：当前上下文、对话流、可恢复状态、配置健康度、低频操作。
- 当前布局以顶部状态栏、主对话 timeline、输入区和上下文 footer 为核心；低频能力通过 overlay/modal 打开，80x24 应可完成输入、阅读、审批、查看失败和退出。
- 顶部状态栏只展示工作区、当前模型、联网开关和当前工作状态。profile、provider、API key、权限、sandbox、session、turn 和原始 state 等诊断字段不得进入常驻状态栏；异常在对应配置、审批、权限或失败界面展示。
- 状态栏按终端 cell 宽度截断：80–119 列优先保留当前目录名，模型可以截断，联网和工作状态必须完整；只给联网与工作状态片段使用语义色，不整行随状态变色。
- 主对话区采用非对称安静结构：用户消息使用低对比表面和细左边线，不显示角色标签；assistant 回答直接落在主背景上，不使用卡片、边框或标题；系统、命令、操作和提示使用紧凑行内通知，失败保留明确符号与错误语义。
- 长 Plan、过程文本和工具诊断必须保留全文，性能问题不得用截断、默认折叠或减少文本掩盖。长正文使用按行缓存的只读渲染；正文区域默认支持普通鼠标拖选和复制，不能拦截左键去展开详情。展开/收起只能由独立的小型控制件触发。
- 性能陷阱：不要把长正文放进单个 `Static` 后依赖其整块 `render_strips`；鼠标悬停、选择或局部状态变化会反复重绘全文。应使用 `RichLog` 或同等的 Line API 按行缓存，普通单条详情只同步目标 block；只有过程组展开、收起这类改变可见条目集合的操作才同步受影响的窗口，禁止重建整个 timeline。
- 工具过程运行时显示中文动作摘要，过程标题按秒只刷新当前块的步骤数与耗时，不触发 timeline 全量重绘；最终回答完成后连同本轮工具失败、任务受阻诊断一起折叠为“已完成 N 步 · 本轮耗时 ›”并冻结耗时。没有最终回答的失败、待审批和待补充输入保持可见。普通界面使用中文工具名，详情同时显示中文名与原始标识；完整 transcript、tool output、stdout、stderr、patch 和 episode trace 仍按需打开。
- 输入区使用多行 `TextArea`：`Enter` 提交，`Ctrl+Enter` 换行，空输入不提交，`Esc` 关闭当前 modal/overlay 或返回上层交互。焦点只使用一格细左边线、轻微背景和光标变化，不显示高亮粗边框。
- 输入框下方、快捷键 footer 上方可显示最近一次真实模型 step 的上下文用量：120 列及以上显示绝对 token 与整数占比，80–119 列优先只显示占比，窗口未知时只显示绝对 token；没有可信 provider usage 时整行隐藏。占比使用实际执行模型的输入上限（`limit.input` 优先于 `limit.context`），Anthropic 输入量包含 cache creation/read token；不得用压缩预算的默认窗口估算。模型或会话切换时清空，恢复会话不回放旧值。
- 聊天 footer 固定为 `/ 命令 · Ctrl+F 搜索 · ? 帮助 · Ctrl+Q 退出`，不显示 `Enter 发送`；运行时用 `Ctrl+X 取消任务` 替换 `/ 命令`。其他上下文 footer 最多四组操作，完整键位进入 `?` 帮助。
- 计划任务未读数不进入顶部状态栏；数量增加时只发一次非持久通知，完整数量留在计划任务界面。
- 运行时请求用户补充信息时，输入区进入回答状态，提交后继续同一个 turn，不能变成新 prompt。
- 补充输入使用 InputDock 内联结构化面板：普通 PromptInput 在交互期间隐藏，slash command、文件引用、图片附件和历史输入逻辑不得接收回答按键。Esc 只关闭本次提问并返回 `dismissed`；Ctrl+X 才取消整个任务。
- 结构化面板支持 1–3 题、单选、多选、自由文本和 Review；方向键/数字选择、Space 切换多选、Ctrl+Enter 插入文本换行、Tab/Shift+Tab 切题。80×24 使用紧凑标题并只展示当前选项说明，120 列以上展示全部说明，resize 后保持焦点、选择和草稿。
- 每个请求只挂载一次 QuestionPrompt；选项移动、草稿编辑、问题导航和 Review 只刷新该 Widget，不得调用应用级 `_refresh()` 或重建 timeline。打开/关闭只定向更新状态栏、footer、焦点和 InputDock。
- `/plan` 或 `/plan <任务>` 进入 Plan Mode。Plan 确认面板默认焦点在反馈输入，Enter 提交反馈、Ctrl+Enter 换行；批准必须通过 Tab 后 Enter 或鼠标点击，不提供单键批准。Esc 仅最小化且不 resolve，Ctrl+X 才取消任务。
- Todo 使用独立只读面板原位刷新，不把状态变化追加到 timeline；80×24 显示单行摘要，大屏可用 Enter 或点击展开完整四态列表。
- 工具审批 modal 必须展示工具名、影响范围和关键参数摘要；文件修改和命令执行等高影响操作默认焦点应放在 Deny。
- 审批 modal 是 focus trap；审批结果必须回到同一个 turn。
- 帮助应以 modal、overlay 或上下文化帮助呈现，不应污染对话流。
- 记忆候选审查必须支持多候选导航和确认/拒绝目标项。
- 所有功能必须键盘可达，鼠标只作为增强；颜色不能作为唯一语义，`NO_COLOR` 下仍应可读。
- 普通用户路径统一使用简体中文；HaAgent、OpenAI、DeepSeek、模型 ID、环境变量、Slash 命令和高级诊断原始值保持原名。
- Failure 展示必须包含 failed\_stage、failure\_category、reason 和 episode\_path；不要静默 fallback，也不要过度推断。
- `assistant_attempt_reset` 只撤销当前逻辑 turn 的 provisional assistant 内容，不改动已固化 intermediate/process item；迟到的旧 attempt delta 必须在 runtime sink 层按代际丢弃。
- 流恢复等待与 background retrieve 期间 `Ctrl+X` 必须可取消；取消后 TUI 回到可输入状态，不留下假运行态。

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

