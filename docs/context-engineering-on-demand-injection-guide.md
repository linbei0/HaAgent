# 按需注入上下文工程指导

本文整理 2026-06-27 对 context engineering、memory、MCP、long-context 资料的调研结论，并转化为 HaAgent 后续改进可直接使用的工程原则。

目标不是追新名词，而是回答一个具体问题：HaAgent 怎么保证“按需加载上下文”是可审计、可测试、可解释的工程行为，而不是靠模型或开发者猜用户这句话复杂不复杂。

## 一句话结论

模型输入应该默认薄，只放本轮必需的高信号内容。完整历史、审计、工具 trace、候选记忆、长文件和大表格应该留在磁盘、工具、执行环境或检索索引里；只有被明确选中的摘要、事实、片段或工具结果摘录才进入 prompt。

“按需”不是判断用户这句话难不难，而是由结构化信号触发：

- 当前入口和 workspace 决定加载哪些项目规则。
- 当前任务状态决定是否需要 session summary、working state、未完成计划。
- 记忆检索结果决定是否注入长期记忆。
- 工具选择和工具 schema 决定是否暴露工具能力。
- 大数据处理结果决定只返回统计、样例或摘录，不把原始全集塞给模型。
- 每次注入或跳过都记录 diagnostics，后续能解释、测试和复盘。

## 外部资料提取

### 1. Context engineering 的核心不是写更长 prompt

Anthropic 将 context engineering 定义为：每次模型推理前，决定哪些 token 应该进入有限上下文窗口。关键点是上下文会持续变化，包括 system prompt、工具、MCP、外部数据、消息历史和运行状态。

对 HaAgent 的启示：

- 不要把 prompt 当成静态模板。
- 每轮调用模型前都应有一个 context assembly 阶段。
- 这个阶段的输出必须是结构化的 `model_input`，而不是在代码里到处拼字符串。
- prompt 变厚必须有来源、预算和原因。

### 2. 长上下文不是免费容量

Chroma 的 Context Rot 研究指出，长上下文窗口虽然能容纳更多 token，但真实应用中模型性能会随着输入变长出现退化。Anthropic 也强调上下文是有限注意力资源，新增 token 会消耗注意力预算。

对 HaAgent 的启示：

- “模型支持 128k / 1M token”不代表应该尽量塞满。
- 可审计数据越完整越好，但模型输入越准越好。
- 低信号内容不是中性内容，它会稀释注意力、增加冲突和误召回。
- 预算不能只防止超长，还要主动维持高信号密度。

### 3. 常见策略可以归为四类：写入、选择、压缩、隔离

LangChain 将 agent context engineering 概括为四类：

- Write：把信息写到上下文窗口之外，例如 scratchpad、session state、memory file。
- Select：按任务选择相关信息，例如检索记忆、按需读取文件、选择工具。
- Compress：压缩历史和工具输出，例如 summary、compaction、摘录。
- Isolate：隔离复杂上下文，例如子任务、执行环境、独立工具过程。

对 HaAgent 的启示：

- `audit.jsonl`、episode、tool trace 属于 Write，不等于要注入 prompt。
- memory retrieval 属于 Select，必须有分数、命中字段和阈值。
- session summary 和 working state 属于 Compress，必须有预算。
- shell/code_run 处理大文件、大表格属于 Isolate，模型只看结果摘要。

### 4. 工具和数据也要按需暴露

Anthropic 的 MCP/code execution 资料强调两个 token 黑洞：一次性暴露大量工具定义，以及把中间工具结果反复传过模型上下文。更好的方式是让 agent 按需发现工具，并在执行环境里过滤、聚合、转换数据，只把必要结果返回模型。

对 HaAgent 的启示：

- 工具注册可以完整，但模型可见工具集应是当前任务需要的最小集合。
- 大文件、大表格、搜索结果不应原样进入 prompt。
- `code_run` / shell 可以作为数据处理隔离层，模型只接收统计、样例、错误和必要摘录。
- 对敏感数据，优先在工具层脱敏或汇总，避免中间明文进入模型输入。

### 5. 记忆需要“两阶段”：临时笔记，再合并成长期记忆

OpenAI Cookbook 的 personalization 示例强调，长期记忆系统要处理去重、冲突、遗忘和低信号剪枝；session notes 到 global memories 的两阶段处理比一次性写长期记忆更可靠。

对 HaAgent 的启示：

- 每轮自动抽取长期记忆风险很高。
- 候选记忆可以多，但正式记忆必须经过证据边界、去重、冲突处理和用户确认。
- 助手回答不能作为用户事实的证据来源。
- 被拒绝、过期、替代的记忆要参与后续抑制，不能无限重复候选。

### 6. 多类型记忆比一锅粥可靠

MIRIX 等 memory-agent 研究把记忆拆成 Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault 等类型。HaAgent 不需要照抄多 agent 结构，但应该接受一个事实：不同记忆的生命周期、可信度和注入条件不同。

对 HaAgent 的启示：

- 用户偏好、工作区事实、会话进度、工具观察、操作流程不应混在一个文件里。
- 不同 scope 的记忆必须物理或逻辑分开：User / Workspace / Session。
- 检索时必须知道命中的是哪类记忆，不能只返回一段正文。

## HaAgent 的目标设计

### Context Assembly 应成为明确阶段

每次调用模型前，HaAgent 应先构造一个 `ContextAssembly` 结果。它至少包含：

- `base_instructions`：稳定、短小、通用的系统规则。
- `workspace_context`：当前 workspace root、允许访问边界、必要项目规则。
- `session_context`：有界 session summary、working state、未完成任务。
- `memory_context`：被选中的长期记忆，带来源和命中原因。
- `tool_context`：当前允许且相关的工具集合或工具入口。
- `observations`：最近工具结果的压缩观察，不是完整 tool trace。
- `diagnostics`：每个 source 注入或跳过的原因、预算、token 估算。

其中 `diagnostics` 默认不进入模型输入，只写入 episode / trace，供 inspect、测试和调试使用。

### Prompt 默认分层

建议把模型输入分成四层：

1. 永远注入：身份、工作区边界、安全规则、输出语言、工具协议。
2. 常规注入：本轮用户输入、最近短摘要、当前 working state。
3. 条件注入：相关记忆、项目规则、任务 scaffold、验证要求、文件摘录。
4. 禁止直接注入：完整 audit、完整 episode、完整 transcript、完整工具输出、完整候选记忆池。

这不是为了省 token 而省 token，而是为了让模型看到的内容更干净。

### “按需”的触发条件

按需加载应由稳定工程信号触发，而不是用户话术猜测。

| 上下文类型 | 注入触发 | 跳过原因 | diagnostics 应记录 |
| --- | --- | --- | --- |
| 项目规则 | 当前 workspace 存在 `AGENTS.md` 或项目 docs 被入口要求 | 普通目录无项目规则 | 文件路径、大小、预算、摘要方式 |
| Session summary | 会话已有历史或正在恢复 | 新会话首轮 | summary 版本、更新时间、token 估算 |
| Working state | 存在未完成目标、当前计划或关键发现 | 无持续任务 | 字段列表、更新时间 |
| 长期记忆 | 检索命中高置信、scope 匹配、预算允许 | 分数低、来源不可信、用户拒绝、预算不足 | query、score、命中字段、source、skip reason |
| 工具说明 | 当前任务需要工具能力，或模型可调用工具入口固定 | 纯闲聊、无需工具 | tool name、选择原因、schema 预算 |
| 文件内容 | 用户显式引用或检索命中具体文件 | 没有文件目标 | path、range、摘要/摘录策略 |
| 工具结果 | 工具刚返回且对下一步必要 | 旧结果、过长、只需落审计 | excerpt 长度、truncated、完整结果位置 |

### 用“选择器”代替“复杂度判断”

不要写一个 `is_complex_prompt("你好")` 之类的判断。更稳的设计是多个小选择器共同决定上下文：

- `WorkspaceContextSelector`：只关心工作区规则和边界。
- `SessionContextSelector`：只关心会话恢复和 working state。
- `MemoryContextSelector`：只关心记忆检索、scope、证据和预算。
- `ToolContextSelector`：只关心工具能力是否需要暴露。
- `ObservationSelector`：只关心哪些工具观察需要给模型继续推理。

每个选择器输入结构化状态，输出 `selected / skipped` 和原因。最后由 `ContextBudgeter` 统一裁剪预算。

## 记忆系统改进指导

### 写入侧

长期记忆写入必须满足证据边界：

- 用户直接声明可以成为候选证据。
- 成功工具结果可以成为候选证据。
- 明确文件内容可以成为候选证据。
- 助手复述、推理、猜测、计划不能成为用户事实证据。

候选到正式记忆必须经过：

- canonical fingerprint。
- semantic duplicate check。
- conflict check。
- rejected tombstone 抑制。
- scope/category 校验。
- 用户确认或明确策略授权。

### 读取侧

记忆注入必须满足：

- scope 匹配，例如 user / workspace / session。
- 来源可信，例如 confirmed memory 优先，candidate 默认不进 prompt。
- 命中可解释，例如 title/tag/summary/body 哪个字段命中。
- 达到最低相关阈值。
- 有 token 预算。
- 不与更高优先级事实冲突。

当前中文单字检索误命中先作为已知风险保留，不用停用字表急修。后续如果修，应该从结构化命中原因、短语级匹配、阈值和 rerank 入手，而不是维护一张脆弱中文词表。

## 可参考和不可照抄

### 可以参考

- Anthropic：把 context 当有限资源，强调 just-in-time retrieval、compaction、structured note-taking、tool result clearing。
- Anthropic MCP/code execution：工具定义和中间结果按需加载，大数据在执行环境处理。
- LangChain：Write / Select / Compress / Isolate 四分法，适合作为 HaAgent context 模块边界。
- OpenAI Cookbook：session trimming、compression、memory note 到 global memory 的两阶段合并。
- OpenHarness：frontmatter metadata、signature、disabled、ttl、supersedes、bounded memory prompt。
- GenericAgent：No Execution, No Memory、working checkpoint、长期记忆与短期状态分开。

### 不建议照抄

- 不照抄 OpenHarness 的中文单字 tokenizer。
- 不照抄“把前几个记忆文件默认塞进 prompt”的做法。
- 不把完整 trace/audit 当成模型上下文。
- 不靠 prompt 规则要求模型“不要乱记忆”。
- 不靠用户话术表判断任务复杂度。
- 不把所有工具 schema 一次性暴露给模型。

## 验收标准

### 行为验收

- 用户只说“你好”时，模型输入保持薄上下文，不注入无关长期记忆。
- 用户问“我的爱好是什么”时，能注入经过确认的相关用户记忆。
- 用户恢复长任务时，只注入 bounded summary 和 working state，不复制完整历史。
- 工具读到大文件时，下一轮只注入必要摘录或统计，不注入完整文件。
- 审计文件增长不会导致模型输入增长。

### 工程验收

- 每次模型调用都有可 inspect 的 context manifest。
- 每条被注入的上下文都有 source、reason、budget、token estimate。
- 每条被跳过的候选上下文都有 skip reason。
- 记忆候选创建、确认、拒绝、抑制都有结构化事件。
- 测试可以断言某轮 prompt 包含什么、不包含什么。

### 回归测试建议

- `test_greeting_does_not_load_unrelated_memory`
- `test_explicit_memory_question_loads_confirmed_relevant_memory`
- `test_final_response_is_not_memory_evidence`
- `test_rejected_memory_fingerprint_suppresses_repeated_candidate`
- `test_context_manifest_records_selected_and_skipped_sources`
- `test_large_tool_result_is_summarized_not_injected_raw`
- `test_audit_growth_does_not_increase_prompt_size`

## 建议实施顺序

1. 先加 context manifest / diagnostics，不改变模型行为也能看到当前真实 prompt 是怎么拼的。
2. 给 `AgentSession` 的模型调用加测试钩子，能断言最终 `model_input`。
3. 收紧记忆写入证据边界，禁止 `final_response` 作为长期用户事实证据。
4. 把 memory retrieval 的 selected/skipped 原因结构化。
5. 把 session summary、working state、memory、tool observations 分成独立 context sources。
6. 增加统一预算器，按 source 优先级裁剪。
7. 最后再处理中文检索质量，不用脆弱停用字表抢修。

## 参考资料

- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Anthropic: Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [LangChain: Context Engineering](https://www.langchain.com/blog/context-engineering-for-agents)
- [LangChain Docs: Context engineering in agents](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [OpenAI Cookbook: Session memory](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)
- [OpenAI Cookbook: Context personalization](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization)
- [Chroma: Context Rot](https://www.trychroma.com/research/context-rot)
- [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957)

