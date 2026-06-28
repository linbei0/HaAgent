# 记忆系统问题记录与改进方向

本文记录 2026-06-27 会话中暴露出的记忆系统问题，作为后续修复的输入。目标是把现象、原因、边界和可参考项目经验固定下来，避免后续只针对单个症状打补丁。

## 背景

当前用户级记忆目录位于 `C:\Users\jiang\.haagent\memory`，其中包含：

- `index.json`：已确认长期记忆索引。
- `user_preferences.jsonl`：用户偏好类长期记忆正文。
- `audit.jsonl`：候选创建、确认、拒绝、索引重建等审计事件。

检查时发现正式长期记忆只有两条，但两条语义重复：

- `喜欢吃饭`：证据来自用户直接声明“我喜欢吃饭”，相对合理。
- `用户爱好`：证据来自“用户询问自己的爱好是什么，助手根据记忆回答爱好是吃饭”，本质是从助手回答和已有记忆中二次抽取出的重复事实。

`audit.jsonl` 共有 114 行，其中 105 行是 `candidate_created`。重复候选集中在“用户身份与爱好”和“饮食喜好”。这说明当前问题不是正式记忆文件损坏，而是候选抽取过于积极、去重不足、审计噪声过大。

## 本次已落地修复

本次修复把写入侧从“每个 completed turn 后都尝试抽取”改为“显式记忆结算”。模型只有在当前任务确实出现长期有价值的信息时，才可调用 `start_memory_update` 申请结算；runtime 只在本轮工具事件中看到该工具成功返回 `memory_update_requested=true` 后，才运行 `MemoryExtractor`。

这个设计参考了 GenericAgent 的 `start_long_term_update` 思路，但 HaAgent 的工具只立一个可审计 flag，不直接写正式记忆、不绕过候选队列、不绕过用户确认。正式长期记忆仍必须先进入候选队列，之后由确定性服务确认和落库。

抽取请求仍保留 `final_response`，但仅作为上下文帮助模型理解本轮任务，不允许作为 evidence。代码层面会拒绝 `assistant_response`、`final_response`、`model_inference`、`memory_recall` 和 `unknown` 等证据来源；每个候选都必须提供 `evidence_source` 与可在对应来源中定位的 `evidence_quote`。

SOP 类候选可以整理助手在任务中形成的解决方案，但不能只凭助手最终回答或用户泛泛要求生成。它必须有成功工具结果、明确文件内容或成功验证结果作为证据，才能进入候选队列。

重复抑制本次只做确定性 fingerprint：由代码根据归一化后的 `category`、`body`、`evidence_source`、`evidence_quote` 计算，不信任模型输出。该 fingerprint 会参与正式记忆、pending/confirmed/rejected 候选的重复判断；已拒绝候选的同 fingerprint 重复提案只计入 diagnostics/rejected count，不再写新的 `candidate_created` 审计事件。

本次没有做 embedding、RAG、reranker、LLM 语义去重，也没有迁移或清理 `C:\Users\jiang\.haagent\memory` 里的历史坏记忆。原因是当前 P0 是先收紧写入证据边界和候选噪声，避免继续制造新坏记忆。

## 已暴露问题

### 1. 助手回答会被二次抽取成长期记忆

当前 `AgentSession` 在每个 completed turn 后运行记忆抽取。抽取请求同时包含用户输入和助手最终回答。

这会产生闭环：

1. 用户说“我喜欢吃饭”。
2. 系统写入长期记忆：用户喜欢吃饭。
3. 用户问“我的爱好是什么”。
4. 助手根据旧记忆回答：你的爱好是吃饭。
5. 抽取器把助手回答再次抽成新记忆：用户爱好是吃饭。

长期记忆的证据边界不应包含助手根据已有记忆生成的回答。助手回答可以作为解释或交付物，但不能作为用户事实的来源。

### 2. 候选去重只覆盖浅层重复

当前去重能处理完全相同的 hash、完全相同正文、近似标题，但挡不住语义重复：

- `用户明确表示自己喜欢吃饭。`
- `用户的爱好是吃饭。`

这类重复在人看来是同一个事实，但在当前判重里会通过。

### 3. Pending / rejected 候选缺少全局抑制

审计日志显示大量重复候选被反复创建。当前逻辑主要检查正式记录和当前 session 的 pending queue，不能有效抑制跨 session 的 pending/rejected 重复。

用户已经拒绝过的候选，如果后续以同义标题或同义正文再次出现，也应该能被识别并降噪。

### 4. 审计数据和模型输入边界容易混淆

审计日志变大本身不是问题。审计数据应该完整落盘，服务检查、回放、调试和 eval。

真正需要避免的是：为了可审计，把完整 episode、完整 tool trace、完整 audit、完整候选历史复制进下一次模型输入。

原则应保持清晰：

- 审计数据可以重。
- 模型输入默认要薄。
- 二者通过 compact observation、manifest 或检索结果连接，而不是直接拼接。

### 5. 普通聊天的上下文加载需要按需，但不能靠猜复杂度

本会话明确过一个改进方向：普通聊天模型输入默认保持薄上下文，只有在出现结构化动作、工具需求、明确记忆查询或高相关上下文时才加载更多内容。

这里的“按需”不能靠猜“任务复杂度”，也不能靠匹配用户话术表。更可靠的工程边界是：

- 工具调用需求由工具协议和 runtime 状态决定。
- 记忆加载由检索分数、候选来源、scope/category、预算和诊断记录决定。
- 项目规则加载由 workspace、文件存在性和显式运行入口决定。
- 每次加载或跳过都写入可审计 diagnostics。

## 暂缓处理项：中文单字检索

本会话发现：当前中文检索按单字切词，`你好` 的 `好` 可能命中 `爱好`、`偏好`，导致无关问候也注入“吃饭”记忆。

这个问题存在，但暂不作为最高优先级修复项。原因：

- 当前更严重的问题是写入侧会把助手回答二次抽成长期记忆。
- 用停用字表、简单过滤“好/你/我”等方式修复不够可靠，容易引入新的误判。
- OpenHarness 的记忆检索也有中文单字切词，不能直接照抄。

后续处理该问题时，应避免只靠脆弱词表。更稳的方向是引入明确的检索阈值、短语级匹配、结构化命中原因、必要时再考虑中文分词或 embedding rerank，并用测试覆盖：

- `你好` 不应注入“吃饭”记忆。
- `我的爱好是什么` 应能查到“吃饭”相关记忆。

## 可参考项目经验

### OpenHarness

可参考：

- markdown 记忆文件加 frontmatter metadata。
- `signature` 用于内容去重。
- `disabled`、`ttl_days`、`supersedes`、`tags` 等字段支持软删除、过期和替代关系。
- `MEMORY.md` 作为索引，正文放 topic 文件。
- prompt 中的 memory entrypoint 有行数和字节预算。
- `latest_user_prompt` 可用于选择相关记忆，而不是无条件加载全部细节。

不建议直接照抄：

- 中文单字 tokenizer。
- ohmo 默认加载前若干记忆文件到 prompt 的做法。

### GenericAgent

可参考：

- “No Execution, No Memory”：长期记忆只记录用户直接事实或工具成功验证过的事实。
- L1/L2/L3 分层：L1 只放存在性指针，L2 放事实库，L3 放专项 SOP 或脚本。
- `working checkpoint`：当前任务短期关键状态与长期记忆分开。
- 任务结束时再触发长期记忆结算，而不是每轮自动写长期记忆。
- 历史压缩保留最近几轮，旧工具结果和 thinking 折叠。

不建议直接照抄：

- 全局记忆每次直接拼进系统提示的方式。
- 依赖自由文本 prompt 规则保证长期记忆质量，而没有代码级证据边界。

## 改进原则

1. 长期记忆证据必须有来源类型。
   - 允许：用户直接声明、成功工具结果、明确文件内容。
   - 不允许：助手根据旧记忆生成的回答、模型推理猜测、未验证计划。

2. 候选和正式记忆分离。
   - 候选可多，可审计。
   - 正式长期记忆必须由确定性服务提交。
   - 用户拒绝过的候选应能参与后续抑制。

3. 审计和 prompt 分离。
   - audit / episode / transcript 可以完整落盘。
   - 模型输入只拿 bounded summary、compact observation 或相关记忆。

4. 按需加载必须有可解释的工程信号。
   - 不通过猜用户话术复杂度决定 prompt 厚度。
   - 每次注入记忆都应记录 query、score、命中字段、预算、跳过原因。

5. 不用 prompt 修 runtime bug。
   - 证据边界、去重、候选状态、检索阈值应尽量由代码和测试保证。

## 建议修复顺序

### P0：先加回归测试

- 助手根据已有记忆回答后，不应产生新的同义长期记忆候选。
- `audit.jsonl` 可记录候选，但同义 rejected 候选不应反复刷屏。
- 普通聊天不应无条件注入完整审计、完整 episode 或完整工具输出。
- memory diagnostics 应能说明某条记忆为什么被注入或为什么被跳过。

### P1：收紧写入侧

- `MemoryExtractionRequest` 不再把 `final_response` 作为长期事实证据。
- 候选 evidence 必须引用用户输入片段或成功工具结果。
- 增加候选 canonical fingerprint，用于识别同义或近义重复。
- rejected 候选留下可比对的 tombstone 或 fingerprint。

### P2：收紧读取侧，但暂不急着改中文分词

- 为每次 memory retrieval 生成 diagnostics。
- 默认 prompt 只注入高置信、预算内、可解释命中的记忆。
- 对普通问候、无任务输入保持薄上下文。
- 中文单字误命中先作为已知风险保留，等写入侧稳定后再处理。

### P3：优化存储形态

- 评估是否从当前 JSONL 正式记录迁移到 markdown + frontmatter。
- 如果迁移，优先借鉴 OpenHarness 的 metadata 字段和 signature 思路。
- 保持用户级、workspace 级、session 级物理分离。

## 本次明确暂不处理

- 普通聊天上下文按需加载：本次只避免普通聊天无条件触发写入侧抽取，不重构 context assembly、工具 schema 按需暴露或 AGENTS.md 注入策略。
- 中文 tokenizer / 单字检索误命中：保留为已知风险，后续用结构化命中原因、短语级匹配、阈值或 rerank 处理，不用停用字表抢修。
- 语义去重：本次只做 deterministic fingerprint 与既有内容/标题近似抑制，不引入 embedding、RAG、reranker 或 LLM 去重。
- 历史数据清理：不迁移、不删除、不自动修复 `C:\Users\jiang\.haagent\memory` 中已经存在的坏记忆。
- 候选确认体验：不改变用户确认、拒绝、提交长期记忆的交互流程。

## 后续 TODO

- 建立 prompt registry 与 prompt snapshot tests，确保记忆结算提示词、工具说明和证据协议变更可审计。
- 设计普通聊天上下文按需加载：context manifest、selected/skipped diagnostics、记忆注入预算和工具 schema 最小暴露。
- 修复中文检索质量问题：覆盖 `你好` 不应命中“爱好/偏好”，`我的爱好是什么` 应命中已确认爱好记忆。
- 为 memory retrieval 增加更完整的 selected/skipped diagnostics，解释每条记忆为什么注入或跳过。

## 验收标准

- 长期记忆不会因为助手复述旧记忆而自我复制。
- 用户拒绝过的候选不会以轻微改写反复出现。
- “可审计”只增加磁盘记录，不导致下一轮模型输入变重。
- 普通聊天默认薄上下文；加载更多内容时有结构化原因。
- 本会话发现的问题都有对应测试或 diagnostics 能复现。
