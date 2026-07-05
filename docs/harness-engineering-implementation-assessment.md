# HaAgent Harness 工程实施评估

评估日期：2026-07-05

## 1. 结论摘要

HaAgent 已经不是 H0 级“聊天式补丁工具”。从代码实现看，它具备真实 agent loop、结构化工具路由、episode package、context manifest、验证命令执行、TUI 会话、工作状态、MCP 接入、可选 Docker sandbox、eval 导出与运行器等能力，整体处于 **H1 到 H2 之间，局部达到 H2，尚未达到 H3**。

当前最明显的 Harness 工程缺口不是“没有工具”或“没有 loop”，而是：

1. **项目事实源不统一**：根目录 `AGENTS.md` 已被确认过时，且引用的 `docs/harness-requirements.md`、`docs/code-governance.md`、`docs/unresolved-risks-and-roadmap.md` 当前不存在；真实规则散落在 `README.md`、`docs/active-rules-summary.md`、`src/haagent/docs/` 和代码测试中。
2. **自然语言任务契约仍过薄**：TUI 普通 turn 会生成临时 task contract，但默认验收标准是通用句，验证命令为空，缺少明确的非目标、风险边界、完成定义和验证策略。
3. **eval 已有原型，但还不是持续改进系统**：内置 eval 更像 smoke/regression seed，缺少真实失败转 dataset、grader 分层、环境元数据、pass@k/pass^k、owner、CI 门禁和 trace 阅读流程。
4. **sandbox 和权限边界偏个人助手形态**：Docker sandbox 已存在但非默认强边界；local subprocess 默认继承环境、网络 unrestricted，只能算降级审计记录，不是企业级隔离。
5. **可观测性可复盘但不够可运营**：episode trace 很强，但尚未形成指标、成本、模型/prompt/tool 版本、dashboard、失败聚合和回流机制。
6. **安全治理缺少 source-sink 闭环**：已有 web 内容外部标记、网络 guard、secret redaction、policy/approval，但尚未把不可信来源与高风险 sink 之间的影响关系结构化记录并执行。
7. **熵增审计与文档机械校验缺位**：没有检测文档漂移、规则过期、AGENTS 映射失效、质量分数退化、过时 plan 或坏模式复制的自动检查。

如果按 `E:\md-note\harness\05-harness成熟度模型与评估模板.md` 的 11 项职责评估，HaAgent 当前最强的是 **Tool Access、Context Selection、Observability 的 episode 层、Permissions 的工具审批层、Task State 的会话层**；最弱的是 **Task Specification、Failure Attribution 的闭环使用、Verification 的 grader 化、Entropy Auditing、Intervention Recording 的审计化运营**。

## 2. 评估依据

### 2.1 本地资料

已系统阅读 `E:\md-note\harness` 下全部资料：

- `README.md`
- `01-核心概念笔记.md`
- `02-agent-runtime-设计方案.md`
- `03-代表性资料精读路线.md`
- `04-精读卡片与设计启发.md`
- `05-harness成熟度模型与评估模板.md`

这些资料提供了本次评估的核心框架：Agent = Model + Harness、agent loop、context engineering、tool/router/sandbox/policy/eval/observability、episode package，以及 11 项 Harness 职责和 H0-H3 成熟度模型。

### 2.2 项目内证据

重点阅读和抽样核对了以下项目资料与实现：

- `README.md`
- `docs/active-rules-summary.md`
- `src/haagent/docs/TASK_POLICY.md`
- `src/haagent/docs/EVAL_CASE_SCHEMA.md`
- `src/haagent/runtime/orchestration/orchestrator.py`
- `src/haagent/runtime/session/agent.py`
- `src/haagent/runtime/session/turn.py`
- `src/haagent/context/builder.py`
- `src/haagent/tools/router.py`
- `src/haagent/runtime/episodes/writer.py`
- `src/haagent/runtime/episodes/validator.py`
- `src/haagent/verification/engine.py`
- `src/haagent/runtime/evaluation/runner.py`
- `src/haagent/runtime/sandbox/local.py`
- `src/haagent/runtime/sandbox/docker_backend.py`
- `examples/evals/*.json`
- `tests/unit`、`tests/integration`、`tests/extended` 中与 runtime、context、tool、episode、eval、sandbox、TUI 相关的测试索引。

### 2.3 联网补充资料

补充核验了以下公开资料。部分 OpenAI 页面直连抓取返回 403，因此使用搜索摘要与可访问文档交叉验证，未把不可核验细节当作唯一事实来源。

- OpenAI: [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- OpenAI: [Agents SDK guide](https://developers.openai.com/api/docs/guides/agents)
- OpenAI: [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)
- OpenAI Agents SDK: [Tracing](https://openai.github.io/openai-agents-python/tracing/)
- Anthropic: [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Anthropic: [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Model Context Protocol: [Authorization specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- Model Context Protocol: [Security and trust & safety](https://modelcontextprotocol.io/specification/2025-11-25/index)
- Model Context Protocol blog: [2026-07-28 release candidate](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/)

## 3. 外部 Harness 基准

本次评估采用以下基准，而不是只看项目是否“能跑”：

1. **Harness 是模型外的运行基座**：包括上下文、工具、状态、权限、沙箱、可观测性、验证、eval 和人工介入。
2. **AGENTS.md 应是地图，不是事实仓库**：长期事实应放在版本化、可索引、可检查的 `docs/`、schemas、plans、quality、安全与架构文档中。
3. **上下文是有限资源**：应 just-in-time 读取、记录来源、预算与 skip reason；工具输出应摘要化或可清理，长任务状态应外部化。
4. **工具是能力边界**：工具 schema、参数校验、权限、审批、错误结构和输出摘要决定 agent 能否稳定行动。
5. **验证和 eval 是不同层**：verification 证明一次任务完成；eval 衡量一类任务在模型、工具、prompt、sandbox 变化后的稳定性。
6. **生产 agent 需要 trace-first，然后 dataset/eval**：先读真实 trace，建立失败分类，再抽取 regression/capability eval。
7. **安全不能靠 prompt 自觉**：MCP、web、shell、文件写入、外部内容和凭证都需要结构化授权、source-sink 控制、最小权限和审计。
8. **sandbox 是企业级分水岭**：本地 subprocess 可用于个人助手，但企业级执行需要明确文件、进程、网络、凭证、资源和产物边界。
9. **H3 是持续改进系统**：trace 回流 dataset、eval 进 CI/CD、红队和权限审计常态化、成本和环境元数据可比较、文档漂移可自动发现。

## 4. 当前已有 Harness 能力

### 4.1 Agent loop 与运行编排

`RunOrchestrator` 已经承担核心 agent loop：加载 task、准备 run、构造 context、调用 model gateway、执行工具、记录 transcript/tool calls、执行 verification、写 failure attribution、处理取消和 human interaction。它不是单次聊天封装，已经是明确的 runtime 编排层。

积极信号：

- 模型调用经由 `ModelGateway`。
- 工具调用经由 `ToolRouter`。
- 每个 run 写 episode package。
- run status、failure category、verification reached 等状态进入 transcript。
- TUI turn 通过 `AgentSession` 进入 runtime，而不是在 UI 层自造执行逻辑。

不足：

- loop 仍偏线性，复杂任务缺少显式 contract compilation、计划审批、独立 evaluator 或 adversarial verification。
- stop/continue 主要围绕 max turns、final answer 和 verification result，尚未形成可配置的任务级状态机与失败升级策略。

### 4.2 工具路由与权限

`ToolRouter` 已具备较好的 Harness 原语：

- allowed tools 白名单。
- registry/schema 对齐校验。
- 参数类型和必填字段校验。
- path policy 和 workspace 边界。
- policy/approval 决策。
- guardrail 阻断。
- dynamic MCP tool dispatch。
- 每次调用写入 `tool-calls.jsonl`。
- 高风险工具未批准时 handler 不执行。

这说明项目已经进入 H1/H2 的关键区域：工具不是随意函数调用，而是受 policy、schema、trace 约束的能力面。

不足：

- approval 目前偏工具级，不够任务风险级、数据来源级、sink 级。
- policy 语义还不覆盖 actor、租户、环境、网络、凭证、生产资源等企业维度。
- 对 MCP tool description 的信任边界和远程 MCP 授权仍需加强；MCP 2025-11-25 规范已把 OAuth 2.1、Protected Resource Metadata、scope、resource indicator 等作为 HTTP transport 授权重点。

### 4.3 Context engineering

`ContextBuilder` 已经实现较成熟的 context assembly：

- 构造 system/task messages。
- 写 contexts 快照和 context manifest。
- 记录 compaction diagnostics。
- 支持 session summary、working state、memory、skills、prompt packs。
- 对 observation 做 compact。
- 有 source diagnostics、预算统计和 skipped/selected 记录。

这与 Anthropic “just-in-time context、tool output offloading、memory/compaction” 方向一致。

不足：

- 项目级事实源分散，导致 ContextBuilder 即使机制存在，也可能读到过时或不完整的规则。
- 缺少面向文档知识库的强索引、freshness、ownership、cross-link 校验。
- 缺少对模型可见工具集的“任务相关性最小化”更细粒度策略，目前更多依赖入口配置和 registry。

### 4.4 Episode package 与可复盘性

`EpisodeWriter` 与 `EpisodeValidator` 是项目很强的部分：

- 创建 `episode.json`、`transcript.jsonl`、`tool-calls.jsonl`、`context-manifest.json`、`plan.json`、`environment.json`、`sandbox.json`、`workspace-preflight.json`、`failure.json`、`verification/commands.jsonl` 等结构。
- validator 对跨文件一致性、tool status、policy、approval、verification、sandbox、environment、failure record 做严格校验。
- `inspect` 和 `export-eval` 复用 package view，避免从散乱日志推断。

这已经接近本地资料中 episode package 模型的核心精神。

不足：

- `environment.json` 当前只记录 Python、platform、created_at、workspace_root；缺少模型版本、采样参数、prompt/skill/tool 版本、CPU、内存、网络策略、并发、镜像 digest、cache 策略等 eval 可比性字段。
- 没有 cost/token 记录文件。
- 没有 artifact 目录的统一规范和生命周期。
- failure attribution 有文件和 category，但尚未形成“失败 -> 分类 owner -> 文档/工具/策略/eval 回流”的运营闭环。

### 4.5 Verification 与 eval

已有能力：

- `VerificationEngine` 能执行 task 中声明的 verification commands，记录 stdout/stderr excerpt、truncation、timeout、redaction。
- `runtime/evaluation/runner.py` 支持 eval case、manifest、deterministic model responses、expected tools、final status、final response、failure category 和 context expectation。
- `export_eval_case()` 能从 episode 导出 eval case。
- `examples/evals` 有内置 smoke/regression seed。

不足：

- 普通 TUI 任务默认 `verification_commands: []`，大量自然语言任务无法独立证明完成。
- eval case 还不是 Terminal-Bench/SWE-bench 风格的 instruction + environment + oracle + grader 完整任务包。
- 当前比较逻辑偏状态、工具名、最终文本和少量 context expectation，缺少可扩展 grader 插件、artifact checks、rubric、partial credit、LLM judge 校准、人审入口。
- eval 没有区分 capability eval 与 regression eval。
- 没有 pass@k/pass^k、重复运行、基础设施噪声记录。
- 没有从真实失败 trace 自动或半自动转 dataset 的流程。

### 4.6 Sandbox 与执行边界

已有能力：

- `LocalSubprocessSandboxBackend` 明确记录 degraded 状态。
- `DockerSandboxBackend` 支持 non-root、cap drop、no-new-privileges、read-only rootfs、CPU/memory/pids/tmpfs、network 配置、minimal env。
- sandbox metadata 写入 episode。
- shell/code_run cwd 受 workspace policy 约束。

不足：

- 默认 sandbox 是 local subprocess，`network_policy=unrestricted`、`credential_policy=inherit_environment`、`process_policy=local_subprocess`。
- Docker 不可用时可 fallback，适合个人体验，但企业口径下不能视为强隔离。
- 缺少 egress allowlist、secret broker、credential lease/revoke、snapshot/restore、artifact export、forensic bundle。
- 缺少 sandbox provider、镜像 digest、资源 request/hard limit 与并发记录。

### 4.7 Session、TUI 与 personal assistant 主路径

HaAgent 的产品方向不是 Codex clone，而是本地个人 AI 助手。当前实现对此比较一致：

- `haagent` 打开 TUI。
- `AgentSession` 管理多轮会话、turns、working state、session compaction、resume。
- TUI 通过 `AssistantService` 驱动 runtime。
- session 不把完整 episode trace 复制进模型输入，而是使用 summary/working state。

这符合“不要增加用户心智负担、不要增加模型输入 token”的产品约束。

不足：

- 普通用户体验优先导致 task contract 偏薄：自然语言直接进入 goal，验收和验证弱。
- TUI 中的 approval/human interaction 已存在事件，但干预记录尚未形成审计报告和后续 eval 语料。
- 长任务的四文件模式 Spec/Plan/Runbook/Status 尚未成为 runtime 原生工作流。

## 5. 11 项 Harness 职责成熟度评估

| 职责 | 当前成熟度 | 证据 | 主要欠缺 |
| --- | --- | --- | --- |
| Task Specification | H1+ | `TaskSpec` 有 goal/constraints/tools/acceptance/verification/policy；TUI 会写临时 task yaml | 自然语言 contract 默认过薄；缺非目标、风险、done-when、澄清、验证编译 |
| Context Selection | H2- | `ContextBuilder`、context manifest、预算、compaction、memory/skills/prompt packs | 事实源不统一；缺文档 freshness/ownership/索引校验 |
| Tool Access | H2 | `ToolRouter` schema/policy/approval/guardrail/trace/workspace/MCP | 缺 source-sink、企业 RBAC、远程 MCP 授权生产化 |
| Project Memory | H1+/H2- | memory candidate/retrieval/navigation、session working state | 长期记忆与项目知识库治理未完全闭环；缺质量与冲突运营 |
| Task State | H2- | `AgentSession`、working_state、turn summaries、episode status | 缺复杂任务里程碑、handoff artifact、计划审批状态机 |
| Observability | H2- | transcript/tool-calls/context/sandbox/failure/verification package | 缺 token/cost、metrics、dashboard、trace 聚合、版本维度 |
| Failure Attribution | H1+ | `failure.json`、failure category、validator | 缺真实 trace 阅读流程、owner、回流为 eval/策略/文档 |
| Verification | H1+ | declared command runner、verification trace | 普通 turn 默认无验证；缺 grader 框架、oracle、artifact checks |
| Permissions | H2- | policy approval、path policy、high-risk denial、interactive approval | 缺 actor/env/data source/sink/tenant 维度；local sandbox 默认弱 |
| Entropy Auditing | H0/H1 | 有 active-rules-summary，但无自动审计 | 无文档漂移检查、质量评分、规则 freshness、架构漂移检测 |
| Intervention Recording | H1 | human interaction events 进入 transcript/tool trace | 缺独立 intervention log、审批理由汇总、人类修改对照和审计查询 |

综合判断：**HaAgent 是 H1/H2 混合态。若只评价个人本地助手，核心 loop 已经可用；若按企业级 Harness Engineering 评估，仍缺少 H2 的验证强度和 H3 的持续治理闭环。**

## 6. 具体欠缺与不足

### 6.1 顶层知识库治理缺口

现状：

- 用户已明确指出 `AGENTS.md` 内容过时。
- `AGENTS.md` 引用的 `docs/harness-requirements.md`、`docs/code-governance.md`、`docs/unresolved-risks-and-roadmap.md` 当前不存在。
- `docs/` 顶层只有 `active-rules-summary.md` 和 `superpowers/` 计划规格文档。
- `src/haagent/docs/` 中有 `TASK_POLICY.md`、`EVAL_CASE_SCHEMA.md`、real LLM smoke/dogfood 文档，但不在顶层知识地图中。

影响：

- 新 agent 进入项目时无法可靠判断哪个文档是事实源。
- 运行时、eval、policy 等关键规则藏在 package 目录，降低 agent-legibility。
- 文档缺口会污染 ContextBuilder 的上游事实，即使 context assembly 机制正确，也可能输入过时知识。

建议：

- 重建顶层 `docs/` system of record：`ARCHITECTURE.md`、`HARNESS_REQUIREMENTS.md`、`RUNTIME_CONTRACTS.md`、`SECURITY.md`、`EVALS.md`、`QUALITY_SCORE.md`、`ROADMAP.md`。
- 将 `src/haagent/docs/` 的稳定规范迁移或镜像到顶层 docs，并在源码包中只保留面向 package 分发的短入口。
- 添加 docs validator：检查 `AGENTS.md` 链接存在、docs 索引覆盖、过时引用、schema 文档和代码字段一致性。

优先级：P0。

### 6.2 自然语言任务缺少 contract compilation

现状：

`write_chat_task_yaml()` 把用户请求写入 `goal`，但默认：

```yaml
constraints: []
acceptance_criteria:
  - Complete the requested chat task.
verification_commands: []
```

影响：

- 普通 TUI 主路径是产品重点，但也是任务规格最弱的路径。
- 模型可能提前宣布完成，runtime 难以独立判断。
- 后续 eval export 虽能导出 task，但验收信息不足，难以生成高质量 regression case。

建议：

- 增加轻量 contract compiler，不增加用户负担：从 prompt、target paths、工具意图和项目类型生成结构化草案。
- 对低风险普通任务可保持自动执行，但 contract 至少包含：goal、non_goals、expected_outputs、risk_level、acceptance_criteria、verification_strategy。
- 当任务涉及文件修改、命令执行、外发数据、长期任务或不明确偏好时，触发结构化澄清或计划确认。
- 不要靠 prompt 指令修补，应由 task schema 和 runtime 状态表达。

优先级：P0/P1。

### 6.3 Verification 过度依赖任务显式命令

现状：

- `VerificationEngine` 只执行 `verification_commands`。
- TUI 普通任务默认没有 verification commands。
- 文件整理、文档编辑、CSV 分析等非代码任务缺少确定性验证策略。

影响：

- “完成”容易退化成模型自评。
- episode package 有 verification 位置，但很多真实任务会没有高信号证据。

建议：

- 建立 verification strategy registry：代码任务、文档任务、CSV/数据任务、文件组织任务分别有最小检查。
- 文件写入类任务至少记录 diff/file existence/content checks。
- 文档总结类任务记录 source coverage、引用文件列表、摘要长度/结构检查。
- 代码任务自动建议或要求运行项目已有测试命令，但需避免增加 token 和用户负担。
- 将 verification result 回流到 agent loop，而不是只作为末尾检查。

优先级：P1。

### 6.4 Eval 系统缺少数据集生命周期

现状：

- `examples/evals` 只有少量内置 case。
- runner 支持 deterministic model responses，但主要比较工具名、状态、最终文本、failure category、context expectation。
- `export_eval_case()` 能从 episode 导出，但导出结构还不是完整可运行 benchmark schema。

影响：

- 不能稳定回答“这次模型、prompt、工具、sandbox、context 改动是否退化”。
- 无法区分能力探索与回归门禁。
- 真实失败不会自然沉淀为组织知识。

建议：

- 定义 eval dataset lifecycle：candidate -> reviewed -> regression/capability -> retired。
- 每周/每阶段人工读 20-50 条 trace，建立 failure taxonomy。
- 增加 grader plugin 接口：command grader、artifact grader、schema grader、LLM rubric grader、人审。
- 支持重复运行和 pass@k/pass^k。
- eval report 必须记录模型、prompt/tool/skill 版本、sandbox、CPU、内存、网络、timeout、并发、日期。
- 将生产失败和 dogfood 失败优先转成 regression eval。

优先级：P1。

### 6.5 Sandbox 默认边界不足

现状：

- local subprocess 是默认降级后端。
- local metadata 明确显示 `network_policy=unrestricted`、`credential_policy=inherit_environment`、`isolation.no_new_privileges=false`。
- Docker backend 有较好隔离参数，但属于可选路径。

影响：

- 对个人本地助手可接受，但不能宣称企业级安全边界。
- agent 执行 shell/code 时仍可能继承过多环境能力。
- eval 结果会受宿主环境影响，不可复现。

建议：

- 将 sandbox mode 在 UI/episode 中明确展示为安全等级，而不是只写 metadata。
- 高风险任务默认要求 Docker 或等价隔离；不可用时显式拒绝或请求用户确认。
- 增加 egress policy、secret allowlist、环境变量最小注入、资源 hard limit、artifact export。
- eval 默认使用固定 sandbox profile，记录镜像 digest 和资源配置。

优先级：P1。

### 6.6 Source-sink 安全控制不足

现状：

- `web_fetch` 会把公网内容标记为外部内容。
- network guard 会拒绝 localhost、私网、metadata 等 URL。
- secret redaction、tool guardrail、policy approval 已存在。

不足：

- 外部内容的 trust level 没有贯穿到后续高风险 tool sink。
- episode 中没有记录“某个 shell/file_write/web/MCP 动作是否受外部内容影响”。
- MCP tool description、网页内容、项目文件、用户输入之间没有统一 trust boundary。

影响：

- prompt injection 防护仍偏局部。
- 当 agent 读取网页后写文件、执行命令或调用 MCP 时，审计人员难以判断风险来源。

建议：

- 为 context source、tool observation、memory item 增加 `trust_level` 和 `source_type`。
- 为高风险 sink 定义额外 policy：shell、file_write、apply_patch、MCP write/action、web external call、credential use。
- episode 记录 source-sink link：哪些不可信输入进入了本次决策上下文，后续触发了哪些 sink。
- 对外部内容影响高风险动作时，默认要求确认或独立验证。

优先级：P1/P2。

### 6.7 可观测性缺少运营指标

现状：

- trace 文件结构扎实。
- inspect/export 能读取 episode package。

不足：

- 缺少 token/cost 统计。
- 缺少模型版本、prompt 版本、tool 版本、skill 版本。
- 缺少跨 episode 聚合：成功率、失败分类、工具错误率、审批拒绝率、验证失败率、平均 turns、平均成本。
- 没有 trace viewer/dashboard 或导出到 OpenTelemetry/Langfuse/LangSmith 等后端的适配。

影响：

- 单次可复盘，批量不可治理。
- 改动收益难以量化。

建议：

- 扩展 `environment.json` 与新增 `cost.json`。
- 给 prompt packs、tools registry、skills registry、model profile 写版本或 hash。
- 增加 `haagent inspect --aggregate` 或 `haagent report`，先输出本地 Markdown/JSON 即可。
- 后续再考虑 OTel 或 Langfuse 集成，不要一开始引入平台复杂度。

优先级：P2。

### 6.8 Failure attribution 尚未形成学习闭环

现状：

- `failure.json` 有 category/stage/reason/evidence。
- validator 会校验 failure record。

不足：

- 没有固定流程要求读 trace、复盘失败、更新 docs/tools/policy/eval。
- failure category 与 eval dataset、roadmap、quality score 没有关联。
- 没有 owner、状态、重复出现次数、关闭条件。

影响：

- 失败可记录，但不一定转化为 harness 改进。

建议：

- 增加 `docs/FAILURE_TAXONOMY.md` 与 `docs/QUALITY_SCORE.md`。
- 定义失败处理流程：classify -> decide fix surface -> add eval/check -> close。
- 从 `failure.json` 批量生成失败周报。

优先级：P2。

### 6.9 Entropy auditing 缺位

现状：

- 有 `docs/active-rules-summary.md`，但它本身更像人工整理结果。
- 没有自动验证文档与代码一致。

影响：

- 当前 `AGENTS.md` 过时就是熵增审计缺位的直接证据。
- 后续 TUI、runtime、eval、memory 快速演进时，文档漂移会继续扩大。

建议：

- 新增 docs/harness lint：
  - `AGENTS.md` 链接必须存在。
  - docs index 必须覆盖顶层规范文档。
  - schema 文档字段必须与代码导出字段一致。
  - README 普通入口说明必须与 CLI parser 一致。
  - 过时入口、迁移 stub、已删除命令不得出现在 active docs 中。
- 将该检查纳入 `uv run haagent check`。

优先级：P0/P1。

### 6.10 人工干预记录不足

现状：

- human interaction request/response 进入 transcript。
- Tool approval 可记录 granted/missing/not_required。

不足：

- 没有独立 `intervention-log.md/jsonl`。
- 没有区分用户澄清、审批、拒绝、计划修改、手工修复、外部状态改变。
- 没有把人工介入作为质量信号统计。

影响：

- 长任务中“人在哪里纠偏”不可聚合。
- 后续很难优化什么时候问用户、什么时候自主研究。

建议：

- episode 增加 `intervention-log.jsonl`。
- 每条干预记录包含：type、stage、reason、request summary、response summary、affected tool/run state、whether resumed。
- eval export 包含 intervention summary。

优先级：P2。

## 7. 与项目产品方向的张力

HaAgent 的目标是本地个人 AI 助手，不是企业平台或 Codex clone。因此不能简单要求它立刻实现完整 H3 平台控制面。更合理的判断是：

- **个人助手主路径优先是正确的**：TUI-first、当前目录 workspace、低心智负担、薄上下文，符合项目方向。
- **Harness 能力不能反客为主**：eval、inspect、task.yaml、dogfood、sandbox 配置应服务稳定性，不应变成普通用户入口。
- **但 Harness 基础证据必须真实**：即使用户不理解 episode，也必须能在后台产生可验证、可复盘、可改进的数据。

因此，建议采用“前台轻、后台硬”的路线：

- 前台：继续让用户只看到 TUI、自然语言、少量审批、清晰失败状态。
- 后台：强化 task contract、verification、episode metadata、eval dataset、docs governance、sandbox profile。

## 8. 优先级路线图

### P0：立即修正事实源和文档漂移

1. 承认 `AGENTS.md` 过时，把它降级为短入口地图。
2. 新建或恢复当前真实存在的顶层 docs：
   - `docs/HARNESS_REQUIREMENTS.md`
   - `docs/RUNTIME_CONTRACTS.md`
   - `docs/SECURITY.md`
   - `docs/EVALS.md`
   - `docs/ARCHITECTURE.md`
   - `docs/QUALITY_SCORE.md`
3. 将 `src/haagent/docs/TASK_POLICY.md` 和 `src/haagent/docs/EVAL_CASE_SCHEMA.md` 纳入顶层索引。
4. 增加 docs validator，并纳入 `haagent check`。

### P1：把普通 TUI turn 推到 H2-

1. 增加轻量 contract compiler。
2. 为常见任务类型生成 verification strategy。
3. 让文件变更类任务至少有 deterministic evidence。
4. 让 Docker sandbox 成为高风险任务的推荐或必需 profile。
5. 扩展 environment/cost metadata。

### P2：建设 eval 和失败学习闭环

1. 从真实 dogfood/失败 episode 抽取 20-50 个 eval candidates。
2. 区分 capability 与 regression。
3. 增加 grader plugin。
4. 增加 aggregate inspect/report。
5. 增加 intervention log。

### P3：向 H3 生产治理靠拢

1. eval 进入 CI/CD。
2. trace 导出到观测后端或本地 dashboard。
3. source-sink policy 完整落地。
4. sandbox 支持 snapshot、artifact export、egress allowlist、secret broker。
5. 引入定期 entropy audit 和质量评分更新。

## 9. 建议的评估目标状态

短期目标不是追求“企业级全功能”，而是把 HaAgent 稳定推进到 **个人助手场景下的 H2-**：

- 普通 TUI 任务自动产生足够清晰的 task contract。
- 每个 episode 都能说明：任务是什么、模型看见了什么、调用了什么工具、验证了什么、失败归因是什么、运行环境是什么。
- 文件和命令工具始终有明确 workspace/sandbox/policy 边界。
- 至少 20 个真实或拟真实 eval case 能在本地重复运行。
- 文档事实源可由 fresh agent session 可靠发现。
- 过时文档和 schema 漂移能被 `haagent check` 发现。

达成这些后，HaAgent 才适合继续投入更复杂的多 agent、自动化任务、长期记忆扩展或平台控制面能力。

## 10. 最终判断

HaAgent 的底层 runtime 已经有相当多正确的 Harness 原语，代码质量和测试覆盖也明显围绕 ToolRouter、ContextBuilder、EpisodeValidator、Sandbox、Eval runner 等关键边界展开。它的问题不是方向错误，而是 **已经实现的 Harness 原语还没有被统一成一套可治理的工程系统**。

换句话说，当前 HaAgent “会做事、会留痕、能部分验证”，但还没有充分做到“每次做事都能被契约约束、被独立验证、被批量评估、被安全边界治理、被失败闭环改进”。这正是下一阶段 Harness 工程最应该补齐的部分。
