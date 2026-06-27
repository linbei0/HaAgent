# HaAgent TUI 体验审计与改进建议

调研日期：2026-06-27

## 1. 结论摘要

当前 HaAgent TUI 已经完成“可运行的显式垂直切片”：它复用 `AssistantService`，用 Textual worker 承载长任务，支持基础对话、工具审批、补充输入、配置错误提示、失败摘要、记忆候选确认，并有一组较完整的 `run_test()` 自动化覆盖。这是很好的工程基础。

但从使用体验看，它仍更像“把运行状态搬进 Textual 的调试面板”，还没有形成成熟 TUI 应有的空间记忆、键盘流畅度、信息层级、状态可恢复性和可发现性。主要短板集中在：

1. 信息架构偏原始状态展示，缺少可扫读的任务时间线和主次层级。
2. 输入体验过弱，仍是单行 `Input`，和既有设计文档中的多行 `TextArea`、`Ctrl+Enter` 发送不一致。
3. 会话恢复、历史 session、搜索、命令面板、文件引用等 Agent 前端关键能力没有暴露到 TUI。
4. 工具执行只显示 started/done/failed 的粗粒度行，缺少进度、取消、详情、diff 和输出摘要查看。
5. 记忆候选模式目前没有上下选择能力，多候选时只能操作第一条，这是功能层面的明显缺口。
6. 帮助系统、视觉主题、响应式布局和可访问性仍停留在首版水平。

建议不要把 `haagent` 默认入口切到 TUI。应继续保持 `haagent tui` 为显式入口，先把它打磨成“个人助手会话工作台”的可靠前端，再单独评估是否提升默认入口优先级。

## 2. 审计范围

本次审计基于以下本地材料：

- `src/haagent/tui/app.py`：当前 Textual TUI 实现。
- `tests/test_tui_app.py`：当前 TUI 自动化测试。
- `src/haagent/app/assistant_service.py`：TUI 可用的服务层能力。
- `docs/tui-first-version-design.md`：首版界面方案。
- `docs/tui-technical-design.md`：TUI 技术设计。
- `docs/harness-requirements.md` 与 `docs/code-governance.md`：产品边界与代码治理。

外部参照：

- [lazygit keybindings](https://lazygit.dev/keybindings/)：多面板、上下文快捷键、`?` 帮助、搜索、撤销。
- [Yazi 官网](https://yazi-rs.github.io/)：异步 I/O、任务调度、实时进度、取消和预览。
- [Posting README](https://github.com/darrenburns/posting)：Textual TUI、jump mode、命令面板、主题、可配置键位。
- [Textual command palette](https://textual.textualize.io/guide/command_palette/) 与 [Textual workers](https://textual.textualize.io/guide/workers/)：命令发现、后台任务和线程安全更新。
- [Aider docs](https://aider.chat/docs/)：终端 AI 助手的 repo map、slash commands、lint/test、undo 和 Git 集成。
- [OpenCode TUI docs](https://opencode.ai/docs/tui/)：`@` 文件引用、slash commands、session 切换、undo、TUI 配置、主题、键位和鼠标设置。
- [Gemini CLI README](https://github.com/google-gemini/gemini-cli)：内置工具、MCP 扩展、终端优先的 Agent 工作流。

## 3. 项目边界判断

HaAgent 的产品方向不是 IDE，也不是纯代码 Agent。项目文档明确要求：

- 普通路径仍是 `haagent setup` 后在任意目录运行 `haagent`。
- 不把 GUI/TUI 作为普通默认路径，除非另有明确决策。
- TUI 不应绕过 `AgentSession`、`ModelGateway`、`ToolRouter`、workspace root 和 episode trace。
- 不把 harness、eval、dogfood 暴露成普通用户主路径。

因此，TUI 改进的合理方向不是做文件树 IDE，也不是复刻 Codex 或 OpenCode，而是成为“当前 workspace 上的个人助手会话工作台”：让用户看得懂当前上下文、任务进度、工具影响、失败原因、可恢复会话和需要自己决策的地方。

## 4. 当前已经做得好的部分

### 4.1 服务层边界正确

`HaAgentTuiApp` 通过 `AssistantService` 驱动会话，没有直接解析 CLI 文本输出，也没有绕过 runtime。`AssistantService` 已提供 profile 状态、session 创建/恢复、session 列表、事件流运行和记忆候选操作。这个分层和 `docs/tui-technical-design.md` 的方向一致。

### 4.2 长任务不会直接卡住 UI

`_run_prompt()` 使用 `@work(thread=True, exclusive=True)`，线程内通过 `call_from_thread()` 回到 UI 线程处理事件。测试也覆盖了 running 状态下 UI 不阻塞。

### 4.3 审批和补充输入保留同一 turn

工具审批通过 `ToolApprovalModal` 展示，用户补充输入通过 `_pending_interaction` 回传给同一个 `run_prompt_events()`。这避免了把“回答 Agent 的追问”误当成新 prompt。

### 4.4 安全意识已经进入 UI

审批摘要会通过 `_safe_summary()` 调用 `redact_secret_like_text()`，测试覆盖了 secret 不出现在 modal 渲染结果中。API key 缺失时也只显示环境变量名，不要求在 TUI 输入真实 key。

### 4.5 自动化测试基础扎实

`tests/test_tui_app.py` 覆盖了启动、配置缺失、API key 缺失、提交 prompt、审批允许/拒绝、用户补充输入、记忆候选、长回复换行、失败展示和 secret redaction。这比很多首版 TUI 更可靠。

## 5. 与成熟 TUI 的差距

### 5.1 空间记忆不足

lazygit、Posting、Yazi 这类成熟 TUI 的共同点是：核心区域位置稳定，用户能形成“某类信息总在某处”的记忆。HaAgent 当前有主对话区和右侧栏，但右侧栏主要是配置字段和最近工具行，缺少稳定的任务结构，例如：

- 当前任务阶段。
- 本轮工具时间线。
- 等待用户决策的事项。
- 最近 session。
- 当前 workspace 的关键上下文。

当前布局有框架，但信息还没有被组织成“工作台”。

### 5.2 键盘模型不完整

成熟 TUI 通常同时提供四层键盘体验：通用方向键、vim motion、上下文动作、命令面板或 slash command。HaAgent 当前有 `Enter`、`m`、`a/y/r`、`Esc`、`PgUp/PgDn`，但缺少：

- 列表上下移动，例如记忆候选的 `j/k`、方向键。
- `/` 搜索。
- `:` 或 `Ctrl+P` 命令面板。
- slash commands，例如 `/sessions`、`/resume`、`/new`、`/tools`。
- `@file` 文件引用。
- `Shift+Tab` 反向焦点。
- 当前模式下的详细帮助 overlay。

这会让功能扩展后迅速变得不可发现。

### 5.3 输入体验明显低于 Agent 类工具预期

当前使用 Textual `Input`，`Enter` 直接发送。既有设计文档建议使用 `TextArea`，`Enter` 换行，`Ctrl+Enter` 发送。对于个人助手任务，用户经常要输入多段说明、列表、文件名、约束和编辑要求，单行输入会很快变成瓶颈。

与 OpenCode 的 `@` 文件引用、Aider 的 slash commands、Posting 的 URL/命令导入相比，HaAgent 当前输入区还缺少“把本地上下文拉入 prompt”的低摩擦机制。

### 5.4 帮助系统会污染对话流

当前 `?` 会调用 `action_help()`，把帮助内容追加到 conversation。成熟 TUI 通常使用 overlay、command palette 或当前 panel 的上下文帮助。把帮助写进对话流会带来两个问题：

- 用户滚动历史时会混入非任务内容。
- 帮助内容不会随焦点和模式细分，长期会变成一段越来越拥挤的说明。

### 5.5 工具执行反馈太粗

当前 tool event 基本显示为：

- `Tool xxx ...`
- `Tool xxx done`
- `Tool xxx failed`

这不足以支撑真实个人助手任务。用户需要知道：

- 工具为什么运行。
- 正在处理哪个文件或命令。
- 已经完成多少。
- 是否可取消。
- 修改了哪些文件。
- 是否有 diff、stdout/stderr 摘要或后续可查看详情。

Yazi 的任务进度和取消、lazygit 的命令透明性、Aider 的 diff/undo 都说明：工具状态不是装饰，而是 Agent 信任感的核心。

### 5.6 会话恢复能力没有形成界面

`AssistantService` 已经有 `list_sessions()`、`resume_session()` 和 `continue_latest_session()`，但当前 TUI 没有 session 列表、恢复、新建、搜索或继续入口。对于 HaAgent 这种本地个人助手，会话恢复是核心体验，不应只停留在 CLI 或服务层。

### 5.7 记忆候选模式存在功能缺口

当前 `_memory_candidates` 有 `_memory_selected`，但没有看到上/下移动绑定或 action。多条候选出现时，用户无法选择第二条及之后的候选，只能确认/拒绝当前默认项。这是需要优先修复的功能问题。

同时，记忆模式会在窄屏下直接替换 conversation 内容。这保证了可读性，但也打断了用户对话上下文。更好的做法是用 overlay 或专门的 review pane，并保持返回位置。

### 5.8 视觉层级偏弱

当前 CSS 主要使用 `$surface`、`$primary` 和默认 Textual 样式。面板标题、状态、工具结果、失败、审批、记忆候选、焦点态之间缺少稳定的语义视觉体系。具体表现：

- 顶部状态栏是一条长字符串，字段权重接近。
- 右侧栏大量 `name: value` 文本，扫读成本高。
- 工具成功、失败、等待审批缺少符号和颜色组合。
- 英文和中文混杂，例如 `Tool Approval`、`Memory Candidates`、`Profile`。
- 侧栏 `Static` 可 focus，但缺少明显焦点提示。

视觉不需要华丽，但需要把状态和动作变得一眼可分。

### 5.9 响应式策略过于单一

当前宽度低于 120 时隐藏右侧栏。这个策略简单可靠，但还不够：

- 顶部状态栏在窄屏仍可能过长。
- 侧栏隐藏后，session、工具、失败摘要没有统一替代入口。
- 没有最小尺寸提示，例如低于 80x24 时显示 resize message。
- 没有 120/160/200 宽的渐进布局策略。

成熟 TUI 往往会使用 priority collapse、stacking、overlay 或 title-only panes，而不是只隐藏信息。

### 5.10 可访问性和终端兼容性未形成测试项

目前没有看到围绕以下场景的明确测试或设计约束：

- `NO_COLOR`。
- 浅色终端主题。
- Windows Terminal、tmux、SSH。
- 颜色不可作为唯一语义。
- 鼠标捕获与终端文本选择。
- 小尺寸终端。

HaAgent 是本地工具，终端环境差异会直接影响第一印象。

## 6. 与开源 AI 终端助手的差距

### 6.1 缺少文件引用和上下文选择

OpenCode 在 TUI 中支持 `@` 文件引用并进行 fuzzy search。Aider 通过命令参数和 `/add` 控制进入上下文的文件，同时有 repo map 辅助理解代码库。HaAgent 当前虽然有 workspace-bound 文件工具，但 TUI 输入区没有提供“我明确指这个文件/这些文件”的轻量入口。

建议新增：

- `@` 文件引用，基于 workspace root 做 fuzzy file search。
- `/add` 或 `/context` 查看本轮显式上下文。
- `/workspace` 查看当前边界和可访问范围。

这不要求把 TUI 做成文件管理器，只是降低自然语言和本地文件之间的摩擦。

### 6.2 缺少操作历史、撤销和变更可视化

Aider 和 OpenCode 都把文件变更可追踪、可撤销作为信任基础。HaAgent 有 episode trace 和工具记录，但 TUI 没有把“本轮改了什么、如何查看、如何回退”展示出来。

建议优先做非破坏性的可视化：

- 本轮 changed files 列表。
- 对 `apply_patch` 展示文件级摘要和可打开的 diff 详情。
- 若 workspace 是 Git repo，提示可用 Git 查看或回滚，但不要自动执行未请求的 Git 操作。
- 后续再评估 `/undo`，必须先定义 Git/non-Git workspace 的真实边界。

### 6.3 缺少命令体系

成熟 Agent CLI/TUI 往往有 slash commands 或命令面板，例如 `/help`、`/sessions`、`/tools`、`/memory`、`/settings`。HaAgent 当前把功能绑定在少量按键上，短期够用，长期不利于扩展。

建议采用“双入口”：

- `Ctrl+P` 或 `:` 打开 Textual command palette，用于发现和执行 UI 命令。
- 输入框内 `/` 开头作为 Agent/TUI 命令，例如 `/sessions`、`/new`、`/resume`、`/memory`、`/tools`、`/clear`。

命令应调用结构化 service 方法，不要靠解析用户自然语言。

### 6.4 缺少注意力管理

长任务、审批、补充问题、完成和失败都需要明确提醒。OpenCode 的 TUI 文档中把 attention 配置作为 TUI 行为的一部分。HaAgent 当前只有状态栏文字变化和聊天行追加，终端失焦时用户可能错过审批或完成。

建议分阶段：

- 先在 TUI 内做显著状态条和 pending badge。
- 再评估桌面通知或声音，默认关闭，配置开启。

## 7. 分级问题清单

### Critical

1. 记忆候选列表不可导航。
   - 位置：`src/haagent/tui/app.py` 的 memory mode。
   - 影响：多候选时无法选择目标，确认/拒绝只能操作第一项。
   - 建议：增加 `up/down`、`j/k`、`g/G` 列表导航，更新 footer，并补多候选测试。

### High

1. 输入区不支持多行和 `Ctrl+Enter`。
   - 影响：复杂任务输入成本高，和设计文档不一致。
   - 建议：从 `Input` 迁移到 `TextArea`，保留补充输入状态，添加空输入、换行、提交测试。

2. 会话管理没有 TUI 界面。
   - 影响：用户无法在 TUI 中恢复上下文，多轮个人助手价值被削弱。
   - 建议：实现 sessions overlay，支持列出、搜索、恢复、继续最新、新建。

3. 工具进度和修改摘要不足。
   - 影响：用户难以判断 Agent 是否可靠、是否卡住、是否改了文件。
   - 建议：右侧栏升级为 tool timeline，支持当前工具详情 overlay 和取消入口。

4. 帮助不是 overlay。
   - 影响：污染对话流，且无法按当前模式上下文化。
   - 建议：`?` 打开帮助 modal，显示当前焦点可用动作；保留 `--help` 做完整参考。

5. 无搜索、过滤、命令面板。
   - 影响：历史对话、工具列表和 session 增长后不可管理。
   - 建议：先做 `/` 当前面板搜索，再接 Textual command palette。

### Medium

1. 状态栏和侧栏信息层级弱。
   - 建议：顶部只保留 workspace basename、profile、model、state、session 短 id；完整信息放详情。

2. 响应式策略只有隐藏侧栏。
   - 建议：增加 80x24 最小门槛、窄屏 overlay、宽屏更丰富侧栏、超宽不继续拉长正文。

3. 视觉系统未语义化。
   - 建议：建立 TUI theme token：default/muted/emphasis/success/warning/error/info/selection/focus。

4. 中英混排不统一。
   - 建议：面向中文项目默认中文化 UI 文案，保留工具名和协议字段原文。

5. 侧栏 focus 缺少明显反馈。
   - 建议：焦点边框、标题高亮或 footer 上下文化。

6. `q` 作为全局退出可能导致误触。
   - 建议：默认显示和主推 `Ctrl+Q`，评估 `q` 是否只在非输入焦点或确认后退出。

### Low

1. 缺少主题配置和 `NO_COLOR` 验证。
2. 缺少浅色终端和 Windows Terminal/tmux/SSH 兼容性清单。
3. 缺少外部编辑器入口。
4. 缺少任务完成后的轻量通知或状态闪烁。

## 8. 改进路线图

### P0：修复可用性硬伤

目标：不改变架构，只让现有 TUI 变得可靠。

- 增加记忆候选多项导航和测试。
- 将 `?` 改为帮助 modal，不再写入 conversation。
- 梳理全局 `q` 行为，避免输入时误退出。
- 压缩状态栏，长 workspace/model/session 做截断。
- 增加最小尺寸提示和 80x24/120x40/200x60 snapshot 测试。

### P1：把会话和输入体验补齐

目标：让 TUI 成为可日常使用的个人助手前端。

- `Input` 迁移到 `TextArea`，`Ctrl+Enter` 发送，`Enter` 换行。
- 增加 sessions overlay：搜索、恢复、继续最新、新建。
- 增加 `/` 搜索当前对话和工具列表。
- 增加命令面板或 slash command 的最小集合：`/help`、`/sessions`、`/memory`、`/tools`、`/new`、`/resume`。
- 增加 `@file` fuzzy reference，但只产生结构化引用，不扩大 workspace 边界。

### P2：提高 Agent 信任感

目标：让用户看懂 Agent 正在做什么、做了什么、哪里失败。

- 右侧栏改为任务状态工作台：阶段、工具时间线、pending decision、changed files、last failure。
- 工具详情 overlay：参数摘要、影响范围、stdout/stderr 摘要、episode path。
- 对文件修改展示 changed files 和 diff 摘要。
- 支持取消当前任务。取消应走 runtime/service 的显式状态，不靠吞异常。
- 失败块给出保守下一步，但不做猜测式自动诊断。

### P3：建立视觉和主题系统

目标：让 TUI 从“能用”变成“清爽、稳定、可扫读”。

- 建立语义 token，并统一状态颜色和符号。
- 给面板标题、焦点态、选中态、危险操作、成功/失败建立一致样式。
- 支持 `NO_COLOR` 下可读。
- 增加暗色/浅色主题，先少量内置，不做主题市场。
- 中文文案统一，减少首屏概念负担。

### P4：谨慎引入高级能力

目标：只加入符合 HaAgent 产品方向的增强。

- Git repo 内可考虑 `/undo`，但必须明确非 Git workspace 的行为。
- 外部编辑器编辑长 prompt。
- 可选桌面通知，默认关闭。
- MCP/tool 能力列表查看，但不做复杂插件市场。
- 只读分析模式或安全模式，可作为个人助手任务的安全边界，不做多 Agent 系统。

## 9. 不建议做的事

1. 不建议把 TUI 立刻改成默认入口。
   - 当前还缺少输入、session、工具详情、响应式和帮助系统。默认入口切换会增加普通用户风险。

2. 不建议把 TUI 改造成 IDE。
   - HaAgent 的目标是本地个人助手。文件树、编辑器、多标签工作台可以作为辅助能力，但不应成为主界面。

3. 不建议用自然语言匹配实现 slash commands 或安全边界。
   - 命令、工具、session、workspace 都应走结构化 service 方法和明确状态字段。

4. 不建议为了视觉效果大量加入装饰。
   - TUI 的美观来自信息密度、空间稳定、语义颜色和键盘流畅度，不来自边框堆叠。

5. 不建议把完整 stdout、patch、episode trace 默认塞进主对话。
   - 默认展示摘要，详情按需打开，避免增加用户心智负担和模型输入 token 诱因。

## 10. 建议的验收标准

完成 P0/P1 后，TUI 至少应满足：

- 80x24 下能完成输入、阅读、审批、查看失败和退出。
- 120x40 下能同时看到对话、状态、工具时间线和 session 摘要。
- 200x60 下不会把正文无限拉长，而是增加有用状态密度。
- 所有功能键盘可达，`?` 显示当前模式帮助。
- 支持多行输入，`Ctrl+Enter` 提交。
- 支持 session 列表、恢复和继续最新。
- 支持多候选记忆审查。
- 支持当前对话搜索。
- 审批 modal 默认焦点仍在拒绝。
- secret 不出现在 UI snapshot。
- Textual `run_test()` 覆盖核心交互，至少包含 80x24、120x40、窄屏侧栏折叠、多候选、session overlay 和帮助 modal。

## 11. 优先任务拆分建议

第一批小任务：

1. 修复 memory candidate 列表导航。
2. 帮助改为 modal，并加当前模式 footer。
3. 状态栏字段压缩和截断。
4. `q` 退出行为收敛。
5. 增加最小尺寸提示。

第二批中等任务：

1. 输入区迁移到 `TextArea`。
2. sessions overlay。
3. 搜索 overlay。
4. tool timeline 侧栏。
5. 工具详情 modal。

第三批体验任务：

1. 命令面板或 slash command。
2. `@file` fuzzy reference。
3. 语义主题 token。
4. changed files/diff 摘要。
5. 可取消当前任务。

这条路线能最大限度复用当前已有的 `AssistantService` 和测试基础，也符合项目“不增加普通用户心智负担、不把 TUI 变成未决策默认主线”的约束。
