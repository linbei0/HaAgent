# HaAgent TUI 首版界面方案

## 1. 信息架构

首版采用“顶部状态栏 + 主对话区 + 可折叠右侧栏 + 输入区 + 快捷键 footer”。核心视图只有一个：个人助手会话。HaAgent 不是 IDE，首版不做代码编辑器、文件树主界面或复杂多标签工作台。

信息优先级：

1. 当前上下文：workspace、profile、model、session、运行状态。
2. 对话流：user / assistant / tool / failure。
3. 可恢复状态：最近 sessions、当前 turn、工具进度。
4. 配置健康度：profile 是否存在、API key env 是否存在。
5. 低频操作：帮助、查看摘要、切换/恢复 session。

## 2. 布局草图

### 2.1 80x24

80x24 使用单栏布局，右侧栏折叠为状态摘要行，保证最小终端尺寸下仍可完成输入、阅读和处理审批。

```text
┌ HaAgent ─ workspace: ...\project  profile: deepseek  key: missing ┐
│ session: session-abc  turn: 2  state: idle                         │
├────────────────────────────────────────────────────────────────────┤
│ You                                                                │
│   整理这个目录里的会议纪要                                           │
│                                                                    │
│ Assistant                                                          │
│   我会先查看当前目录结构，然后总结可整理的文件。                       │
│                                                                    │
│ Tool file_list ✓  . 12 items                                       │
│ Tool file_read ✓  notes/weekly.md  3.2KB                           │
│                                                                    │
│ Failure ! model_call                                               │
│   当前 provider/base_url/model 组合可能不匹配。                     │
├────────────────────────────────────────────────────────────────────┤
│ > 输入消息，Ctrl+Enter 发送                                         │
│   多行输入区域，Esc 清空当前编辑                                    │
├────────────────────────────────────────────────────────────────────┤
│ [Ctrl+Enter]发送 [Tab]焦点 [?]帮助 [/]搜索 [Esc]取消 [q]退出        │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 120x40

120x40 是首版推荐默认形态：左侧为主对话，右侧为状态和会话侧栏。

```text
┌ HaAgent  ws: E:\work\docs  profile: deepseek  openai-chat/deepseek-chat  key: ok  state: running ┐
├──────────────────────────────────────────────────────────────┬────────────────────────────────────┤
│ Conversation                                                 │ Status                             │
│                                                              │ session                            │
│ You                                                          │   session-abc                      │
│   把这些资料整理成摘要。                                      │   turns: 3                         │
│                                                              │   latest: 2026-06-24 14:12         │
│ Assistant                                                    │                                    │
│   我会读取 Markdown 和 CSV，先给你一版结构化摘要。              │ profile                            │
│                                                              │   name: deepseek                   │
│ Tool file_search ✓                                           │   provider: openai-chat            │
│   pattern: *.md  matches: 8                                  │   model: deepseek-chat             │
│ Tool file_read …                                             │   api_key_env: DEEPSEEK_API_KEY    │
│   notes/q2.md                                                │   key: available                   │
│                                                              │                                    │
│ Assistant                                                    │ tools                              │
│   已找到 8 份资料，按主题分为三组：...                         │   file_search ✓                    │
│                                                              │   file_read running                │
│                                                              │   shell pending approval           │
│                                                              │                                    │
│                                                              │ sessions                           │
│                                                              │ > session-abc  3 turns             │
│                                                              │   session-old  1 turn              │
├──────────────────────────────────────────────────────────────┴────────────────────────────────────┤
│ > 输入消息...                                                                                     │
│   Ctrl+Enter 发送；Enter 换行                                                                     │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Ctrl+Enter]发送 [Tab]切换焦点 [?]帮助 [/]搜索 [s]sessions [n]新会话 [Esc]取消 [q]退出              │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 200x60

200x60 可以展示更完整的状态，但仍保持个人助手形态，不扩展成 IDE。

```text
┌ HaAgent ─ workspace: E:\python-project\HaAgent ─ profile: deepseek ─ provider: openai-chat ─ model: deepseek-chat ─ key: ok ─ session: session-abc ─ state: idle ┐
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┤
│ Conversation                                                                                                  │ Workspace                                            │
│                                                                                                               │   E:\python-project\HaAgent                          │
│ You                                                                                                           │   runs: .runs                                        │
│   设计 TUI 首版界面。                                                                                          │                                                      │
│                                                                                                               │ Profile                                              │
│ Assistant                                                                                                     │   name: deepseek                                     │
│   首版应该围绕当前目录的自然语言助手体验，而不是 IDE。                                                           │   provider: openai-chat                              │
│                                                                                                               │   base_url: https://api.deepseek.com                 │
│ Tool file_search ✓                                                                                             │   model: deepseek-chat                               │
│   query: "AssistantService"                                                                                    │   api_key_env: DEEPSEEK_API_KEY                      │
│   result: 4 exact matches                                                                                      │   key: available                                     │
│                                                                                                               │                                                      │
│ Tool apply_patch pending approval                                                                              │ Current Session                                      │
│   action: modify  files: 1                                                                                     │   id: session-abc                                    │
│   scope: inside workspace                                                                                      │   turns: 4                                           │
│                                                                                                               │   updated: 14:12                                     │
│ Assistant                                                                                                     │                                                      │
│   如果你批准，我会修改 docs/tui-plan.md。                                                                       │ Tools This Turn                                      │
│                                                                                                               │   file_search ✓                                      │
│                                                                                                               │   apply_patch approval                               │
│                                                                                                               │                                                      │
│                                                                                                               │ Recent Sessions                                      │
│                                                                                                               │ > session-abc  TUI design                            │
│                                                                                                               │   session-090  summarize docs                        │
│                                                                                                               │   session-071  inspect CSV                           │
│                                                                                                               │                                                      │
│                                                                                                               │ Help                                                 │
│                                                                                                               │   Tab focus  ? help  / search                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┤
│ > 输入给 HaAgent 的下一步请求。Enter 换行，Ctrl+Enter 发送。                                                                                                           │
│                                                                                                                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [Ctrl+Enter]发送 [Tab]焦点 [Shift+Tab]反向 [/]搜索 [?]帮助 [s]会话 [r]恢复 [n]新会话 [Esc]取消/关闭 [q]退出                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. 顶部状态栏字段

80 宽时压缩显示：

- `workspace`：只显示 basename 或尾部路径。
- `profile`：profile name；缺失显示 `profile: missing`。
- `key`：`ok` / `missing`，不显示真实 API key。
- `state`：`idle` / `running` / `waiting approval` / `waiting input` / `failed`。

120 宽及以上显示：

- `workspace_root`
- `profile_name`
- `provider`
- `model`
- `api_key_env` 状态：`key: ok/missing`
- `session_id` 短 id
- `turn_count`
- `state`

200 宽可以补充：

- `base_url` 放侧栏展示，避免顶部过长。
- `runs_root` 放侧栏展示。

## 4. 主对话区事件展示

主对话区按 `ChatEvent.event_type` 做轻量映射：

- `turn_started`：显示用户消息块 `You`，内容是用户 prompt。
- `assistant_message`：显示 `Assistant`，支持 Markdown 渲染，但不做代码编辑体验。
- `tool_started`：单行进度，例如 `Tool file_read ... notes.md`。
- `tool_finished`：摘要行，例如 `Tool file_read ✓ notes.md 3.2KB`。
- `tool_failed`：错误摘要，例如 `Tool shell ! command failed`。
- `approval_requested`：主区显示 pending 行，同时弹审批 modal。
- `user_input_requested`：显示 assistant 需要补充信息，并把输入区切到回答状态。
- `failure`：错误块，展示 `failed_stage`、`failure_category`、`reason`、`episode_path` 摘要。
- `turn_finished`：不单独刷屏，只更新状态栏和侧栏。

默认不展示完整 transcript、完整 tool output、完整 episode trace。需要详情时可以通过 `Enter` 或 `v` 查看当前事件详情摘要，但首版详情也要有长度限制。

## 5. 右侧栏

右侧栏在 120 宽以上显示；80 宽时折叠，通过 `s` 打开 sessions overlay，通过 `?` 查看帮助。

右侧栏分四块：

### Session

- 当前 `session_id`
- `turn_count`
- `updated_at`
- `first_request` 摘要
- 最近 sessions 列表，当前项高亮

### Profile

- `profile_name`
- `provider`
- `base_url`
- `model`
- `api_key_env`
- `api_key_available`
- 如果缺失，显示明确状态，不要求输入 key

### Tools This Turn

- 工具名
- 状态：pending / running / approval / done / failed
- 关键参数摘要：文件路径、命令名、匹配数、写入文件数
- 不展示完整 stdout/stderr

### Workspace

- `workspace_root`
- `runs_root`
- 当前运行状态
- 失败时显示 `episode_path` 的短路径

## 6. 输入区和快捷键 Footer

输入区：

- 多行输入，`Enter` 换行。
- `Ctrl+Enter` 发送。
- 空输入不提交。
- `Esc`：运行中为取消或关闭 modal；编辑中为清空当前草稿或退出当前模式。
- 当 runtime 请求用户补充信息时，输入区标题变为 `Answer required`，提交后继续同一个 turn。

Footer 常驻只放当前可用动作：

- 默认：`[Ctrl+Enter]发送 [Tab]焦点 [?]帮助 [/]搜索 [s]会话 [q]退出`
- 运行中：`[Esc]取消 [Tab]焦点 [?]帮助`
- modal 中：`[y]允许 [n]拒绝 [Tab]切换 [Esc]关闭`
- sessions 焦点：`[Enter]恢复 [n]新会话 [/]搜索 [Esc]返回`

所有功能必须键盘可达，鼠标只作为增强。

## 7. 工具审批 Modal

```text
┌ Tool Approval ─────────────────────────────────────────────┐
│ 工具请求需要确认                                           │
│                                                            │
│ tool      apply_patch                                      │
│ action    修改文件                                         │
│ scope     当前 workspace 内                                │
│ target    docs/tui-plan.md                                 │
│ summary   将新增 TUI 首版方案文档                          │
│                                                            │
│ 风险提示：该操作会修改本地文件。                            │
│                                                            │
│        [ Allow y ]          [ Deny n ]                     │
└────────────────────────────────────────────────────────────┘
```

设计规则：

- modal 是 focus trap，背景不可操作。
- 文件修改、命令执行等高影响操作默认焦点放在 `Deny`。
- 显示工具名、影响范围、关键参数摘要。
- 不显示完整 patch/output，提供 `v` 查看摘要详情。
- 审批结果必须回到同一个 turn，不能变成新 prompt。

## 8. 状态展示文案

### API key 不可用

```text
无法开始运行：API key 不可用

当前 profile 使用 api_key_env=DEEPSEEK_API_KEY。
API key 解析优先级是环境变量 > 系统凭据库 > 明文用户文件。
HaAgent 不会在 TUI 中输入、保存或显示真实 API key。

请先设置环境变量，或在 TUI 内通过 `/model` 重新保存到系统凭据库后重试。
```

### Profile 缺失

```text
未找到默认模型配置

请在 TUI 内打开：
  /model

配置完成后即可继续当前会话。
```

### 运行失败

```text
本轮任务失败

阶段：model_call
类型：provider_error
原因：HTTP 404

当前 provider/base_url/model 组合可能不匹配。
请检查 profile 配置；详细运行记录已写入 episode。
```

如果失败不是 provider 配置问题，不做过度推断；只展示 runtime 给出的 `failed_stage`、`failure_category`、`reason`，再给最保守的下一步建议。

## 9. 颜色语义表

widget 代码不硬编码具体颜色，只引用语义 token。

| Token | 用途 |
| --- | --- |
| `fg.default` | 普通正文、assistant 输出 |
| `fg.muted` | 时间、路径、session id、次要 metadata |
| `fg.emphasis` | 面板标题、当前焦点、重要字段 |
| `bg.base` | 应用基础背景 |
| `bg.surface` | 面板背景 |
| `bg.overlay` | modal 背景 |
| `bg.selection` | 当前选中 session / 按钮 |
| `accent.primary` | 焦点边框、可操作按钮、当前输入状态 |
| `status.info` | running、tool started、普通提示 |
| `status.success` | tool finished、key available、turn completed |
| `status.warning` | waiting approval、API key missing、需要用户输入 |
| `status.error` | failure、tool failed、profile missing |

颜色不能单独承载意义：同时配合文字标签、符号、位置和粗细。支持 `NO_COLOR` 时仍应靠文本和布局可读。

## 10. 首版不做

- 不做 IDE，不做代码编辑器，不做文件树主界面。
- 不做完整 transcript 浏览器。
- 不默认展示完整 tool output、stdout、stderr、patch、episode trace。
- 不做 harness/eval/dogfood 普通用户入口。
- 不做多 Agent 编排。
- 不做长期记忆。
- 不做插件市场或复杂主题市场。
- 不做 Web UI / Electron / 桌面 App。
- 不在 TUI 中输入、保存、复制或展示真实 API key。
- 不做自动安装依赖。
- 不做复杂命令面板；`?` 帮助和少量快捷键足够首版。
