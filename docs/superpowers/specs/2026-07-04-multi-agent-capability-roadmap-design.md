# HaAgent 多智能体能力三阶段设计

## 背景

HaAgent 已经有轻量多智能体基础：主 Agent 可以通过 `agent` 工具启动后台 worker，worker 由 `MultiAgentRuntime` 管理，状态和通知写入 `TeamStore`，工具权限由 `worker_tool_policy` 按 `explorer`、`worker`、`verification` 三类裁剪。

当前不足不是“没有多智能体”，而是能力还停留在“能派一个后台帮手做事”。相比 OpenHarness 的 swarm/coordinator 体系，HaAgent 缺少可配置的 agent profile、运行中通信、结构化权限请求，以及更强的进程和工作区隔离。

本设计把能力分三阶段推进：

1. 做实 `model_profile` 和自定义 agent profile。
2. 补结构化 worker 通知、运行中消息和权限请求。
3. 再考虑 subprocess 和 worktree 隔离。

## 产品原则

- HaAgent 仍然是 TUI-first 的本地个人助手，不变成复杂团队编排平台。
- 普通用户不需要理解“多智能体框架”才能获益。
- 多智能体能力默认隐藏在主 Agent 的自然语言任务执行里，TUI 只展示必要状态。
- 不增加模型输入 token 的常驻负担；只有在主 Agent 需要派 worker 时才加载精简 profile 摘要。
- 所有模型调用继续走 `ModelGateway`，所有工具调用继续走 `ToolRouter`。
- worker 的文件修改和命令执行仍受 workspace root、path policy、approval policy 约束。

## 非目标

- 不在第一阶段引入 subprocess、tmux、远程 agent、浏览器自动化或大型 swarm UI。
- 不把 OpenHarness 的完整 coordinator mode 直接搬进 HaAgent。
- 不让用户手动管理复杂团队、队列、分支拓扑。
- 不为历史 `.runs` 或旧 team 文件增加兼容层；HaAgent 当前仍是 pre-1.0。

## 推荐路线

### 阶段一：做实 model_profile 和自定义 agent profile

`model_profile` 是“这个 worker 用哪个模型连接配置”。例如主会话用默认模型，验证 worker 用更便宜快的模型，复杂代码审查 worker 用更强模型。

`agent profile` 是“这个 worker 是哪类帮手”。它描述这个帮手适合什么任务、默认提示、可用工具、默认模型、最大轮数、是否允许联网、是否允许修改文件。

第一阶段的核心收益是把“派谁去做事”变成稳定配置，而不是只靠主 Agent 临时写 prompt。

建议新增一个小而深的模块：

- `src/haagent/multi_agent/profiles.py`
  - 读取内置 agent profile。
  - 读取用户自定义 agent profile。
  - 校验 profile 字段。
  - 把 profile 解析成 worker 创建所需的明确配置。

内置 profile 只保留少量高价值角色：

- `explorer`：只读探索和总结。
- `worker`：可按父会话策略执行实现任务。
- `verification`：运行验证和读取结果。

允许用户在后续版本增加自定义 profile，但第一版先支持文件加载和校验，不急着做复杂 TUI 编辑器。

### 阶段二：结构化通知、运行中消息和权限请求

当前 worker 更像“派出去后等它回来”。阶段二让 worker 更像一个会汇报和请示的帮手。

结构化通知统一为明确字段：

- `event_type`
- `team_id`
- `agent_id`
- `task_id`
- `status`
- `summary`
- `result_excerpt`
- `episode_path`
- `error`
- `needs_attention`

运行中消息要解决两个问题：

- 主 Agent 能给已运行 worker 补充上下文或改变方向。
- worker 能在 turn 边界读取 mailbox，而不是必须结束后重启。

权限请求要解决“worker 想做高风险操作时怎么问”的问题。第一版不做复杂权限 UI，只要求有结构化请求记录、主 Agent 可读、用户可通过现有 human interaction 流程批准或拒绝。

建议新增或扩展：

- `src/haagent/multi_agent/messages.py`
  - 定义 worker 消息、通知、权限请求的数据结构。
- `src/haagent/multi_agent/team_store.py`
  - 增加 pending message / permission request 的读写方法。
- `src/haagent/multi_agent/runtime.py`
  - worker turn 开始前读取 mailbox。
  - 允许运行中 worker 在安全点接收消息。
  - 写入结构化通知。

阶段二的收益是主 Agent 能指挥、纠偏、收敛结果，而不是只靠后台任务最终输出。

### 阶段三：subprocess 和 worktree 隔离

阶段三服务于更重的代码任务。它不是多智能体的第一步，而是当 worker 真正开始并行修改代码后才有价值。

`subprocess` 隔离表示每个 worker 跑在独立进程里。一个 worker 卡住、崩溃或占用资源，不应拖垮主会话。

`worktree` 隔离表示给代码 worker 创建独立 Git worktree。不同 worker 可以在不同目录修改代码，最后由主 Agent 或用户决定是否合并。

建议新增：

- `src/haagent/multi_agent/backends.py`
  - 定义 `WorkerBackend` 接口。
  - 保留当前 in-process 实现。
  - 后续增加 subprocess 实现。
- `src/haagent/multi_agent/worktree.py`
  - 只负责 Git worktree 创建、校验、清理。

阶段三的收益是并行代码工作更安全，但也最容易增加复杂度，所以必须放在阶段一、二之后。

## 方案选择

### 方案 A：轻量渐进路线（推荐）

先做 profile，再做通信和权限，再做隔离。每一步都能独立带来收益，也能保持 HaAgent 的个人助手定位。

优点：

- 不显著增加用户心智负担。
- 不需要马上改动进程模型。
- 更容易用现有 `AgentSession`、`ToolRouter`、`TeamStore` 测试。

缺点：

- 初期不支持真正强隔离的并行代码修改。

### 方案 B：直接移植 OpenHarness swarm

一次性引入 agent definitions、subprocess backend、team lifecycle、permission sync、worktree。

优点：

- 能快速接近 OpenHarness 的多智能体表面能力。

缺点：

- 会明显偏离 HaAgent 当前 TUI-first 个人助手路线。
- 会增加大量配置、状态、测试和 UI 复杂度。
- 很容易把 harness 复杂度暴露给普通用户。

### 方案 C：只做提示词和内置角色

只在 prompt 里告诉模型可以派 explorer/worker/verification，不新增 profile 存储和通信协议。

优点：

- 改动最少。

缺点：

- 依赖模型临时理解，行为不稳定。
- 很难测试。
- 违反项目“不要用 prompt 补 runtime 边界”的规则。

## 数据模型草案

### AgentProfile

```python
@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    subagent_type: str
    system_prompt: str
    allowed_tools: list[str] | None
    approval_allowed_tools: list[str] | None
    approved_tools: list[str] | None
    model_profile: str | None
    max_turns: int | None
    enable_web: bool | None
```

字段解释：

- `name` 是模型和用户看到的稳定名称。
- `subagent_type` 继续映射到现有 `worker_tool_policy`，避免第一阶段扩大工具权限面。
- `system_prompt` 是 profile 专属的短提示，只注入 worker，不注入主会话。
- `model_profile` 指向现有 provider profile 名称。

### WorkerNotification

```python
@dataclass(frozen=True)
class WorkerNotification:
    event_type: str
    team_id: str
    agent_id: str
    task_id: str
    status: str
    summary: str
    result_excerpt: str
    episode_path: str
    error: str
    needs_attention: bool
```

### WorkerPermissionRequest

```python
@dataclass(frozen=True)
class WorkerPermissionRequest:
    request_id: str
    team_id: str
    agent_id: str
    task_id: str
    tool_name: str
    tool_args_summary: str
    reason: str
    status: str
```

## 用户体验

普通用户不需要直接配置多智能体。主 Agent 可以自然地说：

- “我先派一个只读助手检查项目结构。”
- “验证助手正在跑测试。”
- “代码助手需要修改文件，我会先问你。”

TUI 只需要展示简洁状态：

- worker 名称
- 当前状态
- 一句话摘要
- 需要用户确认的请求

更高级的自定义 agent profile 可以先通过配置文件支持，后续再考虑 `/agents` 或 `/model` 子页面整合。

## 测试策略

阶段一测试重点：

- 加载内置 profile。
- 加载用户 profile。
- 无效字段显式失败。
- `model_profile` 能让 worker 使用对应模型 gateway。
- 未配置模型 profile 时继承主会话。

阶段二测试重点：

- worker 完成、失败、停止都会写结构化通知。
- 主 Agent 能读取 worker 通知，但不会把完整 worker 输出塞进模型输入。
- 运行中消息在 worker 安全点被读取。
- 权限请求写入后可被主 Agent 或 human interaction 处理。

阶段三测试重点：

- backend 接口不破坏现有 in-process worker。
- subprocess worker 可被查询、停止、读取输出。
- worktree 路径校验阻止路径穿越。
- worker 修改只发生在自己的 worktree。

## 成功标准

- 主 Agent 能按 profile 派出不同用途的 worker。
- worker 能使用指定 `model_profile`，未指定时继承主会话模型。
- worker 的通知结构稳定，可被 TUI 和主 Agent 消费。
- 主 Agent 可以向 worker 发送后续消息，并在安全点生效。
- 高风险 worker 操作能形成结构化权限请求。
- subprocess/worktree 能作为后续阶段加入，不要求前两阶段提前实现。

## 自检结果

- 未发现待填写内容。
- 三阶段顺序与 HaAgent 当前产品原则一致。
- 第一阶段不引入 subprocess/worktree，避免过早复杂化。
- 通知和权限请求通过结构化数据实现，不依赖提示词猜测。
