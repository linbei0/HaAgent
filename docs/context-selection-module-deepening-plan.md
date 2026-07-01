# Context Selection Module 深化方案

本文给出 HaAgent 后续开发更深的上下文选择 Module 的实用方案。目标不是做一个大型 RAG 系统，也不是把 prompt 拼装再包一层名字，而是用少量、集中、可测试的代码，把“每次模型调用前应该看见什么”变成可审计、可解释、可控预算的工程行为。

## 背景与目标

HaAgent 的产品目标是本地个人 AI 助手：用户进入任意目录运行 `haagent`，自然地完成文件整理、文档分析、项目理解、代码修改、联网查询和多轮任务。随着能力增加，模型输入会越来越容易混入多类上下文：

- 当前用户请求。
- 项目规则与 workspace 边界。
- session summary 与 working state。
- 长期记忆。
- 工具 schema 与工具 workflow 提示。
- 文件引用、搜索结果、工具观察。
- 人机交互状态、审批结果、取消/失败信息。

如果这些内容继续分散在 `ContextBuilder`、`MemoryRetriever`、`AgentSession`、tool result formatting 和 prompt builder 中独立拼接，短期能跑，长期会出现三个问题：

1. 模型输入为何变厚不可解释。
2. 错误回答无法复盘到底缺了哪段上下文或被哪段上下文污染。
3. 后续增加 repo map、联网 observation、记忆类型或文件引用时，会不断把选择逻辑散落到调用点。

本方案的目标是建立一个轻量但有深度的 `ContextSelection` Module：

- 对外提供一个小接口，调用方只需要给它 task/session/runtime 状态。
- 对内集中处理 source 收集、选择、预算、裁剪、manifest 和 diagnostics。
- 默认不增加普通用户心智负担。
- 默认不增加模型输入 token；任何新增上下文都必须有来源、原因和预算记录。
- 不引入 embedding、向量库、后台索引服务或复杂插件系统。

## 现状判断

当前 HaAgent 已经有可复用地基：

- `src/haagent/context/builder.py` 已负责构造 system/task messages，并写入 `contexts/<id>.json` 与 `contexts/<id>-manifest.json`。
- `src/haagent/context/manifest.py` 已有 `ContextManifest` 和 `ContextIndex`，但字段还偏摘要，不能表达每条上下文的 selected/skipped 原因。
- `src/haagent/memory/retrieval.py` 已有 `MemoryRetrievalResult.to_manifest_dict()`，能记录 used memories、budget 和 diagnostics。
- `src/haagent/context/observation_compaction.py` 已把工具 observation 压缩为稳定摘要。
- `src/haagent/runtime/chat_session.py` 已维护 bounded summaries、working state、interaction state，并通过 `RunOrchestrator` 进入 episode。
- `src/haagent/runtime/episode.py` 已有 transcript、tool-calls 和 context manifest 的写入点。

这说明不需要重建系统。深化方向应是把已有能力收束为一个真正的选择 Module。

## 不解决什么

为避免“上下文选择”膨胀成另一个平台，本阶段明确不做：

- 不做 embedding、向量数据库、语义 RAG。
- 不做全仓库后台索引和文件 watcher。
- 不做复杂 provider marketplace 或 source plugin 生态。
- 不让普通用户配置上下文策略。
- 不用用户话术列表判断任务复杂度。
- 不把完整 episode、完整 transcript、完整 tool trace、完整 audit 注入模型。
- 不把 diagnostics 注入模型输入。

本阶段只做本地、同步、确定性、可测试的选择层。

## 核心设计

### Module 定位

新增 `src/haagent/context/selection.py`，作为上下文选择的主 Module。

它不是新的 runtime，也不直接调用模型或工具。它只做一件事：把可用上下文候选转换成模型可见上下文和审计 manifest。

建议接口：

```python
@dataclass(frozen=True)
class ContextSelectionRequest:
    task: TaskSpec
    workspace_root: Path
    provider_name: str
    session_summary: str | None = None
    working_state: dict | None = None
    interaction_state: list[dict] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    budget: ContextSelectionBudget = field(default_factory=ContextSelectionBudget)


@dataclass(frozen=True)
class ContextSelectionResult:
    system_sections: list[ContextSection]
    task_sections: list[ContextSection]
    selected: list[ContextDecision]
    skipped: list[ContextDecision]
    budget: ContextBudgetReport
```

`ContextBuilder` 仍然负责写文件和构造最终 Chat messages，但不再直接决定所有来源是否进入 prompt。它调用 `ContextSelector.select(request)`，再把 `system_sections` 和 `task_sections` 传给 `messages.py` 渲染。

这样做的好处是：选择逻辑集中，现有 episode 写入不变，模型调用边界不变。

### 基础数据结构

新增 `src/haagent/context/sources.py`，放通用上下文类型。

```python
@dataclass(frozen=True)
class ContextCandidate:
    source_type: str
    source_id: str
    placement: Literal["system", "task"]
    title: str
    content: str
    reason: str
    priority: int
    hard_required: bool = False
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSection:
    source_type: str
    source_id: str
    placement: Literal["system", "task"]
    title: str
    content: str
    chars: int


@dataclass(frozen=True)
class ContextDecision:
    source_type: str
    source_id: str
    title: str
    reason: str
    placement: Literal["system", "task"] | None
    priority: int
    chars: int
    selected: bool
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

关键点：

- `source_type` 是稳定技术面，例如 `project_instructions`、`session_summary`、`working_state`、`memory`、`tool_workflow`、`interaction_state`、`observation`。
- `reason` 是选择原因，例如 `workspace_agents_md_found`、`resumed_session`、`memory_retrieval_match`、`recent_tool_result_needed`。
- `skip_reason` 是跳过原因，例如 `empty`、`over_budget`、`missing_file`、`not_relevant`、`lower_priority_replaced`。
- `hard_required` 只用于安全边界、用户当前任务和工具协议这类不能被预算裁掉的内容，必须少用。

这个类型足够小，但能覆盖后续大部分来源。

## Source 分层

### 永远候选，但不一定很长

这些 source 每次都进入候选池，但内容可以为空或很薄：

| source_type | 来源 | 默认 placement | 选择规则 |
| --- | --- | --- | --- |
| `base_instructions` | `context/instructions.py` | system | hard required，保持短 |
| `workspace_boundary` | workspace root、allowed tools | system/task | hard required，表达边界与工具协议 |
| `tool_workflow` | allowed tools 推导的 workflow hints | system | 按工具集合选择，不全量暴露 |
| `task_contract` | `TaskSpec` | task | hard required |

### 条件候选

这些 source 只有在结构化信号存在时进入候选池：

| source_type | 触发条件 | 内容预算建议 | 跳过原因 |
| --- | --- | --- | --- |
| `project_instructions` | workspace 存在 `AGENTS.md` | 4000 chars | missing_file / over_budget |
| `session_summary` | session 恢复或已有 summaries | 1000 chars | empty |
| `working_state` | working state 非空 | 1600 chars | empty / invalid |
| `memory` | `MemoryRetriever` 返回命中 | 3300 chars 总预算 | no_match / over_budget |
| `interaction_state` | 最近有人机交互、审批或输入 | 1200 chars | empty |
| `observation` | 上轮工具结果对下一步有用 | 2000 chars 总预算 | stale / over_budget |

### 暂不纳入的来源

这些来源先保持在工具或磁盘层，不进入默认选择：

- 完整 transcript。
- 完整 tool-calls。
- 完整 memory audit。
- 完整 `.runs` episode。
- 全仓库 repo map。
- Web search 原始结果全集。

如果后续需要，也应以 compact observation 或 explicit file/tool reference 的形式进入。

## 预算策略

预算不需要一开始用 tokenizer。HaAgent 现有代码多用字符预算，继续采用字符预算即可，后续可替换为 token estimate。

建议新增：

```python
@dataclass(frozen=True)
class ContextSelectionBudget:
    max_system_chars: int = 9000
    max_task_chars: int = 12000
    max_project_instructions_chars: int = 4000
    max_session_summary_chars: int = 1000
    max_working_state_chars: int = 1600
    max_memory_chars: int = 3300
    max_observation_chars: int = 2000
    max_interaction_state_chars: int = 1200
```

选择顺序：

1. 先保留 `hard_required`。
2. 按 `placement` 分 system/task 预算。
3. 同一 placement 内按 `priority` 从小到大选择。
4. 单个 source 超过自身预算时先裁剪内容，并记录 `truncated=true`。
5. 总预算不足时跳过低优先级 source，并记录 `skip_reason=over_budget`。

建议默认优先级：

| priority | source |
| --- | --- |
| 0 | base instructions、workspace boundary、task contract |
| 10 | project instructions |
| 20 | current working state |
| 30 | session summary |
| 40 | explicit interaction state |
| 50 | relevant memory |
| 60 | recent compact observations |
| 70 | tool workflow hints |

说明：tool workflow hints 当前很有用，但它们是规则性提示，容易越来越长。后续应按工具集合选择最少提示，不应无限扩展。

## Diagnostics 与 Manifest

### Manifest 结构

扩展 `ContextManifest`，保持兼容已有字段，同时新增 `selection`：

```json
{
  "context_id": "0001",
  "provider": "openai",
  "workspace_root": "E:\\python-project\\HaAgent",
  "generated_at": "2026-06-29T...",
  "message_count": 2,
  "system_chars": 6420,
  "task_chars": 5300,
  "selection": {
    "budget": {
      "max_system_chars": 9000,
      "max_task_chars": 12000,
      "used_system_chars": 6420,
      "used_task_chars": 5300
    },
    "selected": [
      {
        "source_type": "project_instructions",
        "source_id": "AGENTS.md",
        "reason": "workspace_agents_md_found",
        "placement": "system",
        "chars": 4000,
        "truncated": true
      }
    ],
    "skipped": [
      {
        "source_type": "observation",
        "source_id": "tool-call-17",
        "reason": "recent_tool_result",
        "skip_reason": "over_budget",
        "chars": 3800
      }
    ]
  }
}
```

`selection.selected` 与 `selection.skipped` 默认只进 manifest，不进模型。

### 为什么需要 skipped

只记录 selected 不够。失败复盘时，经常需要知道“为什么没进”：

- 用户显式引用的文件是否没被识别。
- 记忆是否命中但被预算挤掉。
- 工具结果是否因为 stale 被跳过。
- project instructions 是否文件不存在或读取失败。

`skipped` 能让 inspect/eval 直接定位问题，而不是靠猜。

### Manifest 写入位置

沿用现有 episode 结构：

- `contexts/<context_id>.json`：最终模型 messages snapshot。
- `contexts/<context_id>-manifest.json`：该次 context 的详细 manifest。
- `context-manifest.json`：episode 内上下文索引。

不新增全局数据库，不改变普通用户体验。

## 与现有代码的落点

### `ContextBuilder`

当前 `ContextBuilder.build()` 同时做读取、选择、拼接、写 manifest。深化后建议改成：

1. 验证工具。
2. 构造 `ContextSelectionRequest`。
3. 调用 `ContextSelector.select()`。
4. 调用 `build_system_message_from_sections()` 和 `build_task_message_from_sections()`。
5. 写入 context snapshot 与 manifest。

这样 `ContextBuilder` 仍是 episode/context 文件的所有者，但不再包含具体 source 选择规则。

### `messages.py`

保留现有 `build_system_message()`、`build_task_message()` 可短期兼容。新增两个 section 渲染函数：

```python
def render_sections(sections: list[ContextSection]) -> str:
    parts = []
    for section in sections:
        parts.append(f"{section.title}:")
        parts.append(section.content.strip())
    return "\n\n".join(parts)
```

不要让渲染函数自己做选择或预算。

### `MemoryRetriever`

当前 memory manifest 已经包含 used memories、budget 和 diagnostics。深化时不要重写 memory retrieval，只补两点：

1. `RetrievedMemory.to_context_candidate()` 或 adapter 函数，把 memory result 转成 `ContextCandidate`。
2. diagnostics 中补充更多 skip 明细，例如被预算跳过的 memory id、score、scope，而不是只有计数。

短期不修中文单字 tokenizer，避免把本方案变成检索质量项目。

### `observation_compaction.py`

它已经适合作为 observation source 的基础。新增 `ObservationSelector` 时只调用 `observation_summary()`，不直接读原始 result。

第一阶段只选择最近 3 条重要 observation：

- `file_read`：最近一次与当前任务相关的读取片段。
- `file_search`：候选摘要。
- `shell` / `code_run`：失败或验证相关摘要。
- `apply_patch` / `apply_patch_set`：变更结果摘要。

完整工具结果仍保存在 `tool-calls.jsonl`。

### `AgentSession`

`AgentSession` 不应该知道每个 source 的细节。它只负责把现有状态传入：

- `_bounded_summaries(...)` 得到的 session summary。
- `_working_state`。
- `runtime_events` 或 compact 后 observations。
- `interaction_state`。

后续如果 TUI 要显示“本轮模型看到了什么”，也应读取 episode manifest，而不是让 TUI 重算。

## 选择器设计

建议在 `selection.py` 中先实现 5 个小选择器函数，不急着抽象 class。

### `collect_project_instruction_candidates`

输入 workspace root。若 `AGENTS.md` 存在，读取并生成 candidate：

- `source_type=project_instructions`
- `source_id=AGENTS.md`
- `placement=system`
- `reason=workspace_agents_md_found`
- `priority=10`

读取失败应抛 `ContextBuildError`，不要静默跳过项目规则。

### `collect_session_candidates`

输入 session summary、working state、interaction state。

规则：

- summary 非空才候选。
- working state 通过 `format_working_state_for_model()` 成功后才候选。
- interaction state 只保留最近 8 条摘要。

无内容时写 skipped：`skip_reason=empty`。这类 skipped 可按 source 记录一条，不需要每次膨胀很多。

### `collect_memory_candidates`

调用现有 `MemoryRetriever.retrieve()`。每条 retrieved memory 变成一个 candidate。

重要：memory 的选择原因必须来自检索结果，而不是泛泛写 “relevant”。建议 metadata 至少包含：

- `score`
- `scope`
- `category`
- `updated_at`
- `tags`

如果 retrieval diagnostics 有 index missing、invalid、over budget，也合并进 manifest。

### `collect_tool_workflow_candidates`

从 allowed tools 推导 workflow hints。当前 `ContextBuilder._tool_workflow_hints()` 可以整体搬入 selector。

后续可以继续小步优化：

- 没有 file edit 工具时，不注入 apply_patch 提示。
- 没有 web 工具时，不注入 web 安全提示。
- 没有 shell/code_run 时，不注入 cwd 提示。

### `collect_observation_candidates`

输入最近 runtime/tool events，先转 `observation_summary()`，再根据 tool/status 选：

- 失败的 `shell` / `code_run` 优先。
- 最近的 `file_read` / `file_search` 次优先。
- 成功的 `file_write` / patch 作为变更状态。
- 过旧或重复 observation 跳过。

第一阶段可以只接入已有 `interaction_state` 或 runtime event 摘要，不需要追溯完整 tool-calls。

## 渐进实施路线

### 阶段一：结构化 selection，不改变模型内容

目标：把当前已经进入 prompt 的内容都转成 candidates，再由 selector 选回去，保证模型输入等价或近似等价。

改动：

- 新增 `sources.py`、`selection.py`。
- 扩展 `manifest.py`。
- `ContextBuilder` 调用 selector。
- 增加 snapshot/manifest 测试。

验收：

- 现有 chat/run 测试通过。
- `contexts/<id>.json` 内容与改动前保持同类结构。
- `contexts/<id>-manifest.json` 有 `selection.selected/skipped/budget`。

### 阶段二：把 memory diagnostics 做实

目标：回答“为什么这条 memory 进来了，为什么那条没进”。

改动：

- `MemoryRetriever` 对 over budget 的 memory 记录 id、scope、score、chars。
- 对 no match 不记录全部 memory，只记录汇总计数。
- manifest 中 memory diagnostics 与 selection decisions 对齐。

验收：

- 普通问候没有无关 memory 注入。
- 明确问记忆时相关 memory 进入 selected。
- 预算不足时 skipped 能看到具体 memory id。

### 阶段三：接入 compact observations

目标：工具结果进入下一轮时只带压缩观察，不带原始大结果。

改动：

- 从 runtime events 生成 observation candidates。
- 对 file_read/search/shell/code_run/patch 设定不同 priority。
- 大输出只显示 excerpt 与完整结果位置。

验收：

- 大文件读取不会让下一轮 task message 暴涨。
- shell/code_run 失败摘要仍足够模型继续修复。
- 完整结果仍能在 `tool-calls.jsonl` inspect。

### 阶段四：inspect 展示

目标：开发者能直接看本轮上下文选择结果。

改动：

- `cli_inspect` 或 episode renderer 增加 Context Selection 摘要。
- 展示 selected count、skipped count、system/task chars、memory count、over budget count。

验收：

- 一眼能看到本轮 prompt 主要来源。
- 不需要打开 raw model input 也能知道是否 memory/project/session/observation 被选中。

## 测试方案

### 单元测试

新增 `tests/test_context_selection.py`：

- `test_project_instructions_selected_when_agents_md_exists`
- `test_project_instructions_skipped_when_missing`
- `test_session_summary_selected_only_when_non_empty`
- `test_working_state_selected_with_budget`
- `test_low_priority_source_skipped_when_over_budget`
- `test_hard_required_source_survives_budget`
- `test_manifest_records_selected_and_skipped`

### 集成测试

扩展现有 ContextBuilder 测试或新增 `tests/test_context_builder_selection.py`：

- `test_context_builder_writes_selection_manifest`
- `test_context_builder_model_input_snapshot_remains_bounded`
- `test_memory_selection_appears_in_manifest`
- `test_diagnostics_not_in_model_input`

### 回归测试

围绕已知风险：

- `test_greeting_does_not_load_unrelated_user_memory`
- `test_memory_question_loads_relevant_confirmed_memory`
- `test_large_tool_result_uses_compact_observation`
- `test_audit_growth_does_not_change_model_input_size`
- `test_final_response_is_not_memory_evidence`

这些测试比“模型回答看起来对不对”更稳定，因为它们断言 deterministic context selection。

## 示例流程

### 普通问候

用户输入：`你好`

预期 selected：

- base instructions
- workspace boundary
- task contract
- minimal tool workflow

预期 skipped：

- session summary: empty
- working state: empty
- memory: no_match
- observations: empty

模型输入保持薄，不注入“爱好/偏好”等长期记忆。

### 用户问“我的爱好是什么”

预期 selected：

- base instructions
- task contract
- relevant user memory，reason=`memory_retrieval_match`

manifest 应能看到 memory score、scope、category、chars。

### 恢复长任务

预期 selected：

- session summary
- working state
- recent interaction state
- relevant workspace memory

预期 skipped：

- full transcript: not_candidate
- full tool-calls: not_candidate

模型只看 bounded summary 和当前状态，不复制完整历史。

### 工具刚读过大文件

预期 selected：

- file_read compact observation，包含 path、line range、excerpt、truncated=true。

预期 skipped：

- raw file content，reason=`raw_tool_result_kept_in_trace`

完整内容仍在工具结果或文件系统中，模型需要更多时再调用 `file_read`。

## 风险与控制

### 风险：Module 变成 pass-through

如果 selector 只是把所有内容原样返回，那就是浅 Module。控制方式：

- 必须有预算。
- 必须有 selected/skipped。
- 必须有 source priority。
- 必须有测试覆盖跳过行为。

### 风险：过度工程化

如果一开始做 provider/plugin/embedding/indexer，会偏离目标。控制方式：

- 第一版只支持本地同步函数。
- source 类型固定在当前 runtime 已有来源。
- 不引入新服务和新依赖。

### 风险：prompt 行为漂移

控制方式：

- 阶段一保持模型输入结构等价。
- 用 snapshot 或关键片段测试保护。
- 先写 manifest，再逐步改变选择策略。

### 风险：diagnostics 泄漏进模型

控制方式：

- `ContextSelectionResult` 明确区分 sections 与 decisions。
- `messages.py` 只渲染 sections。
- 测试断言 `skip_reason`、`selected`、`diagnostics` 不出现在 model input。

## 推荐文件变更清单

第一批开发建议只改这些文件：

- 新增 `src/haagent/context/sources.py`：candidate、section、decision、budget 类型。
- 新增 `src/haagent/context/selection.py`：收集 source、应用预算、输出 result。
- 修改 `src/haagent/context/manifest.py`：增加 selection 字段。
- 修改 `src/haagent/context/builder.py`：调用 selector，写增强 manifest。
- 修改 `src/haagent/context/messages.py`：支持 section 渲染。
- 修改 `tests/` 中 context builder / memory retrieval 相关测试。

暂不修改：

- `ModelGateway`。
- `ToolRouter`。
- TUI 主流程。
- provider 配置。
- memory 存储格式。

## 成功标准

这项深化完成后，应满足：

- 每次模型调用都能说明模型看到了哪些上下文。
- 每个被选中的上下文都有 source、reason、placement、chars。
- 关键被跳过上下文有 skip_reason。
- 普通聊天不会因为审计、记忆、历史增长而自动变胖。
- memory、session、working state、tool observation 的注入规则能单独测试。
- 后续想接 repo map 或 semantic retrieval 时，只需新增 source adapter，不需要改散落的 prompt 拼接点。

换句话说，Context Selection Module 的收益不是“立刻让模型变聪明”，而是让 HaAgent 的上下文行为可控、可复盘、可演进。它应该是一层薄但有深度的工程结构：接口小，内部能做选择、预算和诊断。
