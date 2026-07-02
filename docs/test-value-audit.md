# HaAgent 测试价值审计清单

本清单用于约束测试瘦身：默认快测只保留高信号、高风险、低成本用例；低价值并不等于删除，慢但有价值的测试迁到显式入口。

## 判定规则

- keep：workspace/path 边界、secret redaction、approval policy、ToolRouter、ModelGateway、episode/transcript schema、CLI/TUI 主路径、历史 bug 回归。
- merge：同一行为的多文案、多状态标签、多枚举错误矩阵，合并为代表场景或单个结构性断言。
- delete：只锁定中文措辞、视觉层级、内部实现细节，且上层已有同等行为保护的测试。
- move_to_e2e：真实模型、长 dogfood、完整 TUI 键盘漫游、慢 smoke、inspect/eval/export 高级 harness 回归。

## tests/tui/test_app.py

原文件为 `tests/test_tui_app.py`，已迁到 `tests/tui/test_app.py`，默认 pytest 不收集，显式运行：

```powershell
uv run pytest tests/tui -q
```

- keep：启动并展示 status、提交 prompt、审批 allow/deny、用户输入续同一 turn、工具摘要、模型配置、文件引用核心流。
- keep：secret redaction、外部路径确认、权限模式、TUI 与 AssistantService 的关键接线。
- merge：主题、no-color、status bar 宽度、帮助弹窗上下文、memory 导航、command suggestion 这类 UI 状态矩阵。
- delete：只断言中文标题、符号、视觉层级、空 summary 是否渲染等低风险文案/样式锁定。
- move_to_e2e：完整键盘漫游、长 streaming 稳定性、真实流程 smoke。

已执行的第一轮瘦身：

- `merge_tool_activity` 从 `ConversationTimeline.add_tool_activity()` 抽为纯函数。
- 新增 `tests/unit/tui/test_tool_activity.py` 覆盖 running/done、pending confirmation、重复同名工具调用。
- `tests/test_tui_memory_presenter.py` 迁到 `tests/unit/tui/test_memory_presenter.py`，去除对 TUI 大文件 helper 的依赖。

## tests/integration/tools/test_tool_router.py

- keep：工具注册一致性、schema validation、policy/approval 顺序、workspace 越界、external root 权限、secret guardrail、trace 写入。
- keep：`file_write`、`apply_patch_set` 的原子性和拒绝后不修改工作区。
- merge：缺失参数、类型错误、extra argument 可合并为一个 validation representative test，保留“不调用 handler”的独立断言。
- merge：shell cwd 多个等价路径场景可合并为一个表驱动测试，但保留 workspace escape、missing dir、timeout 三个独立失败类。
- delete：同一 error message 的逐字段中文措辞断言，改为结构字段和错误类别断言。

本轮未删除 ToolRouter 用例，因为它覆盖安全边界和 tool trace 合同，是默认快测的高价值主体。

## tests/integration/models/test_model_gateway.py

- keep：provider registry 映射、OpenAI Responses/OpenAI-compatible Chat/Anthropic/Gemini wire contract、tool call normalization、streaming、secret 不落盘、不进 transcript。
- keep：explicit failure mapping 和 orchestrator transcript 写入，因为它们是 runtime contract。
- merge：base URL normalization 的多 provider 变体可压成代表场景；invalid tool arguments 的 Responses/Chat 重复路径可共享断言 helper。
- merge：FakeModelGateway 的多个输入记录细节可保留一个综合行为测试。
- delete：只验证默认 endpoint 字符串拼接但不保护 provider contract 的重复断言。

本轮未删除 ModelGateway 用例；后续应先合并重复参数矩阵，再看是否仍超预算。

## tests/integration/runtime/test_episode_validator.py

- keep：episode metadata、required file、tool-call policy/approval、transcript/jsonl、failure/status matching、context manifest、validated package view。
- merge：字段类型错误矩阵可以以参数化 helper 保留覆盖，但减少重复独立测试函数。
- merge：verification command 的 missing/non-string/status/exit-code/timeout 可合并成单个表驱动结构错误测试。
- delete：只锁定错误消息全文的断言，改为包含字段名和错误类别。

本轮未删除 EpisodeValidator 用例；schema 破坏性路径仍是 inspect/eval 可用性的根安全网。

## 默认迁出清单

- `tests/tui/`：完整 Textual app 接线，显式路径运行。
- `tests/e2e/`：真实/长流程 smoke、dogfood、real LLM 手动入口。
- `tests/extended/`：inspect/eval/export/check 这类高级 harness 回归，默认不进入 1 分钟快测。

显式入口：

```powershell
uv run pytest tests/tui -q
uv run pytest tests/extended -q
uv run pytest tests/e2e -q --run-e2e
uv run pytest tests/e2e -q --real-llm
```
