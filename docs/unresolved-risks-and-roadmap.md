# 未解决风险与路线图

## 当前优先级：个人助手启动体验

- `haagent setup` 配置用户级默认 profile。
- 无子命令 `haagent` 直接进入个人助手聊天模式。
- 默认 workspace root 是当前目录。
- 交互式多轮对话由 `AgentSession` 管理。
- `haagent sessions` 和 `haagent --continue` 服务目录相关会话恢复。

## 中期路线

- 长期记忆与用户偏好，按 `docs/superpowers/specs/2026-06-25-memory-system-v1-design.md` 推进：Session/Workspace/User Memory 物理分开，长期记忆先进入候选队列，用户确认后由确定性服务落库，不把完整 episode trace 注入模型输入。
- 记忆系统已记录当前问题与改进方向：见 `docs/memory-system-issues-and-improvement-notes.md`。后续修复优先收紧写入证据边界、候选去重和审计/prompt 分离；中文单字检索误命中暂列已知风险，不用脆弱停用字表急修。
- 更好的文件整理能力和文档处理能力。
- 更自然的任务恢复体验，包括跨目录提示和更清晰的 session 摘要。
- 更丰富的个人助手任务模板，例如 CSV 分析、资料整理、草稿润色和脚本结果解释。
- 普通聊天的模型输入需要从完整任务脚手架改为按需加载：默认保持薄上下文，仅在出现结构化动作或工具需求时加载项目规则、任务 scaffold、验证要求和相关记忆，不通过猜测用户话术复杂度决定 prompt 厚度。具体工程原则见 `docs/context-engineering-on-demand-injection-guide.md`。

## 风险

- 配置体验必须避免把 API key 写入本地 profile、项目配置或 trace；系统凭据库是默认存储，明文用户文件只能显式 opt-in。
- 会话恢复必须继续使用 bounded summary，不能复制完整历史、完整 episode 或完整工具输出进模型输入。
- 可审计数据和模型输入必须保持分离；episode、task contract、plan 和 tool trace 可以完整落盘，但不能无条件进入下一次模型输入。
- Harness/eval/dogfood 仍要可用，但不能重新变成普通用户路径的中心。
