# HaAgent Domain Context

## Model Context Runtime

Model Context Runtime 是每个 session 唯一的模型可见上下文所有者。它持有当前模型消息链和版本化 snapshot，并负责同一 epoch 内 append-only delta、checkpoint、强制 rebuild、恢复校验、diagnostics 以及 `model-context.json` 的原子持久化。

`AgentSession` 只向它提供 session summary、working state、Todo、Plan 等原始事实；`RunOrchestrator` 和 `TurnLoop` 只通过单轮 handle 开始请求、取得模型调用 frame，并提交已完成的模型/工具消息。epoch、revision、snapshot 和 delta 插入位置不属于调用者 interface。
