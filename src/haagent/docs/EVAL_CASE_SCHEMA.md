# Eval case schema v1

`export_eval_case()` 把一个已校验的 episode package 导出为可序列化的 eval case 字典。导出入口会先复用 episode validator / package view；invalid episode 必须在导出前被 validator 拒绝，不能生成半可信 eval case。

Eval export 是后台评估能力，不是普通用户入口。普通用户路径应优先通过 `haagent setup` 配置模型连接，然后在目标目录运行 `haagent`；`haagent chat` 是显式自然语言入口，`task.yaml` 和 eval case 主要服务复现、批处理、smoke 和回归评估。

## 顶层字段

当前 `eval_case_version=1.0` 的真实顶层字段如下：

- `eval_case_version`: eval case schema 版本。
- `episode_version`: episode package schema 版本。
- `task`: 任务事实摘要，来自 episode package 内冻结的任务契约。该契约可能来自 `task.yaml`，也可能来自 `haagent` / `haagent chat` 为自然语言请求生成的临时 task contract。
- `workspace_root`: episode 记录的 workspace root。
- `final_status`: episode 最终状态，例如 `completed` 或 `failed`。
- `failure`: 失败摘要；成功时为 `null`。
- `verification`: verification command evidence summary 列表。
- `sandbox_summary`: 从 `sandbox.json` 扁平化导出的运行边界记录。
- `tool_names_used`: 本 episode 中出现过的工具名，按字典序去重。
- `tool_argument_errors`: `tool_argument_invalid` 错误摘要列表。
- `approval_summary`: 每条 tool call 的 policy approval 摘要。
- `next_actions`: 每轮 context 的结构化 next action 摘要。

`episode_id` 是后续可考虑补充的稳定标识字段，但当前 `export_eval_case()` v1.0 不导出该键。文档和消费者应以当前真实输出为准，避免把缺失字段当作已存在 schema。

## 嵌套字段

### task

`task` 包含：

- `goal`
- `acceptance_criteria`
- `verification_commands`

这些字段来自 episode package 中复制保存的任务契约。即使原始 `task.yaml` 或 chat 输入上下文在工作区中被修改，eval export 也只读取 episode 内冻结快照，保证重复导出确定。

### verification

`verification` 的每项至少包含：

- `command`
- `status`
- `exit_code`
- `timeout`
- `stdout_excerpt`
- `stderr_excerpt`
- `stdout_truncated`
- `stderr_truncated`
- `stdout_original_length`
- `stderr_original_length`
- `redacted`

这些字段来自 `verification/commands.jsonl`，导出时不会重新执行 verification command，也不会重新读取原始命令输出。

### sandbox_summary

`sandbox_summary` 来自 episode package 内的 `sandbox.json`，包含：

- `workspace_root`
- `filesystem_boundary`
- `network_policy`
- `process_policy`
- `credential_policy`
- `command_timeout_seconds`

其中 `command_timeout_seconds` 从 `sandbox.resource_limits.command_timeout_seconds` 扁平化导出。该字段是运行边界记录和审计 metadata，不代表真正容器隔离，也不表示文件系统、网络或进程执行方式被额外沙箱化。

### approval_summary

`approval_summary` 的每项至少包含：

- `tool_name`
- `action`
- `approval_required`
- `approval_status`
- `approval_reason`

`approval_status` 含义：

- `not_required`: 低风险或中风险工具不需要审批。
- `missing`: 高风险工具需要审批，但本次任务没有获得可执行批准。
- `granted`: 高风险工具已在任务契约的 policy 配置中被显式批准，可以进入 handler 执行路径。

## 最小 JSON 示例

```json
{
  "eval_case_version": "1.0",
  "episode_version": "1.0",
  "task": {
    "goal": "Say hello through the fake tool",
    "acceptance_criteria": ["Run reaches completed state"],
    "verification_commands": []
  },
  "workspace_root": "E:\\python-project\\HaAgent\\examples\\tasks",
  "final_status": "completed",
  "failure": null,
  "verification": [],
  "sandbox_summary": {
    "workspace_root": "E:\\python-project\\HaAgent\\examples\\tasks",
    "filesystem_boundary": "workspace_root",
    "network_policy": "unrestricted",
    "process_policy": "local_subprocess",
    "credential_policy": "inherit_environment",
    "command_timeout_seconds": 60
  },
  "tool_names_used": ["fake_tool"],
  "tool_argument_errors": [],
  "approval_summary": [
    {
      "tool_name": "fake_tool",
      "action": "allow",
      "approval_required": false,
      "approval_status": "not_required",
      "approval_reason": "approval not required for low risk tool fake_tool"
    }
  ],
  "next_actions": [
    {
      "context_id": "0001",
      "status": "none",
      "reason": "none",
      "based_on_observation_index": null,
      "based_on_tool_name": null
    }
  ]
}
```
