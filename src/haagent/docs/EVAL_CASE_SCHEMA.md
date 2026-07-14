# Eval case schema v1

`export_eval_case()` 把一个已校验的 episode package 导出为可序列化的 eval case 字典。导出入口会先复用 episode validator / package view；invalid episode 必须在导出前被 validator 拒绝，不能生成半可信 eval case。

Eval export 是后台评估能力，不是普通用户入口。普通用户路径是在目标目录运行 `haagent` 进入 TUI，并在 TUI 内通过 `/connect` 配置连接、通过 `/model` 切换模型；`task.yaml` 和 eval case 主要服务复现、批处理、smoke 和回归评估。

## 顶层字段

当前 `eval_case_version=1.0` 的真实顶层字段如下：

- `eval_case_version`: eval case schema 版本。
- `episode_version`: episode package schema 版本。
- `task`: 任务事实摘要，来自 episode package 内冻结的任务契约。该契约可能来自 `task.yaml`，也可能来自 TUI 会话为自然语言请求生成的临时 task contract。
- `workspace_root`: episode 记录的 workspace root。
- `final_status`: episode 最终状态，例如 `completed` 或 `failed`。
- `expected_tool_uses`: 供 eval runner 比对的预期工具名集合；当前导出值来自实际 tool call。
- `expectations`: 最终状态、失败分类和最终回复 contains 条件。
- `failure`: 失败摘要；成功时为 `null`。
- `verification`: verification command evidence summary 列表。
- `sandbox_summary`: 从 `sandbox.json` 扁平化导出的运行边界记录。
- `environment_summary`: 从 `environment.json` 提取的简洁运行环境、模型和工具数量摘要。
- `cost_summary`: 从 `cost.json` 提取的 usage、估价可用性和 token totals 摘要；没有可靠价格时 `estimated_cost` 保持 `null`。
- `tool_names_used`: 本 episode 中出现过的工具名，按字典序去重。
- `tool_argument_errors`: `tool_argument_invalid` 错误摘要列表。
- `approval_summary`: 每条 tool call 的 policy approval 摘要。
- `human_interactions`: 用户补充输入和审批事件的脱敏摘要。
- `final_response`: 最后一次模型回复的 provider、turn、content 和 tool call 数量；没有模型回复时为 `null`。

## 嵌套字段

### task

`task` 包含：

- `goal`
- `constraints`
- `allowed_tools`
- `acceptance_criteria`
- `verification_commands`
- `policy`

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
- `backend`
- `network_policy`
- `process_policy`
- `credential_policy`
- `command_timeout_seconds`
- `cpu_limit`
- `memory_limit`
- `pids_limit`
- `degraded`
- `availability_reason`
- `sandbox_user`
- `privileged`

其中 `command_timeout_seconds` 从 `sandbox.resource_limits.command_timeout_seconds` 扁平化导出。该字段是运行边界记录和审计 metadata，不代表真正容器隔离，也不表示文件系统、网络或进程执行方式被额外沙箱化。

### environment_summary

`environment_summary` 来自 episode package 内的 `environment.json`，包含：

- `python`
- `platform`
- `haagent_version`
- `model_provider`
- `model`
- `endpoint`
- `allowed_tool_count`

该摘要不包含 API key、Authorization header、完整请求体、完整响应体或完整工具 schema。

### cost_summary

`cost_summary` 来自 episode package 内的 `cost.json`，包含：

- `usage_available`
- `pricing_available`
- `model_call_count`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `estimated_cost`
- `currency`
- `reason`

token 字段只来自 provider usage metadata。没有 usage 时 token totals 为 `null`；没有可靠价格匹配时 `estimated_cost` 和 `currency` 为 `null`，并通过 `reason` 说明不可估价原因。

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
- `not_evaluated`: 工具不存在、未被任务允许或因前序失败跳过，policy 没有进入评估阶段。

### human_interactions 与 final_response

`human_interactions` 只保留事件类型、工具名、问题、审批结果，以及用户回答的字符数；不导出用户回答正文。`final_response` 来自 transcript 中最后一条 `model_response`，包含：

- `provider`
- `turn`
- `content`
- `tool_call_count`

## 最小 JSON 示例

```json
{
  "eval_case_version": "1.0",
  "episode_version": "1.0",
  "task": {
    "goal": "Say hello through the fake tool",
    "constraints": [],
    "allowed_tools": ["fake_tool"],
    "acceptance_criteria": ["Run reaches completed state"],
    "verification_commands": [],
    "policy": {
      "approval_allowed_tools": [],
      "approved_tools": []
    }
  },
  "workspace_root": "E:\\python-project\\HaAgent\\examples\\tasks",
  "final_status": "completed",
  "expected_tool_uses": ["fake_tool"],
  "expectations": {
    "final_status": "completed",
    "failure_category": null,
    "final_response": {
      "mode": "contains",
      "value": "Fake model observed tool results."
    }
  },
  "failure": null,
  "verification": [],
  "sandbox_summary": {
    "workspace_root": "E:\\python-project\\HaAgent\\examples\\tasks",
    "filesystem_boundary": "workspace_root",
    "backend": "local",
    "network_policy": "unrestricted",
    "process_policy": "local_subprocess",
    "credential_policy": "inherit_environment",
    "command_timeout_seconds": 60,
    "cpu_limit": null,
    "memory_limit": null,
    "pids_limit": null,
    "degraded": true,
    "availability_reason": "local backend has no container isolation",
    "sandbox_user": null,
    "privileged": null
  },
  "environment_summary": {
    "python": "3.13.5 (...)",
    "platform": "Windows-...",
    "haagent_version": "0.1.0",
    "model_provider": "fake",
    "model": "fake-model",
    "endpoint": null,
    "allowed_tool_count": 1
  },
  "cost_summary": {
    "usage_available": false,
    "pricing_available": false,
    "model_call_count": 0,
    "input_tokens": null,
    "output_tokens": null,
    "total_tokens": null,
    "estimated_cost": null,
    "currency": null,
    "reason": "model gateway did not provide usage metadata"
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
  "human_interactions": [],
  "final_response": {
    "provider": "fake",
    "turn": 2,
    "content": "Fake model observed tool results.",
    "tool_call_count": 0
  }
}
```
