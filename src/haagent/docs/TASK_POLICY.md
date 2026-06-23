# 任务契约 policy 配置

任务契约可以通过可选的 `policy` 字段声明本次任务的高风险工具审批配置。任务契约可能来自高级 `task.yaml`，也可能来自 `haagent` / `haagent chat` 为自然语言请求生成的临时 task contract。

`task.yaml` 仍是复现、批处理、smoke 和 eval 的高级入口；普通用户路径应优先使用 `haagent setup` 后在目标目录运行 `haagent`。无论入口是什么，policy 都只影响 Policy Engine 对工具调用的决策，不会自动引入交互式审批，也不会改变工具本身的执行方式。

## 字段语义

- `policy.approval_allowed_tools`: 允许哪些高风险工具进入审批申请流程。它只表示“可以申请审批”，不代表工具已经被批准执行。
- `policy.approved_tools`: 显式批准哪些高风险工具在本次任务中执行。被批准的高风险工具会得到 `approval.status=granted`，并进入对应 handler 的执行路径。

如果 `policy` 缺失，两个列表都默认为空。两个字段都必须是 `list[str]`，列表中的工具名必须存在于 Tool Registry；`approved_tools` 中的工具也必须同时出现在 `approval_allowed_tools` 中。

## 行为差异

- 两者为空：高风险工具会被 policy 拒绝，`policy.action=deny`，`approval.status=missing`，原因表示该工具未被允许申请审批。
- 只配置 `approval_allowed_tools`：高风险工具仍会被 policy 拒绝，`policy.action=deny`，`approval.status=missing`，原因表示该工具允许申请但缺少批准。
- 同时配置 `approval_allowed_tools` 和 `approved_tools`：对应高风险工具会被允许，`policy.action=allow`，`approval.status=granted`，并进入 handler 执行路径。

低风险和中风险工具不需要审批，仍会按原有规则允许执行，并记录 `approval.status=not_required`。

## 示例：允许申请但未批准

下面的任务契约允许 `shell` 进入审批申请流程，但没有把它加入 `approved_tools`。如果模型调用 `shell`，预期结果是 policy deny，approval missing，handler 不会执行。

```yaml
goal: Try a high risk shell command without approval grant
constraints:
  - Demonstrate missing approval trace
allowed_tools:
  - shell
acceptance_criteria:
  - Run records policy_denied for shell
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
```

预期 trace 摘要：

- `tool-calls.jsonl` 中 `policy.action=deny`
- `policy.approval.required=true`
- `policy.approval.status=missing`
- `error.type=policy_denied`

## 示例：高风险工具已批准

下面的任务契约同时把 `shell` 放入 `approval_allowed_tools` 和 `approved_tools`。如果模型调用 `shell`，预期结果是 approval granted，并进入 `shell` handler 执行路径。

```yaml
goal: Run an approved high risk shell command
constraints:
  - Demonstrate approval granted trace
allowed_tools:
  - shell
acceptance_criteria:
  - Shell tool reaches handler execution path
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
  approved_tools:
    - shell
```

预期 trace 摘要：

- `tool-calls.jsonl` 中 `policy.action=allow`
- `policy.approval.required=true`
- `policy.approval.status=granted`
- 工具调用进入 handler 执行路径
