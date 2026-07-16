"""
src/haagent/runtime/evaluation/runner.py - 本地确定性 Eval Runner

读取已导出的 eval case，复用 RunOrchestrator 重新运行并生成回归报告。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from haagent.models.model_ref import ModelInvocation
from haagent.models.types import ModelGateway, ModelResponse, ToolCall
from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.types import McpSettings
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.episodes.validator import load_inspect_episode_package
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.settings import DEFAULT_RUN_MAX_TURNS


EVAL_REPORT_VERSION = "1.0"


class EvalRunnerError(RuntimeError):
    """Eval runner 输入路径或报告输出无法处理时抛出。"""


class EvalCaseError(ValueError):
    """单个 eval case 损坏或缺少可运行字段时抛出。"""


def run_eval_path(
    eval_path: Path,
    *,
    runs_root: Path,
    model_gateway: ModelGateway | None = None,
    max_turns: int = DEFAULT_RUN_MAX_TURNS,
) -> dict[str, Any]:
    case_paths = _resolve_case_paths(eval_path)
    results = [
        _run_single_case(case_path, runs_root, model_gateway, max_turns)
        for case_path in case_paths
    ]
    passed_count = sum(1 for result in results if result["status"] == "passed")
    failed_count = sum(1 for result in results if result["status"] == "failed")
    error_count = sum(1 for result in results if result["status"] == "error")
    return {
        "report_version": EVAL_REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(eval_path),
        "total_count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "results": results,
    }


def _run_single_case(
    case_path: Path,
    runs_root: Path,
    model_gateway: ModelGateway | None,
    max_turns: int,
) -> dict[str, Any]:
    try:
        case = _read_case(case_path)
        gateway = _case_model_gateway(case, model_gateway)
        if case.get("case_type") == "chat_session":
            return _run_chat_session_case(case_path, case, runs_root, gateway, max_turns)
        _required_mapping(case, "task")
        with tempfile.TemporaryDirectory(prefix="haagent-eval-") as task_dir:
            task_root = Path(task_dir)
            workspace_root = _materialize_case_workspace(case_path, case, task_root / "workspace")
            task = _case_task(case_path, case, workspace_root=workspace_root)
            task_path = Path(task_dir) / "task.yaml"
            task_path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")
            run_result = RunOrchestrator(
                runs_root=runs_root,
                model_gateway=gateway,
                max_turns=max_turns,
            ).run(task_path)
        return _compare_case(case_path, case, run_result.episode_path, run_result.status.value)
    except Exception as error:
        return _error_result(case_path, str(error))


def _run_chat_session_case(
    case_path: Path,
    case: dict[str, Any],
    runs_root: Path,
    model_gateway: ModelGateway | None,
    max_turns: int,
) -> dict[str, Any]:
    chat = _required_mapping(case, "chat")
    prompts = _required_str_list(chat, "prompts", "chat.prompts")
    if not prompts:
        raise EvalCaseError("chat.prompts must contain at least one prompt")
    # Eval 输入没有 MCP 配置合同；禁止读取或连接用户级 server，保证离线确定性。
    mcp_runtime = SyncMcpRuntime(McpSettings())
    mcp_runtime.start()
    try:
        with tempfile.TemporaryDirectory(prefix="haagent-eval-chat-") as workspace_dir:
            workspace_root = _materialize_case_workspace(case_path, case, Path(workspace_dir) / "workspace")
            session = AgentSession(
                workspace_root=workspace_root,
                runs_root=runs_root,
                model_gateway=model_gateway,
                max_turns=max_turns,
                memory_extraction_enabled=False,
                mcp_runtime=mcp_runtime,
            )
            result = None
            for index, prompt in enumerate(prompts):
                if index > 0:
                    session = AgentSession.resume(
                        session.session_path,
                        model_gateway=model_gateway,
                        max_turns=max_turns,
                        mcp_runtime=mcp_runtime,
                        owns_mcp_runtime=False,
                    )
                result = session.run_prompt(prompt)
    finally:
        mcp_runtime.close()
    if result is None:
        raise EvalCaseError("chat.prompts must contain at least one prompt")
    return _compare_case(case_path, case, result.episode_path, result.status)


def _compare_case(
    case_path: Path,
    case: dict[str, Any],
    episode_path: Path,
    actual_status: str,
) -> dict[str, Any]:
    package_view = load_inspect_episode_package(episode_path)
    expected_tool_uses = _expected_tool_uses(case)
    actual_tool_uses = sorted({str(record["tool_name"]) for record in package_view.tool_calls})
    missing_tool_uses = sorted(set(expected_tool_uses) - set(actual_tool_uses))
    unexpected_tool_uses = sorted(set(actual_tool_uses) - set(expected_tool_uses))
    expected_status = _expected_status(case)
    final_response_match = _final_response_match(case, package_view.transcript)
    failure_category_match = _failure_category_match(case, package_view.failure_record)

    reasons: list[str] = []
    if actual_status != expected_status:
        reasons.append(f"status mismatch: expected {expected_status}, got {actual_status}")
    if missing_tool_uses:
        reasons.append(f"missing expected tool uses: {', '.join(missing_tool_uses)}")
    if unexpected_tool_uses:
        reasons.append(f"unexpected tool uses: {', '.join(unexpected_tool_uses)}")
    if final_response_match is False:
        reasons.append("final response did not contain expected text")
    if failure_category_match is False:
        reasons.append("failure category did not match expected category")
    context_expectation_reason = _context_expectation_reason(case, episode_path)
    if context_expectation_reason is not None:
        reasons.append(context_expectation_reason)

    passed = not reasons
    return {
        "eval_id": str(case.get("eval_id", case_path)),
        "case_path": str(case_path),
        "status": "passed" if passed else "failed",
        "final_response_match": final_response_match,
        "basic_result_match": passed,
        "expected_tool_uses": expected_tool_uses,
        "actual_tool_uses": actual_tool_uses,
        "missing_tool_uses": missing_tool_uses,
        "unexpected_tool_uses": unexpected_tool_uses,
        "episode_path": str(episode_path),
        "failure_reason": None if passed else "; ".join(reasons),
    }


def _resolve_case_paths(eval_path: Path) -> list[Path]:
    if eval_path.is_dir():
        manifest = eval_path / "manifest.json"
        if manifest.exists():
            return _manifest_case_paths(manifest)
        return sorted(path for path in eval_path.glob("*.json") if path.name != "manifest.json")
    if not eval_path.exists():
        raise EvalRunnerError(f"eval path does not exist: {eval_path}")
    raw = _read_json(eval_path)
    if _is_manifest(raw):
        return _manifest_case_paths(eval_path, raw)
    return [eval_path]


def _manifest_case_paths(manifest_path: Path, manifest: dict[str, Any] | None = None) -> list[Path]:
    manifest = manifest or _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise EvalRunnerError("batch manifest records must be a list")
    case_paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        output_file = record.get("output_file")
        if not isinstance(output_file, str):
            continue
        path = Path(output_file)
        if not path.is_absolute():
            path = manifest_path.parent / path
        case_paths.append(path)
    return case_paths


def _read_case(case_path: Path) -> dict[str, Any]:
    case = _read_json(case_path)
    if _is_manifest(case):
        raise EvalCaseError("expected eval case JSON, got batch manifest")
    return case


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvalCaseError(f"{path} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise EvalCaseError(f"{path} must contain a JSON object")
    return value


def _is_manifest(value: dict[str, Any]) -> bool:
    return "manifest_version" in value and "records" in value


def _case_task(case_path: Path, case: dict[str, Any], *, workspace_root: Path | None = None) -> dict[str, Any]:
    task = _required_mapping(case, "task")
    return {
        "goal": _required_str(task, "goal", "task.goal"),
        "workspace_root": str(workspace_root or _case_workspace_root(case_path, case)),
        "constraints": _required_str_list(task, "constraints", "task.constraints"),
        "allowed_tools": _required_str_list(task, "allowed_tools", "task.allowed_tools"),
        "acceptance_criteria": _required_str_list(
            task,
            "acceptance_criteria",
            "task.acceptance_criteria",
        ),
        "verification_commands": _required_str_list(
            task,
            "verification_commands",
            "task.verification_commands",
        ),
        "policy": _case_policy(task.get("policy")),
    }


def _case_workspace_root(case_path: Path, case: dict[str, Any]) -> Path:
    raw = _required_str(case, "workspace_root", "workspace_root")
    workspace_root = Path(raw)
    if workspace_root.is_absolute():
        return workspace_root
    return (case_path.parent / workspace_root).resolve()


def _materialize_case_workspace(case_path: Path, case: dict[str, Any], target_root: Path) -> Path:
    raw = _required_str(case, "workspace_root", "workspace_root")
    workspace_root = Path(raw)
    if workspace_root.is_absolute():
        return workspace_root
    source_root = (case_path.parent / workspace_root).resolve()
    shutil.copytree(source_root, target_root, dirs_exist_ok=True)
    return target_root.resolve()


def _case_policy(value: object) -> dict[str, list[str]]:
    if value is None:
        return {"approval_allowed_tools": [], "approved_tools": []}
    if not isinstance(value, dict):
        raise EvalCaseError("task.policy must be an object")
    return {
        "approval_allowed_tools": _string_list(value.get("approval_allowed_tools"), "task.policy.approval_allowed_tools"),
        "approved_tools": _string_list(value.get("approved_tools"), "task.policy.approved_tools"),
    }


def _expected_tool_uses(case: dict[str, Any]) -> list[str]:
    value = case.get("expected_tool_uses", case.get("tool_names_used", []))
    return sorted(set(_string_list(value, "expected_tool_uses")))


def _expected_status(case: dict[str, Any]) -> str:
    expectations = case.get("expectations")
    if isinstance(expectations, dict) and isinstance(expectations.get("final_status"), str):
        return str(expectations["final_status"])
    return _required_str(case, "final_status", "final_status")


def _final_response_match(case: dict[str, Any], transcript: list[dict[str, Any]]) -> bool | None:
    expectation = _final_response_expectation(case)
    if expectation is None:
        return None
    actual = _last_model_response_content(transcript)
    mode = expectation["mode"]
    expected = expectation["value"]
    if mode == "exact":
        return actual == expected
    if mode == "contains":
        return expected in actual
    raise EvalCaseError(f"unsupported final_response match mode: {mode}")


def _final_response_expectation(case: dict[str, Any]) -> dict[str, str] | None:
    expectations = case.get("expectations")
    if isinstance(expectations, dict):
        final_response = expectations.get("final_response")
        if isinstance(final_response, dict):
            mode = final_response.get("mode", "contains")
            value = final_response.get("value", "")
            if not isinstance(mode, str) or not isinstance(value, str):
                raise EvalCaseError("expectations.final_response mode and value must be strings")
            return {"mode": mode, "value": value}
    final_response = case.get("final_response")
    if isinstance(final_response, dict) and isinstance(final_response.get("content"), str):
        return {"mode": "contains", "value": str(final_response["content"])}
    return None


def _failure_category_match(case: dict[str, Any], failure_record: dict[str, Any]) -> bool | None:
    expectations = case.get("expectations")
    expected = None
    if isinstance(expectations, dict):
        expected = expectations.get("failure_category")
    if expected is None:
        return None
    failure = failure_record.get("failure")
    actual = failure.get("category") if isinstance(failure, dict) else None
    return actual == expected


def _context_expectation_reason(case: dict[str, Any], episode_path: Path) -> str | None:
    expectations = case.get("expectations")
    if not isinstance(expectations, dict):
        return None
    contains = _optional_string_list(expectations.get("context_contains"), "expectations.context_contains")
    not_contains = _optional_string_list(expectations.get("context_not_contains"), "expectations.context_not_contains")
    if not contains and not not_contains:
        return None
    context_text = _combined_context_text(episode_path)
    missing = [text for text in contains if text not in context_text]
    leaked = [text for text in not_contains if text in context_text]
    reasons = []
    if missing:
        reasons.append(f"context missing expected text: {', '.join(missing)}")
    if leaked:
        reasons.append(f"context contained forbidden text: {', '.join(leaked)}")
    return "; ".join(reasons) if reasons else None


def _combined_context_text(episode_path: Path) -> str:
    contexts_dir = episode_path / "contexts"
    if not contexts_dir.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(contexts_dir.glob("*.json"))
        if not path.name.endswith("-manifest.json")
    )


def _last_model_response_content(transcript: list[dict[str, Any]]) -> str:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return str(record.get("content", ""))
    return ""


def _required_mapping(raw: dict[str, Any], field: str) -> dict[str, Any]:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise EvalCaseError(f"missing {field}")
    return value


def _required_str(raw: dict[str, Any], field: str, label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        raise EvalCaseError(f"missing {label}")
    return value


def _required_str_list(raw: dict[str, Any], field: str, label: str) -> list[str]:
    if field not in raw:
        raise EvalCaseError(f"missing {label}")
    return _string_list(raw[field], label)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvalCaseError(f"{label} must be a list of strings")
    return list(value)


def _optional_string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    return _string_list(value, label)


def _case_model_gateway(case: dict[str, Any], fallback: ModelGateway | None) -> ModelGateway | None:
    responses = case.get("model_responses")
    if responses is None:
        return fallback
    if fallback is not None and fallback.provider_name != "fake":
        return fallback
    return DeterministicEvalGateway(responses)


class DeterministicEvalGateway:
    provider_name = "deterministic-eval"

    def __init__(self, responses: object) -> None:
        if not isinstance(responses, list) or not responses:
            raise EvalCaseError("model_responses must be a non-empty list")
        self._responses = responses
        self._index = 0

    def generate(
        self,
        invocation: ModelInvocation,
        **_: object,
    ) -> ModelResponse:
        del invocation
        if self._index >= len(self._responses):
            return ModelResponse("deterministic eval responses exhausted", [])
        raw = self._responses[self._index]
        self._index += 1
        if not isinstance(raw, dict):
            raise EvalCaseError("model_responses items must be objects")
        content = raw.get("content", "")
        if not isinstance(content, str):
            raise EvalCaseError("model_responses.content must be a string")
        tool_calls = raw.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise EvalCaseError("model_responses.tool_calls must be a list")
        return ModelResponse(
            content=content,
            tool_calls=[_deterministic_tool_call(item) for item in tool_calls],
        )


def _deterministic_tool_call(raw: object) -> ToolCall:
    if not isinstance(raw, dict):
        raise EvalCaseError("model_responses.tool_calls items must be objects")
    name = raw.get("name")
    args = raw.get("args", {})
    if not isinstance(name, str) or not name:
        raise EvalCaseError("model_responses.tool_calls.name must be a string")
    if not isinstance(args, dict):
        raise EvalCaseError("model_responses.tool_calls.args must be an object")
    return ToolCall(name=name, args=args)


def _error_result(case_path: Path, failure_reason: str) -> dict[str, Any]:
    return {
        "eval_id": str(case_path),
        "case_path": str(case_path),
        "status": "error",
        "final_response_match": None,
        "basic_result_match": False,
        "expected_tool_uses": [],
        "actual_tool_uses": [],
        "missing_tool_uses": [],
        "unexpected_tool_uses": [],
        "episode_path": None,
        "failure_reason": failure_reason,
    }
