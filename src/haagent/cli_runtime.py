"""
haagent/cli_runtime.py - CLI 运行期依赖集合

集中保存 CLI 入口需要替换的 runtime adapter，并提供 provider 与 smoke 配置构建。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haagent.models.gateway import OpenAIChatCompletionsGateway, OpenAIResponsesGateway
from haagent.models.provider_profile import (
    ProviderProfile,
    ProviderProfileError,
    load_active_provider_profile,
    load_provider_profile,
)
from haagent.runtime.chat_session import AgentSession
from haagent.runtime.orchestrator import RunOrchestrator


@dataclass(frozen=True)
class SmokeDefinition:
    name: str
    task_path: Path
    requires_profile: bool


@dataclass
class CliRuntime:
    project_root: Path
    orchestrator_cls: type = RunOrchestrator
    session_cls: type = AgentSession
    responses_gateway_cls: type = OpenAIResponsesGateway
    chat_gateway_cls: type = OpenAIChatCompletionsGateway

    def smoke_definitions(self) -> list[SmokeDefinition]:
        return [
            SmokeDefinition(
                name="hello",
                task_path=self.project_root / "examples/tasks/hello.yaml",
                requires_profile=False,
            ),
            SmokeDefinition(
                name="real_file_read",
                task_path=self.project_root / "examples/tasks/openai_chat_file_read_smoke.yaml",
                requires_profile=True,
            ),
            SmokeDefinition(
                name="real_edit_verify",
                task_path=self.project_root / "examples/tasks/openai_chat_edit_smoke.yaml",
                requires_profile=True,
            ),
        ]

    def build_run_model_gateway(self, args: argparse.Namespace) -> Any:
        if args.profile is not None:
            if args.provider not in {None, "fake"} or args.model is not None or args.base_url is not None:
                raise ProviderProfileError(
                    "--profile cannot be combined with --provider, --model, or --base-url",
                )
            return self.gateway_from_profile(load_provider_profile(args.profile))

        if args.provider is None:
            if args.model is not None or args.base_url is not None:
                raise ProviderProfileError("--model and --base-url require --provider or --profile")
            return self.gateway_from_profile(load_active_provider_profile())

        if args.provider in {"openai", "openai-chat"}:
            gateway_kwargs = {}
            if args.model is not None:
                gateway_kwargs["model"] = args.model
            if args.base_url is not None:
                gateway_kwargs["base_url"] = args.base_url
            gateway_class = (
                self.responses_gateway_cls
                if args.provider == "openai"
                else self.chat_gateway_cls
            )
            return gateway_class(**gateway_kwargs)
        return None

    def build_dogfood_model_gateway(self, args: argparse.Namespace) -> Any:
        if args.profile is not None:
            if args.provider is not None or args.model is not None or args.base_url is not None:
                raise ProviderProfileError(
                    "--profile cannot be combined with --provider, --model, or --base-url",
                )
            return self.gateway_from_profile(load_provider_profile(args.profile))
        if args.provider is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderProfileError("OPENAI_API_KEY is not set; dogfood skipped")
        return self.build_run_model_gateway(args)

    def gateway_from_profile(self, profile: ProviderProfile) -> Any:
        gateway_kwargs = {
            "api_key": profile.api_key,
            "model": profile.model,
            "base_url": profile.base_url,
        }
        if profile.provider == "openai":
            return self.responses_gateway_cls(**gateway_kwargs)
        if profile.provider == "openai-chat":
            return self.chat_gateway_cls(**gateway_kwargs)
        raise ProviderProfileError(f"unsupported provider in profile: {profile.provider}")
