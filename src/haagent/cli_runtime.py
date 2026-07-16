"""
haagent/cli_runtime.py - CLI 运行期依赖

CLI 与 TUI 共用 ModelRuntime；显式 provider 参数只构造一次临时 ResolvedModel。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haagent.models.gateway_registry import gateway_from_resolved
from haagent.models.config.connections import ProviderProfileError, user_config_dir
from haagent.models.model_ref import ModelRef, ResolvedCredential, ResolvedModel
from haagent.models.model_runtime import ModelRuntime
from haagent.models.model_settings import ModelSettings
from haagent.models.types import ModelGateway
from haagent.runtime.orchestration.orchestrator import RunOrchestrator


GatewayFactory = Callable[[ResolvedModel], ModelGateway]


@dataclass(frozen=True)
class SmokeDefinition:
    name: str
    task_path: Path
    requires_profile: bool


@dataclass
class CliRuntime:
    project_root: Path
    orchestrator_cls: type = RunOrchestrator
    gateway_factory: GatewayFactory = field(default=gateway_from_resolved)

    def smoke_definitions(self) -> list[SmokeDefinition]:
        return [
            SmokeDefinition("hello", self.project_root / "examples/tasks/hello.yaml", False),
            SmokeDefinition("real_file_read", self.project_root / "examples/tasks/openai_chat_file_read_smoke.yaml", True),
            SmokeDefinition("real_edit_verify", self.project_root / "examples/tasks/openai_chat_edit_smoke.yaml", True),
        ]

    def build_run_model_gateway(self, args: argparse.Namespace) -> Any:
        runtime = ModelRuntime.load(config_dir=user_config_dir(), environ=os.environ, gateway_builder=self.gateway_factory)
        if args.profile is not None:
            if args.provider not in {None, "fake"} or args.base_url is not None:
                raise ProviderProfileError("--profile cannot be combined with --provider or --base-url")
            active = runtime.selection_store.load_active()
            return runtime.create_gateway(ModelRef(args.profile, args.model or active.model))
        if args.provider is None:
            if args.model is not None or args.base_url is not None:
                raise ProviderProfileError("--model and --base-url require --provider or --profile")
            return runtime.create_route_gateway()
        if args.provider in {"openai", "openai-chat"}:
            return self.gateway_factory(self._cli_resolved_model(args))
        return None

    def build_dogfood_model_gateway(self, args: argparse.Namespace) -> Any:
        if args.profile is not None:
            if args.provider is not None or args.base_url is not None:
                raise ProviderProfileError("--profile cannot be combined with --provider or --base-url")
            return self.build_run_model_gateway(args)
        if args.provider is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderProfileError("OPENAI_API_KEY is not set; dogfood skipped")
        return self.build_run_model_gateway(args)

    def _cli_resolved_model(self, args: argparse.Namespace) -> ResolvedModel:
        provider = str(args.provider)
        model = args.model or "gpt-4.1-mini"
        ref = ModelRef(f"cli-{provider}", model)
        return ResolvedModel(
            ref=ref,
            provider=provider,
            base_url=args.base_url or "",
            runtime_kind="remote",
            settings=ModelSettings.empty(),
            credential=ResolvedCredential(os.environ.get("OPENAI_API_KEY", ""), "OPENAI_API_KEY", "env", "env"),
        )
