"""
src/haagent/cli_runtime.py - CLI 运行期依赖集合

集中保存 CLI 入口需要替换的 runtime adapter，并提供 provider 与 smoke 配置构建。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from haagent.models.gateway_registry import gateway_from_profile, gateway_from_route
from haagent.models.model_connections import (
    ModelSelection,
    ProviderProfile,
    ProviderProfileError,
    load_active_model_selection,
    load_model_selection_profile,
    load_model_route,
)
from haagent.models.types import ModelGateway
from haagent.runtime.orchestration.orchestrator import RunOrchestrator


GatewayFactory = Callable[[ProviderProfile], ModelGateway]


@dataclass(frozen=True)
class SmokeDefinition:
    name: str
    task_path: Path
    requires_profile: bool


@dataclass
class CliRuntime:
    project_root: Path
    orchestrator_cls: type = RunOrchestrator
    # 测试可注入；默认与 TUI/multi_agent 共用 registry 工厂。
    gateway_factory: GatewayFactory = field(default=gateway_from_profile)

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
            if args.provider not in {None, "fake"} or args.base_url is not None:
                raise ProviderProfileError(
                    "--profile cannot be combined with --provider or --base-url",
                )
            active_selection = load_active_model_selection()
            selection = ModelSelection(
                connection_id=args.profile,
                model=args.model or active_selection.model,
            )
            return self.gateway_factory(load_model_selection_profile(selection))

        if args.provider is None:
            if args.model is not None or args.base_url is not None:
                raise ProviderProfileError("--model and --base-url require --provider or --profile")
            route = load_model_route()
            primary = load_model_selection_profile(route.primary)
            fallback = load_model_selection_profile(route.fallback) if route.fallback is not None else None
            if self.gateway_factory is gateway_from_profile:
                return gateway_from_route(
                    primary,
                    fallback_profile=fallback,
                    cloud_fallback_consent=route.cloud_fallback_consent,
                )
            return self.gateway_factory(primary)

        if args.provider in {"openai", "openai-chat"}:
            return self.gateway_factory(self._cli_provider_profile(args))
        return None

    def build_dogfood_model_gateway(self, args: argparse.Namespace) -> Any:
        if args.profile is not None:
            if args.provider is not None or args.base_url is not None:
                raise ProviderProfileError(
                    "--profile cannot be combined with --provider or --base-url",
                )
            active_selection = load_active_model_selection()
            selection = ModelSelection(
                connection_id=args.profile,
                model=args.model or active_selection.model,
            )
            return self.gateway_factory(load_model_selection_profile(selection))
        if args.provider is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderProfileError("OPENAI_API_KEY is not set; dogfood skipped")
        return self.build_run_model_gateway(args)

    def _cli_provider_profile(self, args: argparse.Namespace) -> ProviderProfile:
        """把 --provider/--model/--base-url 合成临时 profile，统一走 gateway_factory。"""
        provider = str(args.provider)
        default_models = {
            "openai": "gpt-4.1-mini",
            "openai-chat": "gpt-4.1-mini",
        }
        api_key_env = {
            "openai": "OPENAI_API_KEY",
            "openai-chat": "OPENAI_API_KEY",
        }[provider]
        api_key = os.environ.get(api_key_env) or ""
        model = args.model if args.model is not None else default_models[provider]
        base_url = args.base_url if args.base_url is not None else ""
        return ProviderProfile(
            name=f"cli-{provider}",
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            credential_source="env",
            credential_source_used="env",
            api_key=api_key,
        )
