"""
haagent/models/model_runtime.py - 模型配置、解析与 gateway 组合门面

所有进程入口通过一个生命周期内的 ModelRuntime 使用同一 providers 快照。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from haagent.models.config.config_store import ModelConfigStore
from haagent.models.config.connections import provider_connection_credential_status
from haagent.models.config.selection_store import ModelSelectionStore
from haagent.models.gateway_registry import gateway_from_resolved, gateway_from_route
from haagent.models.model_ref import ModelChoice, ModelRef, ResolvedModel
from haagent.models.model_resolution import resolve_model
from haagent.models.types import ModelGateway


GatewayBuilder = Callable[[ResolvedModel], ModelGateway]


class ModelRuntime:
    """模型配置快照、解析和 gateway 创建的单一应用边界。"""

    def __init__(
        self,
        *,
        config_store: ModelConfigStore,
        environ: Mapping[str, str],
        gateway_builder: GatewayBuilder = gateway_from_resolved,
    ) -> None:
        self.config_store = config_store
        self.environ = environ
        self.gateway_builder = gateway_builder
        self.snapshot = config_store.load()
        self.selection_store = ModelSelectionStore(config_store.path.parent)

    @classmethod
    def load(
        cls,
        *,
        config_dir: Path,
        environ: Mapping[str, str],
        gateway_builder: GatewayBuilder = gateway_from_resolved,
    ) -> "ModelRuntime":
        return cls(
            config_store=ModelConfigStore(config_dir / "providers.json"),
            environ=environ,
            gateway_builder=gateway_builder,
        )

    def refresh(self) -> None:
        self.snapshot = self.config_store.load()

    def resolve(self, ref: ModelRef) -> ResolvedModel:
        return resolve_model(
            ref,
            snapshot=self.snapshot,
            environ=self.environ,
        )

    def create_gateway(self, ref: ModelRef) -> ModelGateway:
        return self.gateway_builder(self.resolve(ref))

    def create_route_gateway(self, primary_ref: ModelRef | None = None, *, retry_controller=None, route_event_sink=None) -> ModelGateway:
        route = self.selection_store.load_route()
        if self.gateway_builder is not gateway_from_resolved:
            return self.gateway_builder(self.resolve(primary_ref or route.primary))
        return gateway_from_route(
            self.resolve(primary_ref or route.primary),
            fallback_model=self.resolve(route.fallback) if route.fallback else None,
            cloud_fallback_consent=route.cloud_fallback_consent,
            retry_controller=retry_controller,
            route_event_sink=route_event_sink,
        )

    def list_choices(self) -> list[ModelChoice]:
        self.snapshot.require_valid()
        choices: list[ModelChoice] = []
        for connection in self.snapshot.records:
            diagnostics = self.snapshot.diagnostics_for(connection.id)
            model_ids = self.snapshot.available_models.get(connection.id, tuple(connection.models))
            for model_id in model_ids:
                config = connection.models.get(model_id)
                choices.append(
                    ModelChoice(
                        ref=ModelRef(connection.id, model_id),
                        connection_name=connection.name,
                        provider_name=connection.provider_name,
                        model_name=model_id,
                        variants=tuple(config.variants) if config is not None else (),
                        diagnostics=diagnostics,
                    )
                )
        return choices

    def credential_status(self, connection_id: str):
        connection = self.snapshot.connection(connection_id)
        return provider_connection_credential_status(
            connection,
            environ=self.environ,
            config_dir=self.snapshot.path.parent,
        )

    def list_connections(self):
        self.snapshot.require_valid()
        return list(self.snapshot.records)

    def set_active(self, ref: ModelRef) -> None:
        self.selection_store.save_active(ref)

    def set_fallback(self, ref: ModelRef | None, *, cloud_consent: bool) -> None:
        self.selection_store.save_fallback(ref, cloud_consent=cloud_consent)
