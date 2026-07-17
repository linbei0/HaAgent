"""
haagent/models/model_runtime.py - 模型配置、解析与 gateway 组合门面

所有进程入口通过一个生命周期内的 ModelRuntime 使用同一 providers 快照。
配置快照、选择、可用模型合并、连接事务与 route 构造均为内部实现细节。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from haagent.models.catalog import CatalogFetchResult
from haagent.models.config.config_store import ModelConfigStore
from haagent.models.config.connections import (
    ProviderConnectionRecord,
    ProviderProfileError,
    ProvidersConfigSnapshot,
    provider_connection_credential_status,
    save_connection_api_key,
)
from haagent.models.config.credentials import CredentialError
from haagent.models.config.selection_store import ModelRoute, ModelSelectionStore
from haagent.models.gateway_registry import gateway_from_resolved, gateway_from_route
from haagent.models.local_runtime import LocalRuntimeDiscovery
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
        # 私有状态：外部只经公开方法访问，禁止直接读写 snapshot/store。
        self._config_store = config_store
        self._environ = environ
        self._gateway_builder = gateway_builder
        self._snapshot = config_store.load()
        self._selection_store = ModelSelectionStore(config_store.path.parent)

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

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ

    def refresh(self) -> None:
        self._snapshot = self._config_store.load()

    def resolve(self, ref: ModelRef) -> ResolvedModel:
        return resolve_model(
            ref,
            snapshot=self._snapshot,
            environ=self._environ,
        )

    def create_gateway(self, ref: ModelRef) -> ModelGateway:
        return self._gateway_builder(self.resolve(ref))

    def create_route_gateway(
        self,
        primary_ref: ModelRef | None = None,
        *,
        retry_controller=None,
        route_event_sink=None,
    ) -> ModelGateway:
        route = self._selection_store.load_route()
        if self._gateway_builder is not gateway_from_resolved:
            return self._gateway_builder(self.resolve(primary_ref or route.primary))
        return gateway_from_route(
            self.resolve(primary_ref or route.primary),
            fallback_model=self.resolve(route.fallback) if route.fallback else None,
            cloud_fallback_consent=route.cloud_fallback_consent,
            retry_controller=retry_controller,
            route_event_sink=route_event_sink,
        )

    def list_choices(self) -> list[ModelChoice]:
        self._snapshot.require_valid()
        choices: list[ModelChoice] = []
        for connection in self._snapshot.records:
            diagnostics = self._snapshot.diagnostics_for(connection.id)
            model_ids = self._snapshot.available_models.get(connection.id, tuple(connection.models))
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
        connection = self._snapshot.connection(connection_id)
        return provider_connection_credential_status(
            connection,
            environ=self._environ,
            config_dir=self._snapshot.path.parent,
        )

    def list_connections(self) -> list[ProviderConnectionRecord]:
        self._snapshot.require_valid()
        return list(self._snapshot.records)

    def connection(self, connection_id: str) -> ProviderConnectionRecord:
        return self._snapshot.connection(connection_id)

    def diagnostics_for(self, connection_id: str) -> tuple[str, ...]:
        return self._snapshot.diagnostics_for(connection_id)

    def load_active(self) -> ModelRef:
        return self._selection_store.load_active()

    def load_route(self) -> ModelRoute:
        return self._selection_store.load_route()

    def set_active(self, ref: ModelRef) -> None:
        self._selection_store.save_active(ref)

    def set_fallback(self, ref: ModelRef | None, *, cloud_consent: bool) -> None:
        self._selection_store.save_fallback(ref, cloud_consent=cloud_consent)

    def ref_for_connection(self, connection_id: str, model: str | None = None) -> ModelRef:
        """用指定连接构造 ModelRef；model 缺省时沿用当前 active 模型名。"""
        selected_model = model or self.load_active().model
        return ModelRef(connection_id, selected_model)

    def save_connection(
        self,
        record: ProviderConnectionRecord,
        *,
        api_key: str | None = None,
        available_models: Mapping[str, set[str]] | None = None,
    ) -> ProviderConnectionRecord:
        """写入连接配置；凭据失败时回滚 providers.json，避免半提交。"""
        previous = self._snapshot
        previous_record = next((item for item in previous.records if item.id == record.id), None)
        snapshot = self._config_store.save_connection(record, expected_digest=previous.digest)
        try:
            save_connection_api_key(record, api_key, config_dir=snapshot.path.parent)
        except (ProviderProfileError, CredentialError, OSError) as error:
            # 凭据写入失败：恢复写盘前连接集合，防止配置已落盘但密钥缺失。
            try:
                self._rollback_connection_write(previous, previous_record, snapshot.digest, record.id)
            except ProviderProfileError as rollback_error:
                raise ProviderProfileError(
                    f"credential write failed ({error}); config rollback also failed: {rollback_error}",
                ) from rollback_error
            raise
        if available_models is not None:
            snapshot = snapshot.bind_available_models(available_models)
        self._snapshot = snapshot
        return record

    def delete_connection(self, connection_id: str) -> None:
        """删除连接并同步清理 selection；顺序固定为 config 再 selection。"""
        snapshot = self._config_store.delete_connection(
            connection_id,
            expected_digest=self._snapshot.digest,
        )
        self._selection_store.remove_connection(
            connection_id,
            [record.id for record in snapshot.records],
        )
        self._snapshot = snapshot

    def bind_available_models(
        self,
        available_models: Mapping[str, set[str]],
        *,
        source: str | None = None,
    ) -> None:
        self._snapshot = self._snapshot.bind_available_models(available_models, source=source)

    def bind_local_discoveries(self, discoveries: Sequence[LocalRuntimeDiscovery]) -> None:
        available: dict[str, set[str]] = {}
        for connection in self._snapshot.records:
            discovery = next(
                (
                    item
                    for item in discoveries
                    if item.runtime_kind == connection.runtime_kind and item.status == "available"
                ),
                None,
            )
            if discovery is not None:
                available[connection.id] = {model.id for model in discovery.models}
        self.bind_available_models(available, source="local")

    def bind_remote_catalog(self, catalog: CatalogFetchResult) -> None:
        providers = {provider.id: provider for provider in catalog.providers}
        available = {
            connection.id: {model.id for model in providers[connection.provider_id].models}
            for connection in self._snapshot.records
            if connection.runtime_kind == "remote" and connection.provider_id in providers
        }
        self.bind_available_models(available, source="remote")

    def _rollback_connection_write(
        self,
        previous: ProvidersConfigSnapshot,
        previous_record: ProviderConnectionRecord | None,
        failed_digest: str,
        connection_id: str,
    ) -> None:
        # 磁盘回滚成功后保留 previous 的 catalog/local 内存态；失败必须显式抛错。
        if previous_record is not None:
            restored = self._config_store.save_connection(previous_record, expected_digest=failed_digest)
        else:
            restored = self._config_store.delete_connection(connection_id, expected_digest=failed_digest)
        self._snapshot = ProvidersConfigSnapshot(
            path=restored.path,
            records=restored.records,
            digest=restored.digest,
            load_error=restored.load_error,
            invalid_model_configs=previous.invalid_model_configs,
            available_models=previous.available_models,
        )
