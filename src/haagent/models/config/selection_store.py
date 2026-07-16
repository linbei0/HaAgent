"""
haagent/models/config/selection_store.py - 模型选择持久化

独占 settings.json 中 active、fallback 和云端 fallback 授权的读写。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from haagent.models.config.connections import ProviderProfileError, USER_SETTINGS_FILE, user_config_dir
from haagent.models.model_ref import ModelRef


@dataclass(frozen=True)
class ModelRoute:
    primary: ModelRef
    fallback: ModelRef | None
    cloud_fallback_consent: bool


class ModelSelectionStore:
    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or user_config_dir()
        self.path = self.config_dir / USER_SETTINGS_FILE

    def load_active(self) -> ModelRef:
        if not self.path.exists():
            raise ProviderProfileError("未找到默认模型配置，请运行 haagent 后在 TUI 内输入 /connect 配置供应商")
        value = self._load().get("active_model")
        if not isinstance(value, dict):
            raise ProviderProfileError("settings config must contain active_model")
        try:
            return ModelRef.from_dict(value, field_name="active_model")
        except ValueError as error:
            raise ProviderProfileError(str(error)) from error

    def load_route(self) -> ModelRoute:
        settings = self._load()
        primary = self.load_active()
        fallback_value = settings.get("fallback_model")
        try:
            fallback = ModelRef.from_dict(fallback_value, field_name="fallback_model") if isinstance(fallback_value, dict) else None
        except ValueError as error:
            raise ProviderProfileError(str(error)) from error
        return ModelRoute(primary, fallback, settings.get("cloud_fallback_consent") is True)

    def save_active(self, ref: ModelRef) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        settings = self._load() if self.path.exists() else {}
        settings["active_model"] = ref.to_dict()
        self._write(settings)
        return self.path

    def save_fallback(self, ref: ModelRef | None, *, cloud_consent: bool) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        settings = self._load() if self.path.exists() else {}
        if ref is None:
            settings.pop("fallback_model", None)
            settings["cloud_fallback_consent"] = False
        else:
            settings["fallback_model"] = ref.to_dict()
            settings["cloud_fallback_consent"] = cloud_consent
        self._write(settings)
        return self.path

    def remove_connection(self, connection_id: str, remaining_connection_ids: list[str]) -> None:
        if not self.path.exists():
            return
        settings = self._load()
        fallback = settings.get("fallback_model")
        if isinstance(fallback, dict) and fallback.get("connection_id") == connection_id:
            settings.pop("fallback_model", None)
            settings["cloud_fallback_consent"] = False
        active = settings.get("active_model")
        if isinstance(active, dict) and active.get("connection_id") == connection_id:
            if remaining_connection_ids:
                active["connection_id"] = remaining_connection_ids[0]
            else:
                settings.pop("active_model", None)
        if settings:
            self._write(settings)
        else:
            self.path.unlink()

    def _load(self) -> dict[str, object]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ProviderProfileError(f"settings config is invalid JSON: {self.path}") from error
        if not isinstance(value, dict):
            raise ProviderProfileError("settings config must be a JSON object")
        return value

    def _write(self, value: dict[str, object]) -> None:
        self.path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_active(*, config_dir: Path) -> ModelRef:
    return ModelSelectionStore(config_dir).load_active()


def load_route(*, config_dir: Path) -> ModelRoute:
    return ModelSelectionStore(config_dir).load_route()


def save_active(ref: ModelRef, *, config_dir: Path) -> Path:
    return ModelSelectionStore(config_dir).save_active(ref)


def save_fallback(ref: ModelRef | None, *, config_dir: Path, cloud_fallback_consent: bool = False) -> Path:
    return ModelSelectionStore(config_dir).save_fallback(ref, cloud_consent=cloud_fallback_consent)
