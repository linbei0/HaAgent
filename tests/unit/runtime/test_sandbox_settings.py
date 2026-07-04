"""
tests/unit/runtime/test_sandbox_settings.py - 沙箱 runtime settings 测试

验证沙箱配置默认值、嵌套 runtime 配置读取，以及非法字段会被拒绝。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.sandbox.settings import (
    DockerSandboxSettings,
    SandboxSettings,
    SandboxSettingsError,
    load_sandbox_settings,
)
from haagent.runtime.settings import RuntimeSettings, load_runtime_settings


def test_default_sandbox_settings_use_local_subprocess() -> None:
    settings = load_sandbox_settings(None)

    assert settings.enabled is False
    assert settings.backend == "local_subprocess"
    assert settings.fail_if_unavailable is False
    assert settings.docker.image == "haagent-sandbox:py311"
    assert settings.docker.network == "none"
    assert settings.docker.cpu_limit == 1.0
    assert settings.docker.memory_limit == "1g"
    assert settings.docker.pids_limit == 128


def test_runtime_settings_loads_nested_sandbox_config(tmp_path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps(
            {
                "interactive_max_turns": 12,
                "sandbox": {
                    "enabled": True,
                    "backend": "docker",
                    "fail_if_unavailable": True,
                    "docker": {
                        "image": "custom:latest",
                        "auto_build_image": False,
                        "cpu_limit": 2.0,
                        "memory_limit": "2g",
                        "pids_limit": 256,
                        "network": "none",
                        "read_only_rootfs": True,
                        "tmpfs": ["/tmp:rw,noexec,nosuid,size=512m"],
                        "extra_readonly_mounts": [str(tmp_path / "readonly")],
                        "extra_env_names": ["UV_CACHE_DIR"],
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path=config_path)

    assert settings.interactive_max_turns == 12
    assert settings.sandbox.enabled is True
    assert settings.sandbox.backend == "docker"
    assert settings.sandbox.fail_if_unavailable is True
    assert settings.sandbox.docker.image == "custom:latest"
    assert settings.sandbox.docker.auto_build_image is False
    assert settings.sandbox.docker.cpu_limit == 2.0
    assert settings.sandbox.docker.memory_limit == "2g"
    assert settings.sandbox.docker.pids_limit == 256
    assert settings.sandbox.docker.tmpfs == ["/tmp:rw,noexec,nosuid,size=512m"]
    assert settings.sandbox.docker.extra_env_names == ["UV_CACHE_DIR"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"backend": "vm"}, "sandbox.backend must be local_subprocess or docker"),
        ({"docker": {"network": "bridge"}}, "sandbox.docker.network only supports none"),
        ({"docker": {"cpu_limit": 0}}, "sandbox.docker.cpu_limit must be positive"),
        ({"docker": {"memory_limit": ""}}, "sandbox.docker.memory_limit must be non-empty"),
        ({"docker": {"pids_limit": 0}}, "sandbox.docker.pids_limit must be positive"),
        (
            {"docker": {"extra_env_names": ["OPENAI_API_KEY=secret"]}},
            "sandbox.docker.extra_env_names must contain variable names",
        ),
    ],
)
def test_sandbox_settings_reject_invalid_values(raw: dict[str, object], message: str) -> None:
    with pytest.raises(SandboxSettingsError, match=message):
        load_sandbox_settings(raw)


def test_runtime_settings_dataclass_keeps_sandbox_default() -> None:
    settings = RuntimeSettings()

    assert isinstance(settings.sandbox, SandboxSettings)
    assert isinstance(settings.sandbox.docker, DockerSandboxSettings)
