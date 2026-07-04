"""
tests/unit/runtime/test_sandbox_docker_image.py - Docker 沙箱镜像测试

验证默认沙箱镜像构建命令兼容 Docker Desktop BuildKit。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from haagent.runtime.sandbox import docker_image


def test_build_default_image_uses_temporary_context_with_dockerfile(monkeypatch) -> None:
    calls: list[tuple[list[str], Path | None, str | None]] = []

    def fake_run(argv, **kwargs):
        cwd = kwargs.get("cwd")
        dockerfile = Path(cwd) / "Dockerfile" if cwd is not None else None
        dockerfile_text = dockerfile.read_text(encoding="utf-8") if dockerfile is not None else None
        calls.append((list(argv), Path(cwd) if cwd is not None else None, dockerfile_text))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_image.subprocess, "run", fake_run)

    assert docker_image.build_default_image("haagent-sandbox:py311") is True

    argv, cwd, dockerfile_text = calls[0]
    assert argv == ["docker", "build", "-t", "haagent-sandbox:py311", "."]
    assert cwd is not None
    assert dockerfile_text == docker_image.dockerfile_content()
