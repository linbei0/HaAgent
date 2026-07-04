"""
src/haagent/runtime/sandbox/docker_image.py - Docker 沙箱镜像

提供默认 Dockerfile 内容，以及镜像存在性检查和默认镜像构建入口。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


DEFAULT_DOCKERFILE = """\
FROM python:3.11-slim-bookworm

RUN apt-get update \\
    && apt-get install -y --no-install-recommends bash git ripgrep ca-certificates curl \\
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash haagent
USER haagent
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
"""


def dockerfile_content() -> str:
    return DEFAULT_DOCKERFILE


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0


def build_default_image(image: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="haagent-sandbox-build-") as context_dir:
        Path(context_dir, "Dockerfile").write_text(DEFAULT_DOCKERFILE, encoding="utf-8")
        result = subprocess.run(
            ["docker", "build", "-t", image, "."],
            cwd=context_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    return result.returncode == 0
