"""Build and locate the Rust ingest sidecar binary for oracle/e2e tests.

The sidecar is a member of the root Cargo workspace, so its binary lands in
the workspace target directory (``<repo>/target`` unless ``CARGO_TARGET_DIR``
or ``.cargo/config.toml`` say otherwise). Rather than guessing that path, the
build asks Cargo for the produced executable via ``--message-format=json``.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_PACKAGE_NAME = "minimappr-ingest-sidecar"
SIDECAR_BINARY_NAME = "minimappr-ingest-sidecar"


def cargo_is_available() -> bool:
    return shutil.which("cargo") is not None


@functools.lru_cache(maxsize=1)
def build_rust_sidecar_binary() -> Path:
    """Build the sidecar (debug profile) once per session and return its executable path."""
    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--quiet",
            "--message-format=json",
            "--package",
            SIDECAR_PACKAGE_NAME,
            "--bin",
            SIDECAR_BINARY_NAME,
        ],
        cwd=str(REPOSITORY_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        if message.get("target", {}).get("name") != SIDECAR_BINARY_NAME:
            continue
        executable = message.get("executable")
        if executable:
            return Path(executable)
    raise RuntimeError(
        f"cargo build reported no executable for {SIDECAR_BINARY_NAME!r}; stderr:\n{completed.stderr}"
    )
