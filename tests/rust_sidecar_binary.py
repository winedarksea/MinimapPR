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
    return build_rust_sidecar_binary_in(REPOSITORY_ROOT)


@functools.lru_cache(maxsize=4)
def build_rust_sidecar_binary_in(repository_root: Path) -> Path:
    """Build the sidecar of the checkout at ``repository_root`` (the E10 gate measures other checkouts)."""
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
        cwd=str(repository_root),
        check=False,
        capture_output=True,
        text=True,
    )
    # stdout is captured for the JSON messages, so surface compiler errors here
    # instead of an opaque CalledProcessError.
    if completed.returncode != 0:
        raise RuntimeError(
            f"cargo build of {SIDECAR_BINARY_NAME!r} failed (exit {completed.returncode}); stderr:\n{completed.stderr}"
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
