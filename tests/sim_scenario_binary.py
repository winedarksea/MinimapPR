"""Shared helper: make sure the Rust `minimappr-sim` binary exists for scenario gates.

Used by the Field Mode simulator-scene tests (W2-onset-srp, W3-attribution).
Builds it with cargo when missing; skips the calling module when cargo is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import pytest

from minimappr.sim import sim_scenario_scoring as scoring

CARGO_BUILD_TIMEOUT_S = 1800


def ensure_sim_binary(gate_name: str) -> None:
    if os.environ.get(scoring.SIM_BINARY_ENV) or scoring.DEFAULT_SIM_BINARY.exists():
        return
    cargo = shutil.which("cargo")
    if cargo is None:
        message = f"cargo not found: {gate_name} simulator scenario gate SKIPPED"
        warnings.warn(message)
        pytest.skip(message)
    subprocess.run(
        [cargo, "build", "--release", "--quiet", "-p", "minimappr-sim"],
        cwd=scoring.REPO_ROOT,
        check=True,
        timeout=CARGO_BUILD_TIMEOUT_S,
    )
