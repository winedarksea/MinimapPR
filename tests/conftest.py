from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_config_overrides(monkeypatch, tmp_path):
    """Point the persisted-overrides file at a per-test tmp path.

    Without this, ``PATCH /api/v1/config`` (which now persists to YAML) would
    write the repo's default ``data/config.yml`` and pollute other tests.
    """
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(tmp_path / "config.yml"))

