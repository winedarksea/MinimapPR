from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HEURISTIC_ROUTING_JSON = """{
  "version": 1,
  "classifiers": {"heuristic": {"backend": "heuristic"}},
  "contexts": {
    "detection_trigger": {"run": ["heuristic"]},
    "localized_render": {"run": ["heuristic"]},
    "omni_continuous": {"run": []}
  },
  "chains": [],
  "triggers": []
}
"""


@pytest.fixture
async def temp_storage(tmp_path):
    """An initialized `Storage` on a per-test SQLite file, closed on teardown.

    Building this by hand is repeated across many test modules; new tests should
    take this fixture rather than add another copy.
    """
    from minimappr.storage.db import Storage

    store = Storage(tmp_path / "test.db")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture(autouse=True)
def _isolate_config_overrides(monkeypatch, tmp_path):
    """Point the persisted-overrides file at a per-test tmp path.

    Without this, ``PATCH /api/v1/config`` (which now persists to YAML) would
    write the repo's default ``data/config.yml`` and pollute other tests.
    """
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(tmp_path / "config.yml"))


@pytest.fixture(autouse=True)
def _heuristic_classifier_routing(monkeypatch, tmp_path):
    """Default every test to heuristic-only classifier routing.

    The shipped routing config always runs YAMNet (+BirdNET) — loading real
    models in every TestClient/from_env test would be prohibitively slow.
    Tests exercising real routing write their own config and re-set
    MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH.
    """
    routing_path = tmp_path / "classifier_routing.json"
    routing_path.write_text(HEURISTIC_ROUTING_JSON, encoding="utf-8")
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH", str(routing_path))
    return routing_path

