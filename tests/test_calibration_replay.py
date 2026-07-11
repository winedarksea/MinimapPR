"""Replay gate for field-collected calibration bundles.

Discovers `tests/data/calibration/*.zip`, replays each bundle through the
full localization + classification pipeline, and enforces its expectations.
Skips entirely when no bundles are present (bundles are gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.calibration.bundle import load_bundle
from minimappr.sim.replay import (
    DEFAULT_EXPECTATIONS,
    build_fusion_for_bundle,
    evaluate_bundle,
    replay_bundle,
)

BUNDLES = sorted((Path(__file__).parent / "data" / "calibration").glob("*.zip"))

pytestmark = pytest.mark.skipif(
    not BUNDLES, reason="no calibration bundles in tests/data/calibration"
)


def _skip_unless_classifier_available(expectations: dict) -> None:
    backend = (expectations.get("runtime") or {}).get("classifier_backend", "yamnet")
    if backend == "yamnet":
        pytest.importorskip("tensorflow")
    elif backend == "birdnet":
        pytest.importorskip("birdnet")


@pytest.mark.asyncio
@pytest.mark.parametrize("bundle_path", BUNDLES, ids=lambda p: p.stem)
async def test_calibration_bundle_replay(bundle_path: Path, tmp_path: Path) -> None:
    bundle = load_bundle(bundle_path)
    expectations = bundle.expectations or DEFAULT_EXPECTATIONS
    _skip_unless_classifier_available(expectations)

    fusion, storage, _settings = await build_fusion_for_bundle(bundle, tmp_path)
    try:
        accepted = await replay_bundle(fusion, bundle)
        assert accepted > 0, "no frames were accepted during replay"
    finally:
        await fusion.stop()

    try:
        detections = await storage.list_detections(limit=10_000)
        report = evaluate_bundle(detections, bundle, expectations)
    finally:
        await storage.close()

    assert report.passed, "\n".join(report.errors)
