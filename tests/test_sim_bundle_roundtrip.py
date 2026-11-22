"""Layer 1 acoustic simulator → calibration bundle → load_bundle → replay.

Builds the `minimappr-sim` Rust binary with cargo, renders a small
fixed-seed scenario (two Sirith tetra nodes + one phone, two bird calls)
into a calibration bundle, then checks that `bundle.load_bundle` accepts it
(including the simulator's additive ground-truth keys such as
`individual_id`) and that it replays through a full FusionNode.

Skips loudly when cargo is unavailable. Set MINIMAPPR_SIM_BINARY to use a
prebuilt binary instead of building.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from minimappr.calibration.bundle import load_bundle
from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CRATE_DIR = REPO_ROOT / "minimappr-sim"
SCENARIO_PATH = SIM_CRATE_DIR / "scenarios" / "roundtrip_small.toml"
CARGO_BUILD_TIMEOUT_S = 1800
RENDER_TIMEOUT_S = 120
DETECTION_POLL_TIMEOUT_S = 20.0
DETECTION_POLL_INTERVAL_S = 0.2
STUB_LABEL = "White-throated Sparrow"
EXPECTED_NODE_CHANNELS = {"tetra-a": 4, "tetra-b": 4, "phone-1": 1}
# Optional: path to an already-built minimappr-sim binary (skips cargo build).
PREBUILT_BINARY_ENV = "MINIMAPPR_SIM_BINARY"


class _StubBirdClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        return ClassificationResult(label=STUB_LABEL, confidence=0.9, scores={STUB_LABEL: 0.9}, features={})


def _cargo_or_skip() -> str:
    cargo = shutil.which("cargo")
    if cargo is None:
        message = (
            "cargo not found on PATH: the minimappr-sim bundle round-trip test is SKIPPED. "
            "Install Rust (https://rustup.rs) to run it."
        )
        warnings.warn(message)
        pytest.skip(message)
    return cargo


@pytest.fixture(scope="module")
def sim_binary() -> Path:
    prebuilt = os.environ.get(PREBUILT_BINARY_ENV)
    if prebuilt:
        binary = Path(prebuilt)
        assert binary.exists(), f"{PREBUILT_BINARY_ENV}={prebuilt} does not exist"
        return binary
    cargo = _cargo_or_skip()
    manifest = str(SIM_CRATE_DIR / "Cargo.toml")
    subprocess.run(
        [cargo, "build", "--release", "--quiet", "--manifest-path", manifest],
        check=True,
        timeout=CARGO_BUILD_TIMEOUT_S,
    )
    # Ask cargo where the binary went (crate-local today, workspace root later).
    metadata = json.loads(
        subprocess.run(
            [cargo, "metadata", "--format-version", "1", "--no-deps", "--manifest-path", manifest],
            check=True,
            capture_output=True,
            timeout=RENDER_TIMEOUT_S,
        ).stdout
    )
    binary = Path(metadata["target_directory"]) / "release" / "minimappr-sim"
    assert binary.exists(), f"cargo build did not produce {binary}"
    return binary


@pytest.fixture(scope="module")
def rendered_bundle(sim_binary: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_path = tmp_path_factory.mktemp("sim_bundle") / "roundtrip_small.zip"
    subprocess.run(
        [str(sim_binary), "render-bundle", "--scenario", str(SCENARIO_PATH), "--out", str(out_path)],
        check=True,
        timeout=RENDER_TIMEOUT_S,
    )
    return out_path


def test_rendered_bundle_loads_with_extended_ground_truth(rendered_bundle: Path) -> None:
    bundle = load_bundle(rendered_bundle)
    node_ids = [node["node_id"] for node in bundle.manifest["nodes"]]
    assert node_ids == list(EXPECTED_NODE_CHANNELS)
    for node_id, channel_count in EXPECTED_NODE_CHANNELS.items():
        audio, sample_rate_hz = bundle.node_audio(node_id)
        assert sample_rate_hz == 48_000
        assert audio.shape == (channel_count, 4 * 48_000)
        assert np.all(np.isfinite(audio))
        assert float(np.abs(audio).max()) > 0.0

    assert [event["individual_id"] for event in bundle.events] == ["wtsp-1", "chsp-1"]
    for event in bundle.events:
        assert event["geometry"]["type"] == "static"
        assert event["species"] and event["is_echo_source"] is False
        assert event["end_ns"] > event["start_ns"]
    assert bundle.expectations is not None
    assert bundle.expectations["runtime"]["classifier_backend"] == "stub"


@pytest.mark.asyncio
async def test_rendered_bundle_replays_through_fusion(rendered_bundle: Path, tmp_path: Path) -> None:
    bundle = load_bundle(rendered_bundle)
    fusion, storage, _settings = await build_fusion_for_bundle(
        bundle, tmp_path / "replay", classifier=_StubBirdClassifier()
    )
    try:
        accepted = await replay_bundle(fusion, bundle)
        assert accepted > 0, "no frames were accepted during replay"
        detections: list[dict] = []
        deadline = time.monotonic() + DETECTION_POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            detections = await storage.list_detections(limit=10_000)
            if detections:
                break
            await asyncio.sleep(DETECTION_POLL_INTERVAL_S)
    finally:
        await fusion.stop()
        await storage.close()

    report = evaluate_bundle(detections, bundle)
    assert len(report.event_results) == len(bundle.events)
    assert detections, "replay of the simulated bundle produced no detections"
    assert report.event_recall > 0.0
