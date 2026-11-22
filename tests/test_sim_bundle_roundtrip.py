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

    call_events = [event for event in bundle.events if not event["is_echo_source"]]
    echo_events = [event for event in bundle.events if event["is_echo_source"]]
    assert [event["individual_id"] for event in call_events] == ["wtsp-1", "chsp-1"]
    for event in call_events:
        assert event["geometry"]["type"] == "static"
        assert event["species"]
        assert event["end_ns"] > event["start_ns"]
    call_ids = {event["event_id"] for event in call_events}
    assert echo_events, "trunk/ground reflections should be listed as echo events"
    assert {event["echo_of_event_id"] for event in echo_events} == call_ids
    assert bundle.expectations is not None
    assert bundle.expectations["runtime"]["classifier_backend"] == "stub"


def test_label_override_and_moving_source_load_as_v2(sim_binary: Path, tmp_path: Path) -> None:
    """A `label` that differs from `species`, and a source `trajectory`, must
    still pass the v2 loader's validation (label in species, sorted waypoints)."""
    scenario_text = SCENARIO_PATH.read_text()
    scenario_text = scenario_text.replace(
        'species = "White-throated Sparrow"',
        'species = "White-throated Sparrow"\nlabel = "WTSP"',
    ).replace(
        'position_m = [-4.0, 9.0, 4.0]',
        'position_m = [-4.0, 9.0, 4.0]\ntrajectory = [[2.0, -4.0, 9.0, 4.0], [4.0, 6.0, 9.0, 4.0]]',
    )
    variant_path = tmp_path / "variant.toml"
    variant_path.write_text(scenario_text)
    out_path = tmp_path / "variant.zip"
    subprocess.run(
        [str(sim_binary), "render-bundle", "--scenario", str(variant_path), "--out", str(out_path)],
        check=True,
        timeout=RENDER_TIMEOUT_S,
    )
    bundle = load_bundle(out_path)
    by_individual = {event["individual_id"]: event for event in bundle.events if not event["is_echo_source"]}
    assert by_individual["wtsp-1"]["label"] == "WTSP"
    assert {entry["label"] for entry in by_individual["wtsp-1"]["species"]} == {"WTSP", "White-throated Sparrow"}
    geometry = by_individual["chsp-1"]["geometry"]
    assert geometry["type"] == "trajectory"
    start_ns = bundle.manifest["time_window"]["start_ns"]
    assert [waypoint["t_ns"] for waypoint in geometry["waypoints"]] == [start_ns + 2_000_000_000, start_ns + 4_000_000_000]


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
    # Echo events are excluded from per-event results.
    assert len(report.event_results) == sum(not event["is_echo_source"] for event in bundle.events)
    assert report.entity_metrics is not None and report.entity_metrics.echo_event_count > 0
    assert detections, "replay of the simulated bundle produced no detections"
    assert report.event_recall > 0.0
