"""Score Rust-simulator scenarios through the replay entity metrics.

Glue for W2-onset-srp (and later server-track streams): render a
`minimappr-sim/scenarios/*.toml` scene to a bundle, add echo ground truth from
the simulator's point reflectors, build a frequency-oracle classifier from the
scene's call generators, replay through a full FusionNode and score with
`evaluate_bundle` (entity metrics included).

Echo events: a point scatterer re-radiates from its own position (delayed by
source->trunk / c), so the apparent source of the echo is the trunk itself.
Current simulator bundles already carry those echo events (trunk + ground);
for older bundles that list trunks as reflector metadata only, one echo event
per (call, point reflector) is added here with that geometry.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import subprocess
import time
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from minimappr.calibration.bundle import CalibrationBundle, load_bundle
from minimappr.calibration.ground_truth_events import validate_ground_truth_payload
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle
from minimappr.sim.synthetic_entity_scenarios import DominantFrequencySpeciesClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIRECTORY = REPO_ROOT / "minimappr-sim" / "scenarios"
SIM_BINARY_ENV = "MINIMAPPR_SIM_BINARY"
DEFAULT_SIM_BINARY = REPO_ROOT / "target" / "release" / "minimappr-sim"
RENDER_TIMEOUT_SECONDS = 120
DETECTION_SETTLE_SECONDS = 3.0
DETECTION_DEADLINE_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 0.25
# Simulator defaults for generators whose TOML omits a frequency key.
GENERATOR_DEFAULT_BANDS_HZ = {
    "fm_sweep": (2500.0, 6500.0),
    "trill": (3600.0, 5400.0),
    "tonal_whistle": (3200.0, 4500.0),
    "harsh": (800.0, 9000.0),
}


def sim_binary_path() -> Path:
    binary = Path(os.environ.get(SIM_BINARY_ENV, str(DEFAULT_SIM_BINARY)))
    if not binary.exists():
        raise FileNotFoundError(
            f"minimappr-sim binary not found at {binary}; run `cargo build --release -p minimappr-sim` "
            f"or set {SIM_BINARY_ENV}"
        )
    return binary


def render_scenario(scenario_path: Path, out_path: Path, *, seed: int | None = None) -> Path:
    command = [str(sim_binary_path()), "render-bundle", "--scenario", str(scenario_path), "--out", str(out_path)]
    if seed is not None:
        command += ["--seed", str(seed)]
    subprocess.run(command, check=True, capture_output=True, timeout=RENDER_TIMEOUT_SECONDS)
    return out_path


def generator_band_hz(generator: dict[str, Any]) -> tuple[float, float]:
    kind = generator.get("kind", "")
    default_low, default_high = GENERATOR_DEFAULT_BANDS_HZ.get(kind, (0.0, 1e9))
    if kind == "fm_sweep":
        start = generator.get("start_frequency_hz", default_high)
        end = generator.get("end_frequency_hz", default_low)
        return min(start, end), max(start, end)
    if kind == "trill":
        carrier = generator.get("carrier_frequency_hz", 0.5 * (default_low + default_high))
        half_sweep = 0.5 * generator.get("element_sweep_hz", default_high - default_low)
        return carrier - half_sweep, carrier + half_sweep
    if kind == "tonal_whistle":
        start = generator.get("start_frequency_hz", default_low)
        end = generator.get("end_frequency_hz", start)
        steps = generator.get("frequency_step_ratio", 1.12) ** max(0, generator.get("note_count", 3) - 1)
        return min(start, end), max(start, end) * steps
    return default_low, default_high


def species_bands_from_scenario(scenario_path: Path) -> dict[str, tuple[float, float]]:
    scenario = tomllib.loads(Path(scenario_path).read_text())
    bands: dict[str, tuple[float, float]] = {}
    for source in scenario.get("sources", []):
        label = source.get("label") or source["species"]
        for call in source.get("calls", []):
            low, high = generator_band_hz(call["generator"])
            previous = bands.get(label, (low, high))
            bands[label] = (min(previous[0], low), max(previous[1], high))
    return bands


def echo_events_from_reflectors(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    echoes = []
    existing_event_ids = {event["event_id"] for event in ground_truth.get("events", [])}
    reflectors = [r for r in ground_truth.get("reflectors", []) if r.get("kind") == "point"]
    for event in ground_truth.get("events", []):
        if event.get("is_echo_source"):
            continue
        for reflector in reflectors:
            reflector_id = reflector.get("reflector_id") or "reflector"
            echo_event_id = f"{event['event_id']}-echo-{reflector_id}"
            if echo_event_id in existing_event_ids:
                continue
            echo = copy.deepcopy(event)
            echo.update(
                event_id=echo_event_id,
                individual_id=f"{event.get('individual_id', event['event_id'])}-echo-{reflector_id}",
                is_echo_source=True,
                echo_of_event_id=event["event_id"],
                reflector={"kind": "trunk", "position_m": reflector["position_m"]},
                geometry={"type": "static", "position_m": reflector["position_m"]},
            )
            echoes.append(echo)
    return echoes


def load_bundle_with_echo_truth(bundle_path: Path) -> CalibrationBundle:
    bundle = load_bundle(bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        ground_truth = json.loads(archive.read("ground_truth.json"))
    echoes = echo_events_from_reflectors(ground_truth)
    if not echoes:
        return bundle
    ground_truth["events"] = list(ground_truth["events"]) + echoes
    return dataclasses.replace(bundle, events=validate_ground_truth_payload(ground_truth))


async def collect_settled_detections(storage) -> list[dict]:
    deadline = time.monotonic() + DETECTION_DEADLINE_SECONDS
    last_count, last_change = -1, time.monotonic()
    detections: list[dict] = []
    while time.monotonic() < deadline:
        detections = await storage.list_detections(limit=10_000)
        if len(detections) != last_count:
            last_count, last_change = len(detections), time.monotonic()
        elif detections and time.monotonic() - last_change >= DETECTION_SETTLE_SECONDS:
            break
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    return detections


async def replay_and_score_scenario(
    scenario_path: Path,
    work_dir: Path,
    *,
    settings_overrides: dict[str, Any] | None = None,
    seed: int | None = None,
):
    """Render, replay and score one scene; returns (CalibrationReport, detections)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = render_scenario(scenario_path, work_dir / f"{Path(scenario_path).stem}.zip", seed=seed)
    bundle = load_bundle_with_echo_truth(bundle_path)
    fusion, storage, _ = await build_fusion_for_bundle(
        bundle,
        work_dir / "replay",
        overrides=settings_overrides or {},
        classifier=DominantFrequencySpeciesClassifier(species_bands_from_scenario(scenario_path)),
    )
    try:
        await replay_bundle(fusion, bundle)
        detections = await collect_settled_detections(storage)
    finally:
        await fusion.stop()
        await storage.close()
    return evaluate_bundle(detections, bundle), detections
