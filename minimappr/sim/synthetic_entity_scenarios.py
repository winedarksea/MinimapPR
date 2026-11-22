"""Python-side synthetic entity scenarios rendered as ground-truth v2 bundles.

Stopgap until the Rust `minimappr-sim render-bundle` lands: free-field point
sources (1/r spreading, exact fractional delay via FFT phase shift, shared
sample clock, white noise floor) over synchronized tetrahedral nodes. Echoes are
modelled as attenuated image sources and recorded as `is_echo_source` events.
No drift, absorption or multipath beyond the explicit image source — enough to
baseline the entity metrics, not to validate DSP.
"""

from __future__ import annotations

import io
import json
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.calibration.ground_truth_events import GROUND_TRUTH_SCHEMA_VERSION
from minimappr.classifiers.base import AudioClassifier
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import ClassificationResult, GeoPoint
from minimappr.sim.geometry import regular_tetrahedron_offsets_m

NANOSECONDS_PER_SECOND = 1_000_000_000
SCENARIO_START_NS = 1_800_000_000 * NANOSECONDS_PER_SECOND
SPEED_OF_SOUND_MPS = 343.2
TETRA_EDGE_M = 0.05
SOURCE_REFERENCE_DISTANCE_M = 1.0
CALL_FADE_FRACTION = 0.1
SITE_ORIGIN = {"lat": 45.0, "lon": -93.0, "alt_m": 250.0}
DEFAULT_NODE_ORIGINS_M: dict[str, tuple[float, float, float]] = {
    "node-a": (0.0, 0.0, 0.0),
    "node-b": (20.0, 0.0, 0.0),
    "node-c": (10.0, 18.0, 0.0),
}
STUB_CLASSIFIER_CONFIDENCE = 0.9
STUB_CLASSIFIER_BACKGROUND_LABEL = "background"
STUB_CLASSIFIER_BACKGROUND_CONFIDENCE = 0.05
# Loop-validation thresholds (same as tests/test_calibration_harness_synthetic.py);
# entity gates stay report-only so the run measures, not judges.
BASELINE_EXPECTATIONS: dict[str, Any] = {
    "schema_version": 2,
    "runtime": {"classifier_backend": "stub", "settings_overrides": {}},
    "classification": {"min_label_accuracy": 0.5, "match": "exact"},
    "localization": {
        "min_event_recall": 0.5,
        "max_median_position_error_m": 50.0,
        "max_p90_position_error_m": 75.0,
        "far_field_distance_m": 120.0,
    },
    "entities": {"association_radius_m": 15.0, "association_time_slop_s": 2.0},
}


@dataclass(frozen=True)
class SyntheticCall:
    """One linear-FM call (whistle-like sweep) from a fixed point."""

    event_id: str
    individual_id: str
    species_label: str
    position_m: tuple[float, float, float]
    start_s: float
    duration_s: float
    sweep_start_hz: float
    sweep_end_hz: float
    source_amplitude: float = 2.0
    is_echo_source: bool = False
    echo_of_event_id: str | None = None
    reflector: dict[str, Any] | None = None


@dataclass(frozen=True)
class SyntheticEntityScenario:
    name: str
    description: str
    duration_s: float
    calls: tuple[SyntheticCall, ...]
    node_origins_m: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_NODE_ORIGINS_M)
    )
    noise_rms: float = 0.002


class DominantFrequencySpeciesClassifier(AudioClassifier):
    """Stub classifier: label = species whose band holds the spectral peak."""

    def __init__(self, species_bands_hz: dict[str, tuple[float, float]]) -> None:
        self._species_bands_hz = dict(species_bands_hz)

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        mono = np.asarray(samples, dtype=np.float64)
        if mono.ndim > 1:
            mono = mono.mean(axis=0)
        if mono.size == 0:
            return self._background()
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
        peak_hz = float(np.fft.rfftfreq(mono.size, 1.0 / sample_rate_hz)[int(np.argmax(spectrum))])
        for label, (low_hz, high_hz) in self._species_bands_hz.items():
            if low_hz <= peak_hz <= high_hz:
                return ClassificationResult(
                    label=label,
                    confidence=STUB_CLASSIFIER_CONFIDENCE,
                    scores={label: STUB_CLASSIFIER_CONFIDENCE},
                    features={"peak_hz": peak_hz},
                )
        return self._background()

    @staticmethod
    def _background() -> ClassificationResult:
        label = STUB_CLASSIFIER_BACKGROUND_LABEL
        confidence = STUB_CLASSIFIER_BACKGROUND_CONFIDENCE
        return ClassificationResult(label=label, confidence=confidence, scores={label: confidence})


def species_bands_for(scenario: SyntheticEntityScenario) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for call in scenario.calls:
        low_hz, high_hz = sorted((call.sweep_start_hz, call.sweep_end_hz))
        previous = bands.get(call.species_label, (low_hz, high_hz))
        bands[call.species_label] = (min(previous[0], low_hz), max(previous[1], high_hz))
    return bands


def _call_waveform(call: SyntheticCall, sample_rate_hz: int) -> np.ndarray:
    n_samples = int(round(call.duration_s * sample_rate_hz))
    t = np.arange(n_samples) / sample_rate_hz
    sweep_rate = (call.sweep_end_hz - call.sweep_start_hz) / max(call.duration_s, 1e-9)
    phase = 2.0 * np.pi * (call.sweep_start_hz * t + 0.5 * sweep_rate * t * t)
    envelope = np.ones(n_samples)
    fade = max(1, int(n_samples * CALL_FADE_FRACTION))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
    envelope[:fade], envelope[-fade:] = ramp, ramp[::-1]
    return np.sin(phase) * envelope


def _fractional_delay(signal: np.ndarray, delay_samples: float, out_length: int) -> np.ndarray:
    """Exact band-limited delay by FFT phase ramp (zero-padded to avoid wrap-around)."""
    n_fft = int(2 ** np.ceil(np.log2(out_length + signal.size + 1)))
    spectrum = np.fft.rfft(signal, n_fft)
    frequencies = np.fft.rfftfreq(n_fft)
    delayed = np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequencies * delay_samples), n_fft)
    return delayed[:out_length]


def _sensor_positions(scenario: SyntheticEntityScenario) -> dict[str, list[np.ndarray]]:
    offsets = [np.asarray(o) for o in regular_tetrahedron_offsets_m(TETRA_EDGE_M)]
    return {
        node_id: [np.asarray(origin, dtype=np.float64) + offset for offset in offsets]
        for node_id, origin in scenario.node_origins_m.items()
    }


def render_node_audio(
    scenario: SyntheticEntityScenario, sample_rate_hz: int, seed: int
) -> dict[str, np.ndarray]:
    """Channels-first float32 audio per node on a shared clock starting at t=0."""
    rng = np.random.default_rng(seed)
    total_samples = int(round(scenario.duration_s * sample_rate_hz))
    audio: dict[str, np.ndarray] = {}
    for node_id, sensors in _sensor_positions(scenario).items():
        channels = rng.normal(0.0, scenario.noise_rms, size=(len(sensors), total_samples))
        for call in scenario.calls:
            waveform = _call_waveform(call, sample_rate_hz)
            source_m = np.asarray(call.position_m, dtype=np.float64)
            for channel_index, sensor_m in enumerate(sensors):
                distance_m = float(np.linalg.norm(source_m - sensor_m))
                arrival_s = call.start_s + distance_m / SPEED_OF_SOUND_MPS
                gain = call.source_amplitude * SOURCE_REFERENCE_DISTANCE_M / max(
                    distance_m, SOURCE_REFERENCE_DISTANCE_M
                )
                channels[channel_index] += gain * _fractional_delay(
                    waveform, arrival_s * sample_rate_hz, total_samples
                )
        audio[node_id] = np.clip(channels, -1.0, 1.0).astype(np.float32)
    return audio


def _pcm16_wav_bytes(channels_first: np.ndarray, sample_rate_hz: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels_first.shape[0])
        out.setsampwidth(2)
        out.setframerate(sample_rate_hz)
        out.writeframes((channels_first.T * 32767.0).astype("<i2").tobytes())
    return buffer.getvalue()


def ground_truth_payload(scenario: SyntheticEntityScenario) -> dict:
    events = []
    for call in scenario.calls:
        event: dict[str, Any] = {
            "event_id": call.event_id,
            "individual_id": call.individual_id,
            "label": call.species_label,
            "label_category": "bird",
            "species": [{"label": call.species_label, "label_category": "bird", "status": "true"}],
            "source": "synthetic",
            "geometry": {"type": "static", "position_m": list(call.position_m)},
            "start_ns": SCENARIO_START_NS + int(call.start_s * NANOSECONDS_PER_SECOND),
            "end_ns": SCENARIO_START_NS
            + int((call.start_s + call.duration_s) * NANOSECONDS_PER_SECOND),
        }
        if call.is_echo_source:
            event.update(
                is_echo_source=True, echo_of_event_id=call.echo_of_event_id, reflector=call.reflector
            )
        events.append(event)
    return {"schema_version": GROUND_TRUTH_SCHEMA_VERSION, "events": events}


def render_scenario_bundle(
    scenario: SyntheticEntityScenario,
    out_path: Path,
    *,
    sample_rate_hz: int = 16_000,
    seed: int = 0,
    expectations: dict[str, Any] | None = None,
) -> Path:
    """Write a calibration bundle zip (manifest v1 + ground truth v2) for the scenario."""
    frame = LocalCoordinateFrame(origin=GeoPoint(**SITE_ORIGIN), mode="flat")
    audio = render_node_audio(scenario, sample_rate_hz, seed)
    offsets = [list(o) for o in regular_tetrahedron_offsets_m(TETRA_EDGE_M)]
    nodes = []
    for node_id, origin in scenario.node_origins_m.items():
        geo = frame.local_to_geo(origin)
        nodes.append(
            {
                "node_id": node_id,
                "audio_file": f"audio/{node_id}.wav",
                "sample_rate_hz": sample_rate_hz,
                "audio_start_time_ns": SCENARIO_START_NS,
                "channel_sensor_ids": [f"{node_id}:ch{i}" for i in range(len(offsets))],
                "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
                "position_m": list(origin),
                "sensor_offsets_m": offsets,
                "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            }
        )
    end_ns = SCENARIO_START_NS + int(scenario.duration_s * NANOSECONDS_PER_SECOND)
    manifest = {
        "schema_version": 1,
        "kind": "minimappr_calibration_bundle",
        "session_id": f"synthetic-{scenario.name}",
        "time_window": {"start_ns": SCENARIO_START_NS, "end_ns": end_ns},
        "site": {"origin": SITE_ORIGIN, "coordinate_mode": "flat"},
        "environment": {"temperature_c": 20.0, "humidity_fraction": 0.5, "source": "synthetic"},
        "nodes": nodes,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("ground_truth.json", json.dumps(ground_truth_payload(scenario), indent=2))
        zf.writestr("detections.json", json.dumps({"schema_version": 1, "detections": []}))
        zf.writestr("expectations.json", json.dumps(expectations or BASELINE_EXPECTATIONS, indent=2))
        for node_id, channels_first in audio.items():
            zf.writestr(f"audio/{node_id}.wav", _pcm16_wav_bytes(channels_first, sample_rate_hz))
    return out_path


def _robin(event_id, individual_id, position_m, start_s, **extra) -> SyntheticCall:
    return SyntheticCall(event_id, individual_id, "robin", position_m, start_s, 0.5, 2000.0, 2600.0, **extra)


def _tanager(event_id, individual_id, position_m, start_s, **extra) -> SyntheticCall:
    return SyntheticCall(event_id, individual_id, "tanager", position_m, start_s, 0.5, 2900.0, 3500.0, **extra)


ROBIN_PERCH_M = (6.0, 12.0, 2.0)
TANAGER_PERCH_M = (18.0, 9.0, 3.0)
TRUNK_IMAGE_M = (-8.0, 24.0, 2.0)
TRUNK_ECHO_AMPLITUDE_FRACTION = 0.4  # ≈ −8 dB, inside the test strategy's 6–18 dB range

SYNTHETIC_ENTITY_SCENARIOS: tuple[SyntheticEntityScenario, ...] = (
    SyntheticEntityScenario(
        name="one_bird_two_bouts",
        description="One robin singing two bouts 6 s apart from one perch",
        duration_s=11.0,
        calls=(_robin("bout-1", "robin-1", ROBIN_PERCH_M, 1.0),
               _robin("bout-2", "robin-1", ROBIN_PERCH_M, 7.0)),
    ),
    SyntheticEntityScenario(
        name="two_species_overlap",
        description="Robin and tanager 12 m apart, calls overlapping by 0.2 s, twice",
        duration_s=11.0,
        calls=(_robin("robin-1a", "robin-1", ROBIN_PERCH_M, 1.0),
               _tanager("tanager-1a", "tanager-1", TANAGER_PERCH_M, 1.3),
               _robin("robin-1b", "robin-1", ROBIN_PERCH_M, 7.0),
               _tanager("tanager-1b", "tanager-1", TANAGER_PERCH_M, 7.3)),
    ),
    SyntheticEntityScenario(
        name="one_bird_trunk_echo",
        description="One robin, two bouts, each with a trunk-reflection image source at −8 dB",
        duration_s=11.0,
        calls=tuple(
            call
            for index, start_s in enumerate((1.0, 7.0))
            for call in (
                _robin(f"bout-{index + 1}", "robin-1", ROBIN_PERCH_M, start_s),
                _robin(
                    f"bout-{index + 1}-echo", f"echo-{index + 1}", TRUNK_IMAGE_M, start_s,
                    source_amplitude=2.0 * TRUNK_ECHO_AMPLITUDE_FRACTION, is_echo_source=True,
                    echo_of_event_id=f"bout-{index + 1}",
                    reflector={"kind": "trunk", "position_m": [-1.0, 18.0, 2.0]},
                ),
            )
        ),
    ),
)
