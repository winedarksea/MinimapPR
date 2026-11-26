"""Offline halves of the faint-call admission sweep (no app, no sidecar).

- **Omni scan replay**: the real ``ContinuousOmniScanner._scan_node`` on the
  sweep capture, at several scan phases (``startup`` is the scanner's own
  first deadline, one interval after start). A call counts as omni-classified
  when one scan window fully contains it and that window passes the gate; the
  classifier is a constant bird, as in the full-app runs. The full-app direct
  runs keep the scanner off: it schedules on wall time while the harness
  streams faster than real time, which would distort its coverage.
- **Trigger noise replay**: the Python ingest trigger decision (absolute
  ``trigger_rms`` OR the relative trigger, then the cooldown) over long
  noise-only backgrounds, for false triggers per hour at a resolution the
  minutes-long full-app runs cannot reach, and the per-frame compute cost.
  The Rust relative trigger makes the same decisions for the same levels
  (``relative_trigger_short_block`` parity goldens).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from minimappr.config import Settings
from minimappr.core.ingest_relative_trigger import NoiseFloorRelativeTrigger, peak_short_block_rms
from minimappr.core.omni_scanner import ContinuousOmniScanner
from minimappr.models import NodeSpec, NodeType
from minimappr.utils.audio import trigger_rms
from tests.faint_admission_sweep_fixtures import (
    FRAME_SECONDS,
    Background,
    ScheduledCall,
    render_background,
)
from tests.range_matrix_cells import SAMPLE_RATE_HZ
from tests.realistic_scene_direct_ingest import ConstantBirdClassifier
from tests.realistic_scene_full_app_run import FRAME_SAMPLES

PCM16_FULL_SCALE = 32768.0
SECONDS_PER_HOUR = 3600.0
OMNI_SCAN_PHASE_STARTUP = "startup"
# Scan phases besides startup, as fractions of the interval (scan k at (k + f) x interval).
OMNI_SCAN_PHASE_FRACTIONS = (0.25, 0.5, 0.75)
NOISE_REPLAY_CHUNK_SECONDS = 30.0
# Steady-state false triggers exclude the floor's first convergence.
NOISE_REPLAY_STARTUP_SECONDS = 30.0
DECIBELS_PER_POWER_DECADE = 10.0


def pcm16_received(capture: np.ndarray) -> np.ndarray:
    """What the server decodes from the firmware's PCM16 upload."""
    return (np.clip(np.round(capture.astype(np.float64) * PCM16_FULL_SCALE), -32768, 32767) / PCM16_FULL_SCALE).astype(
        np.float32
    )


# ---------------------------------------------------------------- omni scan replay


class _CaptureWindowBuffer:
    """Serves the scan window ending at ``scan_end_seconds`` of a fixed capture."""

    def __init__(self, sensor_ids: tuple[str, ...], capture: np.ndarray) -> None:
        self._sensor_ids = sensor_ids
        self._capture = capture
        self.scan_end_seconds = 0.0

    async def get_recent_window_for_sensors(self, sensor_ids, window_seconds):
        end = int(round(self.scan_end_seconds * SAMPLE_RATE_HZ))
        start = max(0, end - int(round(window_seconds * SAMPLE_RATE_HZ)))
        if end <= start:
            return None
        aligned = {sensor_id: self._capture[index, start:end] for index, sensor_id in enumerate(self._sensor_ids)}
        return aligned, SAMPLE_RATE_HZ, int(self.scan_end_seconds * 1e9)


class _SingleNodeRegistry:
    def __init__(self, node: NodeSpec, sensor_ids: tuple[str, ...]) -> None:
        self._runtime = SimpleNamespace(spec=node, sensor_ids=list(sensor_ids))

    async def list_nodes(self):
        return [self._runtime]


@dataclass(frozen=True)
class OmniScanReplayResult:
    gate_mode: str
    phase: str
    classified_call_indices: frozenset[int]
    scans: int
    scans_passing_gate_without_call: int
    capture_seconds: float


def _scan_end_times_seconds(phase: str, interval_seconds: float, capture_seconds: float) -> list[float]:
    offset = interval_seconds if phase == OMNI_SCAN_PHASE_STARTUP else float(phase) * interval_seconds
    return list(np.arange(offset, capture_seconds + 1e-9, interval_seconds))


def replay_omni_scans(
    capture: np.ndarray,
    sensor_ids: tuple[str, ...],
    schedule: list[ScheduledCall],
    *,
    clearest_channel_gate: bool,
    settings: Settings,
) -> list[OmniScanReplayResult]:
    received = pcm16_received(capture)
    capture_seconds = received.shape[1] / SAMPLE_RATE_HZ
    scanner_settings = SimpleNamespace(
        omni_scan_enabled=True,
        omni_scan_interval_seconds=settings.omni_scan_interval_seconds,
        omni_scan_window_seconds=settings.omni_scan_window_seconds,
        omni_scan_min_rms=settings.omni_scan_min_rms,
        omni_scan_clearest_channel_gate_enabled=clearest_channel_gate,
    )
    node = NodeSpec(id="omni-replay", node_type=NodeType.SIRITH_TETRA, position_m=(0.0, 0.0, 0.0), sensor_offsets_m=[(0.0, 0.0, 0.0)] * len(sensor_ids))
    results = []
    phases = [OMNI_SCAN_PHASE_STARTUP, *(repr(fraction) for fraction in OMNI_SCAN_PHASE_FRACTIONS)]
    for phase in phases:
        buffer = _CaptureWindowBuffer(sensor_ids, received)
        passed_scan_ends: list[float] = []

        async def _sink(result, ends=passed_scan_ends, current=buffer) -> None:
            ends.append(current.scan_end_seconds)

        scanner = ContinuousOmniScanner(
            settings=scanner_settings,
            classifier=ConstantBirdClassifier(),
            registry=_SingleNodeRegistry(node, sensor_ids),
            buffer=buffer,
            sink=_sink,
        )
        scan_ends = _scan_end_times_seconds(phase, settings.omni_scan_interval_seconds, capture_seconds)

        async def _run_scans(scans=scan_ends, current=buffer, current_scanner=scanner) -> None:
            for scan_end in scans:
                current.scan_end_seconds = scan_end
                await current_scanner._scan_node(node, list(sensor_ids))

        asyncio.run(_run_scans())
        window = settings.omni_scan_window_seconds
        classified = frozenset(
            call.call_index
            for call in schedule
            if any(end - window <= call.arrival_start_seconds and call.arrival_end_seconds <= end for end in passed_scan_ends)
        )
        without_call = sum(
            1
            for end in passed_scan_ends
            if not any(call.arrival_end_seconds > end - window and call.arrival_start_seconds < end for call in schedule)
        )
        results.append(
            OmniScanReplayResult(
                gate_mode="clearest_channel" if clearest_channel_gate else "channel_mean",
                phase=phase,
                classified_call_indices=classified,
                scans=len(scan_ends),
                scans_passing_gate_without_call=without_call,
                capture_seconds=capture_seconds,
            )
        )
    return results


def clearest_over_mean_gain_db(capture: np.ndarray, call: ScheduledCall) -> float:
    """How much louder the clearest channel is than the channel mean over the call
    (phase cancellation of the mean shows as a large positive gain)."""
    received = pcm16_received(capture).astype(np.float64)
    start = int(call.arrival_start_seconds * SAMPLE_RATE_HZ)
    end = int(np.ceil(call.arrival_end_seconds * SAMPLE_RATE_HZ)) + 1
    segment = received[:, start:end]
    clearest_power = float(np.max(np.mean(segment**2, axis=1)))
    mean_power = float(np.mean(np.mean(segment, axis=0) ** 2))
    return DECIBELS_PER_POWER_DECADE * np.log10(clearest_power / mean_power)


# ---------------------------------------------------------------- trigger noise replay


@dataclass(frozen=True)
class TriggerNoiseReplayResult:
    candidate: str
    background: str
    replay_seconds: float
    startup_false_triggers: int
    steady_false_triggers: int
    microseconds_per_frame: float

    @property
    def steady_false_triggers_per_hour(self) -> float:
        return self.steady_false_triggers * SECONDS_PER_HOUR / (self.replay_seconds - NOISE_REPLAY_STARTUP_SECONDS)


def replay_python_trigger_on_noise(
    candidate: str,
    background: Background,
    settings: Settings,
    *,
    relative_enabled: bool,
    short_block_enabled: bool,
    replay_seconds: float,
    channel_count: int = 4,
) -> TriggerNoiseReplayResult:
    """The ``IngestProcessor`` decision per 20 ms frame: absolute gate OR the
    relative trigger, then a plain cooldown (one node, so the geometry-aware
    same-event window is the cooldown)."""
    rng = np.random.default_rng(background.seed + 1000)
    trigger = NoiseFloorRelativeTrigger(settings.relative_trigger_config()) if relative_enabled else None
    chunk_samples = int(NOISE_REPLAY_CHUNK_SECONDS * SAMPLE_RATE_HZ)
    frames_per_chunk = chunk_samples // FRAME_SAMPLES
    last_trigger_seconds = -np.inf
    startup = steady = 0
    frame_index = 0
    decision_seconds = 0.0
    for _ in range(int(np.ceil(replay_seconds / NOISE_REPLAY_CHUNK_SECONDS))):
        chunk = pcm16_received(render_background(background, channel_count, chunk_samples, rng))
        for local_frame in range(frames_per_chunk):
            frame = chunk[:, local_frame * FRAME_SAMPLES : (local_frame + 1) * FRAME_SAMPLES]
            decision_started = time.perf_counter()
            frame_energy = trigger_rms(frame)
            block_rms = peak_short_block_rms(frame, SAMPLE_RATE_HZ, settings.trigger_short_block_seconds) if short_block_enabled else None
            relatively_loud = trigger is not None and trigger.observe("noise", frame_energy, FRAME_SECONDS, block_rms)
            decision_seconds += time.perf_counter() - decision_started
            frame_seconds = frame_index * FRAME_SECONDS
            if (frame_energy >= settings.trigger_rms or relatively_loud) and (
                frame_seconds - last_trigger_seconds >= settings.trigger_cooldown_seconds
            ):
                last_trigger_seconds = frame_seconds
                if frame_seconds < NOISE_REPLAY_STARTUP_SECONDS:
                    startup += 1
                else:
                    steady += 1
            frame_index += 1
    return TriggerNoiseReplayResult(
        candidate=candidate,
        background=background.name,
        replay_seconds=frame_index * FRAME_SECONDS,
        startup_false_triggers=startup,
        steady_false_triggers=steady,
        microseconds_per_frame=decision_seconds * 1e6 / max(frame_index, 1),
    )
