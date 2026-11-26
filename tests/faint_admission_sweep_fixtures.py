"""Fixtures for the faint-call admission sweep (review item F, extends E7).

One long capture per background carries a schedule of bird-like calls at
stepped levels, so each full-app run scores many calls:

- ``chip``      12 ms FM sweep: shorter than one 20 ms ingest frame
- ``syllable``  60 ms FM sweep: a typical song element
- ``sustained`` 1 s whistle with vibrato: narrowband, long

Each level step is placed twice: inside one frame (``frame_aligned``) and
centred on an upload (and frame) boundary (``frame_straddling``), so a short
call's energy is split over two frames and two uploads. Two ``startup`` calls
land inside and just past the relative trigger's warm-up. A noise-only tail
follows the schedule; false triggers are counted wherever no call is near.

Backgrounds: ``quiet`` (MEMS self-noise, steady) and ``rain`` (a louder
steady floor plus independent per-microphone raindrop clicks: impulsive,
fluctuating, the hard case for any trigger that looks at short blocks or the
strongest channel). Levels are in-band SNR at the nearest microphone over the
background's own RMS in ``SOURCE_BAND_HZ``.

The schedule and backgrounds are synthetic (no held-out field recordings are
in the repo); ``docs/faint_call_admission_results.md`` says so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from tests.acoustic_scene_synthesis import AcousticScene, fractionally_delayed
from tests.range_matrix_cells import AIR_ABSORPTION_DB_PER_M, SAMPLE_RATE_HZ, SOURCE_BAND_HZ
from tests.realistic_capture_synthesis import spreading_gains_relative_to_nearest
from tests.realistic_scene_full_app_run import FRAME_SAMPLES, FRAMES_PER_UPLOAD

DECIBELS_PER_AMPLITUDE_DECADE = 20.0
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE_HZ
UPLOAD_SECONDS = FRAME_SECONDS * FRAMES_PER_UPLOAD
ALIGNMENT_FRAME_ALIGNED = "frame_aligned"
ALIGNMENT_FRAME_STRADDLING = "frame_straddling"
ALIGNMENT_STARTUP = "startup"
# An aligned call starts this far into the second frame of its upload.
ALIGNED_START_INTO_FRAME_SECONDS = 0.002
EDGE_TAPER_FRACTION = 0.25
SUSTAINED_EDGE_RAMP_SECONDS = 0.02

SCHEDULE_FIRST_SLOT_SECONDS = 4.0
SCHEDULE_SLOT_SECONDS = 4.0
NOISE_ONLY_TAIL_SECONDS = 20.0
SWEEP_SNR_STEPS_DB = (16.0, 20.0, 24.0, 28.0)
STARTUP_CALL_ARRIVALS_SECONDS = (0.30, 1.30)
STARTUP_CALL_SNR_DB = 26.0
STARTUP_CALL_KIND = "syllable"
# False-trigger accounting: a trigger this close to a call belongs to it.
CALL_EXCLUSION_BEFORE_SECONDS = 0.5
CALL_EXCLUSION_AFTER_SECONDS = 2.0


def _tukey_taper(sample_count: int, taper_fraction: float) -> np.ndarray:
    taper = np.ones(sample_count)
    edge = max(1, int(sample_count * taper_fraction / 2))
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]
    return taper


def _unit_rms(signal: np.ndarray) -> np.ndarray:
    return signal / float(np.sqrt(np.mean(signal**2)))


def fm_sweep(duration_seconds: float, start_hz: float, end_hz: float) -> np.ndarray:
    times = np.arange(int(round(duration_seconds * SAMPLE_RATE_HZ))) / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * (start_hz * times + 0.5 * (end_hz - start_hz) / duration_seconds * times**2)
    return _unit_rms(np.sin(phase) * _tukey_taper(times.size, EDGE_TAPER_FRACTION))


def vibrato_whistle(duration_seconds: float, centre_hz: float, vibrato_hz: float, depth_hz: float) -> np.ndarray:
    times = np.arange(int(round(duration_seconds * SAMPLE_RATE_HZ))) / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * centre_hz * times - (depth_hz / vibrato_hz) * np.cos(2.0 * np.pi * vibrato_hz * times)
    ramp_fraction = 2.0 * SUSTAINED_EDGE_RAMP_SECONDS / duration_seconds
    return _unit_rms(np.sin(phase) * _tukey_taper(times.size, ramp_fraction))


@dataclass(frozen=True)
class CallKind:
    name: str
    duration_seconds: float
    waveform: Callable[[], np.ndarray]


CALL_KINDS = (
    CallKind("chip", 0.012, lambda: fm_sweep(0.012, 3000.0, 2200.0)),
    CallKind("syllable", 0.060, lambda: fm_sweep(0.060, 3200.0, 1800.0)),
    CallKind("sustained", 1.0, lambda: vibrato_whistle(1.0, 2400.0, 6.0, 150.0)),
)
CALL_KIND_BY_NAME = {kind.name: kind for kind in CALL_KINDS}


@dataclass(frozen=True)
class Background:
    name: str
    broadband_noise_rms: float
    raindrop_rate_per_second: float = 0.0
    raindrop_median_amplitude: float = 0.0
    raindrop_amplitude_log_sigma: float = 0.0
    raindrop_decay_seconds: float = 0.0
    seed: int = 0


QUIET_BACKGROUND = Background(name="quiet", broadband_noise_rms=2.5e-5, seed=41)
# Steady floor at a normal site's level plus ~150 raindrop clicks/s per microphone.
RAIN_BACKGROUND = Background(
    name="rain",
    broadband_noise_rms=1.0e-4,
    raindrop_rate_per_second=150.0,
    raindrop_median_amplitude=3.5e-4,
    raindrop_amplitude_log_sigma=0.6,
    raindrop_decay_seconds=0.0015,
    seed=43,
)
BACKGROUNDS = (QUIET_BACKGROUND, RAIN_BACKGROUND)


def render_background(background: Background, channel_count: int, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    """Channels-first float64. Raindrops: an exponentially decaying envelope
    (one-pole over a sparse impulse train) times white noise, per microphone."""
    channels = background.broadband_noise_rms * rng.standard_normal((channel_count, sample_count))
    if background.raindrop_rate_per_second <= 0.0:
        return channels
    decay_per_sample = float(np.exp(-1.0 / (background.raindrop_decay_seconds * SAMPLE_RATE_HZ)))
    drop_probability = background.raindrop_rate_per_second / SAMPLE_RATE_HZ
    for channel in channels:
        impulses = np.zeros(sample_count)
        drop_indices = np.flatnonzero(rng.random(sample_count) < drop_probability)
        impulses[drop_indices] = background.raindrop_median_amplitude * np.exp(
            background.raindrop_amplitude_log_sigma * rng.standard_normal(drop_indices.size)
        )
        envelope = lfilter([1.0], [1.0, -decay_per_sample], impulses)
        channel += envelope * rng.standard_normal(sample_count)
    return channels


def in_band_rms(channels: np.ndarray, band_hz: tuple[float, float]) -> float:
    """Mean over channels of the RMS inside ``band_hz`` (Parseval on the rfft)."""
    spectrum = np.fft.rfft(channels, axis=1)
    frequencies_hz = np.fft.rfftfreq(channels.shape[1], d=1.0 / SAMPLE_RATE_HZ)
    in_band = (frequencies_hz >= band_hz[0]) & (frequencies_hz <= band_hz[1])
    power = 2.0 * np.sum(np.abs(spectrum[:, in_band]) ** 2, axis=1) / channels.shape[1] ** 2
    return float(np.mean(np.sqrt(power)))


@dataclass(frozen=True)
class ScheduledCall:
    call_index: int
    kind_name: str
    in_band_snr_db: float
    alignment: str
    # First arrival at any microphone, seconds from capture start.
    arrival_start_seconds: float

    @property
    def duration_seconds(self) -> float:
        return CALL_KIND_BY_NAME[self.kind_name].duration_seconds

    @property
    def arrival_end_seconds(self) -> float:
        return self.arrival_start_seconds + self.duration_seconds

    def exclusion_zone_seconds(self) -> tuple[float, float]:
        return (
            self.arrival_start_seconds - CALL_EXCLUSION_BEFORE_SECONDS,
            self.arrival_end_seconds + CALL_EXCLUSION_AFTER_SECONDS,
        )


def _placed_arrival_start_seconds(slot_seconds: float, kind: CallKind, alignment: str) -> float:
    if alignment == ALIGNMENT_FRAME_ALIGNED:
        return slot_seconds + FRAME_SECONDS + ALIGNED_START_INTO_FRAME_SECONDS
    # Centred on the boundary between this slot's first and second upload.
    return slot_seconds + UPLOAD_SECONDS - kind.duration_seconds / 2.0


def admission_schedule(snr_steps_db: Sequence[float] = SWEEP_SNR_STEPS_DB) -> list[ScheduledCall]:
    calls = [
        ScheduledCall(index, STARTUP_CALL_KIND, STARTUP_CALL_SNR_DB, ALIGNMENT_STARTUP, arrival_s)
        for index, arrival_s in enumerate(STARTUP_CALL_ARRIVALS_SECONDS)
    ]
    slot_seconds = SCHEDULE_FIRST_SLOT_SECONDS
    for kind in CALL_KINDS:
        for snr_db in snr_steps_db:
            for alignment in (ALIGNMENT_FRAME_ALIGNED, ALIGNMENT_FRAME_STRADDLING):
                arrival_s = _placed_arrival_start_seconds(slot_seconds, kind, alignment)
                calls.append(ScheduledCall(len(calls), kind.name, snr_db, alignment, arrival_s))
                slot_seconds += SCHEDULE_SLOT_SECONDS
    return calls


def schedule_capture_seconds(schedule: Sequence[ScheduledCall], noise_only_tail_seconds: float = NOISE_ONLY_TAIL_SECONDS) -> float:
    return max(call.arrival_end_seconds for call in schedule) + noise_only_tail_seconds


def _add_delayed(channel: np.ndarray, signal: np.ndarray, delay_samples: float) -> None:
    integer_delay = int(np.floor(delay_samples))
    shifted = fractionally_delayed(signal, delay_samples - integer_delay, signal.size + 2)
    end = min(channel.size, integer_delay + shifted.size)
    if integer_delay < end:
        channel[integer_delay:end] += shifted[: end - integer_delay]


def render_admission_capture(
    scene: AcousticScene,
    background: Background,
    schedule: Sequence[ScheduledCall],
    noise_only_tail_seconds: float = NOISE_ONLY_TAIL_SECONDS,
) -> tuple[dict[str, np.ndarray], float]:
    """Channels-first float32 capture per node and the background's in-band
    RMS (the SNR reference). One node's view, as in E7."""
    rng = np.random.default_rng(background.seed)
    sample_count = int(round(schedule_capture_seconds(schedule, noise_only_tail_seconds) * SAMPLE_RATE_HZ))
    delays_s = scene.propagation_delays_s()
    first_delay_s = min(delays_s.values())
    gains = spreading_gains_relative_to_nearest(scene, AIR_ABSORPTION_DB_PER_M)
    waveforms = {kind.name: kind.waveform() for kind in CALL_KINDS}
    captures: dict[str, np.ndarray] = {}
    reference_rms = 0.0
    for node in scene.nodes:
        channels = render_background(background, len(node.sensor_ids), sample_count, rng)
        reference_rms = in_band_rms(channels, SOURCE_BAND_HZ)
        for call in schedule:
            call_rms = reference_rms * 10.0 ** (call.in_band_snr_db / DECIBELS_PER_AMPLITUDE_DECADE)
            emission_s = call.arrival_start_seconds - first_delay_s
            for channel, sensor_id in zip(channels, node.sensor_ids):
                _add_delayed(
                    channel,
                    gains[sensor_id] * call_rms * waveforms[call.kind_name],
                    (emission_s + delays_s[sensor_id]) * SAMPLE_RATE_HZ,
                )
        captures[node.node_id] = channels.astype(np.float32)
    return captures, reference_rms
