"""Objective harness for compact-array FOA and IAMF object subtraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from minimappr.core.ambi_atob import encode_mono_to_bformat
from minimappr.spatial_audio import encode_ambisonics
from minimappr.spatial_audio.geometry import SIRITH_MIC_POSITIONS_M, centroid_corrected_positions
from minimappr.spatial_audio.objects import subtract_rendered_object_bed_stft
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, synthesize_delayed_array_channels


@dataclass(frozen=True, slots=True)
class PlaneWaveScene:
    channels: NDArray[np.float32]
    source_direction: NDArray[np.float64]
    mono: NDArray[np.float32]
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class FoaMetrics:
    xyz_to_w_db: float
    ideal_xyz_to_w_db_error: float
    max_interchannel_correlation: float
    doa_error_deg: float
    virtual_speaker_directivity_db: float
    click_count: int


@dataclass(frozen=True, slots=True)
class ObjectSubtractionMetrics:
    subtraction_depth_db: float
    unselected_retention_db: float
    click_count: int


def synthesize_plane_wave_scene(
    *,
    sample_rate_hz: int = 16_000,
    duration_s: float = 0.75,
    source_direction: tuple[float, float, float] = (0.86, 0.43, 0.27),
    distance_m: float = 12.0,
    seed: int = 20260708,
) -> PlaneWaveScene:
    """Build a deterministic tetra-array point-source scene."""
    n_samples = int(round(sample_rate_hz * duration_s))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)
    mono = (
        0.24 * np.sin(2.0 * math.pi * 700.0 * t)
        + 0.16 * np.sin(2.0 * math.pi * 1800.0 * t + 0.4)
        + 0.035 * rng.standard_normal(n_samples)
    ).astype(np.float32)
    direction = _unit(np.asarray(source_direction, dtype=np.float64))
    corrected, _ = centroid_corrected_positions(SIRITH_MIC_POSITIONS_M)
    source_position_m = tuple((direction * distance_m).tolist())
    channels = synthesize_delayed_array_channels(
        mono,
        sample_rate_hz,
        source_position_m=source_position_m,
        sensor_offsets_m=tuple(tuple(float(axis) for axis in row) for row in corrected),
    )
    return PlaneWaveScene(
        channels=channels,
        source_direction=direction,
        mono=mono,
        sample_rate_hz=sample_rate_hz,
    )


def synthesize_diffuse_noise(
    *,
    sample_rate_hz: int = 16_000,
    duration_s: float = 0.75,
    seed: int = 17,
    source_count: int = 24,
) -> NDArray[np.float32]:
    """Approximate isotropic diffuse noise by summing random plane waves."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(sample_rate_hz * duration_s))
    channels = np.zeros((4, n_samples), dtype=np.float64)
    for _ in range(source_count):
        direction = _unit(rng.normal(size=3))
        mono = (0.04 * rng.standard_normal(n_samples)).astype(np.float32)
        scene = synthesize_delayed_array_channels(
            mono,
            sample_rate_hz,
            source_position_m=tuple((direction * 10.0).tolist()),
            sensor_offsets_m=SIRITH_TETRA_SENSOR_OFFSETS_M,
        )
        channels += scene.astype(np.float64)
    peak = float(np.max(np.abs(channels))) if channels.size else 0.0
    if peak > 0.98:
        channels *= 0.98 / peak
    return channels.astype(np.float32)


def encode_scene_profile(scene: PlaneWaveScene, profile: str) -> NDArray[np.float32]:
    return encode_ambisonics(
        scene.channels,
        scene.sample_rate_hz,
        profile=profile,
        mic_positions_m=SIRITH_MIC_POSITIONS_M,
    )


def evaluate_foa_metrics(
    foa: NDArray[np.float32],
    *,
    source_direction: NDArray[np.float64],
    sample_rate_hz: int,
) -> FoaMetrics:
    stable = _stable_slice(foa.shape[1], sample_rate_hz)
    segment = foa[:, stable].astype(np.float64)
    w_energy = float(np.mean(segment[0] ** 2)) + 1e-12
    xyz_energy = float(np.mean(segment[1:4] ** 2)) + 1e-12
    xyz_to_w_db = 10.0 * math.log10(xyz_energy / w_energy)
    ideal_xyz_to_w_db = 10.0 * math.log10(2.0 / 3.0)

    corr = np.corrcoef(segment)
    off_diag = corr - np.eye(corr.shape[0])
    max_corr = float(np.max(np.abs(off_diag[np.isfinite(off_diag)])))

    estimated_direction = _estimate_foa_direction(segment)
    doa_error_deg = _bidirectional_angle_deg(estimated_direction, source_direction)
    directivity_db = virtual_speaker_directivity_db(segment, source_direction)
    click_count = spectral_flux_click_count(foa, sample_rate_hz)
    return FoaMetrics(
        xyz_to_w_db=xyz_to_w_db,
        ideal_xyz_to_w_db_error=abs(xyz_to_w_db - ideal_xyz_to_w_db),
        max_interchannel_correlation=max_corr,
        doa_error_deg=doa_error_deg,
        virtual_speaker_directivity_db=directivity_db,
        click_count=click_count,
    )


def evaluate_object_subtraction_depth(
    *,
    bed_full: NDArray[np.float32],
    object_bed: NDArray[np.float32],
    unselected_bed: NDArray[np.float32],
    sample_rate_hz: int,
) -> ObjectSubtractionMetrics:
    cleaned = subtract_rendered_object_bed_stft(
        bed_full,
        object_bed,
        bed_full.shape[1],
        sample_rate_hz=sample_rate_hz,
    )
    stable = _stable_slice(bed_full.shape[1], sample_rate_hz)
    selected_before = _rms(object_bed[:, stable])
    residual_selected = _rms((cleaned - unselected_bed)[:, stable])
    unselected_before = _rms(unselected_bed[:, stable])
    unselected_after = _rms(cleaned[:, stable])
    return ObjectSubtractionMetrics(
        subtraction_depth_db=20.0 * math.log10(selected_before / max(residual_selected, 1e-12)),
        unselected_retention_db=20.0 * math.log10(max(unselected_after, 1e-12) / max(unselected_before, 1e-12)),
        click_count=spectral_flux_click_count(cleaned, sample_rate_hz),
    )


def spectral_flux_click_count(
    channels: NDArray[np.float32],
    sample_rate_hz: int,
    *,
    frame_duration_ms: float = 16.0,
    threshold_sigma: float = 8.0,
) -> int:
    frame_size = max(128, int(round(sample_rate_hz * frame_duration_ms / 1000.0)))
    hop = max(1, frame_size // 2)
    if channels.shape[1] < frame_size * 4:
        return 0
    mono = np.mean(channels.astype(np.float64), axis=0)
    window = np.hanning(frame_size)
    spectra = []
    for start in range(0, mono.size - frame_size + 1, hop):
        mag = np.abs(np.fft.rfft(mono[start : start + frame_size] * window))
        spectra.append(mag)
    if len(spectra) < 3:
        return 0
    diff = np.maximum(0.0, np.diff(np.asarray(spectra), axis=0))
    flux = np.sum(diff, axis=1)
    edge_frames = max(1, int(round(0.08 * sample_rate_hz / hop)))
    if flux.size > edge_frames * 2:
        flux = flux[edge_frames:-edge_frames]
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) + 1e-12
    robust_sigma = 1.4826 * mad
    return int(np.sum(flux > median + threshold_sigma * robust_sigma))


def virtual_speaker_directivity_db(
    foa_segment: NDArray[np.float64],
    source_direction: NDArray[np.float64],
) -> float:
    first_axis = decode_virtual_speaker(foa_segment, source_direction)
    second_axis = decode_virtual_speaker(foa_segment, -source_direction)
    if _rms(second_axis) > _rms(first_axis):
        on_axis, opposite = second_axis, first_axis
    else:
        on_axis, opposite = first_axis, second_axis
    return 10.0 * math.log10((_rms(on_axis) ** 2 + 1e-12) / (_rms(opposite) ** 2 + 1e-12))


def decode_virtual_speaker(
    foa_segment: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    unit_direction = _unit(np.asarray(direction, dtype=np.float64))
    return (
        (foa_segment[0] / math.sqrt(2.0))
        + (unit_direction[0] * foa_segment[1])
        + (unit_direction[1] * foa_segment[2])
        + (unit_direction[2] * foa_segment[3])
    )


def build_two_object_bed(
    *,
    sample_rate_hz: int = 16_000,
    duration_s: float = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    n_samples = int(round(sample_rate_hz * duration_s))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    selected = (0.35 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
    unselected = (0.25 * np.sin(2.0 * math.pi * 1700.0 * t)).astype(np.float32)
    selected_bed = encode_mono_to_bformat(selected, (1.0, 0.0, 0.0))
    unselected_bed = encode_mono_to_bformat(unselected, (0.0, 1.0, 0.0))
    return (selected_bed + unselected_bed).astype(np.float32), selected_bed, unselected_bed


def build_weak_directional_foa(
    *,
    sample_rate_hz: int = 16_000,
    duration_s: float = 0.75,
    direction: tuple[float, float, float] = (0.86, 0.43, 0.27),
) -> tuple[NDArray[np.float32], NDArray[np.float64], int]:
    n_samples = int(round(sample_rate_hz * duration_s))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    mono = (
        0.35 * np.sin(2.0 * math.pi * 1200.0 * t)
        + 0.15 * np.sin(2.0 * math.pi * 2200.0 * t)
    ).astype(np.float32)
    unit_direction = _unit(np.asarray(direction, dtype=np.float64))
    foa = np.zeros((4, n_samples), dtype=np.float32)
    foa[0] = mono / math.sqrt(2.0)
    foa[1:4] = (0.08 * unit_direction[:, np.newaxis] * mono[np.newaxis, :]).astype(np.float32)
    return foa, unit_direction, sample_rate_hz


def _estimate_foa_direction(segment: NDArray[np.float64]) -> NDArray[np.float64]:
    intensity = np.asarray(
        [
            float(np.mean(segment[0] * segment[1])),
            float(np.mean(segment[0] * segment[2])),
            float(np.mean(segment[0] * segment[3])),
        ],
        dtype=np.float64,
    )
    return _unit(intensity)


def _stable_slice(n_samples: int, sample_rate_hz: int) -> slice:
    margin = min(max(1024, sample_rate_hz // 8), max(0, n_samples // 3))
    return slice(margin, n_samples - margin)


def _rms(samples: NDArray[np.float32] | NDArray[np.float64]) -> float:
    values = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values) + 1e-12))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _angle_deg(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    dot = float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _bidirectional_angle_deg(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return min(_angle_deg(left, right), _angle_deg(left, -right))
