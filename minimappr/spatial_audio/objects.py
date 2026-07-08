"""Object-layer FOA subtraction helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from minimappr.spatial_audio.stft import istft_channels, sqrt_hann_window, stft_channels


@dataclass(frozen=True, slots=True)
class ObjectSubtractionProfile:
    frame_duration_ms: float = 64.0
    hop_fraction: float = 0.25
    wiener_beta: float = 0.05
    mask_smoothing_ms: float = 50.0
    mask_threshold: float = 0.08


DEFAULT_OBJECT_SUBTRACTION_PROFILE = ObjectSubtractionProfile()
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class ObjectSlotLike(Protocol):
    slot_id: int
    samples: NDArray[np.float32]
    positions_per_unit: list[dict[int, dict]]
    handoff_gap_ranges: list[tuple[int, int]]


class TrackTrajectoryLike(Protocol):
    track_id: str
    waypoints: list[tuple[int, tuple[float, float, float]]]


def subtract_object_slot_from_bed(
    bed_full: NDArray[np.float32],
    slot: ObjectSlotLike | None,
    samples_per_unit: int,
    n_samples: int,
    *,
    sample_rate_hz: int,
    profile: ObjectSubtractionProfile = DEFAULT_OBJECT_SUBTRACTION_PROFILE,
) -> NDArray[np.float32]:
    if slot is None:
        return bed_full.astype(np.float32, copy=True)

    target_samples = min(int(n_samples), int(slot.samples.shape[0]), int(bed_full.shape[1]))
    object_bed = render_slot_object_bed(slot, samples_per_unit, target_samples)
    if target_samples <= 0:
        return bed_full.astype(np.float32, copy=True)

    if _should_use_exact_subtraction(target_samples, sample_rate_hz, profile):
        cleaned = bed_full.astype(np.float64, copy=True)
        cleaned[:, :target_samples] -= object_bed[:, :target_samples].astype(np.float64)
        return cleaned.astype(np.float32)

    return subtract_rendered_object_bed_stft(
        bed_full,
        object_bed,
        target_samples,
        sample_rate_hz=sample_rate_hz,
        profile=profile,
    )


def subtract_objects_from_bed(
    bed_full: NDArray[np.float32],
    object_tracks: dict[str, NDArray[np.float32]],
    trajectories: list[TrackTrajectoryLike],
    n_samples: int,
    *,
    sample_rate_hz: int,
    profile: ObjectSubtractionProfile = DEFAULT_OBJECT_SUBTRACTION_PROFILE,
) -> NDArray[np.float32]:
    traj_by_id = {trajectory.track_id: trajectory for trajectory in trajectories}
    cleaned = bed_full.astype(np.float32, copy=True)
    for track_id, mono in object_tracks.items():
        trajectory = traj_by_id.get(track_id)
        if trajectory is None or not trajectory.waypoints:
            continue
        target_samples = min(int(n_samples), int(mono.shape[0]), int(cleaned.shape[1]))
        object_bed = render_trajectory_object_bed(mono, trajectory.waypoints, target_samples)
        if _should_use_exact_subtraction(target_samples, sample_rate_hz, profile):
            cleaned[:, :target_samples] = (
                cleaned[:, :target_samples].astype(np.float64)
                - object_bed[:, :target_samples].astype(np.float64)
            ).astype(np.float32)
        else:
            cleaned = subtract_rendered_object_bed_stft(
                cleaned,
                object_bed,
                target_samples,
                sample_rate_hz=sample_rate_hz,
                profile=profile,
            )
    return cleaned.astype(np.float32)


def render_slot_object_bed(
    slot: ObjectSlotLike,
    samples_per_unit: int,
    n_samples: int,
) -> NDArray[np.float32]:
    samples_per_unit = max(1, int(samples_per_unit))
    object_bed = np.zeros((4, n_samples), dtype=np.float64)
    for unit_index, unit_positions in enumerate(slot.positions_per_unit):
        position = unit_positions.get(slot.slot_id)
        if not position:
            continue
        start = unit_index * samples_per_unit
        end = min(start + samples_per_unit, n_samples)
        if end <= start:
            continue
        start_direction = _spherical_to_unit_xyz(
            float(position["azimuth_deg"]),
            float(position["elevation_deg"]),
        )
        end_direction = _spherical_to_unit_xyz(
            float(position.get("end_azimuth_deg", position["azimuth_deg"])),
            float(position.get("end_elevation_deg", position["elevation_deg"])),
        )
        _add_steered_block(
            object_bed,
            slot.samples[start:end],
            start,
            start_direction,
            end_direction,
        )
    return object_bed.astype(np.float32)


def render_trajectory_object_bed(
    mono: NDArray[np.float32],
    waypoints: list[tuple[int, tuple[float, float, float]]],
    n_samples: int,
) -> NDArray[np.float32]:
    object_bed = np.zeros((4, n_samples), dtype=np.float64)
    if not waypoints:
        return object_bed.astype(np.float32)
    sorted_waypoints = sorted((int(sample), tuple(pos)) for sample, pos in waypoints)
    for index, (start_sample, start_position) in enumerate(sorted_waypoints):
        end_sample = (
            sorted_waypoints[index + 1][0]
            if index + 1 < len(sorted_waypoints)
            else n_samples
        )
        start = max(0, min(start_sample, n_samples))
        end = max(start, min(end_sample, n_samples))
        if end <= start:
            continue
        end_position = (
            sorted_waypoints[index + 1][1]
            if index + 1 < len(sorted_waypoints)
            else start_position
        )
        _add_steered_block(
            object_bed,
            mono[start:end],
            start,
            _unit_vector(start_position),
            _unit_vector(end_position),
        )
    return object_bed.astype(np.float32)


def subtract_rendered_object_bed_stft(
    bed_full: NDArray[np.float32],
    object_bed: NDArray[np.float32],
    n_samples: int,
    *,
    sample_rate_hz: int,
    profile: ObjectSubtractionProfile = DEFAULT_OBJECT_SUBTRACTION_PROFILE,
) -> NDArray[np.float32]:
    frame_size = _frame_size_for_rate(sample_rate_hz, profile.frame_duration_ms)
    hop_size = max(1, int(round(frame_size * profile.hop_fraction)))
    target_samples = min(int(n_samples), int(bed_full.shape[1]), int(object_bed.shape[1]))

    bed = bed_full.astype(np.float32, copy=True)
    bed_segment = bed[:, :target_samples]
    object_segment = object_bed[:, :target_samples]
    window = sqrt_hann_window(frame_size)
    bed_spectra = stft_channels(
        bed_segment,
        frame_size=frame_size,
        hop_size=hop_size,
        window=window,
    )
    object_spectra = stft_channels(
        object_segment,
        frame_size=frame_size,
        hop_size=hop_size,
        window=window,
    )
    mask = _wiener_object_mask(
        bed_spectra,
        object_spectra,
        hop_size=hop_size,
        sample_rate_hz=sample_rate_hz,
        profile=profile,
    )
    cleaned_spectra = bed_spectra - (object_spectra * mask[np.newaxis, :, :])
    bed[:, :target_samples] = istft_channels(
        cleaned_spectra,
        frame_size=frame_size,
        hop_size=hop_size,
        n_samples=target_samples,
        window=window,
    )
    return bed.astype(np.float32)


def _wiener_object_mask(
    bed_spectra: NDArray[np.complex128],
    object_spectra: NDArray[np.complex128],
    *,
    hop_size: int,
    sample_rate_hz: int,
    profile: ObjectSubtractionProfile,
) -> NDArray[np.float64]:
    object_power = np.sum(np.abs(object_spectra) ** 2, axis=0)
    bed_power = np.sum(np.abs(bed_spectra) ** 2, axis=0)
    mask = object_power / (object_power + (profile.wiener_beta * bed_power) + 1e-12)
    mask = np.where(mask >= profile.mask_threshold, mask, 0.0)
    mask = _blur_mask_frequency(mask)
    return _smooth_mask_time(
        mask,
        hop_size=hop_size,
        sample_rate_hz=sample_rate_hz,
        smoothing_ms=profile.mask_smoothing_ms,
    )


def _blur_mask_frequency(mask: NDArray[np.float64]) -> NDArray[np.float64]:
    if mask.shape[1] < 3:
        return mask
    blurred = mask.copy()
    blurred[:, 1:-1] = (0.25 * mask[:, :-2]) + (0.5 * mask[:, 1:-1]) + (0.25 * mask[:, 2:])
    return blurred


def _smooth_mask_time(
    mask: NDArray[np.float64],
    *,
    hop_size: int,
    sample_rate_hz: int,
    smoothing_ms: float,
) -> NDArray[np.float64]:
    if mask.shape[0] <= 1:
        return np.clip(mask, 0.0, 1.0)
    tau_s = max(1e-3, smoothing_ms / 1000.0)
    alpha = math.exp(-(hop_size / float(sample_rate_hz)) / tau_s)
    smoothed = np.zeros_like(mask)
    previous = mask[0]
    for frame_index in range(mask.shape[0]):
        previous = (alpha * previous) + ((1.0 - alpha) * mask[frame_index])
        smoothed[frame_index] = previous
    return np.clip(smoothed, 0.0, 1.0)


def _add_steered_block(
    destination: NDArray[np.float64],
    mono_block: NDArray[np.float32],
    start_sample: int,
    start_direction: tuple[float, float, float],
    end_direction: tuple[float, float, float],
) -> None:
    block_length = int(mono_block.shape[0])
    if block_length <= 0:
        return
    samples = np.asarray(mono_block, dtype=np.float64)
    if np.allclose(start_direction, end_direction, atol=1e-12):
        direction = np.asarray(_unit_vector(start_direction), dtype=np.float64)
        destination[0, start_sample : start_sample + block_length] += _INV_SQRT2 * samples
        destination[1:4, start_sample : start_sample + block_length] += (
            direction[:, np.newaxis] * samples[np.newaxis, :]
        )
        return

    directions = np.asarray(
        [
            slerp_unit_vectors(
                start_direction,
                end_direction,
                0.0 if block_length == 1 else offset / float(block_length - 1),
            )
            for offset in range(block_length)
        ],
        dtype=np.float64,
    )
    destination[0, start_sample : start_sample + block_length] += _INV_SQRT2 * samples
    destination[1:4, start_sample : start_sample + block_length] += (
        directions.T * samples[np.newaxis, :]
    )


def slerp_unit_vectors(
    start_direction: tuple[float, float, float],
    end_direction: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    start = np.asarray(_unit_vector(start_direction), dtype=np.float64)
    end = np.asarray(_unit_vector(end_direction), dtype=np.float64)
    amount = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if dot > 0.9995:
        return _unit_tuple(((1.0 - amount) * start) + (amount * end))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        return _unit_tuple(start)
    first_weight = math.sin((1.0 - amount) * theta) / sin_theta
    second_weight = math.sin(amount * theta) / sin_theta
    return _unit_tuple((first_weight * start) + (second_weight * end))


def _spherical_to_unit_xyz(
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return (
        cos_elevation * math.cos(azimuth),
        cos_elevation * math.sin(azimuth),
        math.sin(elevation),
    )


def _unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return _unit_tuple(np.asarray(vector, dtype=np.float64))


def _unit_tuple(vector: NDArray[np.float64]) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0)
    unit = vector / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _frame_size_for_rate(sample_rate_hz: int, frame_duration_ms: float) -> int:
    target = max(256, int(round(sample_rate_hz * frame_duration_ms / 1000.0)))
    return 1 << int(math.ceil(math.log2(target)))


def _should_use_exact_subtraction(
    n_samples: int,
    sample_rate_hz: int,
    profile: ObjectSubtractionProfile,
) -> bool:
    return n_samples < _frame_size_for_rate(sample_rate_hz, profile.frame_duration_ms) * 2
