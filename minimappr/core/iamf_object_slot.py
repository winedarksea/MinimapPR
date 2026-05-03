"""IAMF object slot selection for YouTube-compatible exports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class IamfObjectSlot:
    slot_id: int
    track_id: str
    samples: NDArray[np.float32]
    positions_per_unit: list[dict[int, dict]]
    unit_track_ids: list[str | None]
    active_ranges: list[tuple[int, int]]
    handoff_gap_ranges: list[tuple[int, int]]
    score: float


class ObjectSlotTrajectory(Protocol):
    track_id: str
    waypoints: list[tuple[int, tuple[float, float, float]]]
    tqi: float
    label_confidence: float
    localization_confidence: float


def select_iamf_object_slot(
    trajectories: list[ObjectSlotTrajectory],
    object_tracks: dict[str, NDArray[np.float32]],
    n_samples: int,
    samples_per_unit: int,
    *,
    sample_rate_hz: int,
    min_updates: int = 2,
    min_tqi: float = 0.35,
    min_localization_confidence: float = 0.30,
    challenger_margin: float = 0.15,
    challenger_hold_units: int | None = None,
    handoff_gap_seconds: float = 0.25,
    fade_samples: int | None = None,
) -> IamfObjectSlot | None:
    """Choose one stable IAMF object slot from rendered object candidates."""
    samples_per_unit = max(1, samples_per_unit)
    n_units = max(1, math.ceil(max(0, n_samples) / samples_per_unit))
    hold_units = challenger_hold_units or max(1, math.ceil(0.5 * sample_rate_hz / samples_per_unit))
    gap_units = max(1, math.ceil(max(0.0, handoff_gap_seconds) * sample_rate_hz / samples_per_unit))
    fade_len = fade_samples or max(1, int(round(0.1 * sample_rate_hz)))

    eligible: dict[str, tuple[ObjectSlotTrajectory, NDArray[np.float32]]] = {}
    for traj in trajectories:
        mono = object_tracks.get(traj.track_id)
        if mono is None:
            continue
        if len(traj.waypoints) < min_updates:
            continue
        if traj.tqi < min_tqi or traj.localization_confidence < min_localization_confidence:
            continue
        eligible[traj.track_id] = (traj, mono)

    if not eligible:
        return None

    unit_owner: list[str | None] = []
    current_owner: str | None = None
    challenger_id: str | None = None
    challenger_units = 0

    for unit_idx in range(n_units):
        sample_mid = min(n_samples - 1, unit_idx * samples_per_unit + samples_per_unit // 2)
        active_scores = _score_active_tracks_for_unit(
            eligible, sample_mid, unit_idx * samples_per_unit, samples_per_unit, n_samples
        )
        if not active_scores:
            unit_owner.append(None)
            current_owner = None
            challenger_id = None
            challenger_units = 0
            continue

        best_id, best_score = max(active_scores.items(), key=lambda item: item[1])
        if current_owner is None:
            current_owner = best_id
            challenger_id = None
            challenger_units = 0
        elif best_id != current_owner:
            current_score = active_scores.get(current_owner, 0.0)
            if best_score >= current_score + challenger_margin:
                if challenger_id == best_id:
                    challenger_units += 1
                else:
                    challenger_id = best_id
                    challenger_units = 1
                if challenger_units >= hold_units:
                    current_owner = best_id
                    challenger_id = None
                    challenger_units = 0
            else:
                challenger_id = None
                challenger_units = 0
        else:
            challenger_id = None
            challenger_units = 0
        unit_owner.append(current_owner)

    unit_owner, handoff_gap_ranges = _apply_handoff_gaps(
        unit_owner, samples_per_unit, n_samples, gap_units
    )

    if not any(owner is not None for owner in unit_owner):
        return None

    slot_samples = np.zeros(n_samples, dtype=np.float32)
    positions_per_unit: list[dict[int, dict]] = []
    active_ranges: list[tuple[int, int]] = []
    last_owner: str | None = None
    range_start: int | None = None
    selected_scores: list[float] = []

    for unit_idx, owner in enumerate(unit_owner):
        start = unit_idx * samples_per_unit
        end = min(start + samples_per_unit, n_samples)
        if owner is None:
            positions_per_unit.append({})
            if last_owner is not None and range_start is not None:
                active_ranges.append((range_start, start))
                range_start = None
            last_owner = None
            continue

        traj, mono = eligible[owner]
        score = _score_track_for_unit(
            traj, mono, start, samples_per_unit, n_samples, local_rms_norm=1.0
        )
        slot_samples[start:end] = mono[start:end]
        start_pos = _interpolate_waypoints(traj.waypoints, start)
        end_pos = _interpolate_waypoints(traj.waypoints, max(start, end - 1))
        start_az, start_el, start_distance_m = _xyz_to_spherical(start_pos)
        end_az, end_el, end_distance_m = _xyz_to_spherical(end_pos)
        positions_per_unit.append({
            0: {
                "azimuth_deg": start_az,
                "elevation_deg": start_el,
                "distance_norm": _normalize_scene_distance(start_distance_m),
                "end_azimuth_deg": end_az,
                "end_elevation_deg": end_el,
                "end_distance_norm": _normalize_scene_distance(end_distance_m),
            }
        })
        selected_scores.append(score)
        if owner != last_owner:
            if last_owner is not None and range_start is not None:
                active_ranges.append((range_start, start))
            range_start = start
        last_owner = owner

    if last_owner is not None and range_start is not None:
        active_ranges.append((range_start, n_samples))

    _apply_slot_fades(slot_samples, active_ranges, fade_len)
    winning_owner = next(owner for owner in unit_owner if owner is not None)
    return IamfObjectSlot(
        slot_id=0,
        track_id=winning_owner,
        samples=slot_samples,
        positions_per_unit=positions_per_unit,
        unit_track_ids=unit_owner,
        active_ranges=active_ranges,
        handoff_gap_ranges=handoff_gap_ranges,
        score=float(np.mean(selected_scores)) if selected_scores else 0.0,
    )


def _score_active_tracks_for_unit(
    eligible: dict[str, tuple[ObjectSlotTrajectory, NDArray[np.float32]]],
    sample_mid: int,
    sample_start: int,
    samples_per_unit: int,
    n_samples: int,
) -> dict[str, float]:
    active: list[tuple[str, ObjectSlotTrajectory, NDArray[np.float32], float]] = []
    for track_id, (traj, mono) in eligible.items():
        if not _trajectory_active_at_sample(traj, sample_mid):
            continue
        local_rms = _local_rms(mono, sample_start, samples_per_unit, n_samples)
        active.append((track_id, traj, mono, local_rms))

    if not active:
        return {}

    max_rms = max(max(item[3] for item in active), 1e-9)
    return {
        track_id: _score_track_for_unit(
            traj,
            mono,
            sample_start,
            samples_per_unit,
            n_samples,
            local_rms_norm=min(1.0, local_rms / max_rms),
        )
        for track_id, traj, mono, local_rms in active
    }


def _score_track_for_unit(
    traj: ObjectSlotTrajectory,
    mono: NDArray[np.float32],
    sample_start: int,
    samples_per_unit: int,
    n_samples: int,
    *,
    local_rms_norm: float | None = None,
) -> float:
    if local_rms_norm is None:
        local_rms_norm = min(
            1.0,
            _local_rms(mono, sample_start, samples_per_unit, n_samples)
            / max(_local_rms(mono, 0, n_samples, n_samples), 1e-9),
        )
    score = (
        0.30 * float(traj.tqi)
        + 0.20 * float(traj.label_confidence)
        + 0.20 * float(traj.localization_confidence)
        + 0.30 * float(local_rms_norm)
    )
    return float(np.clip(score, 0.0, 1.0))


def _local_rms(
    mono: NDArray[np.float32],
    sample_start: int,
    samples_per_unit: int,
    n_samples: int,
) -> float:
    start = max(0, min(sample_start, n_samples))
    end = max(start, min(start + samples_per_unit, n_samples, mono.shape[0]))
    if end <= start:
        return 0.0
    block = np.asarray(mono[start:end], dtype=np.float64)
    return float(np.sqrt(np.mean(block * block))) if block.size else 0.0


def _apply_handoff_gaps(
    unit_owner: list[str | None],
    samples_per_unit: int,
    n_samples: int,
    gap_units: int,
) -> tuple[list[str | None], list[tuple[int, int]]]:
    if gap_units <= 0:
        return unit_owner, []

    output = list(unit_owner)
    ranges: list[tuple[int, int]] = []
    previous_owner: str | None = None

    for idx, owner in enumerate(unit_owner):
        if (
            previous_owner is not None
            and owner is not None
            and owner != previous_owner
        ):
            gap_start_unit = idx
            gap_end_unit = min(len(output), gap_start_unit + gap_units)
            for gap_idx in range(gap_start_unit, gap_end_unit):
                output[gap_idx] = None
            start = min(n_samples, gap_start_unit * samples_per_unit)
            end = min(n_samples, gap_end_unit * samples_per_unit)
            if end > start:
                ranges.append((start, end))
        if owner is not None:
            previous_owner = owner

    return output, ranges


def _trajectory_active_at_sample(traj: ObjectSlotTrajectory, sample: int) -> bool:
    if len(traj.waypoints) < 2:
        return False
    return traj.waypoints[0][0] <= sample <= traj.waypoints[-1][0]


def _apply_slot_fades(
    samples: NDArray[np.float32],
    active_ranges: list[tuple[int, int]],
    fade_samples: int,
) -> None:
    for start, end in active_ranges:
        if end <= start:
            continue
        length = end - start
        fade = min(fade_samples, max(1, length // 2))
        if fade <= 0:
            continue
        samples[start:start + fade] *= np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        samples[end - fade:end] *= np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)


def _interpolate_waypoints(
    waypoints: list[tuple[int, tuple[float, float, float]]], sample: int
) -> tuple[float, float, float]:
    if len(waypoints) == 1 or sample <= waypoints[0][0]:
        return waypoints[0][1]
    if sample >= waypoints[-1][0]:
        return waypoints[-1][1]
    for idx in range(len(waypoints) - 1):
        s0, p0 = waypoints[idx]
        s1, p1 = waypoints[idx + 1]
        if s0 <= sample <= s1:
            t = (sample - s0) / max(s1 - s0, 1)
            return (
                p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]),
            )
    return waypoints[-1][1]


def _xyz_to_spherical(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pos
    dist = math.sqrt(x * x + y * y + z * z)
    if dist < 1e-9:
        return 0.0, 0.0, 0.0
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.asin(z / dist))
    return az, el, dist


def _normalize_scene_distance(distance_m: float) -> float:
    return min(1.0, max(0.0, distance_m / 30.0))
