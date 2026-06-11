"""Narrowband bearing estimators that contribute angular priors to TDOA."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPSILON = 1.0e-12


@dataclass(frozen=True, slots=True)
class BearingPrior:
    direction: np.ndarray
    strength: float
    quality: float


def _dominant_frequency_hz(
    signal: np.ndarray,
    sample_rate_hz: int,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> float:
    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    tapered = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(tapered))
    frequencies_hz = np.fft.rfftfreq(tapered.size, d=1.0 / sample_rate_hz)
    selected = (
        (frequencies_hz >= minimum_frequency_hz)
        & (frequencies_hz <= maximum_frequency_hz)
    )
    if not np.any(selected):
        raise ValueError("Configured subspace frequency band has no FFT bins")
    selected_spectrum = spectrum[selected]
    if not np.any(selected_spectrum > EPSILON):
        raise ValueError("Configured subspace frequency band has no signal energy")
    return float(frequencies_hz[selected][int(np.argmax(selected_spectrum))])


def _alias_quality(
    *,
    sensor_positions: dict[str, np.ndarray],
    sound_speed_mps: float,
    frequency_hz: float,
) -> float:
    position_matrix = np.vstack(list(sensor_positions.values()))
    centroid_m = np.mean(position_matrix, axis=0)
    aperture_m = float(np.max(np.linalg.norm(position_matrix - centroid_m, axis=1)) * 2.0)
    alias_cutoff_hz = sound_speed_mps / max(2.0 * aperture_m, EPSILON)
    return float(np.clip(alias_cutoff_hz / max(frequency_hz, alias_cutoff_hz), 0.02, 1.0))


def _estimate_source_count_mdl(eigenvalues: np.ndarray, snapshot_count: int) -> int:
    sensor_count = eigenvalues.size
    if sensor_count <= 2 or snapshot_count <= sensor_count:
        return 1
    sorted_eigenvalues = np.clip(
        np.sort(np.real(eigenvalues))[::-1],
        EPSILON,
        None,
    )
    mdl_scores: list[float] = []
    for source_count in range(sensor_count):
        noise_eigenvalues = sorted_eigenvalues[source_count:]
        noise_dimension = noise_eigenvalues.size
        geometric_mean = float(np.exp(np.mean(np.log(noise_eigenvalues))))
        arithmetic_mean = float(np.mean(noise_eigenvalues))
        likelihood_term = -snapshot_count * noise_dimension * math.log(
            max(geometric_mean / arithmetic_mean, EPSILON)
        )
        penalty_term = (
            0.5
            * source_count
            * (2 * sensor_count - source_count)
            * math.log(snapshot_count)
        )
        mdl_scores.append(likelihood_term + penalty_term)
    return max(1, min(sensor_count - 1, int(np.argmin(mdl_scores))))


def music_bearing_prior(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    sensor_ids: list[str],
    sample_rate_hz: int,
    sound_speed_mps: float,
    azimuth_step_deg: float,
    elevation_step_deg: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    source_count: int | None,
) -> BearingPrior:
    signals = np.vstack([sensor_windows[sensor_id] for sensor_id in sensor_ids]).astype(
        np.float64
    )
    dominant_frequency_hz = _dominant_frequency_hz(
        signals[0],
        sample_rate_hz,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
    )
    frame_size = min(1024, signals.shape[1])
    hop_size = max(frame_size // 2, 1)
    snapshot_vectors: list[np.ndarray] = []
    for start_index in range(0, max(signals.shape[1] - frame_size + 1, 1), hop_size):
        frame = signals[:, start_index : start_index + frame_size]
        if frame.shape[1] != frame_size:
            continue
        spectrum = np.fft.rfft(frame * np.hanning(frame_size)[None, :], axis=1)
        frequency_bin = int(round(dominant_frequency_hz * frame_size / sample_rate_hz))
        snapshot_vectors.append(spectrum[:, frequency_bin])
    if not snapshot_vectors:
        raise ValueError("MUSIC could not form frequency-domain snapshots")
    covariance = np.zeros((len(sensor_ids), len(sensor_ids)), dtype=np.complex128)
    for snapshot_vector in snapshot_vectors:
        snapshot_norm = float(np.linalg.norm(snapshot_vector))
        if snapshot_norm <= EPSILON:
            continue
        normalized_snapshot = snapshot_vector / snapshot_norm
        covariance += np.outer(normalized_snapshot, normalized_snapshot.conj())
    covariance /= max(len(snapshot_vectors), 1)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MUSIC covariance eigendecomposition failed") from exc
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    estimated_source_count = (
        int(source_count)
        if source_count is not None
        else _estimate_source_count_mdl(eigenvalues, len(snapshot_vectors))
    )
    estimated_source_count = max(1, min(len(sensor_ids) - 1, estimated_source_count))
    noise_subspace = eigenvectors[:, estimated_source_count:]
    if noise_subspace.size == 0:
        raise ValueError("MUSIC noise subspace is empty")

    position_matrix = np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids])
    offsets_m = position_matrix - np.mean(position_matrix, axis=0)
    scores: list[float] = []
    directions: list[np.ndarray] = []
    for elevation_deg in np.arange(
        -65.0,
        65.0 + max(elevation_step_deg, 1.0) * 0.5,
        max(elevation_step_deg, 1.0),
    ):
        elevation_rad = math.radians(float(elevation_deg))
        for azimuth_deg in np.arange(
            -180.0,
            180.0,
            max(azimuth_step_deg, 1.0),
        ):
            azimuth_rad = math.radians(float(azimuth_deg))
            direction = np.asarray(
                [
                    math.cos(elevation_rad) * math.cos(azimuth_rad),
                    math.cos(elevation_rad) * math.sin(azimuth_rad),
                    math.sin(elevation_rad),
                ],
                dtype=np.float64,
            )
            delays_s = (offsets_m @ direction) / sound_speed_mps
            steering_vector = np.exp(
                1j * 2.0 * np.pi * dominant_frequency_hz * delays_s
            )
            projection = noise_subspace.conj().T @ steering_vector
            score = 1.0 / max(float(np.vdot(projection, projection).real), EPSILON)
            directions.append(direction)
            scores.append(score)
    best_index = int(np.argmax(scores))
    best_score = scores[best_index]
    median_score = float(np.median(scores))
    contrast = float(
        np.clip((best_score - median_score) / max(abs(best_score), EPSILON), 0.0, 1.0)
    )
    alias_quality = _alias_quality(
        sensor_positions=sensor_positions,
        sound_speed_mps=sound_speed_mps,
        frequency_hz=dominant_frequency_hz,
    )
    quality = contrast * alias_quality
    return BearingPrior(
        direction=directions[best_index],
        strength=float(np.clip(quality, 0.01, 1.0)),
        quality=quality,
    )


def esprit_bearing_prior(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    sensor_ids: list[str],
    sample_rate_hz: int,
    sound_speed_mps: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> BearingPrior:
    reference_sensor = sensor_ids[0]
    dominant_frequency_hz = _dominant_frequency_hz(
        sensor_windows[reference_sensor],
        sample_rate_hz,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
    )
    frequency_bin = int(
        round(dominant_frequency_hz * sensor_windows[reference_sensor].size / sample_rate_hz)
    )
    spectra = {
        sensor_id: np.fft.rfft(
            (sensor_windows[sensor_id] - float(np.mean(sensor_windows[sensor_id])))
            * np.hanning(sensor_windows[sensor_id].size)
        )
        for sensor_id in sensor_ids
    }
    reference_phase = float(np.angle(spectra[reference_sensor][frequency_bin]))
    reference_position = sensor_positions[reference_sensor]
    baselines: list[np.ndarray] = []
    projected_distances_m: list[float] = []
    for sensor_id in sensor_ids[1:]:
        phase_delta = float(
            np.angle(
                np.exp(
                    1j
                    * (
                        float(np.angle(spectra[sensor_id][frequency_bin]))
                        - reference_phase
                    )
                )
            )
        )
        delay_s = phase_delta / (2.0 * np.pi * dominant_frequency_hz)
        baselines.append(sensor_positions[sensor_id] - reference_position)
        projected_distances_m.append(sound_speed_mps * delay_s)
    baseline_matrix = np.vstack(baselines)
    projected_distance_vector = np.asarray(projected_distances_m, dtype=np.float64)
    try:
        raw_direction, *_ = np.linalg.lstsq(
            baseline_matrix,
            projected_distance_vector,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("ESPRIT phase-gradient solve failed") from exc
    direction_norm = float(np.linalg.norm(raw_direction))
    if direction_norm < EPSILON:
        raise ValueError("ESPRIT phase-gradient solve is degenerate")
    direction = raw_direction / direction_norm
    normalized_fit_error = float(
        np.linalg.norm(baseline_matrix @ direction - projected_distance_vector)
        / max(np.linalg.norm(projected_distance_vector), EPSILON)
    )
    alias_quality = _alias_quality(
        sensor_positions=sensor_positions,
        sound_speed_mps=sound_speed_mps,
        frequency_hz=dominant_frequency_hz,
    )
    quality = max(0.0, 1.0 - normalized_fit_error) * alias_quality
    return BearingPrior(
        direction=direction,
        strength=float(np.clip(quality, 0.01, 1.0)),
        quality=quality,
    )
