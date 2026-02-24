"""Simple nearest-neighbor track manager.

Track lifecycle:
    tentative  -> confirmed  (2+ corroborating detections)
    confirmed  -> coasting   (no update within stale window)
    coasting   -> confirmed  (new detection re-associates)
    coasting   -> dropped    (exceeds drop timeout)
    tentative  -> dropped    (exceeds drop timeout without confirmation)

Track Quality Index (TQI):
    Composite score from position confidence, classification confidence,
    age-of-last-update, and number of corroborating sensors.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from minimappr.config import Settings
from minimappr.models import TrackState, TrackStatus


@dataclass(slots=True)
class _KalmanTrackState:
    mean: np.ndarray  # [x, y, z, vx, vy, vz]
    covariance: np.ndarray


class TrackManager:
    CONFIRM_THRESHOLD: int = 2  # detections needed to confirm a track
    SUPPORTED_FILTERS: tuple[str, ...] = ("linear", "kalman")

    def __init__(self, settings: Settings) -> None:
        self._tracks: dict[str, TrackState] = {}
        self._kalman_states: dict[str, _KalmanTrackState] = {}
        self._id_counter = itertools.count(1)
        self._assoc_distance_m = settings.association_distance_m
        self._stale_ns = int(settings.track_stale_seconds * 1_000_000_000)
        self._drop_ns = int(settings.track_stale_seconds * 3_000_000_000)  # 3x stale = drop
        self._tracking_filter = settings.tracking_filter.strip().lower()
        if self._tracking_filter not in self.SUPPORTED_FILTERS:
            supported = ", ".join(self.SUPPORTED_FILTERS)
            raise ValueError(
                f"Unsupported tracking_filter '{settings.tracking_filter}'. "
                f"Supported values: {supported}"
            )

        self._kalman_process_noise = float(settings.kalman_process_noise)
        self._kalman_measurement_noise = float(settings.kalman_measurement_noise)
        self._kalman_initial_position_variance = float(settings.kalman_initial_position_variance)
        self._kalman_initial_velocity_variance = float(settings.kalman_initial_velocity_variance)
        if self._kalman_process_noise < 0.0:
            raise ValueError("kalman_process_noise must be >= 0.0")
        if self._kalman_measurement_noise <= 0.0:
            raise ValueError("kalman_measurement_noise must be > 0.0")
        if self._kalman_initial_position_variance <= 0.0:
            raise ValueError("kalman_initial_position_variance must be > 0.0")
        if self._kalman_initial_velocity_variance <= 0.0:
            raise ValueError("kalman_initial_velocity_variance must be > 0.0")
        self._kalman_obs_model = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._kalman_identity = np.eye(6, dtype=np.float64)
        self._kalman_measurement_cov = np.eye(3, dtype=np.float64) * self._kalman_measurement_noise
        self._lock = asyncio.Lock()

    async def update(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        label: str,
        confidence: float,
        label_category: str = "unknown",
        iff_category: str = "unknown",
        sensor_count: int = 1,
        label_id: str | None = None,
        capability_tier: str = "full_3d",
    ) -> TrackState:
        async with self._lock:
            self._age_tracks(timestamp_ns)

            pos = np.asarray(position_m, dtype=np.float64)
            best_track: TrackState | None = None
            best_distance = float("inf")

            for track in self._tracks.values():
                if track.status == TrackStatus.DROPPED.value:
                    continue
                previous = self._association_position(track, timestamp_ns)
                distance = float(np.linalg.norm(pos - previous))
                if distance < self._assoc_distance_m and distance < best_distance:
                    best_distance = distance
                    best_track = track

            if best_track is None:
                track_id = f"trk-{next(self._id_counter):05d}"
                created = TrackState(
                    id=track_id,
                    first_seen_ns=timestamp_ns,
                    last_seen_ns=timestamp_ns,
                    position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                    position_covariance_m2=[
                        [self._kalman_initial_position_variance, 0.0, 0.0],
                        [0.0, self._kalman_initial_position_variance, 0.0],
                        [0.0, 0.0, self._kalman_initial_position_variance],
                    ],
                    velocity_mps=(0.0, 0.0, 0.0),
                    label_id=label_id,
                    label=label,
                    label_category=label_category,
                    iff_category=iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown",
                    confidence=float(confidence),
                    update_count=1,
                    status=TrackStatus.TENTATIVE.value,
                    tqi=self._compute_tqi(confidence, 1, 0.0, sensor_count),
                    capability_tier=capability_tier,
                )
                self._tracks[track_id] = created
                if self._tracking_filter == "kalman":
                    self._init_kalman_state(track_id=track_id, position=pos)
                return created

            if self._tracking_filter == "kalman":
                smoothed_position, smoothed_velocity, covariance = self._update_kalman(best_track, timestamp_ns, pos)
            else:
                smoothed_position, smoothed_velocity, covariance = self._update_linear(best_track, timestamp_ns, pos)

            best_track.last_seen_ns = timestamp_ns
            best_track.position_m = smoothed_position
            best_track.velocity_mps = smoothed_velocity
            best_track.position_covariance_m2 = covariance
            best_track.label_id = label_id
            best_track.label = label
            best_track.label_category = label_category
            best_track.iff_category = (
                iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown"
            )
            best_track.confidence = float(max(best_track.confidence, confidence))
            best_track.update_count += 1
            best_track.capability_tier = capability_tier

            # Lifecycle: tentative -> confirmed after enough detections
            if best_track.status in (TrackStatus.TENTATIVE.value, TrackStatus.COASTING.value):
                if best_track.update_count >= self.CONFIRM_THRESHOLD:
                    best_track.status = TrackStatus.CONFIRMED.value
                elif best_track.status == TrackStatus.COASTING.value:
                    # Re-associated while coasting but not enough updates yet
                    best_track.status = TrackStatus.TENTATIVE.value
            else:
                best_track.status = TrackStatus.CONFIRMED.value

            age_s = (timestamp_ns - best_track.first_seen_ns) / 1_000_000_000.0
            best_track.tqi = self._compute_tqi(
                best_track.confidence, best_track.update_count, age_s, sensor_count
            )
            return best_track

    async def snapshot(self, now_ns: int | None = None) -> list[TrackState]:
        async with self._lock:
            if now_ns is not None:
                self._age_tracks(now_ns)
            return list(self._tracks.values())

    def _age_tracks(self, now_ns: int) -> None:
        for track in self._tracks.values():
            if track.status == TrackStatus.DROPPED.value:
                continue
            gap_ns = now_ns - track.last_seen_ns
            if gap_ns > self._drop_ns:
                track.status = TrackStatus.DROPPED.value
                self._kalman_states.pop(track.id, None)
            elif gap_ns > self._stale_ns:
                if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value):
                    track.status = TrackStatus.COASTING.value

    async def active_ids(self, now_ns: int) -> Iterable[str]:
        async with self._lock:
            self._age_tracks(now_ns)
            return [
                track.id for track in self._tracks.values()
                if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value)
            ]

    @staticmethod
    def _compute_tqi(
        confidence: float,
        update_count: int,
        age_s: float,
        sensor_count: int,
    ) -> float:
        """Composite Track Quality Index.

        Components:
            - classification confidence (0-1)
            - corroboration factor based on update count
            - recency penalty (decays with age)
            - sensor diversity bonus
        """
        corroboration = min(1.0, update_count / 5.0)
        recency = 1.0 / (1.0 + age_s / 30.0)
        sensor_factor = min(1.0, sensor_count / 4.0)
        tqi = 0.3 * confidence + 0.3 * corroboration + 0.2 * recency + 0.2 * sensor_factor
        return float(np.clip(tqi, 0.0, 1.0))

    def _association_position(self, track: TrackState, timestamp_ns: int) -> np.ndarray:
        if self._tracking_filter != "kalman":
            return np.asarray(track.position_m, dtype=np.float64)

        state = self._kalman_states.get(track.id)
        if state is None:
            return np.asarray(track.position_m, dtype=np.float64)

        dt_s = max((timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, 0.0)
        return state.mean[:3] + (state.mean[3:] * dt_s)

    def _update_linear(
        self,
        track: TrackState,
        timestamp_ns: int,
        measured_position: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], list[list[float]]]:
        dt_s = max((timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, 1e-3)
        previous_position = np.asarray(track.position_m, dtype=np.float64)
        previous_velocity = np.asarray(track.velocity_mps, dtype=np.float64)

        measured_velocity = (measured_position - previous_position) / dt_s
        smooth_position = (0.4 * previous_position) + (0.6 * measured_position)
        smooth_velocity = (0.5 * previous_velocity) + (0.5 * measured_velocity)

        # Linear mode does not estimate covariance; expose a conservative diagonal.
        diag = max(1e-3, self._assoc_distance_m * 0.5) ** 2
        covariance = [[diag, 0.0, 0.0], [0.0, diag, 0.0], [0.0, 0.0, diag]]

        return (
            (float(smooth_position[0]), float(smooth_position[1]), float(smooth_position[2])),
            (float(smooth_velocity[0]), float(smooth_velocity[1]), float(smooth_velocity[2])),
            covariance,
        )

    def _init_kalman_state(self, track_id: str, position: np.ndarray) -> None:
        mean = np.zeros(6, dtype=np.float64)
        mean[:3] = position
        covariance = np.diag(
            [
                self._kalman_initial_position_variance,
                self._kalman_initial_position_variance,
                self._kalman_initial_position_variance,
                self._kalman_initial_velocity_variance,
                self._kalman_initial_velocity_variance,
                self._kalman_initial_velocity_variance,
            ]
        ).astype(np.float64)
        self._kalman_states[track_id] = _KalmanTrackState(mean=mean, covariance=covariance)

    def _update_kalman(
        self,
        track: TrackState,
        timestamp_ns: int,
        measured_position: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], list[list[float]]]:
        state = self._kalman_states.get(track.id)
        if state is None:
            self._init_kalman_state(track.id, np.asarray(track.position_m, dtype=np.float64))
            state = self._kalman_states[track.id]

        dt_s = max((timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, 1e-3)
        transition = self._kalman_transition(dt_s)
        process_cov = self._kalman_process_covariance(dt_s)

        prior_mean = transition @ state.mean
        prior_covariance = (transition @ state.covariance @ transition.T) + process_cov

        innovation = measured_position - (self._kalman_obs_model @ prior_mean)
        innovation_covariance = (
            self._kalman_obs_model @ prior_covariance @ self._kalman_obs_model.T
        ) + self._kalman_measurement_cov
        prior_cov_observation = prior_covariance @ self._kalman_obs_model.T
        kalman_gain = np.linalg.solve(innovation_covariance, prior_cov_observation.T).T

        posterior_mean = prior_mean + (kalman_gain @ innovation)
        residual_projection = self._kalman_identity - (kalman_gain @ self._kalman_obs_model)
        posterior_covariance = (
            residual_projection @ prior_covariance @ residual_projection.T
        ) + (kalman_gain @ self._kalman_measurement_cov @ kalman_gain.T)
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)

        state.mean = posterior_mean
        state.covariance = posterior_covariance

        position = posterior_mean[:3]
        velocity = posterior_mean[3:]
        covariance = posterior_covariance[:3, :3]
        return (
            (float(position[0]), float(position[1]), float(position[2])),
            (float(velocity[0]), float(velocity[1]), float(velocity[2])),
            [
                [float(covariance[0, 0]), float(covariance[0, 1]), float(covariance[0, 2])],
                [float(covariance[1, 0]), float(covariance[1, 1]), float(covariance[1, 2])],
                [float(covariance[2, 0]), float(covariance[2, 1]), float(covariance[2, 2])],
            ],
        )

    def _kalman_transition(self, dt_s: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt_s
        transition[1, 4] = dt_s
        transition[2, 5] = dt_s
        return transition

    def _kalman_process_covariance(self, dt_s: float) -> np.ndarray:
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q = self._kalman_process_noise

        block = np.asarray(
            [
                [0.25 * dt4 * q, 0.5 * dt3 * q],
                [0.5 * dt3 * q, dt2 * q],
            ],
            dtype=np.float64,
        )
        covariance = np.zeros((6, 6), dtype=np.float64)
        covariance[0, 0] = block[0, 0]
        covariance[0, 3] = block[0, 1]
        covariance[3, 0] = block[1, 0]
        covariance[3, 3] = block[1, 1]

        covariance[1, 1] = block[0, 0]
        covariance[1, 4] = block[0, 1]
        covariance[4, 1] = block[1, 0]
        covariance[4, 4] = block[1, 1]

        covariance[2, 2] = block[0, 0]
        covariance[2, 5] = block[0, 1]
        covariance[5, 2] = block[1, 0]
        covariance[5, 5] = block[1, 1]
        return covariance
