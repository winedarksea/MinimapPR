"""Track filter strategies implementing the TrackFilter Protocol.

Each filter encapsulates a state estimation algorithm.  The linear
(alpha-beta) filter and 6D Kalman filter are the Phase 1 baselines;
Phase 3 will introduce IMM and particle filter alternatives that
conform to the same ``TrackFilter`` Protocol in ``interfaces.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.models import TrackState


# ---------------------------------------------------------------------------
# Internal Kalman state (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _KalmanInternalState:
    """Per-track internal Kalman state: 6D mean and 6×6 covariance."""

    mean: np.ndarray       # shape (6,): [x, y, z, vx, vy, vz]
    covariance: np.ndarray  # shape (6, 6)
    # Per-track process-noise override (set from the track's label category).
    # ``None`` means "use the filter-wide default".
    process_noise: float | None = None


# ---------------------------------------------------------------------------
# Linear (alpha-beta) filter
# ---------------------------------------------------------------------------

class LinearTrackFilter:
    """Alpha-beta smoothing filter for track state estimation.

    Maintains no per-track internal state — all information is carried
    on the ``TrackState`` object itself.  ``predict()`` returns the
    state unchanged (consistent with the Phase 1 baseline where
    association gating uses the last-observed position for linear mode).
    """

    def __init__(
        self,
        position_alpha: float,
        velocity_alpha: float,
        default_covariance_diagonal: float,
    ) -> None:
        if not 0.0 <= position_alpha <= 1.0:
            raise ValueError("position_alpha must be in [0, 1]")
        if not 0.0 <= velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in [0, 1]")
        self._position_alpha = position_alpha
        self._velocity_alpha = velocity_alpha
        # Linear mode does not estimate covariance; expose a conservative
        # fixed diagonal derived from the association gate radius.
        self._default_covariance_diag_value = max(1e-3, default_covariance_diagonal) ** 2

    # -- TrackFilter Protocol -----------------------------------------------

    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        """Return state unchanged (linear filter uses last-observed position
        for association gating, consistent with Phase 1 behaviour)."""
        return state

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
        measurement_covariance_m2: list[list[float]] | None = None,
    ) -> TrackState:
        """Alpha-beta smoothing: blend previous and measured position/velocity."""
        previous_position = np.asarray(state.position_m, dtype=np.float64)
        previous_velocity = np.asarray(state.velocity_mps, dtype=np.float64)
        measured_position = np.asarray(measurement_m, dtype=np.float64)

        dt_safe = max(dt_s, 1e-3)
        measured_velocity = (measured_position - previous_position) / dt_safe

        smooth_position = (
            self._position_alpha * previous_position
            + (1.0 - self._position_alpha) * measured_position
        )
        smooth_velocity = (
            self._velocity_alpha * previous_velocity
            + (1.0 - self._velocity_alpha) * measured_velocity
        )
        covariance = self._measurement_covariance_output(measurement_covariance_m2)

        return state.model_copy(
            update={
                "position_m": (
                    float(smooth_position[0]),
                    float(smooth_position[1]),
                    float(smooth_position[2]),
                ),
                "velocity_mps": (
                    float(smooth_velocity[0]),
                    float(smooth_velocity[1]),
                    float(smooth_velocity[2]),
                ),
                "position_covariance_m2": covariance,
            },
        )

    def initialize_track(
        self,
        track_id: str,
        position_m: tuple[float, float, float],
    ) -> None:
        """No-op: linear filter has no per-track internal state."""

    def remove_track(self, track_id: str) -> None:
        """No-op: linear filter has no per-track internal state."""

    def _measurement_covariance_output(
        self,
        measurement_covariance_m2: list[list[float]] | None,
    ) -> list[list[float]]:
        if measurement_covariance_m2 is not None:
            try:
                measurement_covariance = np.asarray(measurement_covariance_m2, dtype=np.float64)
            except (TypeError, ValueError):
                measurement_covariance = None
            if (
                measurement_covariance is not None
                and measurement_covariance.shape == (3, 3)
                and np.all(np.isfinite(measurement_covariance))
            ):
                return [
                    [float(measurement_covariance[0, 0]), float(measurement_covariance[0, 1]), float(measurement_covariance[0, 2])],
                    [float(measurement_covariance[1, 0]), float(measurement_covariance[1, 1]), float(measurement_covariance[1, 2])],
                    [float(measurement_covariance[2, 0]), float(measurement_covariance[2, 1]), float(measurement_covariance[2, 2])],
                ]
        d = self._default_covariance_diag_value
        return [[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]]


# ---------------------------------------------------------------------------
# 6D Kalman filter (position + velocity)
# ---------------------------------------------------------------------------

class KalmanTrackFilter:
    """6D Kalman filter (3-position + 3-velocity) for track estimation.

    Maintains per-track internal state (mean and covariance) keyed by
    ``track_id``.  The ``predict`` method performs a non-destructive
    position extrapolation for association gating, while ``update``
    executes the full predict→correct cycle.
    """

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        initial_position_variance: float,
        initial_velocity_variance: float,
        *,
        max_coast_process_seconds: float = 10.0,
        coast_velocity_half_life_seconds: float = 10.0,
    ) -> None:
        if process_noise < 0.0:
            raise ValueError("process_noise must be >= 0.0")
        if measurement_noise <= 0.0:
            raise ValueError("measurement_noise must be > 0.0")
        if initial_position_variance <= 0.0:
            raise ValueError("initial_position_variance must be > 0.0")
        if initial_velocity_variance <= 0.0:
            raise ValueError("initial_velocity_variance must be > 0.0")

        self._process_noise = process_noise
        # Coast guards. Detections for a single real source can be minutes
        # apart; without these a 60 s gap gives Q_pos ~ 0.25*60^4*q (~6.5e6 m^2)
        # and extrapolates a junk velocity 60 s forward, so the predicted mean
        # lands far outside the association gate and the re-detection spawns a
        # new track. ``<= 0`` disables the corresponding guard.
        self._max_coast_process_seconds = float(max_coast_process_seconds)
        self._coast_velocity_half_life_seconds = float(coast_velocity_half_life_seconds)
        self._measurement_noise = measurement_noise
        self._initial_position_variance = initial_position_variance
        self._initial_velocity_variance = initial_velocity_variance

        self._states: dict[str, _KalmanInternalState] = {}

        # Pre-allocated constant matrices
        self._observation_model = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._identity_6 = np.eye(6, dtype=np.float64)
        self._measurement_covariance = np.eye(3, dtype=np.float64) * measurement_noise

    # -- TrackFilter Protocol -----------------------------------------------

    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        """Return predicted position by extrapolating the stored Kalman mean.

        Non-destructive: internal Kalman state is not modified.  This is
        used for association gating — the extrapolated position is more
        accurate than the last-observed position for fast-moving targets.
        """
        internal = self._states.get(state.id)
        if internal is None:
            return state
        dt_safe = max(dt_s, 1e-3)
        transition = self._build_transition_matrix(dt_safe)
        process_covariance = self._build_process_covariance(
            self._process_dt(dt_safe), self._q_for(internal),
        )
        predicted_mean = transition @ internal.mean
        predicted_covariance = (
            transition @ internal.covariance @ transition.T
        ) + process_covariance
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        predicted_position = predicted_mean[:3]
        predicted_position_covariance = predicted_covariance[:3, :3]
        return state.model_copy(
            update={
                "position_m": (
                    float(predicted_position[0]),
                    float(predicted_position[1]),
                    float(predicted_position[2]),
                ),
                "position_covariance_m2": [
                    [
                        float(predicted_position_covariance[0, 0]),
                        float(predicted_position_covariance[0, 1]),
                        float(predicted_position_covariance[0, 2]),
                    ],
                    [
                        float(predicted_position_covariance[1, 0]),
                        float(predicted_position_covariance[1, 1]),
                        float(predicted_position_covariance[1, 2]),
                    ],
                    [
                        float(predicted_position_covariance[2, 0]),
                        float(predicted_position_covariance[2, 1]),
                        float(predicted_position_covariance[2, 2]),
                    ],
                ],
            },
        )

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
        measurement_covariance_m2: list[list[float]] | None = None,
    ) -> TrackState:
        """Full Kalman predict + measurement-update cycle.

        Modifies internal filter state for the track.  Returns a
        ``TrackState`` copy with updated position, velocity, and
        covariance fields.
        """
        internal = self._states.get(state.id)
        if internal is None:
            self.initialize_track(state.id, state.position_m)
            internal = self._states[state.id]

        dt_safe = max(dt_s, 1e-3)
        measured_position = np.asarray(measurement_m, dtype=np.float64)

        # --- Predict (time update) ---
        transition = self._build_transition_matrix(dt_safe)
        process_cov = self._build_process_covariance(
            self._process_dt(dt_safe), self._q_for(internal),
        )
        prior_mean = transition @ internal.mean
        prior_covariance = (transition @ internal.covariance @ transition.T) + process_cov

        # --- Update (measurement) ---
        H = self._observation_model
        innovation = measured_position - (H @ prior_mean)
        measurement_covariance = self._resolved_measurement_covariance(measurement_covariance_m2)
        innovation_cov = (H @ prior_covariance @ H.T) + measurement_covariance
        prior_cross = prior_covariance @ H.T

        try:
            kalman_gain = np.linalg.solve(innovation_cov, prior_cross.T).T
        except np.linalg.LinAlgError:
            try:
                kalman_gain = np.linalg.lstsq(
                    innovation_cov, prior_cross.T, rcond=None
                )[0].T
            except np.linalg.LinAlgError:
                # Singular: preserve prior as current estimate.
                internal.mean = prior_mean
                internal.covariance = prior_covariance
                return self._to_track_state(state, internal)

        posterior_mean = prior_mean + (kalman_gain @ innovation)
        residual_projection = self._identity_6 - (kalman_gain @ H)
        posterior_covariance = (
            residual_projection @ prior_covariance @ residual_projection.T
        ) + (kalman_gain @ measurement_covariance @ kalman_gain.T)
        # Symmetrise to counteract floating-point drift.
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)

        internal.mean = posterior_mean
        internal.covariance = posterior_covariance

        return self._to_track_state(state, internal)

    def initialize_track(
        self,
        track_id: str,
        position_m: tuple[float, float, float],
    ) -> None:
        """Create initial Kalman state for a newly created track."""
        mean = np.zeros(6, dtype=np.float64)
        mean[:3] = position_m
        covariance = np.diag(
            [
                self._initial_position_variance,
                self._initial_position_variance,
                self._initial_position_variance,
                self._initial_velocity_variance,
                self._initial_velocity_variance,
                self._initial_velocity_variance,
            ]
        ).astype(np.float64)
        previous = self._states.get(track_id)
        self._states[track_id] = _KalmanInternalState(
            mean=mean,
            covariance=covariance,
            # A re-initialised track (dormant reacquisition) keeps whatever
            # category-derived q it already had; TrackManager re-applies it
            # anyway, but this keeps the filter correct on its own.
            process_noise=previous.process_noise if previous is not None else None,
        )

    def set_track_process_noise(self, track_id: str, process_noise: float | None) -> None:
        """Override the process noise used for one track.

        Duck-typed hook (deliberately not part of the ``TrackFilter`` Protocol)
        so injected and test filters without it keep working. No-op for unknown
        track ids — TrackManager always initialises the track first.
        """
        internal = self._states.get(track_id)
        if internal is None:
            return
        if process_noise is None or process_noise < 0.0:
            internal.process_noise = None
        else:
            internal.process_noise = float(process_noise)

    def remove_track(self, track_id: str) -> None:
        """Clean up internal state for a dropped or reaped track."""
        self._states.pop(track_id, None)

    # -- Internals ----------------------------------------------------------

    def _to_track_state(
        self,
        state: TrackState,
        internal: _KalmanInternalState,
    ) -> TrackState:
        """Project internal Kalman state onto TrackState fields."""
        pos = internal.mean[:3]
        vel = internal.mean[3:]
        cov = internal.covariance[:3, :3]
        return state.model_copy(
            update={
                "position_m": (float(pos[0]), float(pos[1]), float(pos[2])),
                "velocity_mps": (float(vel[0]), float(vel[1]), float(vel[2])),
                "position_covariance_m2": [
                    [float(cov[0, 0]), float(cov[0, 1]), float(cov[0, 2])],
                    [float(cov[1, 0]), float(cov[1, 1]), float(cov[1, 2])],
                    [float(cov[2, 0]), float(cov[2, 1]), float(cov[2, 2])],
                ],
            },
        )

    def _q_for(self, internal: _KalmanInternalState) -> float:
        q = internal.process_noise
        return self._process_noise if q is None else q

    def _process_dt(self, dt_s: float) -> float:
        """dt used for process-noise growth only (position/velocity still use
        the real dt through the transition matrix)."""
        cap = self._max_coast_process_seconds
        return min(dt_s, cap) if cap > 0.0 else dt_s

    def _build_transition_matrix(self, dt_s: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        half_life = self._coast_velocity_half_life_seconds
        if half_life > 0.0 and dt_s > 0.5 * half_life:
            # Long coast: an Ornstein-Uhlenbeck-style transition that decays the
            # velocity estimate toward zero instead of extrapolating it linearly
            # forever. Displacement is bounded by (half_life / ln2) * |v|.
            decay = float(np.exp(-dt_s * np.log(2.0) / half_life))
            effective_dt = (half_life / np.log(2.0)) * (1.0 - decay)
            transition[0, 3] = effective_dt
            transition[1, 4] = effective_dt
            transition[2, 5] = effective_dt
            transition[3, 3] = decay
            transition[4, 4] = decay
            transition[5, 5] = decay
            return transition
        # Short gaps keep the exact constant-velocity transition.
        transition[0, 3] = dt_s
        transition[1, 4] = dt_s
        transition[2, 5] = dt_s
        return transition

    def _build_process_covariance(self, dt_s: float, process_noise: float | None = None) -> np.ndarray:
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q = self._process_noise if process_noise is None else process_noise

        block = np.array(
            [
                [0.25 * dt4 * q, 0.5 * dt3 * q],
                [0.5 * dt3 * q, dt2 * q],
            ],
            dtype=np.float64,
        )
        covariance = np.zeros((6, 6), dtype=np.float64)
        for axis in range(3):
            pos_idx = axis
            vel_idx = axis + 3
            covariance[pos_idx, pos_idx] = block[0, 0]
            covariance[pos_idx, vel_idx] = block[0, 1]
            covariance[vel_idx, pos_idx] = block[1, 0]
            covariance[vel_idx, vel_idx] = block[1, 1]
        return covariance

    def _resolved_measurement_covariance(
        self,
        measurement_covariance_m2: list[list[float]] | None,
    ) -> np.ndarray:
        if measurement_covariance_m2 is None:
            return self._measurement_covariance
        try:
            covariance = np.asarray(measurement_covariance_m2, dtype=np.float64)
        except (TypeError, ValueError):
            return self._measurement_covariance
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return self._measurement_covariance
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return self._measurement_covariance
        if not np.all(np.isfinite(eigenvalues)):
            return self._measurement_covariance
        variance_floor = max(float(self._measurement_noise), 1.0e-6)
        covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, variance_floor)) @ eigenvectors.T
        covariance = 0.5 * (covariance + covariance.T)
        return covariance + (np.eye(3, dtype=np.float64) * 1e-9)
