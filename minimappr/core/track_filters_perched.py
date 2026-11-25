"""Perched random-walk track filter (Field Mode W2-tracker, S3).

A singing songbird sits still for 5-15 s between bouts, then may hop 10 m. The
6D constant-velocity Kalman reads a metre of fix jitter between bouts as
velocity and coasts the bird away during the silence, splitting one bird into
several tracks. This filter uses a zero-velocity (random-walk) process model:
the state is position only, and position variance grows linearly with the gap
(so 1-sigma grows with sqrt(dt)) instead of drifting along a velocity.

A *relocation*: a fix further than ``relocation_snap_distance_m`` from the
perch is a hop to a new perch, not a noisy sample of the old one. Averaging it
in would smear the track halfway between perches, so the filter re-initialises
at the new fix instead (velocity stays zero either way).

``MotionModelRoutingTrackFilter`` routes each track to this filter or to the
existing constant-velocity filter by its label category's tracking mode
(``core/tracking_mode_config.py``): ``stationary_hold`` categories (wildlife by
default) are perched, ``continuous_motion`` ones (drones, vehicles) keep the 6D
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.tracking_mode_config import TrackingModeConfig
from minimappr.interfaces import TrackFilter
from minimappr.models import TrackState

# Smallest dt (s) used for variance growth; matches the Kalman filter's guard.
_MINIMUM_DT_SECONDS = 1e-3
# A jump past the snap distance only counts as a new perch when it is also
# statistically inconsistent with the perch: chi-square 99% quantile, 3 dof.
# Without this, a distant bird whose fixes carry tens of metres of error would
# "relocate" on every noisy fix and never be averaged.
DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED = 11.34


@dataclass(slots=True)
class _PerchedInternalState:
    position_m: np.ndarray        # shape (3,)
    covariance_m2: np.ndarray     # shape (3, 3)


class PerchedRandomWalkTrackFilter:
    """Position-only Kalman filter with a random-walk process model."""

    def __init__(
        self,
        *,
        position_diffusion_m2_per_second: float,
        measurement_noise_m2: float,
        initial_position_variance_m2: float,
        relocation_snap_distance_m: float,
        relocation_snap_min_mahalanobis_squared: float = DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED,
    ) -> None:
        if position_diffusion_m2_per_second < 0.0:
            raise ValueError("position_diffusion_m2_per_second must be >= 0")
        if measurement_noise_m2 <= 0.0 or initial_position_variance_m2 <= 0.0:
            raise ValueError("perched filter variances must be > 0")
        self._position_diffusion_m2_per_second = float(position_diffusion_m2_per_second)
        self._measurement_noise_m2 = float(measurement_noise_m2)
        self._initial_position_variance_m2 = float(initial_position_variance_m2)
        # <= 0 disables snapping (every fix is fused).
        self._relocation_snap_distance_m = float(relocation_snap_distance_m)
        self._relocation_snap_min_mahalanobis_squared = float(relocation_snap_min_mahalanobis_squared)
        self._states: dict[str, _PerchedInternalState] = {}
        # Tracks whose next fix restarts the filter (a relocation-gate hop).
        self._restart_at_next_fix_track_ids: set[str] = set()
        self.relocation_count = 0

    # -- TrackFilter Protocol -----------------------------------------------

    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        internal = self._states.get(state.id)
        if internal is None:
            return state
        covariance = internal.covariance_m2 + self._diffusion(dt_s)
        return self._to_track_state(state, internal.position_m, covariance)

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
        measurement_covariance_m2: list[list[float]] | None = None,
    ) -> TrackState:
        internal = self._states.get(state.id)
        if internal is None:
            self.initialize_track(state.id, state.position_m)
            internal = self._states[state.id]
        measured = np.asarray(measurement_m, dtype=np.float64)
        measurement_covariance = self._measurement_covariance(measurement_covariance_m2)

        innovation = measured - internal.position_m
        prior_covariance = internal.covariance_m2 + self._diffusion(dt_s)
        innovation_covariance = prior_covariance + measurement_covariance
        gate_restart = state.id in self._restart_at_next_fix_track_ids
        self._restart_at_next_fix_track_ids.discard(state.id)
        if gate_restart or self._is_relocation(innovation, innovation_covariance):
            # New perch: restart at the fix rather than averaging two perches.
            # A gate restart was counted by the relocation gate already.
            if not gate_restart:
                self.relocation_count += 1
            internal.position_m = measured.copy()
            internal.covariance_m2 = measurement_covariance.copy()
            return self._to_track_state(state, internal.position_m, internal.covariance_m2)

        try:
            gain = np.linalg.solve(innovation_covariance.T, prior_covariance.T).T
        except np.linalg.LinAlgError:
            # Singular innovation: keep the prior rather than inventing a fix.
            internal.covariance_m2 = prior_covariance
            return self._to_track_state(state, internal.position_m, internal.covariance_m2)
        identity = np.eye(3, dtype=np.float64)
        residual_projection = identity - gain
        internal.position_m = internal.position_m + gain @ innovation
        posterior = (
            residual_projection @ prior_covariance @ residual_projection.T
            + gain @ measurement_covariance @ gain.T
        )
        internal.covariance_m2 = 0.5 * (posterior + posterior.T)
        return self._to_track_state(state, internal.position_m, internal.covariance_m2)

    def initialize_track(self, track_id: str, position_m: tuple[float, float, float]) -> None:
        self._restart_at_next_fix_track_ids.discard(track_id)
        self._states[track_id] = _PerchedInternalState(
            position_m=np.asarray(position_m, dtype=np.float64).copy(),
            covariance_m2=np.eye(3, dtype=np.float64) * self._initial_position_variance_m2,
        )

    def restart_at_next_fix(self, track_id: str) -> None:
        """The track's next ``update`` restarts the filter at that fix.

        For a relocation-gate hop. Initialising at the fix and then fusing the
        same fix would count it twice and halve its covariance.
        """
        self._restart_at_next_fix_track_ids.add(track_id)

    def remove_track(self, track_id: str) -> None:
        self._states.pop(track_id, None)
        self._restart_at_next_fix_track_ids.discard(track_id)

    def soft_position_update(
        self, track_id: str, corrected_position_m: tuple[float, float, float]
    ) -> bool:
        """Overwrite the stored perch after a partial-constraint nudge.

        Returns whether the track had perched state to update.
        """
        internal = self._states.get(track_id)
        if internal is None:
            return False
        internal.position_m = np.asarray(corrected_position_m, dtype=np.float64).copy()
        return True

    # -- Internals ----------------------------------------------------------

    def _is_relocation(self, innovation: np.ndarray, innovation_covariance: np.ndarray) -> bool:
        if not 0.0 < self._relocation_snap_distance_m < float(np.linalg.norm(innovation)):
            return False
        try:
            mahalanobis_squared = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        except np.linalg.LinAlgError:
            return True
        return mahalanobis_squared > self._relocation_snap_min_mahalanobis_squared

    def _diffusion(self, dt_s: float) -> np.ndarray:
        dt = max(float(dt_s), _MINIMUM_DT_SECONDS)
        return np.eye(3, dtype=np.float64) * (self._position_diffusion_m2_per_second * dt)

    def _measurement_covariance(self, value: list[list[float]] | None) -> np.ndarray:
        default = np.eye(3, dtype=np.float64) * self._measurement_noise_m2
        if value is None:
            return default
        try:
            covariance = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return default
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return default
        covariance = 0.5 * (covariance + covariance.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return default
        floored = np.maximum(eigenvalues, self._measurement_noise_m2)
        return eigenvectors @ np.diag(floored) @ eigenvectors.T

    @staticmethod
    def _to_track_state(
        state: TrackState, position_m: np.ndarray, covariance_m2: np.ndarray
    ) -> TrackState:
        return state.model_copy(
            update={
                "position_m": (float(position_m[0]), float(position_m[1]), float(position_m[2])),
                "velocity_mps": (0.0, 0.0, 0.0),
                "position_covariance_m2": [
                    [float(covariance_m2[row, column]) for column in range(3)] for row in range(3)
                ],
            }
        )


class MotionModelRoutingTrackFilter:
    """Routes each track to the perched or the constant-velocity filter.

    The route is chosen per track from its label category's tracking mode (see
    ``set_track_motion_category``) and defaults to constant velocity, so a
    track without a label moves like any ``continuous_motion`` track.
    """

    def __init__(
        self,
        *,
        constant_velocity_filter: TrackFilter,
        perched_filter: PerchedRandomWalkTrackFilter,
        tracking_modes: TrackingModeConfig,
    ) -> None:
        self._constant_velocity_filter = constant_velocity_filter
        self._perched_filter = perched_filter
        self._tracking_modes = tracking_modes
        self._perched_track_ids: set[str] = set()

    @property
    def perched_filter(self) -> PerchedRandomWalkTrackFilter:
        return self._perched_filter

    def is_perched(self, track_id: str) -> bool:
        return track_id in self._perched_track_ids

    def category_is_perched(self, label_category: str) -> bool:
        return self._tracking_modes.holds_station(label_category)

    def set_track_motion_category(
        self, track_id: str, label_category: str, position_m: tuple[float, float, float]
    ) -> None:
        """(Re)route a track; switching model restarts the new filter at *position_m*."""
        want_perched = self.category_is_perched(label_category)
        if want_perched == self.is_perched(track_id):
            return
        if want_perched:
            self._constant_velocity_filter.remove_track(track_id)
            self._perched_track_ids.add(track_id)
            self._perched_filter.initialize_track(track_id, position_m)
        else:
            self._perched_filter.remove_track(track_id)
            self._perched_track_ids.discard(track_id)
            self._constant_velocity_filter.initialize_track(track_id, position_m)

    def restart_perched_track_at_next_fix(self, track_id: str) -> None:
        """A relocation-gate hop: the perched track restarts at its next fix."""
        if track_id in self._perched_track_ids:
            self._perched_filter.restart_at_next_fix(track_id)

    def _route(self, track_id: str) -> TrackFilter:
        return self._perched_filter if track_id in self._perched_track_ids else self._constant_velocity_filter

    # -- TrackFilter Protocol -----------------------------------------------

    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        return self._route(state.id).predict(state, dt_s)

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
        measurement_covariance_m2: list[list[float]] | None = None,
    ) -> TrackState:
        return self._route(state.id).update(state, measurement_m, dt_s, measurement_covariance_m2)

    def initialize_track(self, track_id: str, position_m: tuple[float, float, float]) -> None:
        self._route(track_id).initialize_track(track_id, position_m)

    def remove_track(self, track_id: str) -> None:
        self._constant_velocity_filter.remove_track(track_id)
        self._perched_filter.remove_track(track_id)
        self._perched_track_ids.discard(track_id)

    def soft_position_update(
        self, track_id: str, corrected_position_m: tuple[float, float, float]
    ) -> bool:
        """Partial-constraint nudge; only perched tracks take one (a CV track's
        velocity state would contradict a bare position jump)."""
        if track_id not in self._perched_track_ids:
            return False
        return self._perched_filter.soft_position_update(track_id, corrected_position_m)

    def set_track_process_noise(self, track_id: str, process_noise: float | None) -> None:
        hook = getattr(self._constant_velocity_filter, "set_track_process_noise", None)
        if callable(hook) and track_id not in self._perched_track_ids:
            hook(track_id, process_noise)
