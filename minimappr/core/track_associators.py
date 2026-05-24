"""Track association strategies implementing the TrackAssociator Protocol.

Each associator determines which existing track (if any) a new detection
should be assigned to.  The nearest-neighbor associator is the Phase 1
baseline; Phase 3 will introduce MHT and JPDA alternatives that conform
to the same ``TrackAssociator`` Protocol defined in ``interfaces.py``.
"""

from __future__ import annotations

import numpy as np

from minimappr.models import TrackState, TrackStatus


class NearestNeighborAssociator:
    """Nearest-neighbor association: match a detection to the closest
    active track within a configurable distance gate.

    This is the default Phase 1 strategy — simple, low-latency, and
    adequate for moderate track density.  It does not handle ambiguous
    association (crossing tracks, high clutter) well; those scenarios
    require MHT or JPDA (Phase 3).
    """

    def __init__(self, association_distance_m: float) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("association_distance_m must be > 0")
        self._association_distance_m = association_distance_m

    def associate(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        existing_tracks: list[TrackState],
        measurement_covariance_m2: list[list[float]] | None = None,
    ) -> str | None:
        """Return the track ID of the closest active track within the
        association gate, or ``None`` if no track is close enough (meaning
        a new track should be created).

        ``existing_tracks`` should contain predicted (extrapolated)
        positions so that the gate comparison uses the expected location
        at ``timestamp_ns``, not the last-observed location.
        """
        measurement = np.asarray(position_m, dtype=np.float64)
        best_track_id: str | None = None
        best_score = float("inf")
        measurement_covariance = self._coerce_covariance(measurement_covariance_m2)

        for track in existing_tracks:
            if track.status == TrackStatus.DROPPED.value:
                continue
            track_position = np.asarray(track.position_m, dtype=np.float64)
            distance = float(np.linalg.norm(measurement - track_position))
            gate_radius = self._adaptive_gate_radius(
                measurement_covariance=measurement_covariance,
                track_covariance=self._coerce_covariance(track.position_covariance_m2),
            )
            if gate_radius <= 0.0:
                continue
            score = distance / gate_radius
            if distance < gate_radius and score < best_score:
                best_score = score
                best_track_id = track.id

        return best_track_id

    def _adaptive_gate_radius(
        self,
        *,
        measurement_covariance: np.ndarray | None,
        track_covariance: np.ndarray | None,
    ) -> float:
        gate_radius = self._association_distance_m
        combined_covariance = None
        if measurement_covariance is not None and track_covariance is not None:
            combined_covariance = measurement_covariance + track_covariance
        elif measurement_covariance is not None:
            combined_covariance = measurement_covariance
        elif track_covariance is not None:
            combined_covariance = track_covariance
        if combined_covariance is None:
            return gate_radius
        try:
            eigenvalues = np.linalg.eigvalsh(combined_covariance)
        except np.linalg.LinAlgError:
            return gate_radius
        largest_variance = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        if not np.isfinite(largest_variance) or largest_variance <= 0.0:
            return gate_radius
        sigma = float(np.sqrt(largest_variance))
        return float(np.clip(max(gate_radius, 2.5 * sigma), gate_radius, gate_radius * 4.0))

    def _coerce_covariance(self, value: list[list[float]] | None) -> np.ndarray | None:
        if value is None:
            return None
        try:
            covariance = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return None
        return covariance
