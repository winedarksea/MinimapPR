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
            residual = measurement - track_position
            distance = float(np.linalg.norm(residual))
            if distance < self._association_distance_m and distance < best_score:
                best_score = distance
                best_track_id = track.id
                continue
            gate = self._association_gate(
                measurement_covariance=measurement_covariance,
                track_covariance=self._coerce_covariance(track.position_covariance_m2),
            )
            if gate is None:
                if distance < self._association_distance_m and distance < best_score:
                    best_score = distance
                    best_track_id = track.id
                continue

            mahalanobis_sq, physical_gate_radius_m = gate
            if distance > physical_gate_radius_m:
                continue
            score = self._mahalanobis_distance_squared(residual, mahalanobis_sq)
            if score is None:
                continue
            if score <= 9.0 and score < best_score:
                best_score = score
                best_track_id = track.id

        return best_track_id

    def _association_gate(
        self,
        *,
        measurement_covariance: np.ndarray | None,
        track_covariance: np.ndarray | None,
    ) -> tuple[np.ndarray, float] | None:
        combined_covariance = None
        if measurement_covariance is not None and track_covariance is not None:
            combined_covariance = measurement_covariance + track_covariance
        elif measurement_covariance is not None:
            combined_covariance = measurement_covariance
        elif track_covariance is not None:
            combined_covariance = track_covariance
        if combined_covariance is None:
            return None
        try:
            covariance = self._positive_definite_covariance(combined_covariance)
            eigenvalues = np.linalg.eigvalsh(covariance)
        except np.linalg.LinAlgError:
            return None
        largest_variance = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        if not np.isfinite(largest_variance) or largest_variance <= 0.0:
            return None
        sigma = float(np.sqrt(largest_variance))
        physical_gate_radius_m = float(
            np.clip(3.0 * sigma, self._association_distance_m, self._association_distance_m * 4.0)
        )
        return covariance, physical_gate_radius_m

    def _mahalanobis_distance_squared(
        self,
        residual: np.ndarray,
        covariance: np.ndarray,
    ) -> float | None:
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            return None
        score = float(residual @ solved)
        return score if np.isfinite(score) else None

    def _positive_definite_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            raise np.linalg.LinAlgError("non-finite covariance eigenvalues")
        clamped = np.maximum(eigenvalues, 1.0e-6)
        stabilized = eigenvectors @ np.diag(clamped) @ eigenvectors.T
        return 0.5 * (stabilized + stabilized.T)

    def _coerce_covariance(self, value: list[list[float]] | None) -> np.ndarray | None:
        if value is None:
            return None
        try:
            covariance = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return None
        try:
            return self._positive_definite_covariance(covariance)
        except np.linalg.LinAlgError:
            return None
