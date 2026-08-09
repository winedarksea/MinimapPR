"""Track association strategies implementing the TrackAssociator Protocol.

Each associator determines which existing track (if any) a new detection
should be assigned to.  The nearest-neighbor associator is the Phase 1
baseline; Phase 3 will introduce MHT and JPDA alternatives that conform
to the same ``TrackAssociator`` Protocol defined in ``interfaces.py``.
"""

from __future__ import annotations

import numpy as np

from minimappr.interfaces import AssociationContext
from minimappr.models import TrackState, TrackStatus


class NearestNeighborAssociator:
    """Nearest-neighbor association: match a detection to the closest
    active track within a configurable distance gate.

    This is the default Phase 1 strategy — simple, low-latency, and
    adequate for moderate track density.  It does not handle ambiguous
    association (crossing tracks, high clutter) well; those scenarios
    require MHT or JPDA (Phase 3).
    """

    def __init__(
        self,
        association_distance_m: float,
        *,
        max_gate_m: float | None = None,
        chi2_gate: float = 9.0,
        category_gate_enabled: bool = True,
        fingerprint_weight: float = 3.0,
    ) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("association_distance_m must be > 0")
        self._association_distance_m = association_distance_m
        # Upper bound on the physical (Euclidean) association gate radius. Defaults to
        # the legacy 4×association_distance_m clamp so behaviour is unchanged unless a
        # deployment widens it for cross-node cone fusion (Phase 3).
        self._max_gate_m = (
            float(max_gate_m)
            if max_gate_m is not None and max_gate_m > 0.0
            else association_distance_m * 4.0
        )
        # Never let the configured max fall below the Euclidean shortcut radius.
        self._max_gate_m = max(self._max_gate_m, association_distance_m)
        self._chi2_gate = float(chi2_gate) if chi2_gate > 0.0 else 9.0
        # Class-aware association (Phase 2). The category gate is a hard reject
        # applied before any distance math; the fingerprint term only re-ranks
        # candidates that already passed the positional gate, so it can never
        # admit a match the geometry rejected nor widen a gate radius.
        self._category_gate_enabled = bool(category_gate_enabled)
        self._fingerprint_weight = max(0.0, float(fingerprint_weight))

    def associate(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        existing_tracks: list[TrackState],
        measurement_covariance_m2: list[list[float]] | None = None,
        context: AssociationContext | None = None,
    ) -> str | None:
        """Return the track ID of the closest active track within the
        association gate, or ``None`` if no track is close enough (meaning
        a new track should be created).

        ``existing_tracks`` should contain predicted (extrapolated)
        positions so that the gate comparison uses the expected location
        at ``timestamp_ns``, not the last-observed location.

        A detection within the plain ``association_distance_m`` Euclidean gate
        is always eligible for a track regardless of that track's covariance
        (an operator-configured wide gate must not be silently tightened by a
        well-converged track's own uncertainty) — but its score is expressed
        in the same chi-squared units as the covariance-aware path, calibrated
        so ``distance == association_distance_m`` sits exactly at the chi2
        gate boundary. That keeps ``best_score`` comparable across every
        candidate track in this call, instead of mixing raw meters with
        chi-squared numbers when both a close plain-distance match and a
        farther covariance-gated match are both in play.

        ``context`` carries the detection's label category and classifier
        scores.  A detection and a track whose categories are both *real*
        (non-``unknown``, non-empty) and different are never associated —
        ``unknown`` stays a wildcard so BLE tracks and low-confidence
        classifications behave exactly as before.  Classifier-score similarity
        against the track's running fingerprint is folded into the score (in
        chi-squared units) purely as a tie-break among gate-passers.
        """
        measurement = np.asarray(position_m, dtype=np.float64)
        best_track_id: str | None = None
        best_score = float("inf")
        measurement_covariance = self._coerce_covariance(measurement_covariance_m2)
        # Isotropic per-axis variance implied by the legacy Euclidean gate: solving
        # distance^2 / fallback_variance == chi2_gate for distance ==
        # association_distance_m.
        fallback_variance = (self._association_distance_m**2) / self._chi2_gate

        detection_category = _real_category(
            context.label_category if context is not None else None
        )
        detection_scores = context.classifier_scores if context is not None else None
        track_fingerprints = context.track_fingerprints if context is not None else {}

        for track in existing_tracks:
            if track.status == TrackStatus.DROPPED.value:
                continue
            if self._category_gate_enabled and detection_category is not None:
                track_category = _real_category(track.label_category)
                if track_category is not None and track_category != detection_category:
                    continue
            track_position = np.asarray(track.position_m, dtype=np.float64)
            residual = measurement - track_position
            distance = float(np.linalg.norm(residual))

            if distance < self._association_distance_m:
                score = (distance * distance) / fallback_variance
                score += self._fingerprint_penalty(
                    detection_scores, track_fingerprints.get(track.id)
                )
                if score < best_score:
                    best_score = score
                    best_track_id = track.id
                continue

            gate = self._association_gate(
                measurement_covariance=measurement_covariance,
                track_covariance=self._coerce_covariance(track.position_covariance_m2),
            )
            if gate is None:
                continue
            covariance, physical_gate_radius_m = gate
            if distance > physical_gate_radius_m:
                continue
            score = self._mahalanobis_distance_squared(residual, covariance)
            if score is None:
                continue
            if score > self._chi2_gate:
                continue
            # Fingerprint is added only after the positional gate has passed.
            score += self._fingerprint_penalty(
                detection_scores, track_fingerprints.get(track.id)
            )
            if score < best_score:
                best_score = score
                best_track_id = track.id

        # TODO(tracking): this is still a per-detection greedy nearest-score match.
        # If simultaneous crossing detections start mis-associating, consider
        # batching all detections in a fusion tick and resolving them jointly with
        # scipy.optimize.linear_sum_assignment (Hungarian algorithm) over the score
        # matrix, the way core/federation.py already does for peer-track
        # deconfliction.
        return best_track_id

    def _fingerprint_penalty(
        self,
        detection_scores: dict[str, float] | None,
        track_fingerprint: dict[str, float] | None,
    ) -> float:
        """Chi-squared-unit penalty for classifier-score dissimilarity.

        Zero when either side is missing, so a track with no fingerprint yet is
        never disadvantaged against one that has one.
        """
        if self._fingerprint_weight <= 0.0:
            return 0.0
        if not detection_scores or not track_fingerprint:
            return 0.0
        similarity = cosine_similarity(detection_scores, track_fingerprint)
        return self._fingerprint_weight * (1.0 - similarity)

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
            np.clip(3.0 * sigma, self._association_distance_m, self._max_gate_m)
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


def _real_category(value: str | None) -> str | None:
    """Normalised category, or ``None`` when it carries no information."""
    if not value:
        return None
    normalised = value.strip().lower()
    if not normalised or normalised == "unknown":
        return None
    return normalised


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosine similarity over two sparse score dicts, clamped to [0, 1].

    Classifier scores are non-negative, so the raw cosine is already in [0, 1];
    the clamp only guards against floating-point overshoot.
    """
    if not left or not right:
        return 0.0
    if len(right) < len(left):
        left, right = right, left
    dot = 0.0
    for key, value in left.items():
        other = right.get(key)
        if other is not None:
            dot += float(value) * float(other)
    if dot <= 0.0:
        return 0.0
    left_norm = float(np.sqrt(sum(float(v) * float(v) for v in left.values())))
    right_norm = float(np.sqrt(sum(float(v) * float(v) for v in right.values())))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(min(1.0, dot / (left_norm * right_norm)))
