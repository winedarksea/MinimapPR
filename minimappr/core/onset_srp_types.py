"""Shared value types for onset-frame SRP clustering (Field Mode S2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minimappr.core.onset_frame_correlation import PairCorrelationCurve

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from minimappr.core.onset_cluster_attribution_evidence import ClusterAttributionEvidence

CLUSTER_STATUS_ACCEPTED = "accepted"
CLUSTER_STATUS_TENTATIVE = "tentative"
CLUSTER_STATUS_ECHO = "echo"


@dataclass(slots=True)
class OnsetFrame:
    """One onset's template-vs-search correlations (the unit of localization)."""

    frame_index: int
    template_node_id: str
    reference_sensor: str
    onset_sample: int  # index into the analysis span (shared timeline)
    strength: float
    band_min_hz: float
    band_max_hz: float
    curves: list[PairCorrelationCurve]
    weight: float = 0.0


@dataclass(frozen=True, slots=True)
class FramePeak:
    frame_index: int
    position_m: np.ndarray
    direction: np.ndarray
    range_m: float
    value: float
    weight: float
    is_frame_primary: bool


@dataclass(slots=True)
class OnsetCluster:
    peaks: list[FramePeak] = field(default_factory=list)
    status: str = CLUSTER_STATUS_ACCEPTED
    status_reasons: list[str] = field(default_factory=list)
    independent_onset_count: int = 0

    @property
    def weight(self) -> float:
        return float(sum(peak.weight for peak in self.peaks))

    @property
    def frame_indices(self) -> set[int]:
        return {peak.frame_index for peak in self.peaks}

    @property
    def led_frame_indices(self) -> set[int]:
        return {peak.frame_index for peak in self.peaks if peak.is_frame_primary}

    def weighted_direction(self) -> np.ndarray:
        stacked = np.vstack([peak.direction * peak.weight for peak in self.peaks]).sum(axis=0)
        norm = float(np.linalg.norm(stacked))
        return stacked / norm if norm > 0.0 else self.peaks[0].direction

    def weighted_position_m(self) -> np.ndarray:
        weights = np.asarray([peak.weight for peak in self.peaks])
        positions = np.vstack([peak.position_m for peak in self.peaks])
        return (weights[:, None] * positions).sum(axis=0) / max(float(weights.sum()), 1e-12)


@dataclass(frozen=True, slots=True)
class LocalizedOnsetCluster:
    """A clustered, refined candidate bird ready for the fusion pipeline."""

    position_m: tuple[float, float, float]
    covariance_m2: list[list[float]] | None
    confidence: float
    gdop: float
    range_projection_mode: str | None
    range_observability: float | None
    residual_rms_seconds: float | None
    reference_sensor: str
    tdoa_s: dict[str, float]
    status: str
    status_reasons: tuple[str, ...]
    onset_samples: tuple[int, ...]  # led onsets at their template nodes (span timeline)
    template_node_ids: tuple[str, ...]
    band_min_hz: float
    band_max_hz: float
    weight: float
    frame_count: int
    independent_onset_count: int
    refined_by_least_squares: bool
    used_envelope_fraction: float
    contributing_node_ids: tuple[str, ...]
    loudest_node_id: str
    # Field Mode W3-attribution: per-node levels/arrivals; None unless attribution is on.
    attribution_evidence: ClusterAttributionEvidence | None = None
    # Field Mode S-fixes: node-plane mirror verdict and the SRP-supported range interval (m from
    # the layout centroid); None interval = range not judged (single array or profile off).
    height_resolution: str | None = None
    range_interval_m: tuple[float, float] | None = None
    # Near edge of a far-open range interval (profile still rising at the search limit): the
    # source is at least this far along the bearing; None = range bounded on both sides or not judged.
    range_lower_bound_m: float | None = None


@dataclass(frozen=True, slots=True)
class OnsetClusteringResult:
    clusters: tuple[LocalizedOnsetCluster, ...]
    onset_count: int
    frame_count: int
    consumed_onset_samples_by_node: dict[str, tuple[int, ...]]
