"""Measured per-node level constraint for partial binding (Field Mode W3-attribution).

Replaces the W2-tracker stand-in (``SnrRankingConstraint``: "the reporting node is
about the nearest") whenever a detection carries real per-node, noise-floor-
relative band SNRs (``onset_cluster_node_band_snr_db``, written by the onset
stage when attribution is on).

For a candidate track position the 1/r spreading model predicts each node's
level up to the unknown source level, which is fitted from the audible nodes.
The residual is the RMS misfit in dB over the level margin. Nodes under the
audible threshold are censored: they only count when the fit says they should
clearly have heard the bird. Faint, distant birds are never rejected on
absolute level -- only the *shape* of the gradient across nodes matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DECIBELS_PER_AMPLITUDE_DECADE = 20.0
MINIMUM_SPREADING_DISTANCE_M = 0.1


@dataclass(frozen=True)
class MeasuredLevelGradientConstraint:
    node_positions_m: tuple[np.ndarray, ...]
    node_band_snr_db: tuple[float, ...]
    audible_snr_db: float
    level_margin_db: float
    kind: str = "level_gradient"
    # The margin is a tolerance on RMS misfit, so (like the stand-in) a hard bound.
    maximum_normalized_residual: float | None = 1.0

    def normalized_residual(self, position_m: np.ndarray) -> float:
        position = np.asarray(position_m, dtype=np.float64)
        spreading = np.asarray(
            [
                DECIBELS_PER_AMPLITUDE_DECADE
                * math.log10(max(float(np.linalg.norm(position - node)), MINIMUM_SPREADING_DISTANCE_M))
                for node in self.node_positions_m
            ]
        )
        measured = np.asarray(self.node_band_snr_db, dtype=np.float64)
        audible = measured >= self.audible_snr_db
        if not np.any(audible):
            return 0.0
        source_level = float(np.mean(measured[audible] + spreading[audible]))
        predicted = source_level - spreading
        misfit = np.where(audible, measured - predicted, np.maximum(predicted - self.audible_snr_db, 0.0))
        return float(np.sqrt(np.mean(np.square(misfit)))) / self.level_margin_db

    def corrected_position(self, position_m: np.ndarray) -> np.ndarray | None:
        return None


def measured_level_constraint(
    *,
    node_band_snr_db: dict[str, float] | None,
    node_centres_m: dict[str, np.ndarray],
    audible_snr_db: float | None,
    level_margin_db: float,
) -> MeasuredLevelGradientConstraint | None:
    """A constraint when >= 2 registered nodes carry readings, else None (keep the stand-in)."""
    if not node_band_snr_db or audible_snr_db is None:
        return None
    readings = [
        (node_centres_m[node], float(snr))
        for node, snr in sorted(node_band_snr_db.items())
        if node in node_centres_m and isinstance(snr, int | float) and math.isfinite(float(snr))
    ]
    if len(readings) < 2:
        return None
    return MeasuredLevelGradientConstraint(
        node_positions_m=tuple(position for position, _ in readings),
        node_band_snr_db=tuple(snr for _, snr in readings),
        audible_snr_db=float(audible_snr_db),
        level_margin_db=level_margin_db,
    )
