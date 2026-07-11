"""Bounded, durable GNSS estimators for node geometry.

The expensive KDE evaluation consumes an immutable reservoir snapshot in a
worker thread.  Ingest only appends one coordinate and returns the last stable
estimate, protecting the audio path from location analytics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class StationaryKdeState:
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    seen_count: int = 0
    estimate: tuple[float, float, float] | None = None
    horizontal_std_m: float | None = None
    last_evaluated_seen_count: int = 0
    last_evaluated_monotonic_s: float = 0.0
    last_checkpoint_monotonic_s: float = 0.0

    def snapshot(self) -> dict:
        return {
            "samples": [list(sample) for sample in self.samples],
            "seen_count": self.seen_count,
            "estimate": list(self.estimate) if self.estimate else None,
            "horizontal_std_m": self.horizontal_std_m,
            "last_evaluated_seen_count": self.last_evaluated_seen_count,
        }

    @classmethod
    def from_snapshot(cls, value: dict) -> "StationaryKdeState":
        samples = [tuple(map(float, sample[:3])) for sample in value.get("samples", []) if len(sample) >= 3]
        estimate = value.get("estimate")
        return cls(
            samples=samples,
            seen_count=max(int(value.get("seen_count", len(samples))), len(samples)),
            estimate=tuple(map(float, estimate[:3])) if isinstance(estimate, list) and len(estimate) >= 3 else None,
            horizontal_std_m=value.get("horizontal_std_m"),
            last_evaluated_seen_count=int(value.get("last_evaluated_seen_count", 0)),
        )

    def add(self, point: tuple[float, float, float], capacity: int) -> None:
        self.seen_count += 1
        if len(self.samples) < capacity:
            self.samples.append(point)
            return
        # Algorithm R gives every historic trusted fix the same probability of
        # remaining in the bounded state, rather than favouring recent weather.
        replace_index = random.randrange(self.seen_count)
        if replace_index < capacity:
            self.samples[replace_index] = point


def compute_stationary_kde(
    samples: list[tuple[float, float, float]], bandwidth_m: float
) -> tuple[tuple[float, float, float], float]:
    """Return a 2-D Gaussian-KDE mode and robust cluster altitude/std."""
    points = np.asarray(samples, dtype=np.float64)
    if len(points) == 1:
        return tuple(points[0]), 0.0
    horizontal = points[:, :2]
    bandwidth = max(float(bandwidth_m), 0.1)
    # Evaluate at observed locations.  The subsequent weighted centroid removes
    # the discrete-sample jitter while retaining the densest multipath cluster.
    delta = horizontal[:, None, :] - horizontal[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    weights = np.exp(-0.5 * squared / (bandwidth * bandwidth))
    peak = int(np.argmax(np.sum(weights, axis=1)))
    cluster_weights = weights[peak]
    cluster_total = float(np.sum(cluster_weights))
    xy = np.sum(horizontal * cluster_weights[:, None], axis=0) / max(cluster_total, 1e-12)
    # Median vertical coordinate resists the much noisier consumer-GNSS height.
    local_cluster = points[cluster_weights >= math.exp(-4.5)]
    altitude = float(np.median(local_cluster[:, 2]))
    radial = np.linalg.norm(local_cluster[:, :2] - xy, axis=1)
    horizontal_std = float(np.quantile(radial, 0.68)) if len(radial) else 0.0
    return (float(xy[0]), float(xy[1]), altitude), horizontal_std
