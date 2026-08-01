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

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint

#: Discriminator for the persisted checkpoint format. Present only on geodetic
#: checkpoints; its absence marks an ENU-era row that cannot be reprojected.
_SNAPSHOT_FORMAT_GEODETIC = "geodetic_v1"
#: 9 decimal degrees is ~0.1 mm, far finer than GNSS, and bounds the JSON size of
#: a full reservoir. Altitude is metres, so 3 decimals is the same scale.
_GEO_LAT_LON_DECIMALS = 9
_GEO_ALT_DECIMALS = 3


def _geo_to_payload(geo: GeoPoint) -> list[float]:
    return [
        round(geo.lat, _GEO_LAT_LON_DECIMALS),
        round(geo.lon, _GEO_LAT_LON_DECIMALS),
        round(geo.alt_m, _GEO_ALT_DECIMALS),
    ]


def _payload_to_geo(entry: object) -> GeoPoint | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        return None
    try:
        return GeoPoint(lat=float(entry[0]), lon=float(entry[1]), alt_m=float(entry[2]))
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class StationaryKdeState:
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    seen_count: int = 0
    estimate: tuple[float, float, float] | None = None
    horizontal_std_m: float | None = None
    last_evaluated_seen_count: int = 0
    last_evaluated_monotonic_s: float = 0.0
    last_checkpoint_monotonic_s: float = 0.0

    def snapshot(self, coordinate_frame: "LocalCoordinateFrame") -> dict:
        """Serialize for persistence in geodetic coordinates.

        Samples are held in ENU metres for the KDE maths, but persisting them
        that way binds the checkpoint to the site origin in force when it was
        written: restore it under a different origin and the acceptance radius
        rejects every correct fix as an outlier, pinning the node at its
        old-frame position forever. Geodetic coordinates are what the fixes
        actually were, so a checkpoint stays valid across any origin change and
        hours of GNSS averaging survive a re-anchor.
        """
        return {
            "format": _SNAPSHOT_FORMAT_GEODETIC,
            "samples": [
                _geo_to_payload(coordinate_frame.local_to_geo(sample)) for sample in self.samples
            ],
            "seen_count": self.seen_count,
            "estimate": (
                _geo_to_payload(coordinate_frame.local_to_geo(self.estimate)) if self.estimate else None
            ),
            "horizontal_std_m": self.horizontal_std_m,
            "last_evaluated_seen_count": self.last_evaluated_seen_count,
        }

    @classmethod
    def from_snapshot(cls, value: dict, coordinate_frame: "LocalCoordinateFrame") -> "StationaryKdeState | None":
        """Rebuild from a geodetic checkpoint, projected into the current frame.

        Returns None for a checkpoint this build cannot interpret — in practice a
        pre-geodetic one holding raw ENU, whose origin is unrecoverable. Those are
        rebuilt from live fixes rather than guessed at, since a wrong restore is
        far more damaging than a cold start (about a second of warmup).
        """
        if value.get("format") != _SNAPSHOT_FORMAT_GEODETIC:
            return None
        samples = [
            coordinate_frame.geo_to_local(geo)
            for geo in (_payload_to_geo(entry) for entry in value.get("samples", []))
            if geo is not None
        ]
        estimate_geo = _payload_to_geo(value.get("estimate"))
        return cls(
            samples=samples,
            seen_count=max(int(value.get("seen_count", len(samples))), len(samples)),
            estimate=coordinate_frame.geo_to_local(estimate_geo) if estimate_geo is not None else None,
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
