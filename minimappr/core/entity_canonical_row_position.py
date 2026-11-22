"""Which position an entity-mode detection row persists (Field Mode S-fixes tracking).

With entity reporting a bird has one canonical row that each positioned bout
updates in place. W3 wrote each bout's *raw* fix into it, so stored history and
``/detections`` jumped to the latest (possibly poor) fix while the entity on
the map followed the tracker's filtered position (S gate: overlap scene median
0.4 -> 15.9 m, omni scene 1.8 -> 52 m). The row now carries the bound track's
fused position and covariance, the same thing the entity shows; the raw fix is
kept in the feature summary and still feeds the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

from minimappr.models import TrackState

RAW_FIX_POSITION_FEATURE_KEY = "raw_fix_position_m"
RAW_FIX_COVARIANCE_FEATURE_KEY = "raw_fix_position_covariance_m2"


@dataclass(frozen=True, slots=True)
class PersistedRowPosition:
    position_m: tuple[float, float, float]
    covariance_m2: list[list[float]] | None
    raw_fix_features: dict[str, object]


def persisted_row_position(
    *,
    use_fused_track_position: bool,
    track: TrackState | None,
    raw_position_m: tuple[float, float, float],
    raw_covariance_m2: list[list[float]] | None,
) -> PersistedRowPosition:
    """The track's fused position when asked for and available, else the raw fix."""
    if not use_fused_track_position or track is None:
        return PersistedRowPosition(position_m=raw_position_m, covariance_m2=raw_covariance_m2, raw_fix_features={})
    return PersistedRowPosition(
        position_m=tuple(float(v) for v in track.position_m[:3]),  # type: ignore[arg-type]
        covariance_m2=track.position_covariance_m2,
        raw_fix_features={
            RAW_FIX_POSITION_FEATURE_KEY: [float(v) for v in raw_position_m[:3]],
            RAW_FIX_COVARIANCE_FEATURE_KEY: raw_covariance_m2,
        },
    )
