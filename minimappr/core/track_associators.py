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
        best_distance = float("inf")

        for track in existing_tracks:
            if track.status == TrackStatus.DROPPED.value:
                continue
            track_position = np.asarray(track.position_m, dtype=np.float64)
            distance = float(np.linalg.norm(measurement - track_position))
            if distance < self._association_distance_m and distance < best_distance:
                best_distance = distance
                best_track_id = track.id

        return best_track_id
