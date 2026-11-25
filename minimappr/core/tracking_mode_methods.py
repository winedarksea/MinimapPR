"""Per-category tracking-mode methods of ``TrackManager`` (kept out of tracking.py).

Mixed into ``TrackManager``; relies on its ``_tracks``, ``_filter``,
``_field_mode``, ``_fingerprints`` and ``_apply_motion_model`` /
``_apply_process_noise``. Modes and defaults: ``core/tracking_mode_config.py``.
"""

from __future__ import annotations

from minimappr.core.tracking_mode_config import TrackingModeConfig


class TrackManagerTrackingModeMethods:
    @property
    def tracking_modes(self) -> TrackingModeConfig:
        return self._field_mode.tracking_modes

    def tracks_category(self, label_category: str | None) -> bool:
        """False for a ``none`` category: its fixes never start or feed a track."""
        return self.tracking_modes.tracks_category(label_category)

    def apply_label_category_tracking_mode(self, track_id: str, label_category: str) -> bool:
        """Apply the mode of a track's new label category; the caller holds the lock.

        Call whenever a track's label category is set or changes. Switching
        between ``continuous_motion`` and ``stationary_hold`` restarts the new
        filter at the track's position. A ``none`` category ends the track at
        once (an unlabeled provisional track turned out to be an untracked
        source) and returns False.
        """
        track = self._tracks.get(track_id)
        if track is None:
            return False
        if not self.tracks_category(label_category):
            self._tracks.pop(track_id, None)
            self._filter.remove_track(track_id)
            self._fingerprints.pop(track_id, None)
            self._field_mode.forget(track_id)
            return False
        self._apply_motion_model(track_id, label_category, track.position_m)
        self._apply_process_noise(track_id, label_category)
        return True
