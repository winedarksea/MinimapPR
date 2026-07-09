"""Track manager with pluggable association and filtering strategies.

Track lifecycle:
    tentative  -> confirmed  (2+ corroborating detections)
    confirmed  -> coasting   (no update within stale window)
    coasting   -> confirmed  (new detection re-associates)
    coasting   -> dropped    (exceeds drop timeout)
    tentative  -> dropped    (exceeds drop timeout without confirmation)

Track Quality Index (TQI):
    Composite score from position confidence, classification confidence,
    age-of-last-update, and number of corroborating sensors.

Association and filtering are delegated to injected strategy objects
conforming to the ``TrackAssociator`` and ``TrackFilter`` Protocols
(see ``interfaces.py``).  When not injected, default implementations
(``NearestNeighborAssociator``, ``LinearTrackFilter`` / ``KalmanTrackFilter``)
are constructed from ``TrackingConfig``.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from typing import Iterable

import numpy as np

from minimappr.config import Settings, TrackingConfig
from minimappr.core.track_associators import NearestNeighborAssociator
from minimappr.core.track_filters import KalmanTrackFilter, LinearTrackFilter
from minimappr.interfaces import TrackAssociator, TrackFilter
from minimappr.models import LabelId, TrackState, TrackStatus


class TrackManager:
    CONFIRM_THRESHOLD: int = 2  # detections needed to confirm a track
    _MAX_CONTRIBUTOR_NODE_IDS: int = 16  # bound on the per-track contributor set

    def __init__(
        self,
        settings: Settings | TrackingConfig,
        *,
        associator: TrackAssociator | None = None,
        track_filter: TrackFilter | None = None,
        default_track_kind: str = "acoustic",
        track_id_prefix: str = "trk-",
    ) -> None:
        cfg = settings.tracking_config() if isinstance(settings, Settings) else settings
        self._default_track_kind = default_track_kind
        self._track_id_prefix = track_id_prefix
        self._tracks: dict[str, TrackState] = {}
        self._id_counter = itertools.count(1)
        self._multi_node_association_count = 0
        self._stale_ns = int(cfg.track_stale_seconds * 1_000_000_000)
        self._drop_ns = int(cfg.track_stale_seconds * cfg.track_drop_multiplier * 1_000_000_000)
        self._reap_ns = int(cfg.track_stale_seconds * cfg.track_reap_multiplier * 1_000_000_000)

        # Initial position covariance for newly created TrackState objects
        # (for external consumers; the filter manages its own internal state).
        self._initial_position_variance = float(cfg.kalman_initial_position_variance)

        # Build default associator if not injected.
        if associator is not None:
            self._associator: TrackAssociator = associator
        else:
            self._associator = NearestNeighborAssociator(
                cfg.association_distance_m,
                max_gate_m=getattr(cfg, "association_max_gate_m", None),
                chi2_gate=getattr(cfg, "association_chi2_gate", 9.0),
            )

        # Build default filter if not injected.
        if track_filter is not None:
            self._filter: TrackFilter = track_filter
        else:
            filter_name = cfg.tracking_filter.strip().lower()
            if filter_name == "kalman":
                if cfg.kalman_process_noise < 0.0:
                    raise ValueError("kalman_process_noise must be >= 0.0")
                if cfg.kalman_measurement_noise <= 0.0:
                    raise ValueError("kalman_measurement_noise must be > 0.0")
                if cfg.kalman_initial_position_variance <= 0.0:
                    raise ValueError("kalman_initial_position_variance must be > 0.0")
                if cfg.kalman_initial_velocity_variance <= 0.0:
                    raise ValueError("kalman_initial_velocity_variance must be > 0.0")
                self._filter = KalmanTrackFilter(
                    process_noise=float(cfg.kalman_process_noise),
                    measurement_noise=float(cfg.kalman_measurement_noise),
                    initial_position_variance=float(cfg.kalman_initial_position_variance),
                    initial_velocity_variance=float(cfg.kalman_initial_velocity_variance),
                )
            elif filter_name == "linear":
                if cfg.linear_position_alpha < 0.0 or cfg.linear_position_alpha > 1.0:
                    raise ValueError("linear_position_alpha must be in [0, 1]")
                if cfg.linear_velocity_alpha < 0.0 or cfg.linear_velocity_alpha > 1.0:
                    raise ValueError("linear_velocity_alpha must be in [0, 1]")
                self._filter = LinearTrackFilter(
                    position_alpha=float(cfg.linear_position_alpha),
                    velocity_alpha=float(cfg.linear_velocity_alpha),
                    default_covariance_diagonal=cfg.association_distance_m * 0.5,
                )
            else:
                raise ValueError(
                    f"Unsupported tracking_filter '{cfg.tracking_filter}'. "
                    f"Supported values: linear, kalman"
                )

        associator_signature = inspect.signature(self._associator.associate)
        self._associator_accepts_measurement_covariance = (
            len(associator_signature.parameters) >= 4
        )
        filter_update_signature = inspect.signature(self._filter.update)
        self._filter_accepts_measurement_covariance = (
            len(filter_update_signature.parameters) >= 4
        )

        # TQI weights (normalised).
        raw_tqi_weights = np.asarray(
            [
                float(cfg.tqi_weight_confidence),
                float(cfg.tqi_weight_corroboration),
                float(cfg.tqi_weight_recency),
                float(cfg.tqi_weight_sensor),
            ],
            dtype=np.float64,
        )
        if np.any(raw_tqi_weights < 0.0):
            raise ValueError("TQI weights must be >= 0")
        weight_total = float(np.sum(raw_tqi_weights))
        if weight_total <= 0.0:
            raise ValueError("TQI weights must sum to > 0")
        self._tqi_weights = tuple(float(value / weight_total) for value in raw_tqi_weights)

        self._lock = asyncio.Lock()

    async def update(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        label: str,
        confidence: float,
        label_category: str = "unknown",
        iff_category: str = "unknown",
        sensor_count: int = 1,
        label_id: LabelId | None = None,
        capability_tier: str = "full_3d",
        measurement_covariance_m2: list[list[float]] | None = None,
        source_node_id: str | None = None,
    ) -> TrackState:
        async with self._lock:
            self._age_tracks(timestamp_ns)

            # Predict existing tracks forward for association gating.
            predicted_tracks: list[TrackState] = []
            for track in self._tracks.values():
                if track.status == TrackStatus.DROPPED.value:
                    continue
                dt_s = max(
                    (timestamp_ns - track.last_seen_ns) / 1_000_000_000.0, 0.0,
                )
                predicted_tracks.append(self._filter.predict(track, dt_s))

            # Delegate association to the pluggable strategy.
            if self._associator_accepts_measurement_covariance:
                matched_id = self._associator.associate(
                    timestamp_ns,
                    position_m,
                    predicted_tracks,
                    measurement_covariance_m2,
                )
            else:
                matched_id = self._associator.associate(
                    timestamp_ns,
                    position_m,
                    predicted_tracks,
                )

            if matched_id is None:
                # --- New track ---
                track_id = f"{self._track_id_prefix}{next(self._id_counter):05d}"
                p_var = self._initial_position_variance
                created = TrackState(
                    id=track_id,
                    first_seen_ns=timestamp_ns,
                    last_seen_ns=timestamp_ns,
                    position_m=(float(position_m[0]), float(position_m[1]), float(position_m[2])),
                    position_covariance_m2=(
                        measurement_covariance_m2
                        if measurement_covariance_m2 is not None
                        else [
                            [p_var, 0.0, 0.0],
                            [0.0, p_var, 0.0],
                            [0.0, 0.0, p_var],
                        ]
                    ),
                    velocity_mps=(0.0, 0.0, 0.0),
                    label_id=label_id,
                    label=label,
                    label_category=label_category,
                    iff_category=iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown",
                    confidence=float(confidence),
                    update_count=1,
                    status=TrackStatus.TENTATIVE.value,
                    tqi=self._compute_tqi(confidence, 1, 0.0, sensor_count, contributor_count=1),
                    capability_tier=capability_tier,
                    track_kind=self._default_track_kind,
                    contributor_node_ids=[source_node_id] if source_node_id else [],
                )
                self._tracks[track_id] = created
                self._filter.initialize_track(track_id, position_m)
                return created

            # --- Update matched track ---
            best_track = self._tracks[matched_id]
            dt_s = max(
                (timestamp_ns - best_track.last_seen_ns) / 1_000_000_000.0, 1e-3,
            )

            # Delegate filtering to the pluggable strategy.
            if self._filter_accepts_measurement_covariance:
                filtered = self._filter.update(
                    best_track,
                    position_m,
                    dt_s,
                    measurement_covariance_m2,
                )
            else:
                filtered = self._filter.update(
                    best_track,
                    position_m,
                    dt_s,
                )

            best_track.last_seen_ns = timestamp_ns
            best_track.position_m = filtered.position_m
            best_track.velocity_mps = filtered.velocity_mps
            best_track.position_covariance_m2 = filtered.position_covariance_m2
            best_track.label_id = label_id
            best_track.label = label
            best_track.label_category = label_category
            best_track.iff_category = (
                iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown"
            )
            best_track.confidence = float(max(best_track.confidence, confidence))
            best_track.update_count += 1
            best_track.capability_tier = capability_tier

            # Phase 3: record the contributing node (bounded, insertion-ordered).
            # A newly-added second distinct node marks this as a cross-node fusion.
            if source_node_id and source_node_id not in best_track.contributor_node_ids:
                had_single_node = len(best_track.contributor_node_ids) <= 1
                best_track.contributor_node_ids.append(source_node_id)
                if len(best_track.contributor_node_ids) > self._MAX_CONTRIBUTOR_NODE_IDS:
                    best_track.contributor_node_ids = best_track.contributor_node_ids[
                        -self._MAX_CONTRIBUTOR_NODE_IDS :
                    ]
                if had_single_node and len(best_track.contributor_node_ids) >= 2:
                    self._multi_node_association_count += 1

            # Lifecycle: tentative -> confirmed after enough detections.
            # Tracks at non-localizable tiers are capped at tentative — the
            # position is just the sensor location so "confirmed" would be
            # misleading.
            can_confirm = capability_tier not in {"classification_only", "alerting_only"}
            if best_track.status == TrackStatus.COASTING.value:
                # A previously confirmed track that re-associates exits coasting.
                best_track.status = TrackStatus.CONFIRMED.value if can_confirm else TrackStatus.TENTATIVE.value
            elif best_track.status == TrackStatus.TENTATIVE.value:
                if can_confirm and best_track.update_count >= self.CONFIRM_THRESHOLD:
                    best_track.status = TrackStatus.CONFIRMED.value
            elif can_confirm:
                best_track.status = TrackStatus.CONFIRMED.value

            age_s = (timestamp_ns - best_track.first_seen_ns) / 1_000_000_000.0
            best_track.tqi = self._compute_tqi(
                best_track.confidence,
                best_track.update_count,
                age_s,
                sensor_count,
                contributor_count=max(len(best_track.contributor_node_ids), 1),
            )
            return best_track

    async def snapshot(self, now_ns: int | None = None) -> list[TrackState]:
        async with self._lock:
            if now_ns is not None:
                self._age_tracks(now_ns)
            return list(self._tracks.values())

    def _age_tracks(self, now_ns: int) -> None:
        reap_ids: list[str] = []
        for track in self._tracks.values():
            gap_ns = now_ns - track.last_seen_ns
            if track.status == TrackStatus.DROPPED.value:
                if gap_ns > self._reap_ns:
                    reap_ids.append(track.id)
                continue
            if gap_ns > self._drop_ns:
                track.status = TrackStatus.DROPPED.value
                self._filter.remove_track(track.id)
            elif gap_ns > self._stale_ns:
                if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value):
                    track.status = TrackStatus.COASTING.value
        for track_id in reap_ids:
            self._tracks.pop(track_id, None)
            self._filter.remove_track(track_id)

    async def active_ids(self, now_ns: int) -> Iterable[str]:
        async with self._lock:
            self._age_tracks(now_ns)
            return [
                track.id for track in self._tracks.values()
                if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value)
            ]

    def _compute_tqi(
        self,
        confidence: float,
        update_count: int,
        age_s: float,
        sensor_count: int,
        contributor_count: int = 1,
    ) -> float:
        """Composite Track Quality Index.

        Components:
            - classification confidence (0-1)
            - corroboration factor based on update count AND distinct contributing
              nodes (a cross-node fusion is more trustworthy than repeat single-node
              detections)
            - recency penalty (decays with age)
            - sensor diversity bonus
        """
        update_corroboration = min(1.0, update_count / 5.0)
        # Distinct contributing nodes strongly corroborate: 2 nodes → full credit.
        node_corroboration = min(1.0, max(contributor_count - 1, 0) / 1.0)
        corroboration = max(update_corroboration, node_corroboration)
        recency = 1.0 / (1.0 + age_s / 30.0)
        sensor_factor = min(1.0, sensor_count / 4.0)
        w_conf, w_corr, w_rec, w_sensor = self._tqi_weights
        tqi = (w_conf * confidence) + (w_corr * corroboration) + (w_rec * recency) + (w_sensor * sensor_factor)
        return float(np.clip(tqi, 0.0, 1.0))

    def multi_node_association_count(self) -> int:
        """Cumulative count of tracks that gained a 2nd distinct contributing node."""
        return self._multi_node_association_count

    def multi_node_active_count(self) -> int:
        """Current number of active tracks corroborated by >=2 distinct nodes."""
        return sum(
            1
            for track in self._tracks.values()
            if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value)
            and len(track.contributor_node_ids) >= 2
        )
