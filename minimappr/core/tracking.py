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
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from minimappr.config import Settings, TrackingConfig
from minimappr.core.track_associators import (
    NearestNeighborAssociator,
    cosine_similarity,
)
from minimappr.core.track_filters import KalmanTrackFilter, LinearTrackFilter
from minimappr.interfaces import AssociationContext, TrackAssociator, TrackFilter
from minimappr.models import LabelId, TrackState, TrackStatus


def _real_category(value: str | None) -> str | None:
    """Normalised label category, or ``None`` when it carries no information."""
    if not value:
        return None
    normalised = value.strip().lower()
    if not normalised or normalised == "unknown":
        return None
    return normalised


@dataclass(slots=True)
class _DormantRecord:
    """Identity of a dropped track, parked for possible reacquisition.

    Sources with long silences (a bird singing every few minutes, a truck that
    idles out of earshot) used to lose their identity permanently at the DROPPED
    transition. Parking the identity lets a later detection at roughly the same
    place, with a compatible class, revive the original track id.

    Memory-only for v1: track ids already reset across process restarts, and the
    dropped track's DB row persists via the housekeeping upsert, so reviving the
    id continues the same row.
    """

    track_id: str
    first_seen_ns: int
    last_seen_ns: int
    position_m: tuple[float, float, float]
    label: str
    label_category: str
    fingerprint: dict[str, float] | None
    confidence: float
    update_count: int
    contributor_node_ids: list[str] = field(default_factory=list)
    track_kind: str = "acoustic"
    label_id: LabelId | None = None
    iff_category: str = "unknown"
    capability_tier: str = "full_3d"


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
        self._cfg = cfg
        self._default_track_kind = default_track_kind
        self._track_id_prefix = track_id_prefix
        self._tracks: dict[str, TrackState] = {}
        self._id_counter = itertools.count(1)
        self._multi_node_association_count = 0
        self._stale_ns = int(cfg.track_stale_seconds * 1_000_000_000)
        self._drop_ns = int(cfg.track_stale_seconds * cfg.track_drop_multiplier * 1_000_000_000)
        self._reap_ns = int(cfg.track_stale_seconds * cfg.track_reap_multiplier * 1_000_000_000)
        # Per-category lifecycle windows. ``unknown`` resolves to the base
        # values above, so BLE tracks (always ``unknown``) are unchanged.
        self._drop_multiplier = float(cfg.track_drop_multiplier)
        self._reap_multiplier = float(cfg.track_reap_multiplier)
        self._lifecycle_ns_cache: dict[str, tuple[int, int, int]] = {}

        # Classifier-score fingerprints, keyed by track id. Deliberately a side
        # dict rather than a TrackState field: TrackState is serialised to the
        # DB, the live hub and federation peers.
        self._fingerprints: dict[str, dict[str, float]] = {}
        self._fingerprint_alpha = float(
            min(1.0, max(0.0, getattr(cfg, "track_fingerprint_alpha", 0.3)))
        )
        self._fingerprint_top_k = max(1, int(getattr(cfg, "track_fingerprint_top_k", 8)))

        # Dormant (dropped-but-remembered) track registry.
        self._dormant: dict[str, _DormantRecord] = {}
        self._dormant_enabled = bool(getattr(cfg, "dormant_reacquire_enabled", True))
        self._dormant_ttl_ns = int(
            float(getattr(cfg, "dormant_ttl_seconds", 1800.0)) * 1_000_000_000
        )
        self._dormant_radius_m = float(getattr(cfg, "dormant_reacquire_radius_m", 20.0))
        self._dormant_min_similarity = float(
            getattr(cfg, "dormant_fingerprint_min_similarity", 0.4)
        )
        self._dormant_confidence_half_life_s = float(
            getattr(cfg, "dormant_confidence_half_life_seconds", 300.0)
        )
        self._dormant_max_records = max(1, int(getattr(cfg, "dormant_max_records", 64)))
        self._dormant_reacquired_count = 0

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
                category_gate_enabled=getattr(
                    cfg, "association_category_gate_enabled", True
                ),
                fingerprint_weight=getattr(cfg, "association_fingerprint_weight", 3.0),
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
                    max_coast_process_seconds=float(
                        getattr(cfg, "kalman_max_coast_process_seconds", 10.0)
                    ),
                    coast_velocity_half_life_seconds=float(
                        getattr(cfg, "kalman_coast_velocity_half_life_seconds", 10.0)
                    ),
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
        # Custom 3- and 4-parameter associators keep working: the association
        # context is only passed to strategies that declare a 5th parameter.
        self._associator_accepts_context = len(associator_signature.parameters) >= 5
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
        classifier_scores: dict[str, float] | None = None,
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

            # Tracks at non-localizable tiers are capped at tentative — the
            # position is just the sensor location so "confirmed" would be
            # misleading.
            can_confirm = capability_tier not in {"classification_only", "alerting_only"}

            # Delegate association to the pluggable strategy.
            if self._associator_accepts_context:
                matched_id = self._associator.associate(
                    timestamp_ns,
                    position_m,
                    predicted_tracks,
                    measurement_covariance_m2,
                    AssociationContext(
                        label=label,
                        label_category=label_category,
                        classifier_scores=classifier_scores,
                        track_fingerprints=self._fingerprints,
                    ),
                )
            elif self._associator_accepts_measurement_covariance:
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
                # --- Reacquire a dormant identity before minting a new one ---
                record = self._find_dormant_match(
                    now_ns=timestamp_ns,
                    position_m=position_m,
                    label=label,
                    label_category=label_category,
                    classifier_scores=classifier_scores,
                )
                if record is not None:
                    return self._revive_dormant(
                        record,
                        timestamp_ns=timestamp_ns,
                        position_m=position_m,
                        label=label,
                        confidence=confidence,
                        label_category=label_category,
                        iff_category=iff_category,
                        sensor_count=sensor_count,
                        label_id=label_id,
                        capability_tier=capability_tier,
                        measurement_covariance_m2=measurement_covariance_m2,
                        source_node_id=source_node_id,
                        classifier_scores=classifier_scores,
                        can_confirm=can_confirm,
                    )

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
                self._apply_process_noise(track_id, label_category)
                self._update_fingerprint(track_id, classifier_scores)
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
            label_changed = label != best_track.label
            category_changed = label_category != best_track.label_category
            best_track.label_id = label_id
            best_track.label = label
            best_track.label_category = label_category
            if category_changed:
                self._apply_process_noise(matched_id, label_category)
            best_track.iff_category = (
                iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown"
            )
            if label_changed:
                best_track.confidence = float(confidence)
            else:
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
            if best_track.status == TrackStatus.COASTING.value:
                # A previously confirmed track that re-associates exits coasting.
                best_track.status = TrackStatus.CONFIRMED.value if can_confirm else TrackStatus.TENTATIVE.value
            elif best_track.status == TrackStatus.TENTATIVE.value:
                if can_confirm and best_track.update_count >= self.CONFIRM_THRESHOLD:
                    best_track.status = TrackStatus.CONFIRMED.value
            elif can_confirm:
                best_track.status = TrackStatus.CONFIRMED.value

            best_track.tqi = self._compute_tqi(
                best_track.confidence,
                best_track.update_count,
                dt_s,
                sensor_count,
                contributor_count=max(len(best_track.contributor_node_ids), 1),
            )
            self._update_fingerprint(matched_id, classifier_scores)
            return best_track

    async def snapshot(self, now_ns: int | None = None) -> list[TrackState]:
        async with self._lock:
            if now_ns is not None:
                self._age_tracks(now_ns)
            return list(self._tracks.values())

    def _lifecycle_ns(self, label_category: str) -> tuple[int, int, int]:
        """(stale, drop, reap) thresholds in ns for *label_category*.

        ``unknown`` (and any unrecognised category) resolves to the base
        ``track_stale_seconds``, so BLE tracks and injected TrackingConfigs
        without the per-category fields behave exactly as before.
        """
        key = (label_category or "unknown").strip().lower() or "unknown"
        cached = self._lifecycle_ns_cache.get(key)
        if cached is not None:
            return cached
        resolver = getattr(self._cfg, "stale_seconds_for", None)
        if callable(resolver):
            stale_seconds = float(resolver(key))
        else:
            stale_seconds = float(self._cfg.track_stale_seconds)
        stale_ns = int(stale_seconds * 1_000_000_000)
        thresholds = (
            stale_ns,
            int(stale_seconds * self._drop_multiplier * 1_000_000_000),
            int(stale_seconds * self._reap_multiplier * 1_000_000_000),
        )
        self._lifecycle_ns_cache[key] = thresholds
        return thresholds

    def _apply_process_noise(self, track_id: str, label_category: str) -> None:
        """Push the category's Kalman ``q`` onto the filter, if it supports it.

        Duck-typed so injected/test filters without the hook keep working.
        """
        hook = getattr(self._filter, "set_track_process_noise", None)
        if not callable(hook):
            return
        resolver = getattr(self._cfg, "process_noise_for", None)
        if not callable(resolver):
            return
        hook(track_id, float(resolver(label_category)))

    def _update_fingerprint(
        self,
        track_id: str,
        classifier_scores: dict[str, float] | None,
    ) -> None:
        """EWMA-merge *classifier_scores* into the track's fingerprint."""
        if not classifier_scores:
            return
        alpha = self._fingerprint_alpha
        previous = self._fingerprints.get(track_id) or {}
        merged: dict[str, float] = {}
        for key in set(previous) | set(classifier_scores):
            value = (alpha * float(classifier_scores.get(key, 0.0))) + (
                (1.0 - alpha) * float(previous.get(key, 0.0))
            )
            if value > 0.0:
                merged[key] = value
        if not merged:
            self._fingerprints.pop(track_id, None)
            return
        top = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        top = top[: self._fingerprint_top_k]
        norm = math.sqrt(sum(value * value for _key, value in top))
        if norm <= 0.0:
            self._fingerprints.pop(track_id, None)
            return
        self._fingerprints[track_id] = {key: value / norm for key, value in top}

    # -- Dormant reacquisition ----------------------------------------------

    def _park_dormant(self, track: TrackState, now_ns: int) -> None:
        """Remember a just-dropped track's identity for later reacquisition."""
        if not self._dormant_enabled:
            return
        # Never park one-hit clutter — only tracks that were corroborated.
        if track.update_count < self.CONFIRM_THRESHOLD:
            return
        fingerprint = self._fingerprints.get(track.id)
        self._dormant[track.id] = _DormantRecord(
            track_id=track.id,
            first_seen_ns=track.first_seen_ns,
            last_seen_ns=track.last_seen_ns,
            position_m=(
                float(track.position_m[0]),
                float(track.position_m[1]),
                float(track.position_m[2]),
            ),
            label=track.label,
            label_category=track.label_category,
            fingerprint=dict(fingerprint) if fingerprint else None,
            confidence=float(track.confidence),
            update_count=int(track.update_count),
            contributor_node_ids=list(track.contributor_node_ids),
            track_kind=track.track_kind,
            label_id=track.label_id,
            iff_category=track.iff_category,
            capability_tier=track.capability_tier,
        )
        while len(self._dormant) > self._dormant_max_records:
            oldest = min(self._dormant.values(), key=lambda record: record.last_seen_ns)
            self._dormant.pop(oldest.track_id, None)

    def _purge_dormant(self, now_ns: int) -> None:
        if not self._dormant:
            return
        expired = [
            record.track_id
            for record in self._dormant.values()
            if now_ns - record.last_seen_ns > self._dormant_ttl_ns
        ]
        for track_id in expired:
            self._dormant.pop(track_id, None)

    def _find_dormant_match(
        self,
        *,
        now_ns: int,
        position_m: tuple[float, float, float],
        label: str,
        label_category: str,
        classifier_scores: dict[str, float] | None,
    ) -> _DormantRecord | None:
        if not self._dormant_enabled or not self._dormant:
            return None
        detection_category = _real_category(label_category)
        radius = self._dormant_radius_m
        if radius <= 0.0:
            return None
        best: _DormantRecord | None = None
        best_score = float("inf")
        for record in self._dormant.values():
            elapsed_ns = now_ns - record.last_seen_ns
            if elapsed_ns < 0 or elapsed_ns > self._dormant_ttl_ns:
                continue
            record_category = _real_category(record.label_category)
            if (
                detection_category is not None
                and record_category is not None
                and record_category != detection_category
            ):
                continue
            distance = math.dist(
                (float(position_m[0]), float(position_m[1]), float(position_m[2])),
                record.position_m,
            )
            if distance > radius:
                continue
            exact_label = bool(label) and label == record.label
            if classifier_scores and record.fingerprint:
                similarity = cosine_similarity(classifier_scores, record.fingerprint)
            else:
                similarity = 1.0 if exact_label else 0.0
            if not exact_label and similarity < self._dormant_min_similarity:
                continue
            score = (distance / radius) + (0.5 * (1.0 - similarity))
            if score < best_score:
                best_score = score
                best = record
        return best

    def _revive_dormant(
        self,
        record: _DormantRecord,
        *,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        label: str,
        confidence: float,
        label_category: str,
        iff_category: str,
        sensor_count: int,
        label_id: LabelId | None,
        capability_tier: str,
        measurement_covariance_m2: list[list[float]] | None,
        source_node_id: str | None,
        classifier_scores: dict[str, float] | None,
        can_confirm: bool,
    ) -> TrackState:
        """Resurrect a dormant identity onto this detection."""
        track_id = record.track_id
        p_var = self._initial_position_variance
        covariance = (
            measurement_covariance_m2
            if measurement_covariance_m2 is not None
            else [
                [p_var, 0.0, 0.0],
                [0.0, p_var, 0.0],
                [0.0, 0.0, p_var],
            ]
        )
        elapsed_s = max((timestamp_ns - record.last_seen_ns) / 1_000_000_000.0, 0.0)
        half_life = self._dormant_confidence_half_life_s
        decayed = (
            float(record.confidence) * (0.5 ** (elapsed_s / half_life))
            if half_life > 0.0
            else 0.0
        )
        revived_confidence = float(max(float(confidence), decayed))
        contributors = list(record.contributor_node_ids)
        if source_node_id and source_node_id not in contributors:
            contributors.append(source_node_id)
        if len(contributors) > self._MAX_CONTRIBUTOR_NODE_IDS:
            contributors = contributors[-self._MAX_CONTRIBUTOR_NODE_IDS :]
        update_count = int(record.update_count) + 1
        status = (
            TrackStatus.CONFIRMED.value if can_confirm else TrackStatus.TENTATIVE.value
        )
        resolved_position = (
            float(position_m[0]),
            float(position_m[1]),
            float(position_m[2]),
        )
        resolved_iff = (
            iff_category if iff_category in {"friendly", "unknown", "hostile"} else "unknown"
        )
        tqi = self._compute_tqi(
            revived_confidence,
            update_count,
            0.0,
            sensor_count,
            contributor_count=max(len(contributors), 1),
        )

        revived = self._tracks.get(track_id)
        if revived is not None:
            # Still present in the table (dropped but not yet reaped): mutate in
            # place so the snapshot keeps exactly one entry for this identity.
            revived.last_seen_ns = timestamp_ns
            revived.position_m = resolved_position
            revived.position_covariance_m2 = covariance
            revived.velocity_mps = (0.0, 0.0, 0.0)
            revived.label_id = label_id
            revived.label = label
            revived.label_category = label_category
            revived.iff_category = resolved_iff
            revived.confidence = revived_confidence
            revived.update_count = update_count
            revived.status = status
            revived.capability_tier = capability_tier
            revived.contributor_node_ids = contributors
            revived.tqi = tqi
        else:
            revived = TrackState(
                id=track_id,
                first_seen_ns=record.first_seen_ns,
                last_seen_ns=timestamp_ns,
                position_m=resolved_position,
                position_covariance_m2=covariance,
                velocity_mps=(0.0, 0.0, 0.0),
                label_id=label_id,
                label=label,
                label_category=label_category,
                iff_category=resolved_iff,
                confidence=revived_confidence,
                update_count=update_count,
                status=status,
                tqi=tqi,
                capability_tier=capability_tier,
                track_kind=record.track_kind,
                contributor_node_ids=contributors,
            )
            self._tracks[track_id] = revived

        # A velocity prior from minutes ago is worthless — start the filter fresh.
        self._filter.initialize_track(track_id, resolved_position)
        self._apply_process_noise(track_id, label_category)
        if record.fingerprint:
            self._fingerprints[track_id] = dict(record.fingerprint)
        self._update_fingerprint(track_id, classifier_scores)
        self._dormant.pop(track_id, None)
        self._dormant_reacquired_count += 1
        return revived

    def _age_tracks(self, now_ns: int) -> None:
        reap_ids: list[str] = []
        for track in self._tracks.values():
            gap_ns = now_ns - track.last_seen_ns
            stale_ns, drop_ns, reap_ns = self._lifecycle_ns(track.label_category)
            if track.status == TrackStatus.DROPPED.value:
                if gap_ns > reap_ns:
                    reap_ids.append(track.id)
                continue
            if gap_ns > drop_ns:
                track.status = TrackStatus.DROPPED.value
                self._park_dormant(track, now_ns)
                self._filter.remove_track(track.id)
            elif gap_ns > stale_ns:
                if track.status in (TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value):
                    track.status = TrackStatus.COASTING.value
        for track_id in reap_ids:
            self._tracks.pop(track_id, None)
            self._filter.remove_track(track_id)
            self._fingerprints.pop(track_id, None)
        self._purge_dormant(now_ns)

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
        seconds_since_last_update: float,
        sensor_count: int,
        contributor_count: int = 1,
    ) -> float:
        """Composite Track Quality Index.

        Components:
            - classification confidence (0-1)
            - corroboration factor based on update count AND distinct contributing
              nodes (a cross-node fusion is more trustworthy than repeat single-node
              detections)
            - recency penalty (decays with the gap since the previous update, not
              total track age)
            - sensor diversity bonus
        """
        update_corroboration = min(1.0, update_count / 5.0)
        # Distinct contributing nodes strongly corroborate: 2 nodes → full credit.
        node_corroboration = min(1.0, max(contributor_count - 1, 0) / 1.0)
        corroboration = max(update_corroboration, node_corroboration)
        recency = 1.0 / (1.0 + seconds_since_last_update / 30.0)
        sensor_factor = min(1.0, sensor_count / 4.0)
        w_conf, w_corr, w_rec, w_sensor = self._tqi_weights
        tqi = (w_conf * confidence) + (w_corr * corroboration) + (w_rec * recency) + (w_sensor * sensor_factor)
        return float(np.clip(tqi, 0.0, 1.0))

    def dormant_reacquired_count(self) -> int:
        """Cumulative count of dropped tracks revived from the dormant registry."""
        return self._dormant_reacquired_count

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
