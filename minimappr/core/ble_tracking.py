"""BLE-device-as-track background estimation.

Turns per-``(mac, node_id)`` RSSI aggregates from the ``BleObservationStore`` into
first-class :class:`~minimappr.models.TrackState` objects via RSSI log-distance
multilateration, feeding a *dedicated* :class:`~minimappr.core.tracking.TrackManager`.

The dedicated manager gives BLE its own track-ID space and independently tunable
association gating, structurally guaranteeing BLE positions never associate or
merge with acoustic tracks (v1 keeps the modalities separate).
"""

from __future__ import annotations

from typing import Any

from minimappr.config import Settings, TrackingConfig
from minimappr.core.ble_multilateration import (
    BleDeviceEstimate,
    estimate_ble_device_position,
)
from minimappr.core.ble_observations import BleObservationRecord, BleObservationStore
from minimappr.core.tracking import TrackManager
from minimappr.models import TrackState


def _device_label(mac: str, observations: list[BleObservationRecord]) -> str:
    """Prefer the most recent advertised device name, falling back to the MAC."""
    for observation in sorted(observations, key=lambda item: item.recv_ns, reverse=True):
        name = (observation.name or "").strip()
        if name:
            return name
    return mac


def _confidence_from_estimate(estimate: BleDeviceEstimate) -> float:
    """Derive a [0, 1] confidence from receiver count and fit residual.

    More receivers and a tighter RSSI-range residual both raise confidence.
    """
    receiver_factor = min(1.0, max(0, estimate.n_receivers) / 5.0)
    residual_factor = 1.0 / (1.0 + max(0.0, estimate.residual_rms_m))
    return round(receiver_factor * residual_factor, 4)


class BleTracker:
    """Owns a dedicated :class:`TrackManager` and runs BLE estimation ticks."""

    def __init__(self, tracking_config: Settings | TrackingConfig) -> None:
        # Dedicated ID prefix keeps BLE tracks in a separate namespace from the
        # acoustic manager's ``trk-`` IDs so they never collide in the shared
        # ``tracks`` table (both managers count IDs from 1 independently).
        self._tracker = TrackManager(
            tracking_config,
            default_track_kind="ble",
            track_id_prefix="ble-",
        )

    @property
    def tracker(self) -> TrackManager:
        return self._tracker

    async def run_tick(
        self,
        *,
        storage: Any,
        observation_store: BleObservationStore,
        node_positions: dict[str, tuple[float, float, float]],
        now_ns: int,
        live_hub: Any | None = None,
        coordinate_frame: Any | None = None,
    ) -> list[TrackState]:
        """Estimate every localizable BLE device once and persist/broadcast tracks.

        Returns the list of :class:`TrackState` objects updated this tick. Devices
        seen by fewer than three non-collinear nodes are skipped (no estimate).
        """
        grouped = await observation_store.latest_by_device(now_ns=now_ns)
        updated: list[TrackState] = []
        for mac, observations in grouped.items():
            estimate = estimate_ble_device_position(
                mac,
                observations,
                node_positions,
                now_ns=now_ns,
            )
            if estimate is None:
                continue

            track = await self._tracker.update(
                timestamp_ns=estimate.last_seen_ns,
                position_m=estimate.position_m,
                label=_device_label(mac, observations),
                confidence=_confidence_from_estimate(estimate),
                label_category="unknown",
                iff_category="unknown",
                capability_tier="2d",
                measurement_covariance_m2=estimate.covariance,
                source_node_id=(
                    estimate.contributor_node_ids[0]
                    if estimate.contributor_node_ids
                    else None
                ),
            )
            if coordinate_frame is not None:
                track.position_geo = coordinate_frame.local_to_geo(track.position_m)

            await storage.upsert_track(track)
            if live_hub is not None:
                await live_hub.broadcast(
                    {
                        "type": "track_updated",
                        "track": track.model_dump(mode="json"),
                        "server_time_ns": now_ns,
                    }
                )
            updated.append(track)
        return updated
