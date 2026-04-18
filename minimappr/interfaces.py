"""Core interface contracts for pluggable MinimapPR subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Protocol, runtime_checkable

import numpy as np

from minimappr.models import (
    ClassificationResult,
    DetectionEvent,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    LabelId,
    LocalizationResult,
    NodeSpec,
    StoreForwardIngestRequest,
    StoreForwardIngestResponse,
    TrackState,
)


Vec3Array = np.ndarray


@dataclass(slots=True)
class EnvironmentReading:
    temperature_c: float
    humidity_fraction: float
    pressure_pa: float | None = None
    wind_speed_mps: float | None = None
    wind_dir_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionDescriptor:
    action_type: str
    destination: str
    priority: str = "normal"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleEvaluationResult:
    rule_id: str
    matched: bool
    descriptors: list[ActionDescriptor] = field(default_factory=list)
    reason: str | None = None


@runtime_checkable
class Localizer(Protocol):
    def localize(
        self,
        sensor_positions: dict[str, Vec3Array],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        ...


@runtime_checkable
class TrackAssociator(Protocol):
    def associate(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        existing_tracks: list[TrackState],
    ) -> str | None:
        ...


@runtime_checkable
class TrackFilter(Protocol):
    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        """Return predicted state extrapolated by *dt_s*.

        Non-destructive — internal filter state must not be modified.
        Used by TrackManager for association gating (the predicted
        position is compared against new measurements).
        """
        ...

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
    ) -> TrackState:
        """Apply measurement update and return the updated state.

        For filters with internal state (e.g. Kalman), this performs
        the full predict-then-correct cycle.  Returns a TrackState copy
        with updated position, velocity, and covariance fields.
        """
        ...

    def initialize_track(
        self,
        track_id: str,
        position_m: tuple[float, float, float],
    ) -> None:
        """Initialize internal filter state for a newly created track."""
        ...

    def remove_track(self, track_id: str) -> None:
        """Clean up internal filter state for a dropped/reaped track."""
        ...


@runtime_checkable
class StorageBackend(Protocol):
    async def initialize(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def upsert_node(
        self,
        spec: NodeSpec,
        last_seen_ns: int,
        position_geo: GeoPoint | None = None,
    ) -> None:
        ...

    async def insert_observation(
        self,
        *,
        node_id: str,
        sensor_id: str,
        sensor_type: str,
        source_type: str,
        toa_ns: int,
        tor_ns: int,
        time_quality: str,
        sample_rate_hz: int | None,
        channel_index: int | None,
        frame_sequence: int | None,
        retention_tier: str = "short",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> str:
        ...

    async def upsert_label(
        self,
        *,
        name: str,
        category: str,
        source: str,
        parent_label_id: LabelId | None = None,
        external_taxonomy: str | None = None,
        external_label_id: str | None = None,
        created_ns: int,
    ) -> LabelId:
        ...

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> None:
        ...

    async def update_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> bool:
        ...

    async def find_detection_for_reporting_window(
        self,
        *,
        source_node_id: str | None,
        label: str,
        report_window_start_ns: int,
        report_window_end_ns: int,
    ) -> dict[str, Any] | None:
        ...

    async def upsert_track(self, track: TrackState) -> None:
        ...

    async def insert_track_update(
        self,
        *,
        track: TrackState,
        timestamp_ns: int,
        event_id: str | None,
        update_type: str,
        detection_id: str | None,
        observation_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...

    async def insert_alert(
        self,
        *,
        timestamp_ns: int,
        rule_id: str | None,
        detection_id: str | None,
        track_id: str | None,
        destination: str,
        priority: str,
        status: str,
        payload: dict[str, Any] | None = None,
        alert_id: str | None = None,
    ) -> str:
        ...

    async def update_alert_status(
        self,
        *,
        alert_id: str,
        status: str,
        updated_ns: int,
        payload_patch: dict[str, Any] | None = None,
    ) -> bool:
        ...

    async def insert_ping(
        self,
        *,
        timestamp_ns: int,
        ping_type: str,
        label: str | None,
        label_id: LabelId | None,
        spl_db: float | None,
        position_m: tuple[float, float, float] | None,
        position_geo: GeoPoint | None,
        source_detection_id: str | None,
        source_observation_id: str | None,
        source_track_id: str | None,
        retention_tier: str,
        metadata: dict[str, Any] | None = None,
        ping_id: str | None = None,
    ) -> str:
        ...

    async def insert_environment(
        self,
        *,
        node_id: str,
        timestamp_ns: int,
        temperature_c: float | None,
        pressure_pa: float | None,
        humidity_fraction: float | None,
        wind_speed_mps: float | None,
        wind_dir_deg: float | None,
        solar_lux: float | None,
        metadata: dict[str, Any] | None = None,
        environment_id: str | None = None,
    ) -> str:
        ...

    async def list_environment(self, limit: int = 500, node_id: str | None = None) -> list[dict[str, Any]]:
        ...

    async def list_latest_environment_per_node(self, limit: int = 500) -> list[dict[str, Any]]:
        ...

    async def list_labels(self) -> list[dict[str, Any]]:
        ...

    async def get_node_by_id(self, node_id: str) -> dict[str, Any] | None:
        ...

    async def get_detection_by_event_id(self, event_id: str) -> dict[str, Any] | None:
        ...

    async def list_observations_by_ids(self, observation_ids: list[str]) -> list[dict[str, Any]]:
        ...

    async def latest_detection_audio_for_track(self, track_id: str) -> tuple[str, str] | None:
        ...

    async def list_track_updates(
        self,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        event_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    async def list_alerts(
        self,
        limit: int = 100,
        *,
        detection_id: str | None = None,
        track_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def has_ingested_frame(
        self,
        *,
        node_id: str,
        frame_sequence: int | None,
        start_time_ns: int,
        source_type: str,
    ) -> bool:
        ...

    async def register_ingested_frame(
        self,
        *,
        node_id: str,
        frame_sequence: int | None,
        start_time_ns: int,
        toa_ns: int,
        tor_ns: int,
        source_type: str,
    ) -> bool:
        ...

    def begin_batch(self) -> AsyncContextManager[None]:
        ...


@runtime_checkable
class EnvironmentProvider(Protocol):
    def get_speed_of_sound(self, location_m: tuple[float, float, float] | None = None) -> float:
        ...

    def get_conditions(self, location_m: tuple[float, float, float] | None = None) -> EnvironmentReading:
        ...


@runtime_checkable
class IngestTransport(Protocol):
    async def deliver_frame(self, payload: IngestFrameRequest) -> IngestFrameResponse:
        ...

    async def deliver_store_forward(self, payload: StoreForwardIngestRequest) -> StoreForwardIngestResponse:
        ...


@runtime_checkable
class AudioPreprocessor(Protocol):
    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        ...


@runtime_checkable
class Beamformer(Protocol):
    def beamform(
        self,
        sensor_positions: dict[str, Vec3Array],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


@runtime_checkable
class TaxonomyProvider(Protocol):
    def category_for_label(self, label: str) -> str:
        ...

    def iff_for_category(self, category: str) -> str:
        ...


@runtime_checkable
class RuleActionHandler(Protocol):
    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class RuleEngine(Protocol):
    async def evaluate(
        self,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
    ) -> list[RuleEvaluationResult]:
        ...
