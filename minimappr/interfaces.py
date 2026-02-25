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
    LocalizationResult,
    NodeSpec,
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
        ...

    def update(self, state: TrackState, measurement_m: tuple[float, float, float]) -> TrackState:
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
        parent_label_id: str | None = None,
        external_taxonomy: str | None = None,
        external_label_id: str | None = None,
        created_ns: int,
    ) -> str:
        ...

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> None:
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
        label_id: str | None,
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

    async def list_labels(self) -> list[dict[str, Any]]:
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
    async def deliver_frame(self, payload: IngestFrameRequest) -> Any:
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
