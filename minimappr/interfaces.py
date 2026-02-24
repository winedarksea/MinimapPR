"""Core interface contracts for pluggable MinimapPR subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from minimappr.models import ClassificationResult, DetectionEvent, IngestFrameRequest, LocalizationResult, TrackState


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

    async def upsert_node(self, *args: Any, **kwargs: Any) -> None:
        ...

    async def insert_observation(self, *args: Any, **kwargs: Any) -> str:
        ...

    async def insert_detection(self, *args: Any, **kwargs: Any) -> None:
        ...

    async def upsert_track(self, *args: Any, **kwargs: Any) -> None:
        ...

    async def insert_track_update(self, *args: Any, **kwargs: Any) -> str:
        ...

    async def insert_alert(self, *args: Any, **kwargs: Any) -> str:
        ...

    async def update_alert_status(self, *args: Any, **kwargs: Any) -> bool:
        ...

    async def insert_ping(self, *args: Any, **kwargs: Any) -> str:
        ...

    async def list_labels(self) -> list[dict]:
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
