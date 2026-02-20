"""Shared API and pipeline models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Vec3 = tuple[float, float, float]


class NodeType(str, Enum):
    POINT = "point"
    SIRITH_TETRA = "sirith_tetra"


class NodeSpec(BaseModel):
    id: str = Field(min_length=1)
    node_type: NodeType
    position_m: Vec3
    sensor_offsets_m: list[Vec3] = Field(default_factory=lambda: [(0.0, 0.0, 0.0)])
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "NodeSpec":
        if not self.sensor_offsets_m:
            raise ValueError("sensor_offsets_m cannot be empty")
        return self


class AudioFrameIn(BaseModel):
    start_time_ns: int = Field(gt=0)
    sample_rate_hz: int = Field(ge=8000, le=192000)
    channels: int = Field(ge=1, le=32)
    encoding: Literal["pcm16le"] = "pcm16le"
    samples_b64: str = Field(min_length=1)
    sequence: int | None = None


class IngestFrameRequest(BaseModel):
    node: NodeSpec
    frame: AudioFrameIn

    @model_validator(mode="after")
    def _validate(self) -> "IngestFrameRequest":
        if self.frame.channels != len(self.node.sensor_offsets_m):
            raise ValueError("frame.channels must equal len(node.sensor_offsets_m)")
        return self


class IngestFrameResponse(BaseModel):
    accepted: bool
    triggered: bool
    frame_energy: float
    detection_id: str | None = None
    queued_event_id: str | None = None
    queue_depth: int | None = None


class ClassificationResult(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)


class LocalizationResult(BaseModel):
    position_m: Vec3
    confidence: float = Field(ge=0.0, le=1.0)
    gdop: float = Field(ge=0.0)
    reference_sensor: str
    tdoa_s: dict[str, float] = Field(default_factory=dict)


class DetectionEvent(BaseModel):
    id: str
    timestamp_ns: int
    position_m: Vec3
    confidence: float
    gdop: float
    label: str
    label_confidence: float
    track_id: str | None = None
    source_sensors: list[str] = Field(default_factory=list)
    reference_sensor: str
    tdoa_s: dict[str, float] = Field(default_factory=dict)
    classifier_scores: dict[str, float] = Field(default_factory=dict)
    feature_summary: dict[str, float] = Field(default_factory=dict)
    snippet_path: str | None = None


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DROPPED = "dropped"


class TrackState(BaseModel):
    id: str
    first_seen_ns: int
    last_seen_ns: int
    position_m: Vec3
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    label: str = "unknown"
    confidence: float = 0.0
    update_count: int = 0
    status: str = TrackStatus.TENTATIVE.value
    tqi: float = Field(default=0.0, ge=0.0, description="Track Quality Index")
