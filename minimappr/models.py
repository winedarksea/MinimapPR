"""Shared API and pipeline models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


Vec3 = tuple[float, float, float]
LabelId: TypeAlias = str


class TimeQuality(str, Enum):
    GPS_LOCKED = "gps_locked"
    GPS_HOLDOVER = "gps_holdover"
    NTP_DISCIPLINED = "ntp_disciplined"
    FREE_RUNNING = "free_running"
    NTP_SYNC = "ntp_sync"  # Legacy database values; converted to NTP_DISCIPLINED on load


class RetentionTier(str, Enum):
    EPHEMERAL = "ephemeral"
    SHORT = "short"
    LONG = "long"
    CONFIG = "config"
    EXPERIMENT = "experiment"
    PERMANENT = "permanent"


class NodeType(str, Enum):
    POINT = "point"
    SIRITH_TETRA = "sirith_tetra"
    ARRAY = "array"
    GATEWAY = "gateway"


class SyncGrade(str, Enum):
    """Per-sensor time-synchronisation quality, ordered high → low."""
    GPS_PPS = "gps_pps"
    PTP     = "ptp"
    NTP     = "ntp"
    FREE    = "free"

    def weight(self) -> float:
        """Localization residual weight for this sync grade."""
        return _SYNC_GRADE_WEIGHTS[self]


_SYNC_GRADE_WEIGHTS: dict[str, float] = {
    SyncGrade.GPS_PPS: 1.0,
    SyncGrade.PTP:     1.0,
    SyncGrade.NTP:     0.25,
    SyncGrade.FREE:    0.05,
}


def sync_grade_from_time_quality(tq: "TimeQuality") -> SyncGrade:
    """Map a runtime TimeQuality to the equivalent SyncGrade."""
    if tq in (TimeQuality.GPS_LOCKED, TimeQuality.GPS_HOLDOVER):
        return SyncGrade.GPS_PPS
    if tq in (TimeQuality.NTP_DISCIPLINED, TimeQuality.NTP_SYNC):
        return SyncGrade.NTP
    return SyncGrade.FREE


class IamfRenderMode(str, Enum):
    """IAMF output layout for a node cluster."""
    FOA_BED        = "foa_bed"
    N_MONO_OBJECTS = "n_mono_objects"
    AUTO           = "auto"


class ClusterSpec(BaseModel):
    """Logical grouping of nodes that form a TDOA array."""
    id: str = Field(min_length=1)
    member_node_ids: list[str] = Field(min_length=1)
    declared_sync_grade: SyncGrade
    iamf_render_mode: IamfRenderMode = IamfRenderMode.AUTO
    max_baseline_m_for_foa: float = Field(default=0.50, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    alt_m: float = 0.0


class NodeSpec(BaseModel):
    id: str = Field(min_length=1)
    node_type: NodeType
    position_m: Vec3 | None = None
    position_geo: GeoPoint | None = None
    sensor_offsets_m: list[Vec3] = Field(default_factory=lambda: [(0.0, 0.0, 0.0)])
    capabilities: list[str] = Field(default_factory=list)
    mobility: Literal["stationary", "mobile"] = "stationary"
    metadata: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    cluster_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "NodeSpec":
        if not self.sensor_offsets_m:
            raise ValueError("sensor_offsets_m cannot be empty")
        if self.position_m is None and self.position_geo is None:
            raise ValueError("Either position_m or position_geo must be provided")
        return self


class AudioFrameIn(BaseModel):
    start_time_ns: int = Field(gt=0)
    utc_start_ns: int | None = Field(default=None, gt=0)
    utc_end_ns: int | None = Field(default=None, gt=0)
    start_sample_index: int | None = Field(default=None, ge=0)
    end_sample_index: int | None = Field(default=None, ge=0)
    sample_rate_hz: int = Field(ge=8000, le=192000)
    channels: int = Field(ge=1, le=32)
    encoding: Literal["pcm16le"] = "pcm16le"
    samples_per_channel: int | None = Field(default=None, gt=0)
    samples_b64: str = Field(min_length=1)
    sequence: int | None = None
    time_quality: TimeQuality = TimeQuality.FREE_RUNNING
    toa_ns: int | None = None
    tor_ns: int | None = None
    timing_diagnostics: dict[str, Any] = Field(default_factory=dict)
    source_type: Literal["raw_sensor", "local_track", "peer_track"] = "raw_sensor"

    @model_validator(mode="after")
    def _defaults(self) -> "AudioFrameIn":
        if self.utc_start_ns is None:
            self.utc_start_ns = self.start_time_ns
        self.start_time_ns = self.utc_start_ns
        if self.toa_ns is None:
            self.toa_ns = self.start_time_ns
        if self.samples_per_channel is not None:
            if self.end_sample_index is None and self.start_sample_index is not None:
                self.end_sample_index = self.start_sample_index + self.samples_per_channel
            if self.utc_end_ns is None:
                duration_ns = int(round((self.samples_per_channel / self.sample_rate_hz) * 1_000_000_000))
                self.utc_end_ns = self.start_time_ns + duration_ns
        if self.end_sample_index is not None and self.start_sample_index is not None:
            if self.end_sample_index < self.start_sample_index:
                raise ValueError("end_sample_index must be >= start_sample_index")
            expected_samples = self.end_sample_index - self.start_sample_index
            if self.samples_per_channel is None:
                self.samples_per_channel = expected_samples
            elif self.samples_per_channel != expected_samples:
                raise ValueError("samples_per_channel must match sample index coverage")
        if self.utc_end_ns is not None and self.utc_end_ns < self.start_time_ns:
            raise ValueError("utc_end_ns must be >= start_time_ns")
        return self


class EnvironmentSampleIn(BaseModel):
    timestamp_ns: int | None = Field(default=None, gt=0)
    temperature_c: float | None = None
    humidity_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    humidity_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    pressure_pa: float | None = Field(default=None, gt=0.0)
    wind_speed_mps: float | None = Field(default=None, ge=0.0)
    wind_dir_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    solar_lux: float | None = Field(default=None, ge=0.0)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "EnvironmentSampleIn":
        if self.humidity_fraction is None and self.humidity_percent is not None:
            self.humidity_fraction = self.humidity_percent / 100.0
        return self

    def has_any_measurement(self) -> bool:
        return any(
            value is not None
            for value in (
                self.temperature_c,
                self.humidity_fraction,
                self.pressure_pa,
                self.wind_speed_mps,
                self.wind_dir_deg,
                self.solar_lux,
            )
        )


class IngestFrameRequest(BaseModel):
    node: NodeSpec
    frame: AudioFrameIn
    environment: EnvironmentSampleIn | None = None

    @model_validator(mode="after")
    def _validate(self) -> "IngestFrameRequest":
        if self.frame.channels != len(self.node.sensor_offsets_m):
            raise ValueError("frame.channels must equal len(node.sensor_offsets_m)")
        return self


class IngestFrameResponse(BaseModel):
    accepted: bool
    duplicate: bool = False
    triggered: bool
    frame_energy: float
    detection_id: str | None = None
    queued_event_id: str | None = None
    queue_depth: int | None = None


class StoreForwardBufferedFrameRequest(BaseModel):
    frame: AudioFrameIn
    environment: EnvironmentSampleIn | None = None


class StoreForwardIngestRequest(BaseModel):
    node: NodeSpec
    buffered_frames: list[StoreForwardBufferedFrameRequest] = Field(min_length=1, max_length=2048)
    sort_by_toa: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "StoreForwardIngestRequest":
        for item in self.buffered_frames:
            if item.frame.channels != len(self.node.sensor_offsets_m):
                raise ValueError("buffered frame.channels must equal len(node.sensor_offsets_m)")
        return self


class StoreForwardBufferedFrameResponse(BaseModel):
    sequence: int | None = None
    start_time_ns: int
    accepted: bool
    duplicate: bool = False
    triggered: bool = False
    frame_energy: float = 0.0
    queued_event_id: str | None = None
    detail: str | None = None


class StoreForwardIngestResponse(BaseModel):
    accepted: bool
    total_frames: int
    accepted_frames: int
    duplicate_frames: int
    rejected_frames: int
    queued_events: int
    results: list[StoreForwardBufferedFrameResponse] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)


class LocalizationResult(BaseModel):
    position_m: Vec3
    confidence: float = Field(ge=0.0, le=1.0)
    gdop: float = Field(ge=0.0)
    reference_sensor: str
    tdoa_s: dict[str, float] = Field(default_factory=dict)
    attempted_algorithm: str | None = None
    resolved_algorithm: str | None = None
    wavelength_factor: float | None = Field(default=None, ge=0.0, le=1.0)
    dominant_frequency_hz: float | None = Field(default=None, ge=0.0)
    alias_cutoff_hz: float | None = Field(default=None, ge=0.0)


class DetectionEvent(BaseModel):
    id: str
    event_id: str | None = None
    source_type: Literal["raw_sensor", "local_track", "peer_track"] = "raw_sensor"
    source_node_id: str | None = None
    timestamp_ns: int
    toa_ns: int | None = None
    tor_ns: int | None = None
    time_quality: TimeQuality = TimeQuality.FREE_RUNNING
    stale_ns: int | None = None
    report_window_start_ns: int | None = None
    report_window_end_ns: int | None = None
    reporting_modality: Literal["localized", "omni"] = "localized"
    position_m: Vec3
    position_geo: GeoPoint | None = None
    position_covariance_m2: list[list[float]] | None = None
    confidence: float
    gdop: float
    label_id: LabelId | None = None
    label: str
    label_category: str = "unknown"
    iff_category: Literal["friendly", "unknown", "hostile"] = "unknown"
    label_confidence: float
    spl_db: float | None = None
    track_id: str | None = None
    source_sensors: list[str] = Field(default_factory=list)
    source_observation_ids: list[str] = Field(default_factory=list)
    zone_ids: list[str] = Field(default_factory=list)
    reference_sensor: str
    tdoa_s: dict[str, float] = Field(default_factory=dict)
    classifier_scores: dict[str, float] = Field(default_factory=dict)
    feature_summary: dict[str, Any] = Field(default_factory=dict)
    retention_tier: RetentionTier = RetentionTier.SHORT
    snippet_path: str | None = None

    @model_validator(mode="after")
    def _defaults(self) -> "DetectionEvent":
        if self.event_id is None:
            self.event_id = self.id
        if self.toa_ns is None:
            self.toa_ns = self.timestamp_ns
        if self.tor_ns is None:
            self.tor_ns = self.timestamp_ns
        # Normalize legacy "ntp_sync" to canonical "ntp_disciplined"
        if self.time_quality == TimeQuality.NTP_SYNC:
            self.time_quality = TimeQuality.NTP_DISCIPLINED
        return self


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
    position_geo: GeoPoint | None = None
    position_covariance_m2: list[list[float]] | None = None
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    label_id: LabelId | None = None
    label: str = "unknown"
    label_category: str = "unknown"
    iff_category: Literal["friendly", "unknown", "hostile"] = "unknown"
    confidence: float = 0.0
    update_count: int = 0
    status: str = TrackStatus.TENTATIVE.value
    tqi: float = Field(default=0.0, ge=0.0, description="Track Quality Index")
    capability_tier: Literal["full_3d", "2d", "classification_only", "alerting_only"] = "full_3d"


class FederationHeartbeat(BaseModel):
    server_id: str = Field(min_length=1)
    sent_ns: int = Field(gt=0)


class FederationTrackSnapshot(BaseModel):
    server_id: str = Field(min_length=1)
    generated_ns: int = Field(gt=0)
    source_type: Literal["local_track", "peer_track"] = "local_track"
    tracks: list[TrackState] = Field(default_factory=list)


class FederationAck(BaseModel):
    ok: bool = True
    server_id: str = Field(min_length=1)
    received_ns: int = Field(gt=0)
    peer_state: str = "active"


class FusionStatusResponse(BaseModel):
    started: bool
    queue: dict[str, int]
    workers: dict[str, int]
    last_trigger_ns: int
    last_error: str | None = None
    registered_nodes: int
    metrics: dict[str, Any]
    realtime: dict[str, Any]
    offline_replay_mode: bool
    drop_on_backpressure: bool


class FederationStatusResponse(BaseModel):
    enabled: bool
    server_id: str
    peer_count: int
    peer_track_count: int
    peers: list[dict[str, Any]]
    metrics: dict[str, int]
    publish_interval_seconds: float
    heartbeat_interval_seconds: float
    link_timeout_seconds: float
    track_ttl_seconds: float


class CopStatusResponse(BaseModel):
    active_nodes: int
    degraded_nodes: int
    offline_nodes: int
    active_tracks: int
    recent_alert_count: int


class ZoneType(str, Enum):
    ALERT = "alert_zone"
    EXCLUSION = "exclusion_zone"
    COVERAGE = "coverage_zone"
    INTEREST = "interest_zone"


class ZoneSpec(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    zone_type: ZoneType
    polygon_geo: list[list[float]] = Field(min_length=3)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_polygon(self) -> "ZoneSpec":
        for point in self.polygon_geo:
            if not isinstance(point, list) or len(point) < 2:
                raise ValueError("polygon_geo points must be [lat, lon]")
            lat = float(point[0])
            lon = float(point[1])
            if lat < -90.0 or lat > 90.0:
                raise ValueError("zone latitude must be in [-90, 90]")
            if lon < -180.0 or lon > 180.0:
                raise ValueError("zone longitude must be in [-180, 180]")
        return self


class ZoneOccupancyState(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    occupied: bool
    occupying_track_ids: list[str] = Field(default_factory=list)
    occupying_labels: list[str] = Field(default_factory=list)
    updated_ns: int


class ContextSnapshot(BaseModel):
    generated_ns: int
    active_tracks: list[dict[str, Any]] = Field(default_factory=list)
    zone_occupancy: list[ZoneOccupancyState] = Field(default_factory=list)
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list)
    node_health: dict[str, int] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    system_health: str = "ok"


class AlertStatus(str, Enum):
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Built-In Test (BIT) Reporting
# ---------------------------------------------------------------------------


class BITType(str, Enum):
    """Category of Built-In Test."""

    PBIT = "pbit"   # Power-On BIT — executed at node startup
    CBIT = "cbit"   # Continuous BIT — periodic self-test during operation
    IBIT = "ibit"   # Initiated BIT — on-demand test triggered by operator


class BITStatus(str, Enum):
    """Aggregate outcome of a BIT report or individual test."""

    PASS = "pass"
    FAIL = "fail"
    DEGRADED = "degraded"


class BITTestResult(BaseModel):
    """Single sub-test within a BIT report.

    Failure codes follow the convention ``<BIT_TYPE>_FAIL: <SUBSYSTEM>_<SYMPTOM>``
    e.g. ``CBIT_FAIL: MIC_CH3_CLIP``, ``PBIT_FAIL: LORA_CORE_TIMEOUT``.
    """

    test_name: str = Field(min_length=1, description="Human-readable test identifier")
    status: BITStatus
    failure_code: str | None = Field(
        default=None,
        description="Structured failure code for fault isolation, e.g. CBIT_FAIL: MIC_CH3_CLIP",
    )
    detail: str | None = Field(default=None, description="Free-form diagnostic detail")
    measured_value: float | None = Field(default=None, description="Observed metric value")
    threshold: float | None = Field(default=None, description="Pass/fail threshold for measured_value")
    subsystem: str | None = Field(default=None, description="Subsystem under test, e.g. audio, lora, gps")


class BITReportIn(BaseModel):
    """Inbound BIT report submitted by a sensor node."""

    report_type: BITType
    timestamp_ns: int | None = Field(default=None, gt=0)
    results: list[BITTestResult] = Field(min_length=1)
    firmware_version: str | None = None
    uptime_seconds: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BITReport(BaseModel):
    """Stored / returned BIT report with server-assigned fields."""

    id: str
    node_id: str
    report_type: BITType
    overall_status: BITStatus
    timestamp_ns: int
    received_ns: int
    results: list[BITTestResult]
    failure_codes: list[str] = Field(
        default_factory=list,
        description="Convenience list of all non-null failure_code values from results",
    )
    firmware_version: str | None = None
    uptime_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeHealthStatus(str, Enum):
    """Unified node health incorporating heartbeat staleness and BIT status."""

    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    BIT_FAIL = "bit_fail"  # BIT failure overrides heartbeat-based status


# ---------------------------------------------------------------------------
# Pipeline / Nodes view — per-node audio pipeline state and settings
# ---------------------------------------------------------------------------


class NodeAudioOverride(BaseModel):
    """Per-node audio DSP overrides stored in Settings.node_audio_overrides."""

    mic_gains_db: list[float] | None = None
    hp_hz: float | None = None
    lp_hz: float | None = None
    smoothing: Literal["off", "ema_50ms", "ema_200ms"] | None = None


class PipelineStageView(BaseModel):
    name: str
    count_in: int = 0
    count_out: int = 0
    drops: int = 0
    queue_depth: int = 0
    queue_max: int = 0
    lag_s: float | None = None


class MicView(BaseModel):
    index: int
    label: str
    gain_db: float = 0.0
    hp_hz: float = 0.0
    lp_hz: float = 0.0
    smoothing: str = "off"
    rms_recent: list[float] = Field(default_factory=list)


class PipelineNodeView(BaseModel):
    node_id: str
    node_type: str
    mics: list[MicView] = Field(default_factory=list)
    stages: list[PipelineStageView] = Field(default_factory=list)
    frame_gaps: int = 0
    zero_padded_degraded: int = 0
    last_frame_ns: int | None = None
    sample_rate_hz: int | None = None
    audio_status: str = "unknown"


class PipelineNodesResponse(BaseModel):
    active_pipeline: Literal["python", "rust"]
    nodes: list[PipelineNodeView] = Field(default_factory=list)
    pipeline_seconds_behind_realtime: float | None = None
