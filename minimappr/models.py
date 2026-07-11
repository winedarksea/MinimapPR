"""Shared API and pipeline models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


Vec3 = tuple[float, float, float]
LabelId: TypeAlias = str
SpatialDisplayMode: TypeAlias = Literal["node_only", "bearing_only", "localized"]


WEAK_RANGE_OBSERVABILITY_THRESHOLD = 0.10
UNOBSERVABLE_RANGE_DISPLAY_MODES = {"range_asymptotic", "range_boundary"}


def spatial_display_mode_for_detection(
    *,
    reporting_modality: str,
    feature_summary: dict[str, Any] | None,
) -> SpatialDisplayMode:
    if reporting_modality == "omni":
        return "node_only"
    features = feature_summary or {}
    range_mode = features.get("localization_range_projection_mode")
    if isinstance(range_mode, str) and range_mode in UNOBSERVABLE_RANGE_DISPLAY_MODES:
        return "bearing_only"
    # bearing_projected has a valid projected position along the bearing ray — render
    # as localized so the covariance ellipsoid (elongated radially) appears on the COP.
    if range_mode == "range_bearing_projected":
        return "localized"
    range_observability = features.get("localization_range_observability")
    if isinstance(range_observability, int | float) and range_observability <= WEAK_RANGE_OBSERVABILITY_THRESHOLD:
        return "bearing_only"
    return "localized"


def _validate_overlay_bounds(bounds: list[list[float]]) -> None:
    if not bounds:
        return
    if len(bounds) != 4:
        raise ValueError("overlay bounds must contain exactly four [lat, lon] corners")
    for corner in bounds:
        if len(corner) != 2:
            raise ValueError("overlay bound corners must be [lat, lon]")
        lat, lon = float(corner[0]), float(corner[1])
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ValueError("overlay bounds contain invalid latitude or longitude")


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


class NodeCapability(str, Enum):
    # Extending the node capability vocabulary means adding one enum member here.
    AUDIO = "audio"
    ARRAY_LOCALIZATION = "array_localization"
    ENVIRONMENT = "environment"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    BLE_RSSI = "ble_rssi"
    SEISMIC = "seismic"
    RADAR_24G = "radar_24g"
    SDR = "sdr"
    SPEECH = "speech"
    PTZ_CAMERA = "ptz_camera"
    RELAY = "relay"
    GPS = "gps"
    GPS_OPTIONAL = "gps_optional"


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
    # Earth surface through the Kármán line. Acoustic detections cannot be
    # physically meaningful outside this envelope, and bounding it keeps corrupt
    # local-to-geographic conversions from entering persistence or API payloads.
    alt_m: float = Field(default=0.0, ge=-12_000.0, le=100_000.0)


class NodeOrientation(BaseModel):
    """Node-local sensor frame orientation in the site east/north/up frame."""

    yaw_deg: float = Field(ge=-360.0, le=360.0, default=0.0)
    pitch_deg: float = Field(ge=-90.0, le=90.0, default=0.0)
    roll_deg: float = Field(ge=-180.0, le=180.0, default=0.0)


class NodeSafetyConfig(BaseModel):
    require_arm_for_action: bool = False
    min_action_interval_seconds: float | None = Field(default=None, ge=0.0)
    require_arm_for_slew: bool | None = None
    min_slew_interval_seconds: float | None = Field(default=None, ge=0.0)
    no_go_zone_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_effector_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "require_arm_for_action" not in migrated and "require_arm_for_slew" in migrated:
            migrated["require_arm_for_action"] = migrated.get("require_arm_for_slew")
        if "min_action_interval_seconds" not in migrated and "min_slew_interval_seconds" in migrated:
            migrated["min_action_interval_seconds"] = migrated.get("min_slew_interval_seconds")
        return migrated

    @model_validator(mode="after")
    def _normalize_zone_ids(self) -> "NodeSafetyConfig":
        if self.require_arm_for_slew is not None:
            self.require_arm_for_action = self.require_arm_for_slew
        else:
            self.require_arm_for_slew = self.require_arm_for_action
        if self.min_slew_interval_seconds is not None:
            self.min_action_interval_seconds = self.min_slew_interval_seconds
        else:
            self.min_slew_interval_seconds = self.min_action_interval_seconds
        self.no_go_zone_ids = sorted(
            {
                str(zone_id).strip()
                for zone_id in self.no_go_zone_ids
                if str(zone_id).strip()
            }
        )
        return self


class NodeOverrides(BaseModel):
    position_m: Vec3 | None = None
    position_geo: GeoPoint | None = None
    orientation: NodeOrientation | None = None
    mobility: Literal["stationary", "mobile"] | None = None
    # Absent means the server derives the filter from mobility: KDE for fixed
    # installations and Kalman for mobile nodes.
    position_filter: Literal["raw", "kalman", "kde"] | None = None
    capabilities: list[NodeCapability] | None = None
    updated_ns: int | None = Field(default=None, ge=0)


class NodeSpec(BaseModel):
    id: str = Field(min_length=1)
    node_type: NodeType
    position_m: Vec3 | None = None
    position_geo: GeoPoint | None = None
    sensor_offsets_m: list[Vec3] = Field(default_factory=lambda: [(0.0, 0.0, 0.0)])
    orientation: NodeOrientation = Field(default_factory=NodeOrientation)
    capabilities: list[NodeCapability] = Field(default_factory=list)
    mobility: Literal["stationary", "mobile"] = "stationary"
    metadata: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    cluster_id: str | None = None
    transport: dict[str, Any] = Field(default_factory=dict)
    capability_config: dict[str, Any] = Field(default_factory=dict)
    safety: NodeSafetyConfig = Field(default_factory=NodeSafetyConfig)
    permissions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_capabilities = value.get("capabilities")
        if not isinstance(raw_capabilities, list):
            return value
        recognized_values = {capability.value for capability in NodeCapability}
        recognized: list[str] = []
        unrecognized: list[str] = []
        for capability in raw_capabilities:
            capability_value = capability.value if isinstance(capability, NodeCapability) else str(capability)
            if capability_value in recognized_values:
                recognized.append(capability_value)
            else:
                unrecognized.append(capability_value)
        if not unrecognized:
            return value
        normalized = dict(value)
        metadata = dict(normalized.get("metadata") or {})
        existing = metadata.get("unrecognized_capabilities")
        existing_list = existing if isinstance(existing, list) else []
        metadata["unrecognized_capabilities"] = sorted({*map(str, existing_list), *unrecognized})
        normalized["metadata"] = metadata
        normalized["capabilities"] = recognized
        return normalized

    @model_validator(mode="after")
    def _validate(self) -> "NodeSpec":
        if not self.sensor_offsets_m:
            raise ValueError("sensor_offsets_m cannot be empty")
        if self.position_m is None and self.position_geo is None:
            raise ValueError("Either position_m or position_geo must be provided")
        return self


class NodeRegistrationRequest(BaseModel):
    id: str = Field(min_length=1)
    node_type: NodeType = NodeType.POINT
    position_m: Vec3 | None = None
    position_geo: GeoPoint | None = None
    sensor_offsets_m: list[Vec3] = Field(default_factory=lambda: [(0.0, 0.0, 0.0)])
    orientation: NodeOrientation = Field(default_factory=NodeOrientation)
    capabilities: list[NodeCapability] = Field(default_factory=list)
    mobility: Literal["stationary", "mobile"] = "stationary"
    metadata: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    transport: dict[str, Any] = Field(default_factory=dict)
    capability_config: dict[str, Any] = Field(default_factory=dict)
    safety: NodeSafetyConfig = Field(default_factory=NodeSafetyConfig)
    permissions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "NodeRegistrationRequest":
        if not self.sensor_offsets_m:
            raise ValueError("sensor_offsets_m cannot be empty")
        if self.position_m is None and self.position_geo is None:
            raise ValueError("Either position_m or position_geo must be provided")
        return self

    def to_node_spec(self) -> NodeSpec:
        return NodeSpec(
            id=self.id,
            node_type=self.node_type,
            position_m=self.position_m,
            position_geo=self.position_geo,
            sensor_offsets_m=self.sensor_offsets_m,
            orientation=self.orientation,
            capabilities=self.capabilities,
            mobility=self.mobility,
            metadata=self.metadata,
            properties=self.properties,
            transport=self.transport,
            capability_config=self.capability_config,
            safety=self.safety,
            permissions=self.permissions,
        )


class NodePatchRequest(BaseModel):
    overrides: NodeOverrides | None = None
    clear_overrides: list[str] = Field(default_factory=list)
    transport: dict[str, Any] | None = None
    capability_config: dict[str, Any] | None = None
    safety: NodeSafetyConfig | None = None
    permissions: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


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


class BleObservationIn(BaseModel):
    mac: str = Field(min_length=1)
    addr_type: int = Field(ge=0, le=3)
    rssi_last: int = Field(ge=-127, le=20)
    rssi_ewma: float = Field(ge=-127.0, le=20.0)
    rssi_min: int = Field(ge=-127, le=20)
    rssi_max: int = Field(ge=-127, le=20)
    count: int = Field(ge=1)
    age_ms: int = Field(default=0, ge=0)
    adv_type: int | None = Field(default=None, ge=0, le=255)
    name: str | None = Field(default=None, max_length=64)


class BleIngestRequest(BaseModel):
    node_id: str = Field(min_length=1)
    boot_count: int | None = Field(default=None, ge=0)
    boot_id: int | None = Field(default=None, ge=0)
    monotonic_ms: int | None = Field(default=None, ge=0)
    observations: list[BleObservationIn] = Field(default_factory=list)


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
    position_covariance_m2: list[list[float]] | None = None
    range_observability: float | None = Field(default=None, ge=0.0, le=1.0)
    residual_rms_seconds: float | None = Field(default=None, ge=0.0)
    range_projection_mode: str | None = None
    attempted_algorithm: str | None = None
    resolved_algorithm: str | None = None
    wavelength_factor: float | None = Field(default=None, ge=0.0, le=1.0)
    dominant_frequency_hz: float | None = Field(default=None, ge=0.0)
    alias_cutoff_hz: float | None = Field(default=None, ge=0.0)
    condition_number: float | None = Field(default=None, ge=0.0)


class ContributorSummary(BaseModel):
    node_id: str
    roles: list[str] = Field(default_factory=list)
    sensor_ids: list[str] = Field(default_factory=list)
    contribution_count: int = Field(default=0, ge=0)
    last_contributed_ns: int | None = None


class DetectionReviewState(str, Enum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class TrainingExampleKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


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
    spatial_display_mode: SpatialDisplayMode | None = None
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
    review_state: DetectionReviewState = DetectionReviewState.UNREVIEWED
    review_label_id: LabelId | None = None
    review_label: str | None = None
    review_label_category: str | None = None
    review_notes: str | None = None
    review_updated_ns: int | None = None
    promote_to_training: bool = False
    training_example_kind: TrainingExampleKind | None = None
    retention_tier: RetentionTier = RetentionTier.SHORT
    snippet_path: str | None = None
    contributors: list[ContributorSummary] = Field(default_factory=list)

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
        if self.review_label is not None and self.review_label_category is None:
            self.review_label_category = self.label_category
        if self.spatial_display_mode is None:
            self.spatial_display_mode = spatial_display_mode_for_detection(
                reporting_modality=self.reporting_modality,
                feature_summary=self.feature_summary,
            )
        return self


class DetectionReviewUpdateRequest(BaseModel):
    review_state: DetectionReviewState | None = None
    review_label: str | None = Field(default=None, min_length=1)
    review_label_category: str | None = Field(default=None, min_length=1)
    review_notes: str | None = Field(default=None, max_length=4000)
    promote_to_training: bool | None = None
    training_example_kind: TrainingExampleKind | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "DetectionReviewUpdateRequest":
        if self.review_label is not None:
            self.review_label = self.review_label.strip() or None
        if self.review_label_category is not None:
            self.review_label_category = self.review_label_category.strip() or None
        if self.review_notes is not None:
            stripped_notes = self.review_notes.strip()
            self.review_notes = stripped_notes or None
        if self.review_label_category is not None and self.review_label is None:
            raise ValueError("review_label_category requires review_label")
        if self.training_example_kind is not None and self.promote_to_training is False:
            raise ValueError("training_example_kind requires promote_to_training")
        return self


class ReviewedDetectionExportItem(BaseModel):
    detection_id: str
    event_id: str
    timestamp_ns: int
    observed_at_iso: str
    track_id: str | None = None
    source_node_id: str | None = None
    reporting_modality: Literal["localized", "omni"]
    position_geo: GeoPoint | None = None
    original_label: str
    original_label_category: str = "unknown"
    reviewed_label: str | None = None
    reviewed_label_category: str | None = None
    effective_label: str
    effective_label_category: str
    review_state: DetectionReviewState
    review_notes: str | None = None
    promote_to_training: bool = False
    audio_url: str | None = None
    has_audio: bool = False


class ReviewedDetectionExportPackage(BaseModel):
    export_type: Literal["ebird_review_package"] = "ebird_review_package"
    generated_at_ns: int
    generated_at_iso: str
    detection_count: int = Field(ge=0)
    detections: list[ReviewedDetectionExportItem] = Field(default_factory=list)


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
    # Provenance of the sensing modality that produced this track. "acoustic" is
    # the default so existing tracks and serialization are unchanged; "ble"
    # marks tracks produced by the BLE RSSI multilateration loop.
    track_kind: Literal["acoustic", "ble"] = "acoustic"
    contributor_count: int = Field(default=0, ge=0)
    contributors: list[ContributorSummary] = Field(default_factory=list)
    # Phase 3: distinct source node IDs that have contributed detections to this
    # track (bounded, insertion-ordered). A track corroborated by ≥2 nodes is a
    # cross-node fusion and carries higher confidence than a single-array cone.
    contributor_node_ids: list[str] = Field(default_factory=list)


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


class MapOverlayKind(str, Enum):
    IMAGE = "image"
    SVG = "svg"
    GEOJSON = "geojson"


class MapOverlaySpec(BaseModel):
    id: str
    name: str
    kind: MapOverlayKind
    content_url: str
    mime: str
    bounds: list[list[float]] = Field(default_factory=list)
    opacity: float = Field(default=0.75, ge=0.0, le=1.0)
    storey: str | None = None
    enabled: bool = True
    created_ns: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class MapOverlayUpdate(BaseModel):
    name: str | None = None
    bounds: list[list[float]] | None = None
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    storey: str | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "MapOverlayUpdate":
        if self.bounds is not None:
            _validate_overlay_bounds(self.bounds)
        return self


class RuleActionModel(BaseModel):
    type: str = "alert"
    destination: str = "cop"
    priority: str = "normal"
    payload: dict[str, Any] = Field(default_factory=dict)


class RuleConditionModel(BaseModel):
    label_categories: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    reporting_modalities: list[str] = Field(default_factory=list)
    zone_ids: list[str] = Field(default_factory=list)
    track_statuses: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RuleModel(BaseModel):
    id: str = Field(min_length=1)
    enabled: bool = True
    scope: Literal["detection", "track"] = "detection"
    when: RuleConditionModel = Field(default_factory=RuleConditionModel)
    actions: list[RuleActionModel] = Field(default_factory=lambda: [RuleActionModel()])
    cooldown_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_actions(self) -> "RuleModel":
        if not self.actions:
            raise ValueError("rules must define at least one action")
        return self


class RulesConfigUpdate(BaseModel):
    rules: list[RuleModel] = Field(default_factory=list)


class RulesConfigResponse(BaseModel):
    rules: list[RuleModel]
    path: str
    source: Literal["file", "default"]


class ContextSnapshot(BaseModel):
    generated_ns: int
    active_tracks: list[dict[str, Any]] = Field(default_factory=list)
    zone_occupancy: list[ZoneOccupancyState] = Field(default_factory=list)
    recent_alerts: list[dict[str, Any]] = Field(default_factory=list)
    node_health: dict[str, int] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    system_health: str = "ok"


class PtzStatus(BaseModel):
    node_id: str
    state: Literal["idle", "slewing", "streaming", "error", "offline"] = "offline"
    pan_deg: float | None = None
    tilt_deg: float | None = None
    zoom: float | None = None
    armed: bool = False
    last_seen_ns: int | None = None
    active_track_id: str | None = None


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
    """Per-node audio DSP overrides stored in Settings.node_audio_overrides.

    The legacy flat fields (`mic_gains_db`, `hp_hz`, `lp_hz`, `smoothing`)
    coexist with the new `stages` field, which carries an ordered chain of
    [`PreprocessStage`-shaped][1] dicts. When `stages` is present, the legacy
    fields are ignored — this mirrors the Rust sidecar's `NodeAudioConfig`
    behavior so the JSON schema is identical across both languages.

    [1]: minimappr-ingest-sidecar/src/dsp_worker.rs::PreprocessStage
    """

    mic_gains_db: list[float] | None = None
    hp_hz: float | None = None
    lp_hz: float | None = None
    smoothing: Literal["off", "ema_50ms", "ema_200ms"] | None = None
    stages: list[dict] | None = None


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
    audio_override: NodeAudioOverride | None = None
    frame_gaps: int = 0
    zero_padded_degraded: int = 0
    last_frame_ns: int | None = None
    sample_rate_hz: int | None = None
    audio_status: str = "unknown"


class PipelineNodesResponse(BaseModel):
    active_pipeline: Literal["python", "rust"]
    nodes: list[PipelineNodeView] = Field(default_factory=list)
    pipeline_seconds_behind_realtime: float | None = None


# ── Calibration ground truth (training-data bundles) ─────────────────────────

class CalibrationGroundTruthIn(BaseModel):
    """Operator-entered ground-truth event for a calibration capture session.

    v1 supports static sources only; the bundle schema reserves a trajectory
    geometry kind for future GPX/CSV import.
    """

    label: str
    label_category: str = "unknown"
    lat: float
    lon: float
    alt_m: float = 0.0
    start_ns: int
    end_ns: int
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> "CalibrationGroundTruthIn":
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must be >= start_ns")
        return self


class CalibrationGroundTruthUpdate(BaseModel):
    label: str | None = None
    label_category: str | None = None
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    notes: str | None = None


class CalibrationGroundTruthEvent(BaseModel):
    event_id: str
    session_id: str
    label: str
    label_category: str = "unknown"
    geometry_kind: str = "static"
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    start_ns: int
    end_ns: int
    notes: str | None = None
    created_ns: int = 0
    updated_ns: int = 0
