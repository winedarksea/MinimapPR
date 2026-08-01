use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeStatus {
    #[serde(default, alias = "id")]
    pub node_id: String,
    #[serde(default = "default_node_health", alias = "health_status")]
    pub health: String,
    #[serde(alias = "last_seen_ns")]
    pub last_seen_ns: Option<i64>,
    pub last_seen_seconds_ago: Option<f64>,
    pub node_type: Option<String>,
    pub mobility: Option<String>,
    pub capabilities: Option<Vec<String>>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub reported_position_m: Option<Vec<f64>>,
    #[serde(default)]
    pub overrides: serde_json::Value,
    #[serde(default)]
    pub safety: serde_json::Value,
    #[serde(default)]
    pub permissions: serde_json::Value,
    pub ptz_status: Option<PtzStatusData>,
    pub gps_fix: Option<bool>,
    pub temperature_c: Option<f64>,
    #[serde(alias = "humidity_fraction")]
    pub humidity: Option<f64>,
    pub rms_history: Option<Vec<f64>>,
    pub channel_count: Option<u32>,
    pub firmware_version: Option<String>,
    pub bit_failure_codes: Option<Vec<String>>,
    pub audio_debug: Option<NodeAudioDebug>,
    pub latest_environment: Option<NodeEnvironment>,
    pub metadata: Option<NodeMetadata>,
    pub latest_time_quality: Option<String>,
    #[serde(default)]
    pub latest_timing_diagnostics: Option<serde_json::Value>,
}

impl NodeStatus {
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities
            .as_ref()
            .map(|capabilities| capabilities.iter().any(|item| item == capability))
            .unwrap_or(false)
    }
}

fn default_node_health() -> String {
    "unknown".to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeAudioDebug {
    pub sensor_count: Option<u32>,
    pub active_sensor_count: Option<u32>,
    pub sample_rate_hz: Option<f64>,
    pub last_sample_time_ns: Option<i64>,
    pub age_seconds: Option<f64>,
    pub rms: Option<f64>,
    pub status: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeEnvironment {
    pub timestamp_ns: Option<i64>,
    pub temperature_c: Option<f64>,
    pub pressure_pa: Option<f64>,
    pub humidity_fraction: Option<f64>,
    pub wind_speed_mps: Option<f64>,
    pub wind_dir_deg: Option<f64>,
    pub solar_lux: Option<f64>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeMetadata {
    pub gps: Option<NodeGpsMeta>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeGpsMeta {
    pub signal: Option<String>,
    pub position_source: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Track {
    #[serde(alias = "id")]
    pub track_id: String,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    pub tqi: Option<f64>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub position_covariance_m2: Option<Vec<Vec<f64>>>,
    pub velocity_mps: Option<Vec<f64>>,
    #[serde(alias = "last_seen_ns")]
    pub last_update_ns: Option<i64>,
    #[serde(alias = "update_count")]
    pub sensor_count: Option<u32>,
    #[serde(default)]
    pub contributor_count: u32,
    #[serde(default)]
    pub contributors: Vec<ContributorSummary>,
    pub status: Option<String>,
    /// Sensing modality provenance: "acoustic" (default) or "ble".
    #[serde(default)]
    pub track_kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ContributorSummary {
    pub node_id: String,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub sensor_ids: Vec<String>,
    pub contribution_count: u32,
    pub last_contributed_ns: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: Option<f64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Detection {
    pub event_id: String,
    pub node_id: Option<String>,
    pub reporting_modality: Option<String>,
    pub spatial_display_mode: Option<String>,
    pub localization_range_projection_mode: Option<String>,
    pub localization_range_observability: Option<f64>,
    pub localization_method: Option<String>,
    pub capability_tier: Option<String>,
    pub position_geo_uncertainty: Option<serde_json::Value>,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    pub label_confidence: Option<f64>,
    pub received_ns: Option<i64>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub position_covariance_m2: Option<Vec<Vec<f64>>>,
    pub has_audio: Option<bool>,
    pub snippet_path: Option<String>,
    pub track_id: Option<String>,
    pub contributors: Vec<ContributorSummary>,
}

#[derive(Deserialize)]
struct DetectionWire {
    event_id: Option<String>,
    id: Option<String>,
    #[serde(alias = "source_node_id")]
    node_id: Option<String>,
    reporting_modality: Option<String>,
    spatial_display_mode: Option<String>,
    feature_summary: Option<serde_json::Value>,
    label: Option<String>,
    confidence: Option<f64>,
    label_confidence: Option<f64>,
    #[serde(alias = "tor_ns")]
    received_ns: Option<i64>,
    position_m: Option<Vec<f64>>,
    position_geo: Option<GeoPoint>,
    position_covariance_m2: Option<Vec<Vec<f64>>>,
    has_audio: Option<bool>,
    snippet_path: Option<String>,
    track_id: Option<String>,
    #[serde(default)]
    contributors: Vec<ContributorSummary>,
}

impl<'de> Deserialize<'de> for Detection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DetectionWire::deserialize(deserializer)?;
        let feature_summary = wire.feature_summary.as_ref();
        Ok(Self {
            event_id: wire.event_id.or(wire.id).unwrap_or_default(),
            node_id: wire.node_id,
            reporting_modality: wire.reporting_modality,
            spatial_display_mode: wire.spatial_display_mode,
            localization_range_projection_mode: feature_summary
                .and_then(|features| features.get("localization_range_projection_mode"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
            localization_range_observability: feature_summary
                .and_then(|features| features.get("localization_range_observability"))
                .and_then(|value| value.as_f64()),
            localization_method: feature_summary
                .and_then(|features| features.get("localization_method"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
            capability_tier: feature_summary
                .and_then(|features| features.get("capability_tier"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
            position_geo_uncertainty: feature_summary
                .and_then(|features| features.get("position_geo_uncertainty"))
                .cloned(),
            label: wire.label,
            confidence: wire.confidence,
            label_confidence: wire.label_confidence,
            received_ns: wire.received_ns,
            position_m: wire.position_m,
            position_geo: wire.position_geo,
            position_covariance_m2: wire.position_covariance_m2,
            has_audio: wire.has_audio,
            snippet_path: wire.snippet_path,
            track_id: wire.track_id,
            contributors: wire.contributors,
        })
    }
}

impl Detection {
    pub fn spatial_display_mode(&self) -> &str {
        self.spatial_display_mode
            .as_deref()
            .unwrap_or(match self.reporting_modality.as_deref() {
                Some("omni") => "node_only",
                _ => "localized",
            })
    }

    pub fn uncertainty_summary(&self) -> &'static str {
        match self.spatial_display_mode() {
            "node_only" => "Node heard this; no bearing/range localization.",
            "bearing_only" => "Bearing/elevation estimate; range weak.",
            _ => "Localized estimate; ellipse shows selected covariance.",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OmniLabelSummary {
    pub label: String,
    pub count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OmniCategorySummary {
    pub category: String,
    pub count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct NodeOmniDetectionSummary {
    pub node_id: String,
    pub active_count: u32,
    pub recent_count: u32,
    pub last_detection_ns: Option<i64>,
    pub top_labels: Vec<OmniLabelSummary>,
    pub top_categories: Vec<OmniCategorySummary>,
    pub max_confidence: Option<f64>,
    pub sample_detection_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct PtzStatusData {
    #[serde(default)]
    pub state: String,
    pub pan_deg: Option<f64>,
    pub tilt_deg: Option<f64>,
    pub zoom: Option<f64>,
    #[serde(default)]
    pub armed: bool,
    pub last_seen_ns: Option<i64>,
    pub active_track_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NodeCapabilityStatusEvent {
    pub node_id: String,
    pub capability: String,
    pub status: PtzStatusData,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NodeUpdatedEvent {
    pub node_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Alert {
    /// Defaulted so a live alert from a handler that has no row id still renders
    /// rather than failing the whole event's deserialization.
    #[serde(alias = "id", default)]
    pub alert_id: String,
    #[serde(alias = "rule_id")]
    pub rule_name: Option<String>,
    #[serde(default)]
    pub detection_id: Option<String>,
    #[serde(default)]
    pub track_id: Option<String>,
    #[serde(default)]
    pub destination: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(alias = "priority")]
    pub severity: Option<String>,
    pub status: Option<String>,
    #[serde(alias = "timestamp_ns")]
    pub triggered_ns: Option<i64>,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ZoneSpec {
    pub id: String,
    pub name: String,
    pub zone_type: String,
    pub polygon_geo: Vec<Vec<f64>>,
    #[serde(default)]
    pub properties: serde_json::Value,
    pub created_ns: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ZoneOccupancyState {
    pub zone_id: String,
    pub zone_name: String,
    pub zone_type: String,
    #[serde(default)]
    pub occupied: bool,
    #[serde(default)]
    pub occupying_track_ids: Vec<String>,
    #[serde(default)]
    pub occupying_labels: Vec<String>,
    pub updated_ns: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CopStatus {
    pub active_nodes: u32,
    pub degraded_nodes: u32,
    pub offline_nodes: u32,
    pub active_tracks: u32,
    #[serde(default, alias = "recent_alert_count")]
    pub open_alerts: u32,
    #[serde(default)]
    pub detections_last_60s: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct FusionHealth {
    #[serde(default)]
    pub active_drought: Option<bool>,
    #[serde(default)]
    pub seconds_since_last_emission: Option<f64>,
    #[serde(default)]
    pub seconds_since_last_trigger: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct FusionStatus {
    #[serde(default)]
    pub realtime: FusionRealtime,
    #[serde(default)]
    pub metrics: serde_json::Value,
    #[serde(default)]
    pub health: FusionHealth,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct FusionRealtime {
    pub pipeline_seconds_behind_realtime: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ConfigSnapshot {
    pub trigger_rms: f64,
    pub trigger_cooldown_seconds: f64,
    pub localization_window_seconds: f64,
    pub preprocess_enabled: bool,
    pub audio_highpass_hz: f64,
    pub audio_lowpass_hz: f64,
    pub localization_algorithm: String,
    pub localization_strategy: String,
    #[serde(default = "default_classification_audio_source")]
    pub classification_audio_source: String,
    #[serde(default)]
    pub min_localization_confidence: f64,
    #[serde(default)]
    pub skip_localization_for_classification: bool,
    #[serde(default)]
    pub localization_band_min_hz: f64,
    #[serde(default)]
    pub localization_band_max_hz: f64,
    #[serde(default)]
    pub birdnet_chunked_dispatch_enabled: bool,
    #[serde(default)]
    pub birdnet_trigger_min_confidence: f64,
    #[serde(default)]
    pub birdnet_geo_min_confidence: f64,
    #[serde(default)]
    pub birdnet_enabled: bool,
    #[serde(default)]
    pub drone_head_enabled: bool,
    #[serde(default)]
    pub drone_head_min_confidence: f64,
    #[serde(default)]
    pub drone_head_min_frame_fraction: f64,
    #[serde(default)]
    pub stt_enabled: bool,
    #[serde(default)]
    pub stt_trigger_min_confidence: f64,
    #[serde(default)]
    pub transcript_retention_seconds: f64,
    #[serde(default)]
    pub retention_yamnet_audio_seconds: f64,
    #[serde(default)]
    pub retention_birdnet_audio_seconds: f64,
    #[serde(default)]
    pub retention_drone_audio_seconds: f64,
    #[serde(default)]
    pub retention_alert_audio_seconds: f64,
    #[serde(default)]
    pub retention_detection_metadata_seconds: f64,
    #[serde(default)]
    pub omni_scan_enabled: bool,
    #[serde(default)]
    pub omni_scan_interval_seconds: f64,
    #[serde(default)]
    pub omni_scan_window_seconds: f64,
    #[serde(default)]
    pub omni_scan_min_rms: f64,
    #[serde(default)]
    pub persisted_override_keys: Vec<String>,
    pub yamnet_min_confidence: f64,
    pub detection_min_confidence: f64,
    pub beamformer_type: String,
    pub tracking_filter: String,
    pub fusion_worker_count: u32,
    pub coordinate_mode: String,
    #[serde(default)]
    pub hass: HassConfigSnapshot,
    #[serde(default)]
    pub site_origin: Option<SiteOriginSnapshot>,
    /// Populated only on PATCH responses: sidecar-restart-required patched keys.
    #[serde(default)]
    pub restart_required: Vec<String>,
}

fn default_classification_audio_source() -> String {
    "beamformed".to_string()
}

/// Every field is `#[serde(default)]` with a default matching the backend, so an
/// older backend that does not yet send the MQTT-bridge fields still
/// deserializes rather than blanking the whole config snapshot.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct HassConfigSnapshot {
    #[serde(default)]
    pub enabled: bool,
    /// Inbound enrichment client only — unused by the outbound MQTT bridge.
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub mqtt_host: String,
    #[serde(default = "default_hass_mqtt_port")]
    pub mqtt_port: u32,
    #[serde(default)]
    pub mqtt_username: String,
    #[serde(default)]
    pub mqtt_password: String,
    #[serde(default = "default_hass_client_id")]
    pub mqtt_client_id: String,
    #[serde(default = "default_hass_keepalive")]
    pub mqtt_keepalive_seconds: u32,
    #[serde(default)]
    pub mqtt_tls_enabled: bool,
    #[serde(default)]
    pub mqtt_tls_insecure: bool,
    #[serde(default = "default_hass_discovery_prefix")]
    pub discovery_prefix: String,
    #[serde(default = "default_hass_base_topic")]
    pub base_topic: String,
    #[serde(default = "default_hass_base_topic")]
    pub device_id: String,
    #[serde(default = "default_hass_device_name")]
    pub device_name: String,
    #[serde(default = "default_hass_publish_interval")]
    pub publish_interval_seconds: f64,
    #[serde(default = "default_hass_publish_min_interval")]
    pub publish_min_interval_seconds: f64,
    #[serde(default = "default_hass_reconcile_interval")]
    pub reconcile_interval_seconds: f64,
    #[serde(default = "default_hass_queue_size")]
    pub queue_size: u32,
    #[serde(default = "default_hass_backoff_initial")]
    pub reconnect_backoff_initial_seconds: f64,
    #[serde(default = "default_hass_backoff_max")]
    pub reconnect_backoff_max_seconds: f64,
    #[serde(default = "default_hass_off_delay")]
    pub detection_off_delay_seconds: u32,
    #[serde(default)]
    pub detection_classes: Vec<String>,
    #[serde(default = "default_hass_track_slot_count")]
    pub track_slot_count: u32,
    #[serde(default = "default_hass_spl_window")]
    pub zone_spl_window_seconds: f64,
    #[serde(default = "default_true")]
    pub publish_zone_occupancy: bool,
    #[serde(default = "default_true")]
    pub publish_zone_spl: bool,
    #[serde(default = "default_true")]
    pub publish_detection_classes: bool,
    #[serde(default = "default_true")]
    pub publish_node_status: bool,
    #[serde(default = "default_true")]
    pub publish_system_health: bool,
    #[serde(default = "default_true")]
    pub publish_events: bool,
    #[serde(default)]
    pub publish_track_slots: bool,
}

fn default_hass_mqtt_port() -> u32 {
    1883
}

fn default_hass_client_id() -> String {
    "minimappr".to_string()
}

fn default_hass_keepalive() -> u32 {
    60
}

fn default_hass_discovery_prefix() -> String {
    "homeassistant".to_string()
}

fn default_hass_base_topic() -> String {
    "minimappr".to_string()
}

fn default_hass_device_name() -> String {
    "MinimapPR".to_string()
}

fn default_hass_publish_interval() -> f64 {
    5.0
}

fn default_hass_publish_min_interval() -> f64 {
    1.0
}

fn default_hass_reconcile_interval() -> f64 {
    60.0
}

fn default_hass_queue_size() -> u32 {
    2000
}

fn default_hass_backoff_initial() -> f64 {
    1.0
}

fn default_hass_backoff_max() -> f64 {
    60.0
}

fn default_hass_off_delay() -> u32 {
    30
}

fn default_hass_track_slot_count() -> u32 {
    8
}

fn default_hass_spl_window() -> f64 {
    60.0
}

fn default_true() -> bool {
    true
}

/// 1:1 with the backend `HassBridgeStatusResponse`.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct HassBridgeStatus {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_hass_connection_state")]
    pub connection_state: String,
    #[serde(default)]
    pub transport: Option<String>,
    #[serde(default)]
    pub transport_available: bool,
    #[serde(default)]
    pub mqtt_host: String,
    #[serde(default = "default_hass_mqtt_port")]
    pub mqtt_port: u32,
    #[serde(default)]
    pub mqtt_tls_enabled: bool,
    #[serde(default = "default_hass_discovery_prefix")]
    pub discovery_prefix: String,
    #[serde(default = "default_hass_base_topic")]
    pub base_topic: String,
    #[serde(default = "default_hass_base_topic")]
    pub device_id: String,
    #[serde(default)]
    pub connected_since_ns: Option<i64>,
    #[serde(default)]
    pub last_connect_error: Option<String>,
    #[serde(default)]
    pub last_publish_ns: Option<i64>,
    #[serde(default)]
    pub last_reconcile_ns: Option<i64>,
    #[serde(default)]
    pub queue_depth: u32,
    #[serde(default)]
    pub queue_capacity: u32,
    #[serde(default)]
    pub discovery_entity_count: u32,
    #[serde(default)]
    pub published_state_topic_count: u32,
    #[serde(default)]
    pub metrics: std::collections::BTreeMap<String, i64>,
}

fn default_hass_connection_state() -> String {
    "disabled".to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SiteOriginSnapshot {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: Option<f64>,
    pub reconcile_delay_seconds: Option<f64>,
    pub mode: Option<String>,
    pub source: Option<String>,
}
