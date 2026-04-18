use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Debug, PartialEq, Default)]
pub enum WsStatus {
    Connected,
    Reconnecting,
    #[default]
    Disconnected,
}

// ── Backend model types ─────────────────────────────────────────

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
    pub velocity_mps: Option<Vec<f64>>,
    #[serde(alias = "last_seen_ns")]
    pub last_update_ns: Option<i64>,
    #[serde(alias = "update_count")]
    pub sensor_count: Option<u32>,
    pub status: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Detection {
    #[serde(alias = "id")]
    pub event_id: String,
    #[serde(alias = "source_node_id")]
    pub node_id: Option<String>,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    pub label_confidence: Option<f64>,
    #[serde(alias = "tor_ns")]
    pub received_ns: Option<i64>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub has_audio: Option<bool>,
    pub snippet_path: Option<String>,
    pub track_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Alert {
    #[serde(alias = "id")]
    pub alert_id: String,
    #[serde(alias = "rule_id")]
    pub rule_name: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(alias = "priority")]
    pub severity: Option<String>,
    pub status: Option<String>,
    #[serde(alias = "timestamp_ns")]
    pub triggered_ns: Option<i64>,
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

// ── Config snapshot ─────────────────────────────────────────────

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
    pub classifier_backend: String,
    pub yamnet_min_confidence: f64,
    pub detection_min_confidence: f64,
    pub beamformer_type: String,
    pub tracking_filter: String,
    pub fusion_worker_count: u32,
    pub coordinate_mode: String,
}

// ── Filter state for WS server-side filtering ───────────────────

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FilterState {
    pub show_tracks: bool,
    pub show_detections: bool,
    pub show_alerts: bool,
}

// ── AppState ────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub nodes:      RwSignal<Vec<NodeStatus>>,
    pub tracks:     RwSignal<Vec<Track>>,
    pub detections: RwSignal<VecDeque<Detection>>,
    pub alerts:     RwSignal<VecDeque<Alert>>,
    pub config:     RwSignal<Option<ConfigSnapshot>>,
    pub cop_status: RwSignal<Option<CopStatus>>,
    pub ws_status:  RwSignal<WsStatus>,
    pub active_tab: RwSignal<Tab>,
    pub filter:     RwSignal<FilterState>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub enum Tab {
    #[default]
    Tracks,
    Detections,
    Alerts,
    Config,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            nodes:      RwSignal::new(vec![]),
            tracks:     RwSignal::new(vec![]),
            detections: RwSignal::new(VecDeque::new()),
            alerts:     RwSignal::new(VecDeque::new()),
            config:     RwSignal::new(None),
            cop_status: RwSignal::new(None),
            ws_status:  RwSignal::new(WsStatus::Disconnected),
            active_tab: RwSignal::new(Tab::Tracks),
            filter:     RwSignal::new(FilterState::default()),
        }
    }
}

// ── Live event enum ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveEvent {
    Detection(Detection),
    Alert(Alert),
    TrackUpdate(Track),
    ConfigUpdated { config: ConfigSnapshot },
    SetFilter,
    BitReport {
        #[serde(rename = "node_id")]
        _node_id: String,
    },
}

pub const MAX_FEED_LEN: usize = 50;
