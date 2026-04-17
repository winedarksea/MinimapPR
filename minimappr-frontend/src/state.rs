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
    pub node_id: String,
    pub health: String,
    pub last_seen_seconds_ago: Option<f64>,
    pub position_m: Option<Vec<f64>>,
    pub gps_fix: Option<bool>,
    pub temperature_c: Option<f64>,
    pub humidity: Option<f64>,
    pub rms_history: Option<Vec<f64>>,
    pub channel_count: Option<u32>,
    pub firmware_version: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Track {
    pub track_id: String,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    pub tqi: Option<f64>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub velocity_mps: Option<Vec<f64>>,
    pub last_update_ns: Option<i64>,
    pub sensor_count: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Detection {
    pub event_id: String,
    pub node_id: Option<String>,
    pub label: Option<String>,
    pub confidence: Option<f64>,
    pub received_ns: Option<i64>,
    pub position_m: Option<Vec<f64>>,
    pub position_geo: Option<GeoPoint>,
    pub has_audio: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Alert {
    pub alert_id: String,
    pub rule_name: Option<String>,
    pub message: Option<String>,
    pub severity: Option<String>,
    pub status: Option<String>,
    pub triggered_ns: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CopStatus {
    pub active_nodes: u32,
    pub degraded_nodes: u32,
    pub offline_nodes: u32,
    pub active_tracks: u32,
    pub open_alerts: u32,
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
    BitReport { node_id: String },
}

pub const MAX_FEED_LEN: usize = 50;
