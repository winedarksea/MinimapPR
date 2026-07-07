use crate::api::MapOverlay;
use crate::devices::schema::DeviceRecord;
use crate::recording::RecordingSession;
use leptos::prelude::RwSignal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub mod feeds;
pub mod models;
pub use feeds::*;
pub use models::*;

#[derive(Clone, Debug, PartialEq, Default)]
pub enum WsStatus {
    Connected,
    Reconnecting,
    #[default]
    Disconnected,
}

// ── Filter state for WS server-side filtering ───────────────────

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FilterState {
    pub show_tracks: bool,
    pub show_detections: bool,
    pub show_alerts: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopItemKind {
    Track,
    Detection,
    Alert,
    Node,
    Effector,
    Zone,
}

impl CopItemKind {
    pub fn as_js_kind(self) -> &'static str {
        match self {
            Self::Track => "track",
            Self::Detection => "detection",
            Self::Alert => "alert",
            Self::Node => "node",
            Self::Effector => "effector",
            Self::Zone => "zone",
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Track => "Track",
            Self::Detection => "Detection",
            Self::Alert => "Alert",
            Self::Node => "Node",
            Self::Effector => "Effector",
            Self::Zone => "Zone",
        }
    }

    pub fn from_js_kind(kind: &str) -> Option<Self> {
        match kind {
            "track" => Some(Self::Track),
            "detection" => Some(Self::Detection),
            "alert" => Some(Self::Alert),
            "node" => Some(Self::Node),
            "effector" => Some(Self::Effector),
            "zone" => Some(Self::Zone),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopSelection {
    pub kind: CopItemKind,
    pub id: String,
    /// Clicks pin the selection; hover-only highlights clear on mouse leave.
    pub pinned: bool,
}

impl CopSelection {
    pub fn hovered(kind: CopItemKind, id: String) -> Self {
        Self {
            kind,
            id,
            pinned: false,
        }
    }

    pub fn pinned(kind: CopItemKind, id: String) -> Self {
        Self {
            kind,
            id,
            pinned: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MapContextMenuState {
    pub lat: f64,
    pub lon: f64,
    pub screen_x: f64,
    pub screen_y: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZoneDraft {
    pub id: String,
    pub name: String,
    pub zone_type: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_m: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MapLayerVisibility {
    pub nodes: bool,
    pub tracks: bool,
    pub detections: bool,
    pub effectors: bool,
    pub omni: bool,
    pub zones: bool,
    pub overlays: bool,
    pub acoustic: bool,
    pub future_modalities: bool,
}

impl Default for MapLayerVisibility {
    fn default() -> Self {
        Self {
            nodes: true,
            tracks: true,
            detections: true,
            effectors: true,
            omni: true,
            zones: true,
            overlays: true,
            acoustic: true,
            future_modalities: true,
        }
    }
}

impl MapLayerVisibility {
    pub fn load() -> Self {
        crate::prefs::get_json(crate::prefs::KEY_LAYERS).unwrap_or_default()
    }

    pub fn save(&self) {
        crate::prefs::set_json(crate::prefs::KEY_LAYERS, self);
    }
}

// ── AppState ────────────────────────────────────────────────────

/// **LiveState** — signals fed by WS + periodic polling. Always live.
/// **QueryCache** is a separate concern (see `query_cache.rs` / page-local).
/// The router owns the "active page" — no `active_tab` lives here anymore.
#[derive(Clone)]
pub struct AppState {
    pub nodes: RwSignal<Vec<NodeStatus>>,
    pub tracks: RwSignal<Vec<Track>>,
    /// Registered PTZ effectors. Empty unless a camera has been registered —
    /// all effector UI is gated on this being non-empty.
    pub effectors: RwSignal<Vec<Effector>>,
    pub detections: RwSignal<VecDeque<Detection>>,
    pub omni_detection_summaries: RwSignal<Vec<NodeOmniDetectionSummary>>,
    pub alerts: RwSignal<VecDeque<Alert>>,
    pub zones: RwSignal<Vec<ZoneSpec>>,
    pub zone_occupancy: RwSignal<Vec<ZoneOccupancyState>>,
    pub overlays: RwSignal<Vec<MapOverlay>>,
    pub devices: RwSignal<Vec<DeviceRecord>>,
    pub modality: ModalityFeeds,
    pub live_heatmap_enabled: RwSignal<bool>,
    pub live_heatmap_window: RwSignal<String>,
    pub live_heatmap_refresh_tick: RwSignal<u64>,
    pub live_heatmap_bin_count: RwSignal<usize>,
    pub live_heatmap_error: RwSignal<Option<String>>,
    pub map_layers: RwSignal<MapLayerVisibility>,
    pub config: RwSignal<Option<ConfigSnapshot>>,
    pub cop_status: RwSignal<Option<CopStatus>>,
    pub fusion_status: RwSignal<Option<FusionStatus>>,
    pub ws_status: RwSignal<WsStatus>,
    pub filter: RwSignal<FilterState>,
    pub theme: RwSignal<String>,
    pub audio_drawer_open: RwSignal<bool>,
    pub audio_drawer_detection_id: RwSignal<Option<String>>,
    pub audio_drawer_track_id: RwSignal<Option<String>>,
    /// Active recording session — persists across navigation so the backend
    /// session isn't lost if the user leaves the recording page.
    pub active_recording: RwSignal<Option<RecordingSession>>,
    /// COP item currently hovered or clicked; drives map and sidebar highlighting.
    pub selected_cop_item: RwSignal<Option<CopSelection>>,
    pub map_context_menu: RwSignal<Option<MapContextMenuState>>,
    pub zone_draft: RwSignal<Option<ZoneDraft>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            nodes: RwSignal::new(vec![]),
            tracks: RwSignal::new(vec![]),
            effectors: RwSignal::new(vec![]),
            detections: RwSignal::new(VecDeque::new()),
            omni_detection_summaries: RwSignal::new(vec![]),
            alerts: RwSignal::new(VecDeque::new()),
            zones: RwSignal::new(vec![]),
            zone_occupancy: RwSignal::new(vec![]),
            overlays: RwSignal::new(vec![]),
            devices: RwSignal::new(crate::devices::registry::load_devices()),
            modality: ModalityFeeds::new(),
            live_heatmap_enabled: RwSignal::new(false),
            live_heatmap_window: RwSignal::new("1h".to_string()),
            live_heatmap_refresh_tick: RwSignal::new(0),
            live_heatmap_bin_count: RwSignal::new(0),
            live_heatmap_error: RwSignal::new(None),
            map_layers: RwSignal::new(MapLayerVisibility::load()),
            config: RwSignal::new(None),
            cop_status: RwSignal::new(None),
            fusion_status: RwSignal::new(None),
            ws_status: RwSignal::new(WsStatus::Disconnected),
            filter: RwSignal::new(FilterState::default()),
            theme: RwSignal::new(crate::prefs::current_theme()),
            audio_drawer_open: RwSignal::new(false),
            audio_drawer_detection_id: RwSignal::new(None),
            audio_drawer_track_id: RwSignal::new(None),
            active_recording: RwSignal::new(None),
            selected_cop_item: RwSignal::new(None),
            map_context_menu: RwSignal::new(None),
            zone_draft: RwSignal::new(None),
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
    ConfigUpdated {
        config: ConfigSnapshot,
    },
    EffectorStatus(EffectorStatusEvent),
    RecordingStatus {
        session: RecordingSession,
    },
    OverlayUpdated {
        #[serde(rename = "overlay_id")]
        _overlay_id: Option<String>,
    },
    RulesUpdated,
    SetFilter,
    BitReport {
        #[serde(rename = "node_id")]
        _node_id: String,
    },
}

pub const MAX_FEED_LEN: usize = 50;
