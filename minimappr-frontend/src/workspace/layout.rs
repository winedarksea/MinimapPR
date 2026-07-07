use serde::{Deserialize, Serialize};

pub const DEFAULT_DOCK_WIDTH_PX: f64 = 360.0;
pub const MIN_DOCK_WIDTH_PX: f64 = 280.0;
pub const MAX_DOCK_WIDTH_PX: f64 = 520.0;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct WorkspaceLayout {
    pub nodes_open: bool,
    pub tracks_open: bool,
    pub detections_open: bool,
    pub alerts_open: bool,
    pub zones_open: bool,
    pub audio_open: bool,
    pub rf_open: bool,
    pub seismic_open: bool,
    pub speech_open: bool,
    #[serde(default = "default_left_dock_width_px")]
    pub left_dock_width_px: f64,
    #[serde(default = "default_right_dock_width_px")]
    pub right_dock_width_px: f64,
}

fn default_left_dock_width_px() -> f64 {
    DEFAULT_DOCK_WIDTH_PX
}

fn default_right_dock_width_px() -> f64 {
    DEFAULT_DOCK_WIDTH_PX
}

pub fn clamp_dock_width(width_px: f64) -> f64 {
    if width_px.is_finite() {
        width_px.clamp(MIN_DOCK_WIDTH_PX, MAX_DOCK_WIDTH_PX)
    } else {
        DEFAULT_DOCK_WIDTH_PX
    }
}

impl Default for WorkspaceLayout {
    fn default() -> Self {
        Self {
            nodes_open: true,
            tracks_open: true,
            detections_open: true,
            alerts_open: true,
            zones_open: false,
            audio_open: false,
            rf_open: false,
            seismic_open: false,
            speech_open: false,
            left_dock_width_px: DEFAULT_DOCK_WIDTH_PX,
            right_dock_width_px: DEFAULT_DOCK_WIDTH_PX,
        }
    }
}

impl WorkspaceLayout {
    pub fn load() -> Self {
        crate::prefs::get_json(crate::prefs::KEY_WORKSPACE).unwrap_or_default()
    }

    pub fn save(&self) {
        crate::prefs::set_json(crate::prefs::KEY_WORKSPACE, self);
    }
}
