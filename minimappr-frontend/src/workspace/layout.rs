use serde::{Deserialize, Serialize};

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
