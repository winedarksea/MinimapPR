//! Explicitly scripted sound sources (birds) and the sync-chirp schedule.

use serde::{Deserialize, Serialize};

use crate::physics::geometry::Vec3;
use crate::signals::calls::CallGenerator;
use crate::signals::chirps::ChirpWaveformParams;

/// Typical songbird song level, RMS dB SPL at 1 m.
pub const DEFAULT_CALL_LEVEL_DB_SPL_AT_1M: f64 = 85.0;
pub const DEFAULT_LABEL_CATEGORY: &str = "bird";

fn default_call_level() -> f64 {
    DEFAULT_CALL_LEVEL_DB_SPL_AT_1M
}

fn default_label_category() -> String {
    DEFAULT_LABEL_CATEGORY.to_string()
}

/// One vocalisation; becomes one ground-truth event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallConfig {
    /// Emission start in true session seconds.
    pub start_s: f64,
    /// RMS level over the call's active samples, dB SPL at 1 m.
    #[serde(default = "default_call_level")]
    pub level_db_spl_at_1m: f64,
    pub generator: CallGenerator,
    /// Optional fixed event id; generated otherwise.
    #[serde(default)]
    pub event_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    /// Links every bout of one bird across events (entity-aware scoring).
    pub individual_id: String,
    pub species: String,
    /// Ground-truth label; defaults to `species`.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default = "default_label_category")]
    pub label_category: String,
    pub position_m: Vec3,
    /// Optional `[t_s, x, y, z]` rows (moving source → Doppler).
    #[serde(default)]
    pub trajectory: Option<Vec<[f64; 4]>>,
    #[serde(default)]
    pub calls: Vec<CallConfig>,
}

/// Round-robin chirp sync rounds: in each round every phone (in device
/// order) emits one chirp in its own slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChirpScheduleConfig {
    pub round_start_times_s: Vec<f64>,
    pub slot_duration_s: f64,
    pub waveform: ChirpWaveformParams,
    /// Speaker output, RMS dB SPL at 1 m (over the chirp).
    pub level_db_spl_at_1m: f64,
}

impl Default for ChirpScheduleConfig {
    fn default() -> Self {
        Self {
            round_start_times_s: vec![1.0],
            slot_duration_s: 0.25,
            waveform: ChirpWaveformParams::default(),
            level_db_spl_at_1m: 90.0,
        }
    }
}
