//! Schema of record for a chirp-lab bundle.
//!
//! Per phone (`getExternalFilesDir("chirp_lab")/<run_id>/`, pulled into
//! `<run_dir>/<phone_dir>/`):
//! - `capture.wav`: PCM16 mono 48 kHz; sample 0 = AudioRecord frame 0. The
//!   header may be incomplete (app killed): file length is trusted.
//! - `session.json`: [`SessionDescription`].
//! - `events.jsonl`: one [`BundleEvent`] per line, each with `t_ns`
//!   (System.nanoTime) and `type`.
//!
//! Per run, written by the laptop at pull time: `run_manifest.json`
//! ([`RunManifest`]), the ground truth (tape distance, temperature, pose,
//! surfaces) and which phone directory played which role.
//!
//! Every field the app might omit is optional: an old or partial bundle
//! still analyses, and `validate` reports what is missing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WaveformDescription {
    /// `updown_hfm`, `hfm` or `lfm`.
    pub kind: String,
    pub band_low_hz: f64,
    pub band_high_hz: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScheduleDescription {
    pub period_ms: u64,
    pub offset_ms: u64,
    pub lead_ms: u64,
    pub waveform: WaveformDescription,
    pub channel: String,
    pub level_dbfs: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceDescription {
    pub manufacturer: String,
    pub model: String,
    pub sdk_int: u32,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionDescription {
    pub run_id: String,
    pub role: String,
    pub mode: String,
    pub platform_name: String,
    pub device: DeviceDescription,
    pub schedule: ScheduleDescription,
    pub capture: serde_json::Value,
    pub playback: serde_json::Value,
    pub active_microphones: serde_json::Value,
    pub unprocessed_supported: Option<bool>,
    pub near_ultrasound: serde_json::Value,
    pub routed_input: serde_json::Value,
    pub routed_output: serde_json::Value,
    pub volume: serde_json::Value,
    pub started_ns: Option<i64>,
    pub started_wall_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundleEvent {
    Emission { t_ns: i64, emission_index: u64, output_frame_index: u64, target_ns: Option<i64>, played: bool, channel: Option<String> },
    CaptureTimestamp { t_ns: i64, frame_position: u64, nano_time: i64, wall_ms: Option<i64> },
    PlaybackTimestamp { t_ns: i64, frame_position: u64, nano_time: i64 },
    Attitude { t_ns: i64, quaternion: [f64; 4] },
    Screen { t_ns: i64, on: bool },
    RouteChange { t_ns: i64 },
    CaptureGap { t_ns: i64, expected_frame: u64, got_frame: u64, missing_frames: u64 },
    Marker { t_ns: i64, label: String },
    Thermal { t_ns: i64, status: i64, headroom: Option<f64> },
    Volume { t_ns: i64, music_index: i64, music_max: i64 },
    WriterOverflow { t_ns: i64, dropped_samples: u64 },
    /// Anything else (newer app versions): kept so `validate` can count it.
    Other { t_ns: i64, kind: String },
}

impl BundleEvent {
    pub fn t_ns(&self) -> i64 {
        match self {
            Self::Emission { t_ns, .. }
            | Self::CaptureTimestamp { t_ns, .. }
            | Self::PlaybackTimestamp { t_ns, .. }
            | Self::Attitude { t_ns, .. }
            | Self::Screen { t_ns, .. }
            | Self::RouteChange { t_ns }
            | Self::CaptureGap { t_ns, .. }
            | Self::Marker { t_ns, .. }
            | Self::Thermal { t_ns, .. }
            | Self::Volume { t_ns, .. }
            | Self::WriterOverflow { t_ns, .. }
            | Self::Other { t_ns, .. } => *t_ns,
        }
    }

    /// Parse one JSONL line; `None` for lines that are not an event object.
    pub fn from_json_line(line: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(line).ok()?;
        let t_ns = value.get("t_ns")?.as_i64()?;
        let kind = value.get("type")?.as_str()?.to_string();
        let u = |key: &str| value.get(key).and_then(serde_json::Value::as_u64);
        let i = |key: &str| value.get(key).and_then(serde_json::Value::as_i64);
        let f = |key: &str| value.get(key).and_then(serde_json::Value::as_f64);
        let s = |key: &str| value.get(key).and_then(serde_json::Value::as_str).map(str::to_string);
        Some(match kind.as_str() {
            "emission" => Self::Emission {
                t_ns,
                emission_index: u("emission_index")?,
                output_frame_index: u("output_frame_index")?,
                target_ns: i("target_ns"),
                played: value.get("played").and_then(serde_json::Value::as_bool).unwrap_or(true),
                channel: s("channel"),
            },
            "capture_timestamp" => Self::CaptureTimestamp { t_ns, frame_position: u("frame_position")?, nano_time: i("nano_time")?, wall_ms: i("wall_ms") },
            "playback_timestamp" => Self::PlaybackTimestamp { t_ns, frame_position: u("frame_position")?, nano_time: i("nano_time")? },
            "attitude" => {
                let q = value.get("quaternion")?.as_array()?;
                let component = |index: usize| q.get(index).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                Self::Attitude { t_ns, quaternion: [component(0), component(1), component(2), component(3)] }
            }
            "screen" => Self::Screen { t_ns, on: value.get("on")?.as_bool()? },
            "route_change" => Self::RouteChange { t_ns },
            "capture_gap" => Self::CaptureGap {
                t_ns,
                expected_frame: u("expected_frame").unwrap_or(0),
                got_frame: u("got_frame").unwrap_or(0),
                missing_frames: u("missing_frames").unwrap_or(0),
            },
            "marker" => Self::Marker { t_ns, label: s("label").unwrap_or_default() },
            "thermal" => Self::Thermal { t_ns, status: i("status").unwrap_or(0), headroom: f("headroom") },
            "volume" => Self::Volume { t_ns, music_index: i("music_index").unwrap_or(-1), music_max: i("music_max").unwrap_or(-1) },
            "writer_overflow" => Self::WriterOverflow { t_ns, dropped_samples: u("dropped_samples").unwrap_or(0) },
            _ => Self::Other { t_ns, kind },
        })
    }

    pub fn to_json_line(&self) -> String {
        let value = match self {
            Self::Emission { t_ns, emission_index, output_frame_index, target_ns, played, channel } => serde_json::json!({
                "t_ns": t_ns, "type": "emission", "emission_index": emission_index, "output_frame_index": output_frame_index,
                "target_ns": target_ns, "played": played, "channel": channel,
            }),
            Self::CaptureTimestamp { t_ns, frame_position, nano_time, wall_ms } => serde_json::json!({
                "t_ns": t_ns, "type": "capture_timestamp", "frame_position": frame_position, "nano_time": nano_time, "wall_ms": wall_ms,
            }),
            Self::Marker { t_ns, label } => serde_json::json!({"t_ns": t_ns, "type": "marker", "label": label}),
            Self::Screen { t_ns, on } => serde_json::json!({"t_ns": t_ns, "type": "screen", "on": on}),
            Self::Volume { t_ns, music_index, music_max } => serde_json::json!({"t_ns": t_ns, "type": "volume", "music_index": music_index, "music_max": music_max}),
            other => serde_json::json!({"t_ns": other.t_ns(), "type": "other"}),
        };
        value.to_string()
    }
}

/// One phone's role in the run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoleAssignment {
    pub serial: String,
    /// Directory under the run dir holding this phone's bundle.
    pub dir: String,
    /// Measured d_ii (speaker → ranging mic), if known.
    pub self_path_m: Option<f64>,
    /// Speaker offset from the ranging mic in the world frame (m), if the
    /// pose and the per-model speaker vector are known.
    pub speaker_offset_world_m: Option<[f64; 3]>,
    /// Free text: portrait / landscape / face up …
    pub pose: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunManifest {
    pub label: String,
    /// Matrix run name (A0 … A9) for the gate table.
    pub matrix_run: String,
    pub tape_distance_m: Option<f64>,
    pub temperature_c: Option<f64>,
    pub surface: String,
    /// Marker label → surface description (A2).
    pub marker_surfaces: BTreeMap<String, String>,
    pub roles: BTreeMap<String, RoleAssignment>,
    pub notes: String,
}

/// Dry-air ideal-gas law, same as the phone's `speed_of_sound_mps`.
const SOUND_SPEED_AT_ZERO_CELSIUS_MPS: f64 = 331.3;
const ZERO_CELSIUS_IN_KELVIN: f64 = 273.15;
/// Assumed when the manifest has no temperature (flagged in the report).
pub const DEFAULT_AIR_TEMPERATURE_C: f64 = 20.0;

impl RunManifest {
    pub fn speed_of_sound_mps(&self) -> f64 {
        let temperature_c = self.temperature_c.unwrap_or(DEFAULT_AIR_TEMPERATURE_C);
        SOUND_SPEED_AT_ZERO_CELSIUS_MPS * (1.0 + temperature_c / ZERO_CELSIUS_IN_KELVIN).sqrt()
    }
}
