pub mod api;

use serde::{Deserialize, Serialize};

/// Status of a recording session as reported by the backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RecordingStatus {
    #[default]
    Idle,
    Starting,
    Active,
    AwaitingFinalTracks,
    Stopping,
    Completed,
    Failed,
}

impl RecordingStatus {
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Starting | Self::Active | Self::AwaitingFinalTracks | Self::Stopping
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Starting => "Starting…",
            Self::Active => "Recording",
            Self::AwaitingFinalTracks => "Awaiting final tracks…",
            Self::Stopping => "Stopping…",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        }
    }
}

/// Live session state returned by POST /api/v1/recordings and
/// GET /api/v1/recordings/{session_id}.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordingSession {
    pub session_id: String,
    pub status: RecordingStatus,
    pub listener_node_id: String,
    pub include_ambisonics: bool,
    pub include_iamf: bool,
    pub include_video: bool,
    pub camera_source: Option<String>,
    /// Unix epoch milliseconds when the session started.
    pub started_at_ms: f64,
    pub ended_at_ms: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub ambisonics_ready: bool,
    pub iamf_ready: bool,
    #[serde(default)]
    pub object_ready: bool,
    #[serde(default)]
    pub visual_ready: bool,
    pub video_ready: bool,
    #[serde(default)]
    pub error_message: Option<String>,
}

/// Request body for POST /api/v1/recordings.
///
/// The backend should:
/// 1. Resolve `listener_node_id` to a node position from /api/v1/nodes.
/// 2. Begin buffering audio from all active nodes.
/// 3. If `include_video`, start an ffmpeg capture from `camera_source`.
/// 4. Return a session with status=Starting immediately; transition to Active
///    once capture is confirmed.
#[derive(Clone, Debug, Serialize)]
pub struct StartRecordingRequest {
    pub listener_node_id: String,
    /// Always true — ambisonics B-Format is the base output format.
    pub include_ambisonics: bool,
    /// Encode ambisonics output through iamf-tools to produce an IAMF file.
    pub include_iamf: bool,
    /// Simultaneously capture video via ffmpeg from `camera_source`.
    pub include_video: bool,
    /// Device ID or path passed to ffmpeg `-i` for video capture.
    /// On macOS: AVFoundation device index string (e.g., "0").
    /// On Linux/RPi: v4l2 device path (e.g., "/dev/video0").
    pub camera_source: Option<String>,
}

/// A completed or in-progress recording returned by GET /api/v1/recordings.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RecordingLibraryEntry {
    pub session_id: String,
    /// Unix epoch milliseconds.
    pub started_at_ms: f64,
    pub ended_at_ms: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub listener_node_id: String,
    pub ambisonics_available: bool,
    pub iamf_available: bool,
    #[serde(default)]
    pub object_available: bool,
    #[serde(default)]
    pub visual_available: bool,
    pub video_available: bool,
    /// Approximate combined size of all output files in bytes.
    pub size_bytes: Option<u64>,
    pub status: RecordingStatus,
    #[serde(default)]
    pub error_message: Option<String>,
}

/// A camera/video capture device returned by GET /api/v1/cameras.
///
/// Backend should enumerate:
/// - macOS: AVFoundation devices via `ffmpeg -f avfoundation -list_devices true -i ""`
/// - Linux/RPi: v4l2 devices under /dev/video*
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CameraDevice {
    /// Device ID to pass to ffmpeg (e.g., "0" on macOS, "/dev/video0" on Linux).
    pub id: String,
    /// Human-readable label (e.g., "FaceTime HD Camera", "USB Camera").
    pub label: String,
}
