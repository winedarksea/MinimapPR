//! Detection, clip and bird-entity message bodies.
//!
//! Sample indices are on the named node's own 48 kHz device clock: the host
//! converts with its clock model before addressing a node, so a node never
//! needs to know anyone else's clock.

use serde::{Deserialize, Serialize};

use super::wire_codes::{BirdEntityState, FrequencyBandHz, NodeId, SpeciesScore};

/// Node -> host: one classification window's result. `(node_id,
/// local_sequence)` is the idempotency key for queued uploads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub node_id: NodeId,
    pub local_sequence: u64,
    pub window_first_device_sample_index: u64,
    pub window_sample_count: u32,
    pub classifier_model_id: u16,
    pub species_scores: Vec<SpeciesScore>,
    pub stage1_contrast_db: f32,
    pub stage1_active_band: Option<FrequencyBandHz>,
}

/// Host -> island node: is your audio around this index usable?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessQuery {
    pub query_id: u32,
    pub target_node_id: NodeId,
    pub centre_device_sample_index: u64,
    pub half_window_samples: u32,
    pub band: FrequencyBandHz,
}

/// Node -> host. In-band level only ranks; it never excludes a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessAnswer {
    pub query_id: u32,
    pub node_id: NodeId,
    pub clock_in_sync: bool,
    pub audio_still_buffered: bool,
    pub speech_free: bool,
    pub handling_noise_free: bool,
    pub in_band_level_over_floor_db: f32,
}

/// Array node -> host: per-call direction of arrival.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoaBearing {
    pub node_id: NodeId,
    pub centre_device_sample_index: u64,
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub azimuth_standard_deviation_deg: f32,
    pub band: FrequencyBandHz,
}

/// Host -> node: send this shared window as a clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipRequest {
    pub request_id: u32,
    pub target_node_id: NodeId,
    pub window_first_device_sample_index: u64,
    pub window_sample_count: u32,
    /// `clip_codec::DecimationFilterId` wire byte.
    pub decimation_filter_id: u8,
    /// Clips later than this are abandoned, not retried.
    pub abandon_after_milliseconds: u32,
}

/// One Stage-1 onset carried with a clip.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipOnset {
    pub device_sample_index: u64,
    pub positive_flux_db: f32,
}

/// One sensor frame on the audio timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFrameSample {
    pub device_sample_index: u64,
    /// Rotation vector as unit quaternion (w, x, y, z).
    pub orientation_quaternion: [f32; 4],
    pub linear_acceleration_m_per_s2: [f32; 3],
    pub pressure_hectopascal: Option<f32>,
}

/// Node -> host: MMCL clip container (24 kHz FLAC) + onsets + sensor slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipPayload {
    pub request_id: u32,
    pub node_id: NodeId,
    pub clip_container: Vec<u8>,
    pub onsets: Vec<ClipOnset>,
    pub sensor_frames: Vec<SensorFrameSample>,
}

/// One chunk of an encoded `ClipPayload`/`TapToHearPayload` frame, so large
/// clips interleave below control messages on one ordered stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipPayloadChunk {
    pub request_id: u32,
    pub node_id: NodeId,
    pub chunk_index: u16,
    pub chunk_count: u16,
    pub chunk_bytes: Vec<u8>,
}

/// Stage-1 frame range (relative to the classification window start).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameIndexRange {
    pub first_frame: u32,
    pub end_frame: u32,
}

/// Stage-1 band range `[first_band, end_band)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandIndexRange {
    pub first_band: u8,
    pub end_band: u8,
}

/// Frames x bands owned by one spatial cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterOcclusionMask {
    pub cluster_id: u16,
    pub frame_ranges: Vec<FrameIndexRange>,
    pub band_ranges: Vec<BandIndexRange>,
}

/// Host -> classifying node: noise-fill each cluster and rerun the classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocoOcclusionRequest {
    pub request_id: u32,
    pub target_node_id: NodeId,
    pub classification_window_first_device_sample_index: u64,
    pub cluster_masks: Vec<ClusterOcclusionMask>,
    pub maximum_reruns: u8,
}

/// Scores with one cluster occluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterRescore {
    pub occluded_cluster_id: u16,
    pub species_scores: Vec<SpeciesScore>,
}

/// Node -> host: LOCO result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocoRescoredSpecies {
    pub request_id: u32,
    pub node_id: NodeId,
    pub rescores: Vec<ClusterRescore>,
    pub reruns_used: u8,
}

/// Any phone -> classifying node (via host): play this bird's window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TapToHearRequest {
    pub request_id: u32,
    pub target_node_id: NodeId,
    pub bird_entity_id: u64,
    pub window_first_device_sample_index: u64,
}

/// Classifying node -> requester: 2-3 s MMCL clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TapToHearPayload {
    pub request_id: u32,
    pub node_id: NodeId,
    pub clip_container: Vec<u8>,
}

/// Bearing-only entity location.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityBearing {
    pub from_node_id: NodeId,
    pub azimuth_deg: f32,
    pub azimuth_standard_deviation_deg: f32,
}

/// Host -> session: bird entity state (rides the event log).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BirdEntityUpdate {
    pub bird_entity_id: u64,
    pub event_log_sequence: u64,
    pub state: BirdEntityState,
    pub island_id: u32,
    pub position_east_north_up_m: Option<[f32; 3]>,
    pub bearing: Option<EntityBearing>,
    /// Closed polygon (ENU metres) of the likelihood contour, if any.
    pub likelihood_contour_east_north_m: Vec<[f32; 2]>,
    pub species_belief: Vec<SpeciesScore>,
    /// "Just sang" pulse for the map.
    pub just_sang: bool,
    pub last_heard_reference_sample_index: u64,
}

/// Node -> host after solo fallback: detections held while disconnected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueuedDetectionUpload {
    pub node_id: NodeId,
    pub detections: Vec<DetectionEvent>,
}
