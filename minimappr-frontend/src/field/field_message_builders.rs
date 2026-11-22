//! Builds the `FieldMessage`s this phone sends (core protocol types).

use minimappr_core::protocol::detection_messages::DetectionEvent;
use minimappr_core::protocol::session_messages::Heartbeat;
use minimappr_core::protocol::wire_codes::{FrequencyBandHz, MotionState, PhonePlacement, SpeciesScore};
use minimappr_core::protocol::{FieldMessage, NodeId};

use super::browser_bindings::BirdnetClassification;
use super::field_configuration::BIRDNET_V3_CLASSIFIER_MODEL_ID;
use super::stage1_capture_pipeline::ClassificationWindow;

/// Unknown until the PWA reads Battery/thermal APIs (Safari exposes neither).
const UNKNOWN_FRACTION: f32 = -1.0;

pub fn build_heartbeat(node_id: u32, local_sequence: u64, device_sample_index_now: u64, capture_gap_count: u64) -> FieldMessage {
    FieldMessage::Heartbeat(Heartbeat {
        node_id: NodeId(node_id),
        local_sequence,
        device_sample_index_now,
        thermal_headroom_fraction: UNKNOWN_FRACTION,
        battery_fraction: UNKNOWN_FRACTION,
        placement: PhonePlacement::Handheld,
        motion_state: MotionState::Unknown,
        mic_only_mode: false,
        capture_overrun_count: u32::try_from(capture_gap_count).unwrap_or(u32::MAX),
        gnss_fix: None,
    })
}

/// Label ids are BirdNET V3 class indices (the session's agreed label table
/// is the V3 label CSV).
pub fn build_detection_event(
    node_id: u32,
    local_sequence: u64,
    window: &ClassificationWindow,
    classification: &BirdnetClassification,
) -> FieldMessage {
    FieldMessage::DetectionEvent(DetectionEvent {
        node_id: NodeId(node_id),
        local_sequence,
        window_first_device_sample_index: window.first_device_sample_index,
        window_sample_count: u32::try_from(window.samples.len()).unwrap_or(u32::MAX),
        classifier_model_id: BIRDNET_V3_CLASSIFIER_MODEL_ID,
        species_scores: classification
            .top
            .iter()
            .map(|class| SpeciesScore {
                label_id: class.class_index,
                score: class.score,
            })
            .collect(),
        stage1_contrast_db: window.stage1_contrast_db,
        stage1_active_band: window
            .stage1_active_band_hz
            .map(|(low_hz, high_hz)| FrequencyBandHz { low_hz, high_hz }),
    })
}
