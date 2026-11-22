//! Reactive state of the Field PWA page (capture, classifier, relay).
//! Detections themselves go to the shared `AppState` feed via the local shim.

use std::collections::VecDeque;

use leptos::prelude::{RwSignal, Update};

use super::browser_bindings::{BirdnetLoadReport, NegotiatedCaptureSettings};
use super::field_configuration::RELAY_MESSAGE_LOG_CAPACITY;

#[derive(Debug, Clone, PartialEq)]
pub struct LatestClassification {
    pub common_name: String,
    pub scientific_name: String,
    pub score: f32,
    pub inference_milliseconds: f64,
    pub window_first_device_sample_index: u64,
}

#[derive(Clone, Copy)]
pub struct FieldSignals {
    pub capture_status: RwSignal<String>,
    pub capture_running: RwSignal<bool>,
    pub capture_settings: RwSignal<Option<NegotiatedCaptureSettings>>,
    pub capture_route_change_count: RwSignal<u32>,
    pub stage1_trigger_count: RwSignal<u64>,
    pub stage1_onset_count: RwSignal<u64>,
    pub stage1_trigger_open: RwSignal<bool>,
    pub stage1_last_contrast_db: RwSignal<f64>,
    pub capture_gap_count: RwSignal<u64>,
    pub windows_dropped_count: RwSignal<u64>,
    pub birdnet_status: RwSignal<String>,
    pub birdnet_load_report: RwSignal<Option<BirdnetLoadReport>>,
    pub latest_classification: RwSignal<Option<LatestClassification>>,
    pub classification_count: RwSignal<u64>,
    pub last_clip_bytes: RwSignal<Option<usize>>,
    pub relay_status: RwSignal<String>,
    pub relay_role_description: RwSignal<Option<String>>,
    pub join_code_display: RwSignal<Option<String>>,
    pub relay_session_key_installed: RwSignal<bool>,
    pub relay_message_log: RwSignal<VecDeque<String>>,
    pub relay_sealed_messages_received: RwSignal<u64>,
}

impl FieldSignals {
    pub fn new() -> Self {
        Self {
            capture_status: RwSignal::new("Microphone off".to_string()),
            capture_running: RwSignal::new(false),
            capture_settings: RwSignal::new(None),
            capture_route_change_count: RwSignal::new(0),
            stage1_trigger_count: RwSignal::new(0),
            stage1_onset_count: RwSignal::new(0),
            stage1_trigger_open: RwSignal::new(false),
            stage1_last_contrast_db: RwSignal::new(0.0),
            capture_gap_count: RwSignal::new(0),
            windows_dropped_count: RwSignal::new(0),
            birdnet_status: RwSignal::new("Model not loaded".to_string()),
            birdnet_load_report: RwSignal::new(None),
            latest_classification: RwSignal::new(None),
            classification_count: RwSignal::new(0),
            last_clip_bytes: RwSignal::new(None),
            relay_status: RwSignal::new("Not connected".to_string()),
            relay_role_description: RwSignal::new(None),
            join_code_display: RwSignal::new(None),
            relay_session_key_installed: RwSignal::new(false),
            relay_message_log: RwSignal::new(VecDeque::new()),
            relay_sealed_messages_received: RwSignal::new(0),
        }
    }

    pub fn log_relay(&self, line: String) {
        self.relay_message_log.update(|log| {
            log.push_front(line);
            log.truncate(RELAY_MESSAGE_LOG_CAPACITY);
        });
    }
}
