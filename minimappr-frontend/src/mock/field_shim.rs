//! Local shim standing in for `api.rs` + `ws.rs` in the Field PWA build.
//!
//! The server-backed COP fills `AppState` from REST polling and `/ws/live`.
//! The field build has no server: detections come from the phone's own
//! Stage-1 + BirdNET (and, later, from relay peers), and connection status
//! comes from the relay. This shim is the seam that writes those into the
//! same `AppState` signals, so shared components (feeds, map layers, bird
//! entity tray) can render field data without knowing where it came from.

use std::collections::VecDeque;

use leptos::prelude::{Set, Update};

use crate::state::{AppState, Detection, WsStatus};

const NANOSECONDS_PER_MILLISECOND: f64 = 1_000_000.0;
const LOCAL_REPORTING_MODALITY: &str = "acoustic";
const LOCAL_LOCALIZATION_METHOD: &str = "none_single_phone";

/// One locally produced (or relay-received) classification.
pub struct FieldDetectionReport {
    pub event_id: String,
    pub reporting_node_label: String,
    pub label: String,
    pub confidence: f64,
    pub has_clip: bool,
}

/// Replaces `ws::start_ws` + `api::start_polling` in the field build: there is
/// nothing to poll, so the shim only primes the feeds to their empty state.
pub fn start_field_local_feeds(state: &AppState) {
    state.detections.set(VecDeque::new());
    state.ws_status.set(WsStatus::Disconnected);
}

/// Relay phase stands in for the COP's server WebSocket status.
pub fn publish_relay_connection(state: &AppState, connected: bool, reconnecting: bool) {
    let status = match (connected, reconnecting) {
        (true, _) => WsStatus::Connected,
        (false, true) => WsStatus::Reconnecting,
        (false, false) => WsStatus::Disconnected,
    };
    state.ws_status.set(status);
}

/// Pushes a detection into the shared feed (newest first, bounded).
pub fn publish_detection(state: &AppState, report: FieldDetectionReport, feed_capacity: usize) {
    let detection = Detection {
        event_id: report.event_id,
        node_id: Some(report.reporting_node_label),
        reporting_modality: Some(LOCAL_REPORTING_MODALITY.to_string()),
        spatial_display_mode: None,
        localization_range_projection_mode: None,
        localization_range_observability: None,
        localization_method: Some(LOCAL_LOCALIZATION_METHOD.to_string()),
        capability_tier: None,
        solve_bearing: None,
        label: Some(report.label),
        confidence: Some(report.confidence),
        label_confidence: Some(report.confidence),
        received_ns: Some((js_sys::Date::now() * NANOSECONDS_PER_MILLISECOND) as i64),
        position_m: None,
        position_geo: None,
        position_covariance_m2: None,
        has_audio: Some(report.has_clip),
        snippet_path: None,
        track_id: None,
        contributors: Vec::new(),
    };
    state.detections.update(|feed| {
        feed.push_front(detection);
        feed.truncate(feed_capacity);
    });
}
