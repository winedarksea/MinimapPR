use crate::state::{
    Alert, AppState, CopStatus, Detection, FusionStatus, NodeOmniDetectionSummary, NodeStatus,
    Track, TranscriptLine, ZoneOccupancyState, ZoneSpec, MAX_FEED_LEN,
};
use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use wasm_bindgen_futures::spawn_local;

fn default_rule_action_type() -> String {
    "alert".to_string()
}

fn default_rule_destination() -> String {
    "cop".to_string()
}

fn default_rule_priority() -> String {
    "normal".to_string()
}

fn default_rule_payload() -> serde_json::Value {
    serde_json::json!({})
}

fn default_rule_enabled() -> bool {
    true
}

fn default_rule_scope() -> String {
    "detection".to_string()
}

fn default_rule_actions() -> Vec<RuleAction> {
    vec![RuleAction::default()]
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuleAction {
    #[serde(rename = "type")]
    #[serde(default = "default_rule_action_type")]
    pub action_type: String,
    #[serde(default = "default_rule_destination")]
    pub destination: String,
    #[serde(default = "default_rule_priority")]
    pub priority: String,
    #[serde(default = "default_rule_payload")]
    pub payload: serde_json::Value,
}

impl Default for RuleAction {
    fn default() -> Self {
        Self {
            action_type: "alert".to_string(),
            destination: "cop".to_string(),
            priority: "normal".to_string(),
            payload: serde_json::json!({}),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RuleCondition {
    #[serde(default)]
    pub label_categories: Vec<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub reporting_modalities: Vec<String>,
    #[serde(default)]
    pub zone_ids: Vec<String>,
    #[serde(default)]
    pub track_statuses: Vec<String>,
    #[serde(default)]
    pub source_types: Vec<String>,
    #[serde(default)]
    pub transcript_contains: Vec<String>,
    #[serde(default)]
    pub min_confidence: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuleModel {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_rule_enabled")]
    pub enabled: bool,
    #[serde(default = "default_rule_scope")]
    pub scope: String,
    #[serde(default)]
    pub when: RuleCondition,
    #[serde(default = "default_rule_actions")]
    pub actions: Vec<RuleAction>,
    #[serde(default)]
    pub cooldown_seconds: f64,
}

impl Default for RuleModel {
    fn default() -> Self {
        Self {
            id: String::new(),
            enabled: true,
            scope: "detection".to_string(),
            when: RuleCondition::default(),
            actions: vec![RuleAction::default()],
            cooldown_seconds: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RulesConfigResponse {
    pub rules: Vec<RuleModel>,
    pub path: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TranscriptResponse {
    id: String,
    node_id: Option<String>,
    end_ns: i64,
    text: String,
    trigger_confidence: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MapOverlay {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub content_url: String,
    pub mime: String,
    #[serde(default)]
    pub bounds: Vec<Vec<f64>>,
    pub opacity: f64,
    pub storey: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    pub created_ns: i64,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MapOverlayUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storey: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HeatmapBin {
    pub lat: f64,
    pub lon: f64,
    pub weight: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HeatmapResponse {
    pub window: String,
    pub bin_deg: f64,
    pub now_ns: i64,
    pub bins: Vec<HeatmapBin>,
}

async fn fetch_json<T: DeserializeOwned>(url: &str) -> Option<T> {
    let resp = Request::get(url).send().await.ok()?;
    if resp.ok() {
        resp.json().await.ok()
    } else {
        None
    }
}

pub(crate) fn upsert_transcript_line(state: &AppState, line: TranscriptLine) {
    state.modality.speech_lines.update(|lines| {
        if let Some(index) = lines
            .iter()
            .position(|existing| existing.line_id == line.line_id)
        {
            lines.remove(index);
        }
        lines.push_front(line);
        while lines.len() > MAX_FEED_LEN {
            lines.pop_back();
        }
    });
}

async fn fetch_transcript_history(state: &AppState) {
    let Some(transcripts) =
        fetch_json::<Vec<TranscriptResponse>>("/api/v1/transcripts?limit=50").await
    else {
        return;
    };
    // The server returns newest first. Insert oldest first so the shared
    // upsert helper preserves the same newest-first ordering as live events.
    for transcript in transcripts.into_iter().rev() {
        upsert_transcript_line(
            state,
            TranscriptLine {
                line_id: transcript.id,
                device_id: transcript.node_id.unwrap_or_default(),
                text: transcript.text,
                confidence: transcript.trigger_confidence.unwrap_or(0.0),
                timestamp_ns: transcript.end_ns,
            },
        );
    }
}

async fn poll_once(state: AppState) {
    if let Some(nodes) = fetch_json::<Vec<NodeStatus>>("/api/v1/nodes?limit=64").await {
        state.nodes.set(nodes);
    }
    if let Some(tracks) = fetch_json::<Vec<Track>>("/api/v1/tracks?limit=100").await {
        state.tracks.set(tracks);
    }
    if let Some(dets) = fetch_json::<Vec<Detection>>("/api/v1/detections?limit=50").await {
        state.detections.update(|d| {
            d.clear();
            for det in dets {
                d.push_back(det);
            }
            while d.len() > MAX_FEED_LEN {
                d.pop_front();
            }
        });
    }
    if let Some(summaries) =
        fetch_json::<Vec<NodeOmniDetectionSummary>>("/api/v1/nodes/omni-detection-summary").await
    {
        state.omni_detection_summaries.set(summaries);
    }
    if let Some(cop) = fetch_json::<CopStatus>("/api/v1/cop/status").await {
        state.cop_status.set(Some(cop));
    }
    if let Some(fusion) = fetch_json::<FusionStatus>("/api/v1/fusion/status").await {
        state.fusion_status.set(Some(fusion));
    }
    if let Some(als) = fetch_json::<Vec<Alert>>("/api/v1/alerts?limit=50").await {
        state.alerts.update(|a| {
            a.clear();
            for alert in als {
                a.push_back(alert);
            }
            while a.len() > MAX_FEED_LEN {
                a.pop_front();
            }
        });
    }
    if state.config.get_untracked().is_none() {
        if let Some(cfg) = fetch_json("/api/v1/config").await {
            state.config.set(Some(cfg));
        }
    }
    if let Some(zones) = fetch_json::<Vec<ZoneSpec>>("/api/v1/zones").await {
        state.zones.set(zones);
    }
    if let Some(overlays) = fetch_json::<Vec<MapOverlay>>("/api/v1/overlays").await {
        state.overlays.set(overlays);
    }
    if let Some(occupancy) = fetch_json::<Vec<ZoneOccupancyState>>("/api/v1/zones/occupancy").await
    {
        state.zone_occupancy.set(occupancy);
    }
    state
        .live_heatmap_refresh_tick
        .update(|tick| *tick = tick.wrapping_add(1));
}

pub fn start_polling(state: AppState) {
    let s = state.clone();
    spawn_local(async move {
        fetch_transcript_history(&s).await;
        poll_once(s.clone()).await;
        let mut interval = IntervalStream::new(3_000);
        while interval.next().await.is_some() {
            poll_once(s.clone()).await;
        }
    });
}

/// Delete an offline node and its records. The backend rejects (409) any node
/// that is still active, so this only sticks for truly-stale nodes.
pub async fn delete_node(node_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}");
    let resp = Request::delete(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() || resp.status() == 204 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            body
        })
    }
}

pub async fn patch_config(
    updates: serde_json::Value,
) -> Result<crate::state::ConfigSnapshot, String> {
    let resp = Request::patch("/api/v1/config")
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&updates).unwrap_or_default())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.ok() {
        resp.json::<crate::state::ConfigSnapshot>()
            .await
            .map_err(|e| e.to_string())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(body)
    }
}

pub async fn get_rules() -> Result<RulesConfigResponse, String> {
    let resp = Request::get("/api/v1/rules")
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if resp.ok() {
        resp.json::<RulesConfigResponse>()
            .await
            .map_err(|error| error.to_string())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            body
        })
    }
}

pub async fn put_rules(rules: Vec<RuleModel>) -> Result<RulesConfigResponse, String> {
    let body = serde_json::json!({ "rules": rules });
    let resp = Request::put("/api/v1/rules")
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if resp.ok() {
        resp.json::<RulesConfigResponse>()
            .await
            .map_err(|error| error.to_string())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            body
        })
    }
}

pub async fn list_overlays() -> Result<Vec<MapOverlay>, String> {
    let resp = Request::get("/api/v1/overlays")
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<Vec<MapOverlay>>()
            .await
            .map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn upload_overlay(
    file: web_sys::File,
    name: &str,
    kind: &str,
    bounds_json: &str,
    opacity: f64,
    storey: &str,
) -> Result<MapOverlay, String> {
    let form = web_sys::FormData::new().map_err(|error| format!("{error:?}"))?;
    form.append_with_str("name", name)
        .map_err(|error| format!("{error:?}"))?;
    form.append_with_str("kind", kind)
        .map_err(|error| format!("{error:?}"))?;
    form.append_with_str("opacity", &opacity.to_string())
        .map_err(|error| format!("{error:?}"))?;
    if !bounds_json.trim().is_empty() {
        form.append_with_str("bounds", bounds_json)
            .map_err(|error| format!("{error:?}"))?;
    }
    if !storey.trim().is_empty() {
        form.append_with_str("storey", storey.trim())
            .map_err(|error| format!("{error:?}"))?;
    }
    let filename = file.name();
    form.append_with_blob_and_filename("file", &file, &filename)
        .map_err(|error| format!("{error:?}"))?;

    let resp = Request::post("/api/v1/overlays")
        .body(form)
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<MapOverlay>()
            .await
            .map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn patch_overlay(
    overlay_id: &str,
    update: MapOverlayUpdate,
) -> Result<MapOverlay, String> {
    let encoded = js_sys::encode_uri_component(overlay_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/overlays/{encoded}");
    let resp = Request::patch(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&update).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<MapOverlay>()
            .await
            .map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn delete_overlay(overlay_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(overlay_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/overlays/{encoded}");
    let resp = Request::delete(&url)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn upsert_zone(zone: ZoneSpec) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(&zone.id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/zones/{encoded}");
    let resp = Request::put(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&zone).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if resp.ok() {
        Ok(())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn fetch_heatmap(window: &str) -> Result<HeatmapResponse, String> {
    let encoded_window = js_sys::encode_uri_component(window)
        .as_string()
        .unwrap_or_else(|| "1h".to_string());
    let url = format!("/api/v1/analytics/heatmap?window={encoded_window}&bin=0.0001&max_bins=5000");
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<HeatmapResponse>()
            .await
            .map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn patch_detection_review(
    detection_id: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let encoded = js_sys::encode_uri_component(detection_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/detections/{encoded}/review");
    let resp = Request::patch(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&payload).unwrap_or_default())
        .map_err(|error| error.to_string())?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if resp.ok() {
        resp.json::<serde_json::Value>()
            .await
            .map_err(|error| error.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn patch_alert_status(
    alert_id: &str,
    status: &str,
    reason: Option<&str>,
) -> Result<String, String> {
    let encoded_alert_id = js_sys::encode_uri_component(alert_id);
    let encoded_status = js_sys::encode_uri_component(status);
    let mut url = format!("/api/v1/alerts/{encoded_alert_id}?status={encoded_status}");
    if let Some(reason) = reason.filter(|value| !value.is_empty()) {
        let encoded_reason = js_sys::encode_uri_component(reason)
            .as_string()
            .unwrap_or_default();
        url.push_str("&reason=");
        url.push_str(&encoded_reason);
    }

    let resp = Request::patch(&url)
        .send()
        .await
        .map_err(|error| error.to_string())?;

    if resp.ok() {
        let payload = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|error| error.to_string())?;
        let next_status = payload
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or(status);
        Ok(next_status.to_string())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(body)
    }
}

// ── Effectors (PTZ cameras) ─────────────────────────────────────

async fn post_json(url: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
    let resp = Request::post(url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(body).unwrap_or_default())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| e.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn register_node(payload: serde_json::Value) -> Result<NodeStatus, String> {
    let resp = Request::post("/api/v1/nodes")
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&payload).unwrap_or_default())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        resp.json::<NodeStatus>().await.map_err(|e| e.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

/// Slew a registered camera at a track's current position.
pub async fn aim_ptz_node_at_track(node_id: &str, track_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}/effector/aim");
    let body = serde_json::json!({ "track_id": track_id });
    let result = post_json(&url, &body).await?;
    match result.get("status").and_then(|v| v.as_str()) {
        Some("COMPLETED") => Ok(()),
        _ => Err(result
            .get("failure_class")
            .and_then(|v| v.as_str())
            .unwrap_or("aim failed")
            .to_string()),
    }
}

/// Slew a registered camera at an explicit local ENU position.
pub async fn aim_ptz_node_at_position(node_id: &str, target_m: [f64; 3]) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}/effector/aim");
    let body = serde_json::json!({ "target": target_m });
    let result = post_json(&url, &body).await?;
    match result.get("status").and_then(|v| v.as_str()) {
        Some("COMPLETED") => Ok(()),
        _ => Err(result
            .get("failure_class")
            .and_then(|v| v.as_str())
            .unwrap_or("aim failed")
            .to_string()),
    }
}

/// Capture and persist a still from a camera, linked to a track/detection.
pub async fn snapshot_ptz_node(node_id: &str, track_id: Option<&str>) -> Result<String, String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}/effector/snapshot");
    let body = serde_json::json!({ "track_id": track_id });
    let result = post_json(&url, &body).await?;
    match result.get("status").and_then(|v| v.as_str()) {
        Some("COMPLETED") => Ok(result
            .get("artifact_ids")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()),
        _ => Err(result
            .get("failure_class")
            .and_then(|v| v.as_str())
            .unwrap_or("snapshot failed")
            .to_string()),
    }
}

/// Arm a camera node's PTZ effector so it will accept slew/aim actions.
pub async fn arm_ptz_node(node_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}/effector/arm");
    ptz_action(&url, &serde_json::json!({})).await
}

/// Disarm a camera node's PTZ effector.
pub async fn disarm_ptz_node(node_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(node_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/nodes/{encoded}/effector/disarm");
    ptz_action(&url, &serde_json::json!({})).await
}

/// POST a PTZ action and interpret the `{status, failure_class}` result. The
/// backend returns 409 (not 2xx) when an action is refused, so read the body in
/// both cases rather than treating non-2xx as an opaque error.
async fn ptz_action(url: &str, body: &serde_json::Value) -> Result<(), String> {
    let resp = Request::post(url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(body).unwrap_or_default())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let result = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())?;
    match result.get("status").and_then(|v| v.as_str()) {
        Some("COMPLETED") => Ok(()),
        _ => Err(result
            .get("failure_class")
            .and_then(|v| v.as_str())
            .unwrap_or("action failed")
            .to_string()),
    }
}
