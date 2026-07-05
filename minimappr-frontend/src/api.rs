use crate::state::{
    Alert, AppState, CopStatus, Detection, Effector, FusionStatus, NodeOmniDetectionSummary,
    NodeStatus, Track, MAX_FEED_LEN,
};
use futures::StreamExt;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use leptos::prelude::*;
use serde::de::DeserializeOwned;
use wasm_bindgen_futures::spawn_local;

async fn fetch_json<T: DeserializeOwned>(url: &str) -> Option<T> {
    let resp = Request::get(url).send().await.ok()?;
    if resp.ok() {
        resp.json().await.ok()
    } else {
        None
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
    if let Some(effectors) = fetch_json::<Vec<Effector>>("/api/v1/effectors").await {
        state.effectors.set(effectors);
    }
}

pub fn start_polling(state: AppState) {
    let s = state.clone();
    spawn_local(async move {
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
        resp.json::<serde_json::Value>().await.map_err(|e| e.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn register_effector(payload: serde_json::Value) -> Result<Effector, String> {
    let resp = Request::post("/api/v1/effectors")
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&payload).unwrap_or_default())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        resp.json::<Effector>().await.map_err(|e| e.to_string())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

pub async fn delete_effector(effector_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(effector_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/effectors/{encoded}");
    let resp = Request::delete(&url).send().await.map_err(|e| e.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        Err(resp.text().await.unwrap_or_default())
    }
}

/// Slew a registered camera at a track's current position.
pub async fn aim_effector_at_track(effector_id: &str, track_id: &str) -> Result<(), String> {
    let encoded = js_sys::encode_uri_component(effector_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/effectors/{encoded}/aim");
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

/// Capture and persist a still from a camera, linked to a track/detection.
pub async fn snapshot_effector(
    effector_id: &str,
    track_id: Option<&str>,
) -> Result<String, String> {
    let encoded = js_sys::encode_uri_component(effector_id)
        .as_string()
        .unwrap_or_default();
    let url = format!("/api/v1/effectors/{encoded}/snapshot");
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
