use crate::state::{Alert, AppState, CopStatus, Detection, FusionStatus, NodeStatus, Track, MAX_FEED_LEN};
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
