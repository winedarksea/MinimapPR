use crate::state::{AppState, CopStatus, Detection, NodeStatus, Track, MAX_FEED_LEN};
use leptos::prelude::*;
use gloo_net::http::Request;
use gloo_timers::future::IntervalStream;
use futures::StreamExt;
use serde::de::DeserializeOwned;
use wasm_bindgen_futures::spawn_local;

async fn fetch_json<T: DeserializeOwned>(url: &str) -> Option<T> {
    let resp = Request::get(url).send().await.ok()?;
    if resp.ok() { resp.json().await.ok() } else { None }
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
            for det in dets { d.push_back(det); }
            while d.len() > MAX_FEED_LEN { d.pop_front(); }
        });
    }
    if let Some(cop) = fetch_json::<CopStatus>("/api/v1/cop/status").await {
        state.cop_status.set(Some(cop));
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

pub async fn patch_config(updates: serde_json::Value) -> Result<crate::state::ConfigSnapshot, String> {
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
