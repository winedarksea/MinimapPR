use crate::api::list_overlays;
use crate::map::bindings::{pulse_track_marker, trigger_node_omni_ripple};
use crate::state::{AppState, LiveEvent, NodeStatus, WsStatus, MAX_FEED_LEN};
use gloo_net::http::Request;
use gloo_timers::future::TimeoutFuture;
use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

fn ws_url() -> String {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let proto = if location.protocol().unwrap_or_default() == "https:" {
        "wss"
    } else {
        "ws"
    };
    let host = location.host().unwrap_or_default();
    format!("{proto}://{host}/ws/live")
}

fn connect(state: AppState) -> WebSocket {
    let url = ws_url();
    let ws = WebSocket::new(&url).expect("WebSocket::new");

    // onopen
    {
        let s = state.clone();
        let ws_ref = ws.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            s.ws_status.set(WsStatus::Connected);
            let msg = r#"{"action":"set_filter","tracks":true,"detections":true,"alerts":true}"#;
            let _ = ws_ref.send_with_str(msg);
        });
        ws.set_onopen(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    // onmessage
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
            if let Some(text) = ev.data().as_string() {
                handle_message(&s, &text);
            }
        });
        ws.set_onmessage(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    }

    ws
}

fn handle_message(state: &AppState, text: &str) {
    let event: LiveEvent = match serde_json::from_str(text) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("WS parse error: {e}: {text}");
            return;
        }
    };
    match event {
        LiveEvent::Detection(det) => {
            if let Some(track_id) = det.track_id.as_deref() {
                let should_pulse = if det.spatial_display_mode() == "bearing_only" {
                    state
                        .tracks
                        .with(|tracks| tracks.iter().any(|track| track.track_id == track_id))
                } else {
                    det.spatial_display_mode() == "localized"
                };
                if should_pulse {
                    pulse_track_marker(track_id);
                }
            }
            if det.spatial_display_mode() == "node_only" {
                if let Some(node_id) = det.node_id.as_deref() {
                    let node_geo = state.nodes.with(|nodes| {
                        nodes
                            .iter()
                            .find(|node| node.node_id == node_id)
                            .and_then(|node| node.position_geo.clone())
                    });
                    if let Some(geo) = node_geo {
                        trigger_node_omni_ripple(
                            node_id,
                            geo.lat,
                            geo.lon,
                            det.label.as_deref().unwrap_or("omni detection"),
                        );
                    }
                }
            }
            state.detections.update(|d| {
                d.push_front(det);
                while d.len() > MAX_FEED_LEN {
                    d.pop_back();
                }
            });
        }
        LiveEvent::Alert(alert) => {
            state.alerts.update(|a| {
                a.push_front(alert);
                while a.len() > MAX_FEED_LEN {
                    a.pop_back();
                }
            });
        }
        LiveEvent::TrackUpdate(track) => {
            state.tracks.update(|ts| {
                if let Some(existing) = ts.iter_mut().find(|t| t.track_id == track.track_id) {
                    *existing = track;
                } else {
                    ts.push(track);
                }
            });
        }
        LiveEvent::ConfigUpdated { config } => {
            state.config.set(Some(config));
        }
        LiveEvent::NodeCapabilityStatus(update) => {
            if update.capability == "ptz_camera" {
                state.nodes.update(|nodes| {
                    if let Some(node) = nodes.iter_mut().find(|node| node.node_id == update.node_id) {
                        node.ptz_status = Some(update.status.clone());
                    }
                });
            }
        }
        LiveEvent::NodeUpdated(update) => {
            let nodes_signal = state.nodes;
            spawn_local(async move {
                match Request::get("/api/v1/nodes?limit=64").send().await {
                    Ok(resp) if resp.ok() => match resp.json::<Vec<NodeStatus>>().await {
                        Ok(nodes) => nodes_signal.set(nodes),
                        Err(error) => log::warn!("node refresh parse failed after websocket update for {}: {error}", update.node_id),
                    },
                    Ok(resp) => log::warn!("node refresh failed after websocket update for {}: HTTP {}", update.node_id, resp.status()),
                    Err(error) => log::warn!("node refresh failed after websocket update for {}: {error}", update.node_id),
                }
            });
        }
        LiveEvent::RecordingStatus { session } => {
            if session.status.is_active() {
                state.active_recording.set(Some(session));
            } else {
                state.active_recording.set(None);
            }
        }
        LiveEvent::OverlayUpdated { .. } => {
            let overlays = state.overlays;
            spawn_local(async move {
                match list_overlays().await {
                    Ok(items) => overlays.set(items),
                    Err(error) => {
                        log::warn!("overlay refresh failed after websocket update: {error}")
                    }
                }
            });
        }
        LiveEvent::RulesUpdated | LiveEvent::SetFilter | LiveEvent::BitReport { .. } => {}
    }
}

pub fn start_ws(state: AppState) {
    spawn_local(async move {
        let mut backoff_ms = 1_000u32;
        loop {
            state.ws_status.set(WsStatus::Reconnecting);
            let ws = connect(state.clone());

            // Poll until closed
            let closed = std::rc::Rc::new(std::cell::Cell::new(false));
            {
                let flag = closed.clone();
                let cb = Closure::<dyn FnMut(CloseEvent)>::new(move |_: CloseEvent| {
                    flag.set(true);
                });
                ws.set_onclose(Some(cb.as_ref().unchecked_ref()));
                cb.forget();
            }
            {
                let flag = closed.clone();
                let s = state.clone();
                let cb = Closure::<dyn FnMut(ErrorEvent)>::new(move |_: ErrorEvent| {
                    flag.set(true);
                    s.ws_status.set(WsStatus::Disconnected);
                });
                ws.set_onerror(Some(cb.as_ref().unchecked_ref()));
                cb.forget();
            }

            // Wait for close by polling
            loop {
                TimeoutFuture::new(200).await;
                if closed.get() {
                    break;
                }
            }

            state.ws_status.set(WsStatus::Disconnected);
            let _ = ws.close();

            TimeoutFuture::new(backoff_ms).await;
            backoff_ms = (backoff_ms * 2).min(30_000);
        }
    });
}

#[allow(dead_code)]
pub fn send_filter(state: &AppState) {
    let filter = state.filter.get_untracked();
    let msg = serde_json::json!({
        "action": "set_filter",
        "tracks": filter.show_tracks,
        "detections": filter.show_detections,
        "alerts": filter.show_alerts,
    });
    // Accessing the WS handle from outside the closure is tricky with this simple
    // architecture; the subscription_ack flow handles the initial subscribe, and
    // the backend keeps the filter. Re-connect automatically re-subscribes.
    let _ = msg;
}
