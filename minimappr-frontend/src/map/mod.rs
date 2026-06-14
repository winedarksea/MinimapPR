pub mod bindings;

use crate::state::AppState;
use bindings::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::collections::HashSet;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};

fn covariance_to_js_value(covariance: &[Vec<f64>]) -> JsValue {
    let rows = js_sys::Array::new();
    for row in covariance {
        let values = js_sys::Array::new();
        for value in row {
            values.push(&JsValue::from_f64(*value));
        }
        rows.push(&values);
    }
    rows.into()
}

#[component]
pub fn LeafletMapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let tracks = state.tracks;
    let detections = state.detections;
    let theme = state.theme;
    let selected_cop_item = state.selected_cop_item;

    // Init Leaflet once after the component first mounts.
    Effect::new(move |init_done: Option<bool>| {
        if init_done.is_none() {
            init(44.987, -93.258, 17);
            let selected_cop_item = selected_cop_item;
            let callback =
                Closure::<dyn FnMut(JsValue, JsValue)>::new(move |kind: JsValue, id: JsValue| {
                    let Some(kind) = kind.as_string() else {
                        return;
                    };
                    let Some(id) = id.as_string() else {
                        return;
                    };
                    if let Some(kind) = crate::state::CopItemKind::from_js_kind(&kind) {
                        selected_cop_item.set(Some(crate::state::CopSelection::pinned(kind, id)));
                    }
                });
            set_cop_selection_callback(callback.as_ref().unchecked_ref());
            callback.forget();
            // Second invalidateSize after flex layout settles, matching heatmap timing.
            spawn_local(async move {
                gloo_timers::future::TimeoutFuture::new(250).await;
                invalidate_map_size();
            });
        }
        true
    });

    // Sync nodes → map markers (geodetic mode only; position_m is local-frame, use position_geo).
    {
        let config = state.config;
        Effect::new(move |_| {
            let _ = theme.get();
            let ns = nodes.get();
            let is_geo = config
                .get()
                .map(|c| c.coordinate_mode == "geodetic")
                .unwrap_or(false);
            if is_geo {
                for n in &ns {
                    if let Some(geo) = &n.position_geo {
                        set_node_marker(&n.node_id, geo.lat, geo.lon, &n.health);
                    }
                }
            }
        });
    }

    // Auto-center the map on first load to the most relevant geo data, instead of
    // leaving it parked at the hardcoded default. Priority:
    //   1. most recent track with a geo position,
    //   2. most recent detection with a geo position,
    //   3. otherwise a node position (most recently seen).
    // Runs on every data update until it succeeds once, so it works regardless of
    // whether tracks, detections, or only node positions arrive first.
    {
        Effect::new(move |already_centered: Option<bool>| {
            if already_centered.unwrap_or(false) {
                return true;
            }

            let latest_track = tracks
                .get()
                .into_iter()
                .filter(|t| t.position_geo.is_some())
                .max_by_key(|t| t.last_update_ns.unwrap_or(i64::MIN))
                .and_then(|t| {
                    t.position_geo
                        .as_ref()
                        .map(|g| (t.last_update_ns.unwrap_or(i64::MIN), g.lat, g.lon))
                });
            let latest_detection = detections
                .get()
                .into_iter()
                .filter(|d| d.position_geo.is_some())
                .max_by_key(|d| d.received_ns.unwrap_or(i64::MIN))
                .and_then(|d| {
                    d.position_geo
                        .as_ref()
                        .map(|g| (d.received_ns.unwrap_or(i64::MIN), g.lat, g.lon))
                });

            // Prefer whichever of track/detection is the more recent observation.
            let target = match (latest_track, latest_detection) {
                (Some(t), Some(d)) => Some(if t.0 >= d.0 { t } else { d }),
                (Some(t), None) => Some(t),
                (None, Some(d)) => Some(d),
                (None, None) => None,
            }
            .map(|(_, lat, lon)| (lat, lon))
            .or_else(|| {
                // Fall back to the most recently seen node with a geo position.
                nodes
                    .get()
                    .into_iter()
                    .filter(|n| n.position_geo.is_some())
                    .max_by_key(|n| n.last_seen_ns.unwrap_or(i64::MIN))
                    .and_then(|n| n.position_geo.as_ref().map(|g| (g.lat, g.lon)))
            });

            if let Some((lat, lon)) = target {
                pan_to(lat, lon);
                return true;
            }
            false
        });
    }

    // Sync tracks → map markers.
    // Carries the previous set of track IDs as Effect state so stale markers
    // (tracks that disappeared from the polled list) are removed promptly.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            let ts = tracks.get();

            let current_ids: HashSet<String> = ts.iter().map(|t| t.track_id.clone()).collect();

            // Remove markers for tracks no longer in the server list.
            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_track(id);
                }
            }

            for t in &ts {
                if let Some(geo) = &t.position_geo {
                    let label = t.label.as_deref().unwrap_or("");
                    let tqi = t.tqi.unwrap_or(0.0);
                    let status = t.status.as_deref().unwrap_or("active");
                    set_track_marker(&t.track_id, geo.lat, geo.lon, label, tqi, status);
                    if let Some(vel) = &t.velocity_mps {
                        if vel.len() >= 2 {
                            // Rough local velocity → geo delta (1 m ≈ 9e-6 deg)
                            let dlat = vel[1] * 9e-6;
                            let dlon = vel[0] * 9e-6 / (geo.lat.to_radians().cos()).max(0.01);
                            set_track_velocity_vector(
                                &t.track_id,
                                geo.lat,
                                geo.lon,
                                dlat,
                                dlon,
                                status,
                            );
                        }
                    }
                }
            }

            current_ids
        });
    }

    // Highlight the COP item hovered/clicked in the sidebar or clicked on the map.
    {
        let selected_cop_item = state.selected_cop_item;
        Effect::new(move |_| match selected_cop_item.get() {
            Some(selection) => {
                let current_tracks = tracks.get();
                let current_detections = detections.get();
                highlight_cop_item(selection.kind.as_js_kind(), &selection.id);
                clear_all_cop_uncertainty();
                match selection.kind {
                    crate::state::CopItemKind::Track => {
                        if let Some(track) = current_tracks.iter().find(|track| track.track_id == selection.id) {
                            if let (Some(geo), Some(covariance)) =
                                (&track.position_geo, &track.position_covariance_m2)
                            {
                                let covariance_js = covariance_to_js_value(covariance);
                                set_cop_uncertainty(
                                    "track",
                                    &track.track_id,
                                    geo.lat,
                                    geo.lon,
                                    &covariance_js,
                                );
                            }
                        }
                    }
                    crate::state::CopItemKind::Detection => {
                        if let Some(detection) = current_detections
                            .iter()
                            .find(|detection| detection.event_id == selection.id)
                        {
                            if let (Some(geo), Some(covariance)) =
                                (&detection.position_geo, &detection.position_covariance_m2)
                            {
                                let covariance_js = covariance_to_js_value(covariance);
                                set_cop_uncertainty(
                                    "detection",
                                    &detection.event_id,
                                    geo.lat,
                                    geo.lon,
                                    &covariance_js,
                                );
                            }
                        }
                    }
                    crate::state::CopItemKind::Alert => {}
                }
            }
            None => {
                clear_cop_highlight();
                clear_all_cop_uncertainty();
            }
        });
    }

    // Sync detections → map markers.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            let ds = detections.get();

            let current_ids: HashSet<String> = ds
                .iter()
                .filter(|d| d.position_geo.is_some())
                .map(|d| d.event_id.clone())
                .collect();

            if let Some(ref prev) = prev_ids {
                for id in prev.difference(&current_ids) {
                    remove_detection_marker(id);
                }
            }

            for d in &ds {
                if let Some(geo) = &d.position_geo {
                    let label = d.label.as_deref().unwrap_or("detection");
                    add_detection_marker(&d.event_id, geo.lat, geo.lon, label);
                }
            }

            current_ids
        });
    }

    view! {
        <div class="panel map-column">
            <div class="panel-header">"Map"</div>
            <div class="leaflet-container-wrap">
                <div id="leaflet-map"></div>
                <div class="map-overlay-stack map-overlay-bottom-right">
                    <section class="map-floating-panel">
                        <div class="map-floating-title">"Overlay Legend"</div>
                        <div class="legend">
                            // Nodes
                            <div class="legend-group">
                                <div class="legend-group-label">"Nodes"</div>
                                <span class="legend-item">
                                    <span class="legend-shape-node"></span>
                                    "Online"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-node degraded"></span>
                                    "Degraded"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-node offline"></span>
                                    "Offline"
                                </span>
                            </div>
                            // Tracks
                            <div class="legend-group">
                                <div class="legend-group-label">"Tracks"</div>
                                <span class="legend-item">
                                    <span class="legend-shape-track"></span>
                                    "Active"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-track coasting"></span>
                                    "Coasting"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-track dropped"></span>
                                    "Dropped"
                                </span>
                            </div>
                            // Events
                            <div class="legend-group">
                                <div class="legend-group-label">"Events"</div>
                                <span class="legend-item">
                                    <span class="legend-shape-detection"></span>
                                    "Detection"
                                </span>
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    }
}
