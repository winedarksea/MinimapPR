pub mod bindings;

use crate::state::AppState;
use bindings::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::collections::HashSet;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};

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
    // On the first update that carries any GPS position, pan to the average of all node positions.
    {
        let config = state.config;
        Effect::new(move |already_centered: Option<bool>| {
            let _ = theme.get();
            let ns = nodes.get();
            let is_geo = config
                .get()
                .map(|c| c.coordinate_mode == "geodetic")
                .unwrap_or(false);
            let was_centered = already_centered.unwrap_or(false);
            if is_geo {
                let mut lat_sum = 0.0f64;
                let mut lon_sum = 0.0f64;
                let mut geo_count = 0usize;
                for n in &ns {
                    if let Some(geo) = &n.position_geo {
                        set_node_marker(&n.node_id, geo.lat, geo.lon, &n.health);
                        if !was_centered {
                            lat_sum += geo.lat;
                            lon_sum += geo.lon;
                            geo_count += 1;
                        }
                    }
                }
                if !was_centered && geo_count > 0 {
                    pan_to(lat_sum / geo_count as f64, lon_sum / geo_count as f64);
                    return true;
                }
            }
            was_centered
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
            Some(selection) => highlight_cop_item(selection.kind.as_js_kind(), &selection.id),
            None => clear_cop_highlight(),
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
