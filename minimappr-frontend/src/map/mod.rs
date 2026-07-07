pub mod bindings;

use crate::state::{AppState, MapContextMenuState};
use bindings::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};

const DEFAULT_MAP_LAT: f64 = 44.987;
const DEFAULT_MAP_LON: f64 = -93.258;
const DEFAULT_MAP_ZOOM: u32 = 17;
fn map_initial_center(config: Option<crate::state::ConfigSnapshot>) -> (f64, f64) {
    config
        .and_then(|snapshot| snapshot.site_origin.map(|origin| (origin.lat, origin.lon)))
        .unwrap_or((DEFAULT_MAP_LAT, DEFAULT_MAP_LON))
}

#[component]
pub fn MapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let tracks = state.tracks;
    let effectors = state.effectors;
    let detections = state.detections;
    let selected_cop_item = state.selected_cop_item;
    let config = state.config;
    crate::layers::mount_core_marker_layers(&state);

    // Init MapLibre once after the component first mounts.
    Effect::new(move |init_done: Option<bool>| {
        if init_done.is_none() {
            let (lat, lon) = map_initial_center(config.get_untracked());
            init(lat, lon, DEFAULT_MAP_ZOOM);
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
            let callback = StoredValue::new_local(callback);
            callback.with_value(|callback| {
                set_cop_selection_callback(callback.as_ref().unchecked_ref());
            });
            let context_menu_state = state.map_context_menu;
            let context_callback = Closure::<dyn FnMut(JsValue, JsValue, JsValue, JsValue)>::new(
                move |lat: JsValue, lon: JsValue, screen_x: JsValue, screen_y: JsValue| {
                    let Some(lat) = lat.as_f64() else {
                        return;
                    };
                    let Some(lon) = lon.as_f64() else {
                        return;
                    };
                    context_menu_state.set(Some(MapContextMenuState {
                        lat,
                        lon,
                        screen_x: screen_x.as_f64().unwrap_or(0.0),
                        screen_y: screen_y.as_f64().unwrap_or(0.0),
                    }));
                },
            );
            let context_callback = StoredValue::new_local(context_callback);
            context_callback.with_value(|callback| {
                set_context_menu_callback(callback.as_ref().unchecked_ref());
            });
            on_cleanup(move || {
                set_cop_selection_callback_value(&JsValue::NULL);
                set_context_menu_callback_value(&JsValue::NULL);
                callback.dispose();
                context_callback.dispose();
            });
            // Second invalidateSize after flex layout settles, matching heatmap timing.
            spawn_local(async move {
                gloo_timers::future::TimeoutFuture::new(250).await;
                invalidate_map_size();
            });
        }
        true
    });

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

            let latest_track = tracks.with(|tracks| {
                tracks
                    .iter()
                    .filter_map(|t| {
                        t.position_geo
                            .as_ref()
                            .map(|g| (t.last_update_ns.unwrap_or(i64::MIN), g.lat, g.lon))
                    })
                    .max_by_key(|(last_update_ns, _, _)| *last_update_ns)
            });
            let latest_detection = detections.with(|detections| {
                detections
                    .iter()
                    .filter_map(|d| {
                        d.position_geo
                            .as_ref()
                            .map(|g| (d.received_ns.unwrap_or(i64::MIN), g.lat, g.lon))
                    })
                    .max_by_key(|(received_ns, _, _)| *received_ns)
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
                nodes.with(|nodes| {
                    nodes
                        .iter()
                        .filter_map(|n| {
                            n.position_geo
                                .as_ref()
                                .map(|g| (n.last_seen_ns.unwrap_or(i64::MIN), g.lat, g.lon))
                        })
                        .max_by_key(|(last_seen_ns, _, _)| *last_seen_ns)
                        .map(|(_, lat, lon)| (lat, lon))
                })
            });

            if let Some((lat, lon)) = target {
                pan_to(lat, lon);
                return true;
            }
            false
        });
    }

    view! {
        <div class="panel map-column">
            <div class="panel-header">"Map"</div>
            <div class="leaflet-container-wrap">
                <div id="mmp-map"></div>
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
                            // Effectors (PTZ cameras) — only shown once one exists.
                            {move || (!effectors.get().is_empty()).then(|| view! {
                                <div class="legend-group">
                                    <div class="legend-group-label">"Effectors"</div>
                                    <span class="legend-item">
                                        <span class="legend-shape-node"></span>
                                        "PTZ camera"
                                    </span>
                                </div>
                            })}
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
                                    "Localized detection"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-bearing-detection"></span>
                                    "Bearing / weak range"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-bearing-wedge"></span>
                                    "Bearing uncertainty fan"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-omni-halo"></span>
                                    "Node omni activity"
                                </span>
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    }
}
