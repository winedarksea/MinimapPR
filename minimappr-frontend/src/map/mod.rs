pub mod bindings;

use crate::layers::{all_layer_defs, LayerDef};
use crate::state::MapLayerVisibility;
use crate::state::{AppState, MapContextMenuState};
use bindings::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::hooks::use_location;
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
    let legend_open = RwSignal::new(false);
    let location = use_location();
    crate::layers::mount_core_marker_layers(&state);

    let selection_callback =
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
    let selection_callback = StoredValue::new_local(selection_callback);

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

    // Some router transitions keep the Rust owner alive while replacing the map
    // placeholder. Tie MapLibre binding to the route, not only component creation.
    Effect::new(move |_| {
        let pathname = location.pathname.get();
        if pathname == "/" || pathname.starts_with("/cop") {
            let (lat, lon) = map_initial_center(config.get_untracked());
            init(lat, lon, DEFAULT_MAP_ZOOM);
            selection_callback.with_value(|callback| {
                set_cop_selection_callback(callback.as_ref().unchecked_ref());
            });
            context_callback.with_value(|callback| {
                set_context_menu_callback(callback.as_ref().unchecked_ref());
            });
            // Second invalidateSize after flex layout settles, matching heatmap timing.
            spawn_local(async move {
                gloo_timers::future::TimeoutFuture::new(250).await;
                invalidate_map_size();
            });
        } else {
            set_cop_selection_callback_value(&JsValue::NULL);
            set_context_menu_callback_value(&JsValue::NULL);
            dispose_cop_map();
        }
    });

    on_cleanup(move || {
        set_cop_selection_callback_value(&JsValue::NULL);
        set_context_menu_callback_value(&JsValue::NULL);
        dispose_cop_map();
        selection_callback.dispose();
        context_callback.dispose();
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
                <div class="map-overlay-stack map-overlay-bottom-left">
                    <MapLayerControls />
                    <button
                        type="button"
                        class="map-toolbar-toggle"
                        aria-expanded=move || legend_open.get().to_string()
                        aria-label="Overlay legend"
                        on:click=move |_| legend_open.update(|open| *open = !*open)
                    >
                        <span class="material-symbols-rounded" aria-hidden="true">"legend_toggle"</span>
                        <span>"Legend"</span>
                    </button>
                    {move || legend_open.get().then(|| view! {
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
                    })}
                </div>
            </div>
        </div>
    }
}

#[component]
fn MapLayerControls() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let heatmap_enabled = state.live_heatmap_enabled;
    let heatmap_window = state.live_heatmap_window;
    let heatmap_bin_count = state.live_heatmap_bin_count;
    let heatmap_error = state.live_heatmap_error;
    let map_layers = state.map_layers;
    let menu_open = RwSignal::new(false);

    Effect::new(move |_| {
        map_layers.get().save();
    });

    view! {
        <div class="workspace-map-controls-anchor">
            <button
                type="button"
                class="map-toolbar-toggle"
                aria-expanded=move || menu_open.get().to_string()
                aria-label="Map layers"
                on:click=move |_| menu_open.update(|open| *open = !*open)
            >
                <span class="material-symbols-rounded" aria-hidden="true">"layers"</span>
                <span>"Layers"</span>
            </button>
            {move || menu_open.get().then(|| view! {
                <section class="workspace-map-controls" aria-label="Map layers">
                    <div class="map-control-layer-grid">
                        {all_layer_defs()
                            .iter()
                            .copied()
                            .map(|layer| view! {
                                <LayerToggle layer map_layers />
                            })
                            .collect_view()}
                    </div>
                    <div class="map-control-row">
                        <label class="map-control-toggle">
                            <input
                                type="checkbox"
                                prop:checked=move || heatmap_enabled.get()
                                on:change=move |event| heatmap_enabled.set(event_target_checked(&event))
                            />
                            <span>"Heatmap"</span>
                        </label>
                        <span class="tone-badge neutral">
                            {move || format!("{} bins", heatmap_bin_count.get())}
                        </span>
                    </div>
                    <div class="map-control-segments">
                        <HeatmapWindowButton label="5m" heatmap_window />
                        <HeatmapWindowButton label="1h" heatmap_window />
                        <HeatmapWindowButton label="24h" heatmap_window />
                        <HeatmapWindowButton label="7d" heatmap_window />
                    </div>
                    {move || heatmap_error.get().map(|message| view! {
                        <span class="daily-error map-control-error">{message}</span>
                    })}
                </section>
            })}
        </div>
    }
}

#[component]
fn LayerToggle(layer: LayerDef, map_layers: RwSignal<MapLayerVisibility>) -> impl IntoView {
    let checked = move || (layer.get_visible)(&map_layers.get());
    let input_id = format!("map-layer-{}", layer.id);
    let label_for = input_id.clone();
    view! {
        <label
            class="map-control-toggle map-control-toggle-compact"
            for=label_for
            data-layer-id=layer.id
            data-layer-group=layer.group.as_str()
            data-default-visible=layer.default_visible.to_string()
        >
            <input
                id=input_id
                type="checkbox"
                prop:checked=checked
                on:change=move |event| {
                    let next_checked = event_target_checked(&event);
                    map_layers.update(|layers| (layer.set_visible)(layers, next_checked));
                }
            />
            <span>{layer.title}</span>
        </label>
    }
}

#[component]
fn HeatmapWindowButton(label: &'static str, heatmap_window: RwSignal<String>) -> impl IntoView {
    view! {
        <button
            type="button"
            class=move || {
                if heatmap_window.get() == label {
                    "btn-sm active"
                } else {
                    "btn-sm"
                }
            }
            on:click=move |_| heatmap_window.set(label.to_string())
        >
            {label}
        </button>
    }
}
