pub mod bindings;

use crate::state::AppState;
use bindings::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::collections::HashSet;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};

const DEFAULT_MAP_LAT: f64 = 44.987;
const DEFAULT_MAP_LON: f64 = -93.258;
const DEFAULT_MAP_ZOOM: u32 = 17;
const APPROX_DEGREES_LATITUDE_PER_METER: f64 = 9e-6;
const MIN_VELOCITY_LONGITUDE_COSINE: f64 = 0.01;

fn covariance_to_js_value(covariance: &[Vec<f64>]) -> JsValue {
    serde_wasm_bindgen::to_value(covariance).unwrap_or(JsValue::NULL)
}

fn map_initial_center(config: Option<crate::state::ConfigSnapshot>) -> (f64, f64) {
    config
        .and_then(|snapshot| snapshot.site_origin.map(|origin| (origin.lat, origin.lon)))
        .unwrap_or((DEFAULT_MAP_LAT, DEFAULT_MAP_LON))
}

fn velocity_mps_to_geo_delta(lat: f64, velocity_mps: &[f64]) -> Option<(f64, f64)> {
    if velocity_mps.len() < 2 {
        return None;
    }
    let dlat = velocity_mps[1] * APPROX_DEGREES_LATITUDE_PER_METER;
    let dlon = velocity_mps[0] * APPROX_DEGREES_LATITUDE_PER_METER
        / lat.to_radians().cos().max(MIN_VELOCITY_LONGITUDE_COSINE);
    Some((dlat, dlon))
}

#[component]
pub fn LeafletMapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let tracks = state.tracks;
    let detections = state.detections;
    let omni_detection_summaries = state.omni_detection_summaries;
    let theme = state.theme;
    let selected_cop_item = state.selected_cop_item;
    let config = state.config;

    // Init Leaflet once after the component first mounts.
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
            on_cleanup(move || {
                set_cop_selection_callback_value(&JsValue::NULL);
                callback.dispose();
            });
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
        Effect::new(move |_| {
            let _ = theme.get();
            let is_geo = config
                .get()
                .map(|c| c.coordinate_mode == "geodetic")
                .unwrap_or(false);
            if is_geo {
                nodes.with(|ns| {
                    for n in ns {
                        if let Some(geo) = &n.position_geo {
                            set_node_marker(&n.node_id, geo.lat, geo.lon, &n.health);
                        }
                    }
                });
            }
        });
    }

    // Sync node-attached omnidirectional detection summaries -> halo overlays.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            let node_positions = nodes.with(|ns| {
                ns.iter()
                    .filter_map(|node| {
                        node.position_geo
                            .as_ref()
                            .map(|geo| (node.node_id.clone(), geo.lat, geo.lon))
                    })
                    .collect::<Vec<_>>()
            });
            omni_detection_summaries.with(|summaries| {
                let mut current_ids = HashSet::new();
                for summary in summaries {
                    let Some((_, lat, lon)) = node_positions
                        .iter()
                        .find(|(node_id, _, _)| node_id == &summary.node_id)
                    else {
                        continue;
                    };
                    current_ids.insert(summary.node_id.clone());
                    if let Ok(summary_js) = serde_wasm_bindgen::to_value(summary) {
                        set_node_omni_halo(&summary.node_id, *lat, *lon, &summary_js);
                    }
                }

                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_node_omni_halo(id);
                    }
                }

                current_ids
            })
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

    // Sync tracks → map markers.
    // Carries the previous set of track IDs as Effect state so stale markers
    // (tracks that disappeared from the polled list) are removed promptly.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            tracks.with(|ts| {
                let current_ids: HashSet<String> = ts.iter().map(|t| t.track_id.clone()).collect();

                // Remove markers for tracks no longer in the server list.
                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_track(id);
                    }
                }

                for t in ts {
                    if let Some(geo) = &t.position_geo {
                        let label = t.label.as_deref().unwrap_or("");
                        let tqi = t.tqi.unwrap_or(0.0);
                        let status = t.status.as_deref().unwrap_or("active");
                        let last_update_ns = t.last_update_ns.unwrap_or(i64::MIN) as f64;
                        set_track_marker(
                            &t.track_id,
                            geo.lat,
                            geo.lon,
                            label,
                            tqi,
                            status,
                            last_update_ns,
                        );
                        if let Some((dlat, dlon)) = t
                            .velocity_mps
                            .as_deref()
                            .and_then(|velocity| velocity_mps_to_geo_delta(geo.lat, velocity))
                        {
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

                current_ids
            })
        });
    }

    // Highlight the COP item hovered/clicked in the sidebar or clicked on the map.
    {
        let selected_cop_item = state.selected_cop_item;
        Effect::new(move |_| match selected_cop_item.get() {
            Some(selection) => {
                highlight_cop_item(selection.kind.as_js_kind(), &selection.id);
                clear_all_cop_uncertainty();
                match selection.kind {
                    crate::state::CopItemKind::Track => {
                        tracks.with(|current_tracks| {
                            if let Some(track) = current_tracks
                                .iter()
                                .find(|track| track.track_id == selection.id)
                            {
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
                        });
                    }
                    crate::state::CopItemKind::Detection => {
                        detections.with(|current_detections| {
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
                        });
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
            detections.with(|ds| {
                let current_ids: HashSet<String> = ds
                    .iter()
                    .filter(|d| {
                        d.position_geo.is_some()
                            && d.track_id.is_none()
                            && d.spatial_display_mode() != "node_only"
                    })
                    .map(|d| d.event_id.clone())
                    .collect();

                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_detection_marker(id);
                    }
                }

                let node_positions = nodes.with(|ns| {
                    ns.iter()
                        .filter_map(|node| {
                            node.position_geo
                                .as_ref()
                                .map(|geo| (node.node_id.clone(), geo.lat, geo.lon))
                        })
                        .collect::<Vec<_>>()
                });
                for d in ds {
                    if d.track_id.is_some() || d.spatial_display_mode() == "node_only" {
                        continue;
                    }
                    if let Some(geo) = &d.position_geo {
                        let base_label = d.label.as_deref().unwrap_or("detection");
                        if d.spatial_display_mode() == "bearing_only" {
                            let source = d.node_id.as_ref().and_then(|node_id| {
                                node_positions
                                    .iter()
                                    .find(|(candidate, _, _)| candidate == node_id)
                                    .map(|(_, lat, lon)| (*lat, *lon))
                            });
                            let (source_lat, source_lon, has_source) = source
                                .map(|(lat, lon)| (lat, lon, true))
                                .unwrap_or((0.0, 0.0, false));
                            let label = format!("{base_label} · bearing/elevation, range weak");
                            add_bearing_only_detection_marker(
                                &d.event_id,
                                geo.lat,
                                geo.lon,
                                &label,
                                source_lat,
                                source_lon,
                                has_source,
                                d.received_ns.unwrap_or(i64::MIN) as f64,
                            );
                        } else {
                            let label = format!("{base_label} · localized");
                            add_detection_marker(
                                &d.event_id,
                                geo.lat,
                                geo.lon,
                                &label,
                                d.received_ns.unwrap_or(i64::MIN) as f64,
                            );
                        }
                    }
                }

                current_ids
            })
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
                                    "Localized detection"
                                </span>
                                <span class="legend-item">
                                    <span class="legend-shape-bearing-detection"></span>
                                    "Bearing / weak range"
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
