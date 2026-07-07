pub mod bindings;

use crate::devices::schema::DeviceKind;
use crate::state::{AppState, Detection, MapContextMenuState};
use bindings::*;
use gloo_net::http::Request;
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
const DEFAULT_BEARING_HALF_ANGLE_DEG: f64 = 15.0;
const DEFAULT_BEARING_RANGE_M: f64 = 500.0;
const EARTH_RADIUS_M: f64 = 6_371_000.0;
const FUTURE_MODALITIES_LAYER_ID: &str = "future-modalities";

fn covariance_to_js_value(covariance: &[Vec<f64>]) -> JsValue {
    serde_wasm_bindgen::to_value(covariance).unwrap_or(JsValue::NULL)
}

fn overlay_layer_id(overlay_id: &str) -> String {
    format!("overlay:{overlay_id}")
}

fn overlay_has_four_corners(bounds: &[Vec<f64>]) -> bool {
    bounds.len() == 4 && bounds.iter().all(|corner| corner.len() == 2)
}

fn normalize_geojson_overlay(value: serde_json::Value) -> Option<serde_json::Value> {
    let geometry_type = value
        .get("type")
        .and_then(|item| item.as_str())
        .map(ToString::to_string);
    match geometry_type.as_deref() {
        Some("FeatureCollection") => Some(value),
        Some("Feature") => Some(serde_json::json!({
            "type": "FeatureCollection",
            "features": [value],
        })),
        Some(
            "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon",
        ) => Some(serde_json::json!({
            "type": "FeatureCollection",
            "features": [{ "type": "Feature", "properties": {}, "geometry": value }],
        })),
        _ => None,
    }
}

fn heatmap_bins_to_js_points(bins: &[crate::api::HeatmapBin]) -> JsValue {
    let points = js_sys::Array::new();
    for bin in bins {
        let triple = js_sys::Array::new();
        triple.push(&JsValue::from_f64(bin.lat));
        triple.push(&JsValue::from_f64(bin.lon));
        triple.push(&JsValue::from_f64(bin.weight as f64));
        points.push(&triple);
    }
    points.into()
}

fn future_modalities_layer_spec() -> serde_json::Value {
    serde_json::json!({
        "type": "geojson",
        "layer": {
            "type": "circle",
            "paint": {
                "circle-color": [
                    "match", ["get", "kind"],
                    "rf_emitter", "#4cc9f0",
                    "sdr_rf", "#4cc9f0",
                    "radar_24g", "#ffb703",
                    "seismic", "#f28482",
                    "speech_node", "#2ec4b6",
                    "#b8c0ff"
                ],
                "circle-radius": [
                    "case",
                    ["==", ["get", "surface"], "emitter"], 7,
                    6
                ],
                "circle-opacity": 0.78,
                "circle-stroke-color": "#111827",
                "circle-stroke-width": 1.8
            }
        }
    })
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

fn bearing_degrees_between_points(
    source_lat: f64,
    source_lon: f64,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    let source_lat_rad = source_lat.to_radians();
    let target_lat_rad = target_lat.to_radians();
    let delta_lon_rad = (target_lon - source_lon).to_radians();
    let y = delta_lon_rad.sin() * target_lat_rad.cos();
    let x = source_lat_rad.cos() * target_lat_rad.sin()
        - source_lat_rad.sin() * target_lat_rad.cos() * delta_lon_rad.cos();
    y.atan2(x).to_degrees().rem_euclid(360.0)
}

fn distance_m_between_points(
    source_lat: f64,
    source_lon: f64,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    let delta_lat = (target_lat - source_lat).to_radians();
    let delta_lon = (target_lon - source_lon).to_radians();
    let source_lat = source_lat.to_radians();
    let target_lat = target_lat.to_radians();
    let a = (delta_lat / 2.0).sin().powi(2)
        + source_lat.cos() * target_lat.cos() * (delta_lon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().atan2((1.0 - a).sqrt())
}

fn bearing_half_angle_degrees(detection: &Detection) -> f64 {
    detection
        .position_geo_uncertainty
        .as_ref()
        .and_then(|uncertainty| {
            uncertainty
                .get("bearing_half_angle_deg")
                .or_else(|| uncertainty.get("half_angle_deg"))
                .or_else(|| uncertainty.get("bearing_uncertainty_deg"))
        })
        .and_then(|value| value.as_f64())
        .or_else(|| {
            detection
                .localization_range_observability
                .map(|observability| {
                    let clamped = observability.clamp(0.0, 1.0);
                    30.0 - (clamped * 20.0)
                })
        })
        .unwrap_or(DEFAULT_BEARING_HALF_ANGLE_DEG)
        .clamp(5.0, 45.0)
}

#[component]
pub fn MapPanel() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let nodes = state.nodes;
    let tracks = state.tracks;
    let effectors = state.effectors;
    let detections = state.detections;
    let omni_detection_summaries = state.omni_detection_summaries;
    let zones = state.zones;
    let overlays = state.overlays;
    let devices = state.devices;
    let rf_spectra = state.modality.rf_spectra;
    let seismic_traces = state.modality.seismic_traces;
    let speech_lines = state.modality.speech_lines;
    let live_heatmap_enabled = state.live_heatmap_enabled;
    let live_heatmap_window = state.live_heatmap_window;
    let live_heatmap_refresh_tick = state.live_heatmap_refresh_tick;
    let live_heatmap_bin_count = state.live_heatmap_bin_count;
    let live_heatmap_error = state.live_heatmap_error;
    let theme = state.theme;
    let selected_cop_item = state.selected_cop_item;
    let config = state.config;

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

    // Sync effectors (PTZ cameras) → map markers. Hidden entirely — the map
    // simply never receives markers — when no effectors are registered.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            effectors.with(|es| {
                let current_ids: HashSet<String> = es
                    .iter()
                    .filter(|e| e.position_geo.is_some())
                    .map(|e| e.id.clone())
                    .collect();

                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_effector_marker(id);
                    }
                }

                for e in es {
                    if let Some(geo) = &e.position_geo {
                        // pan_deg is relative to the camera's registered home
                        // yaw; the wedge needs an absolute compass bearing.
                        let home_yaw = e.orientation.as_ref().map(|o| o.yaw_deg).unwrap_or(0.0);
                        let pan = e.status.as_ref().and_then(|s| s.pan_deg).unwrap_or(0.0);
                        let bearing = (home_yaw + pan).rem_euclid(360.0);
                        let effector_state = e
                            .status
                            .as_ref()
                            .map(|s| s.state.as_str())
                            .unwrap_or("offline");
                        set_effector_marker(&e.id, geo.lat, geo.lon, bearing, effector_state);
                    }
                }

                current_ids
            })
        });
    }

    // Future-modality placeholder layer. These artifacts are generated from
    // the local device registry plus explicit mock feeds, so the whole effect
    // can be replaced by backend-fed signals without changing map lifecycle.
    {
        Effect::new(move |prev_wedges: Option<HashSet<String>>| {
            let _ = theme.get();
            let registered_devices = devices.get();
            let rf_frames = rf_spectra.get();
            let seismic_trace_device_ids = seismic_traces
                .get()
                .into_iter()
                .map(|trace| trace.device_id)
                .collect::<HashSet<_>>();
            let speech_line_device_ids = speech_lines
                .get()
                .into_iter()
                .map(|line| line.device_id)
                .collect::<HashSet<_>>();

            let mut features = Vec::new();
            let mut current_wedges = HashSet::new();

            for device in &registered_devices {
                let Some(lat) = device.lat else {
                    continue;
                };
                let Some(lon) = device.lon else {
                    continue;
                };
                let active = match device.kind {
                    DeviceKind::Radar24G | DeviceKind::SdrRf => {
                        rf_frames.iter().any(|frame| frame.device_id == device.id)
                    }
                    DeviceKind::Seismic => seismic_trace_device_ids.contains(&device.id),
                    DeviceKind::SpeechNode => speech_line_device_ids.contains(&device.id),
                };

                features.push(serde_json::json!({
                    "type": "Feature",
                    "properties": {
                        "id": device.id,
                        "label": device.display_label(),
                        "kind": device.kind.as_str(),
                        "surface": "device",
                        "active": active,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                }));

                if device.kind == DeviceKind::Radar24G {
                    let wedge_id = format!("modality-radar:{}", device.id);
                    let bearing = device.yaw_deg.unwrap_or(0.0).rem_euclid(360.0);
                    set_bearing_wedge(&wedge_id, lat, lon, bearing, 24.0, 240.0);
                    current_wedges.insert(wedge_id);
                }
            }

            for frame in rf_frames {
                for emitter in frame.emitters {
                    let (Some(lat), Some(lon)) = (emitter.lat, emitter.lon) else {
                        continue;
                    };
                    features.push(serde_json::json!({
                        "type": "Feature",
                        "properties": {
                            "id": emitter.emitter_id,
                            "label": emitter.emitter_id,
                            "kind": "rf_emitter",
                            "surface": "emitter",
                            "rssi_dbm": emitter.rssi_dbm,
                            "drone_likelihood": emitter.drone_likelihood,
                            "source_device_id": frame.device_id,
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                    }));
                }
            }

            if features.is_empty() {
                remove_layer(FUTURE_MODALITIES_LAYER_ID);
            } else {
                let collection = serde_json::json!({
                    "type": "FeatureCollection",
                    "features": features,
                });
                if let (Ok(spec), Ok(data)) = (
                    serde_wasm_bindgen::to_value(&future_modalities_layer_spec()),
                    serde_wasm_bindgen::to_value(&collection),
                ) {
                    ensure_layer(FUTURE_MODALITIES_LAYER_ID, &spec);
                    set_layer_data(FUTURE_MODALITIES_LAYER_ID, &data);
                }
            }

            if let Some(ref previous) = prev_wedges {
                for id in previous.difference(&current_wedges) {
                    remove_bearing_wedge(id);
                }
            }

            current_wedges
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

    // Sync configured zones -> map polygons. Zones are persisted backend
    // spatial context; keeping them visible in the COP makes rule/alert
    // behavior legible before the full zone editor lands.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            zones.with(|current_zones| {
                let current_ids: HashSet<String> =
                    current_zones.iter().map(|zone| zone.id.clone()).collect();

                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_zone(id);
                    }
                }

                for zone in current_zones {
                    if zone.polygon_geo.len() >= 3 {
                        if let Ok(latlngs) = serde_wasm_bindgen::to_value(&zone.polygon_geo) {
                            set_zone(&zone.id, &latlngs, &zone.name);
                        }
                    }
                }

                current_ids
            })
        });
    }

    // Sync operator-imported map overlays. Images/SVGs need four geographic
    // corners; GeoJSON can render immediately because coordinates are already
    // geodetic.
    {
        Effect::new(move |prev_ids: Option<HashSet<String>>| {
            let _ = theme.get();
            overlays.with(|current_overlays| {
                let mut current_ids = HashSet::new();

                for overlay in current_overlays {
                    let layer_id = overlay_layer_id(&overlay.id);
                    if !overlay.enabled {
                        remove_map_overlay(&layer_id);
                        continue;
                    }

                    match overlay.kind.as_str() {
                        "image" | "svg" if overlay_has_four_corners(&overlay.bounds) => {
                            if let Ok(corners) = serde_wasm_bindgen::to_value(&overlay.bounds) {
                                set_image_overlay(
                                    &layer_id,
                                    &overlay.content_url,
                                    &corners,
                                    overlay.opacity,
                                );
                                current_ids.insert(layer_id);
                            }
                        }
                        "geojson" => {
                            current_ids.insert(layer_id.clone());
                            let url = overlay.content_url.clone();
                            let opacity = overlay.opacity;
                            spawn_local(async move {
                                match Request::get(&url).send().await {
                                    Ok(resp) if resp.ok() => {
                                        match resp.json::<serde_json::Value>().await {
                                            Ok(value) => {
                                                if let Some(collection) =
                                                    normalize_geojson_overlay(value)
                                                {
                                                    if let Ok(js_value) =
                                                        serde_wasm_bindgen::to_value(&collection)
                                                    {
                                                        set_geojson_overlay(
                                                            &layer_id, &js_value, opacity,
                                                        );
                                                    }
                                                }
                                            }
                                            Err(error) => {
                                                log::warn!("overlay GeoJSON parse failed: {error}");
                                            }
                                        }
                                    }
                                    Ok(resp) => {
                                        log::warn!("overlay GeoJSON fetch HTTP {}", resp.status());
                                    }
                                    Err(error) => {
                                        log::warn!("overlay GeoJSON fetch failed: {error}");
                                    }
                                }
                            });
                        }
                        _ => {
                            remove_map_overlay(&layer_id);
                        }
                    }
                }

                if let Some(ref prev) = prev_ids {
                    for id in prev.difference(&current_ids) {
                        remove_map_overlay(id);
                    }
                }

                current_ids
            })
        });
    }

    // Optional live density layer for recent geolocated detections. This uses
    // the existing analytics endpoint and keeps the heatmap off by default so
    // the main COP remains uncluttered until the operator asks for it.
    {
        Effect::new(move |_| {
            let enabled = live_heatmap_enabled.get();
            let window = live_heatmap_window.get();
            if !enabled {
                clear_heatmap();
                live_heatmap_bin_count.set(0);
                live_heatmap_error.set(None);
                return;
            }

            let _refresh_tick = live_heatmap_refresh_tick.get();
            live_heatmap_error.set(None);
            spawn_local(async move {
                match crate::api::fetch_heatmap(&window).await {
                    Ok(response) => {
                        if response.bins.is_empty() {
                            clear_heatmap();
                            live_heatmap_bin_count.set(0);
                        } else {
                            let max_weight = response
                                .bins
                                .iter()
                                .map(|bin| bin.weight)
                                .max()
                                .unwrap_or(1)
                                .max(1);
                            let points = heatmap_bins_to_js_points(&response.bins);
                            set_heatmap_points(&points, max_weight as f64);
                            live_heatmap_bin_count.set(response.bins.len());
                        }
                    }
                    Err(message) => {
                        clear_heatmap();
                        live_heatmap_bin_count.set(0);
                        live_heatmap_error.set(Some(message));
                    }
                }
            });
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
                    crate::state::CopItemKind::Alert
                    | crate::state::CopItemKind::Node
                    | crate::state::CopItemKind::Effector
                    | crate::state::CopItemKind::Zone => {}
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
        Effect::new(
            move |prev_state: Option<(HashSet<String>, HashSet<String>)>| {
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
                    let mut current_bearing_wedge_ids = HashSet::new();

                    let previous_bearing_wedge_ids = prev_state
                        .as_ref()
                        .map(|(_, prev_wedges)| prev_wedges.clone())
                        .unwrap_or_default();

                    if let Some((ref prev_markers, _)) = prev_state {
                        for id in prev_markers.difference(&current_ids) {
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
                                if let Some((source_lat, source_lon)) = source {
                                    let bearing_deg = bearing_degrees_between_points(
                                        source_lat, source_lon, geo.lat, geo.lon,
                                    );
                                    let range_m = distance_m_between_points(
                                        source_lat, source_lon, geo.lat, geo.lon,
                                    )
                                    .clamp(DEFAULT_BEARING_RANGE_M, 10_000.0);
                                    set_bearing_wedge(
                                        &d.event_id,
                                        source_lat,
                                        source_lon,
                                        bearing_deg,
                                        bearing_half_angle_degrees(d),
                                        range_m,
                                    );
                                    current_bearing_wedge_ids.insert(d.event_id.clone());
                                }
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

                    for id in previous_bearing_wedge_ids.difference(&current_bearing_wedge_ids) {
                        remove_bearing_wedge(id);
                    }

                    (current_ids, current_bearing_wedge_ids)
                })
            },
        );
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
