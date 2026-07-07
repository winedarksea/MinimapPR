use crate::map::bindings::{
    clear_detection_layer, remove_bearing_wedge, set_bearing_wedge, set_detection_layer_data,
};
use crate::state::{AppState, Detection};
use leptos::prelude::*;
use std::collections::HashSet;

const DEFAULT_BEARING_HALF_ANGLE_DEG: f64 = 15.0;
const DEFAULT_BEARING_RANGE_M: f64 = 500.0;
const EARTH_RADIUS_M: f64 = 6_371_000.0;

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

pub fn mount(state: &AppState) {
    let detections = state.detections;
    let nodes = state.nodes;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |previous_bearing_wedge_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().detections;
        detections.with(|ds| {
            let mut current_bearing_wedge_ids = HashSet::new();
            let mut features = Vec::new();

            let node_positions = nodes.with(|ns| {
                ns.iter()
                    .filter_map(|node| {
                        node.position_geo
                            .as_ref()
                            .map(|geo| (node.node_id.clone(), geo.lat, geo.lon))
                    })
                    .collect::<Vec<_>>()
            });
            if visible {
                for d in ds {
                    if d.track_id.is_some() || d.spatial_display_mode() == "node_only" {
                        continue;
                    }
                    if let Some(geo) = &d.position_geo {
                        let base_label = d.label.as_deref().unwrap_or("detection");
                        let display_mode = d.spatial_display_mode();
                        let mut source_lat = 0.0;
                        let mut source_lon = 0.0;
                        let mut has_source = false;
                        if d.spatial_display_mode() == "bearing_only" {
                            let source = d.node_id.as_ref().and_then(|node_id| {
                                node_positions
                                    .iter()
                                    .find(|(candidate, _, _)| candidate == node_id)
                                    .map(|(_, lat, lon)| (*lat, *lon))
                            });
                            (source_lat, source_lon, has_source) = source
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
                        }
                        features.push(serde_json::json!({
                            "type": "Feature",
                            "properties": {
                                "id": &d.event_id,
                                "label": base_label,
                                "confidence": d.confidence.or(d.label_confidence).unwrap_or(0.5),
                                "received_ns": d.received_ns.unwrap_or(i64::MIN),
                                "display_mode": display_mode,
                                "reporting_modality": d.reporting_modality.as_deref(),
                                "node_id": d.node_id.as_deref(),
                                "has_source": has_source,
                                "source_lat": source_lat,
                                "source_lon": source_lon,
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [geo.lon, geo.lat],
                            },
                        }));
                    }
                }
            }

            if !visible || features.is_empty() {
                clear_detection_layer();
            } else {
                let collection = serde_json::json!({
                    "type": "FeatureCollection",
                    "features": features,
                });
                if let Ok(data) = serde_wasm_bindgen::to_value(&collection) {
                    set_detection_layer_data(&data);
                }
            }

            if let Some(ref previous) = previous_bearing_wedge_ids {
                for id in previous.difference(&current_bearing_wedge_ids) {
                    remove_bearing_wedge(id);
                }
            }

            current_bearing_wedge_ids
        })
    });
}
