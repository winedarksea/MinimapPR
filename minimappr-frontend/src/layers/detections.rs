use crate::map::bindings::{
    clear_detection_layer, remove_bearing_wedge, set_bearing_wedge, set_detection_layer_data,
};
use crate::state::{AppState, Detection, GeoPoint};
use leptos::prelude::*;
use std::collections::HashSet;

const DEFAULT_BEARING_HALF_ANGLE_DEG: f64 = 15.0;
const DEFAULT_BEARING_RANGE_M: f64 = 500.0;
const MAX_BEARING_RANGE_M: f64 = 10_000.0;
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

/// Half angle for a detection without a stored solve bearing (Python-ingest
/// path, rows stored before the bearing was): a guess from range observability.
fn legacy_bearing_half_angle_degrees(detection: &Detection) -> f64 {
    detection
        .localization_range_observability
        .map(|observability| {
            let clamped = observability.clamp(0.0, 1.0);
            30.0 - (clamped * 20.0)
        })
        .unwrap_or(DEFAULT_BEARING_HALF_ANGLE_DEG)
        .clamp(5.0, 45.0)
}

struct BearingWedge {
    source_lat: f64,
    source_lon: f64,
    bearing_deg: f64,
    half_angle_deg: f64,
    range_m: f64,
}

fn wedge_range_m(source_lat: f64, source_lon: f64, target: &GeoPoint) -> f64 {
    distance_m_between_points(source_lat, source_lon, target.lat, target.lon)
        .clamp(DEFAULT_BEARING_RANGE_M, MAX_BEARING_RANGE_M)
}

/// The stored solve bearing (origin = contributing-sensor centroid, width from
/// the covariance), else the reporting node toward the projected point.
fn bearing_wedge(
    detection: &Detection,
    position_geo: &GeoPoint,
    node_positions: &[(String, f64, f64)],
) -> Option<BearingWedge> {
    if let Some(bearing) = &detection.solve_bearing {
        if let Some(origin) = &bearing.origin_geo {
            return Some(BearingWedge {
                source_lat: origin.lat,
                source_lon: origin.lon,
                bearing_deg: bearing.bearing_deg,
                half_angle_deg: bearing.half_width_deg,
                range_m: bearing
                    .range_max_m
                    .filter(|range_m| bearing.range_observable && *range_m > 0.0)
                    .unwrap_or_else(|| wedge_range_m(origin.lat, origin.lon, position_geo)),
            });
        }
    }
    let node_id = detection.node_id.as_ref()?;
    let (_, source_lat, source_lon) = node_positions
        .iter()
        .find(|(candidate, _, _)| candidate == node_id)?;
    Some(BearingWedge {
        source_lat: *source_lat,
        source_lon: *source_lon,
        bearing_deg: bearing_degrees_between_points(
            *source_lat,
            *source_lon,
            position_geo.lat,
            position_geo.lon,
        ),
        half_angle_deg: legacy_bearing_half_angle_degrees(detection),
        range_m: wedge_range_m(*source_lat, *source_lon, position_geo),
    })
}

pub fn mount(state: &AppState) {
    let detections = state.detections;
    let nodes = state.nodes;
    let map_layers = state.map_layers;
    let theme = state.theme;
    let bird_entities = state.bird_entities;
    let bird_entities_enabled = state.bird_entities_enabled;

    Effect::new(move |previous_bearing_wedge_ids: Option<HashSet<String>>| {
        let _ = theme.get();
        let visible = map_layers.get().detections;
        // Entity mode: a species the entity layer already draws is not drawn
        // again as a raw detection (one bird, one thing on screen).
        let entity_species: std::collections::HashSet<String> = if bird_entities_enabled.get() {
            bird_entities.with(|entities| {
                entities.iter().map(|e| e.species_top_label.to_lowercase()).collect()
            })
        } else {
            std::collections::HashSet::new()
        };
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
                    let drawn_as_entity = d
                        .label
                        .as_deref()
                        .is_some_and(|label| entity_species.contains(&label.to_lowercase()));
                    if d.track_id.is_some() || d.spatial_display_mode() == "node_only" || drawn_as_entity {
                        continue;
                    }
                    if let Some(geo) = &d.position_geo {
                        let base_label = d.label.as_deref().unwrap_or("detection");
                        let display_mode = d.spatial_display_mode();
                        let mut source_lat = 0.0;
                        let mut source_lon = 0.0;
                        let mut has_source = false;
                        if display_mode == "bearing_only" {
                            if let Some(wedge) = bearing_wedge(d, geo, &node_positions) {
                                (source_lat, source_lon, has_source) =
                                    (wedge.source_lat, wedge.source_lon, true);
                                set_bearing_wedge(
                                    &d.event_id,
                                    wedge.source_lat,
                                    wedge.source_lon,
                                    wedge.bearing_deg,
                                    wedge.half_angle_deg,
                                    wedge.range_m,
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
                set_detection_layer_data(&crate::map::bindings::json_value_to_js(&collection));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(feature_summary: serde_json::Value) -> Detection {
        serde_json::from_value(serde_json::json!({
            "id": "det-1",
            "source_node_id": "tetra-d",
            "reporting_modality": "localized",
            "spatial_display_mode": "bearing_only",
            "position_geo": {"lat": 37.001, "lon": -122.0, "alt_m": 10.0},
            "feature_summary": feature_summary,
        }))
        .expect("detection")
    }

    #[test]
    fn stored_solve_bearing_sets_the_wedge_origin_direction_and_width() {
        let d = detection(serde_json::json!({
            "localization_solve_bearing": {
                "origin_node_id": "tetra-d",
                "origin_position_m": [1.0, 2.0, 3.0],
                "origin_geo": {"lat": 37.0, "lon": -122.0001, "alt_m": 3.0},
                "bearing_deg": 56.5,
                "half_width_deg": 6.8,
                "elevation_deg": 20.0,
                "range_min_m": null,
                "range_max_m": null,
                "range_projection_mode": "range_boundary",
                "range_observable": false
            }
        }));
        let node_positions = vec![("tetra-d".to_string(), 36.9, -121.9)];
        let wedge = bearing_wedge(&d, d.position_geo.as_ref().unwrap(), &node_positions).unwrap();
        assert_eq!((wedge.source_lat, wedge.source_lon), (37.0, -122.0001));
        assert_eq!(wedge.bearing_deg, 56.5);
        assert_eq!(wedge.half_angle_deg, 6.8);
        assert_eq!(wedge.range_m, DEFAULT_BEARING_RANGE_M);
    }

    #[test]
    fn without_a_stored_bearing_the_wedge_starts_at_the_reporting_node() {
        let d = detection(serde_json::json!({"localization_range_observability": 0.0}));
        let node_positions = vec![("tetra-d".to_string(), 37.0, -122.0)];
        let wedge = bearing_wedge(&d, d.position_geo.as_ref().unwrap(), &node_positions).unwrap();
        assert_eq!((wedge.source_lat, wedge.source_lon), (37.0, -122.0));
        assert!(wedge.bearing_deg.abs() < 1e-6);
        assert_eq!(wedge.half_angle_deg, 30.0);
    }
}
