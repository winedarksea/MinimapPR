//! Field Mode bird entities on the map — spatial states only.
//!
//! Direction known: a bearing fan from the reporting node (annular sector when
//! the range band is known). Pinpointed: a point plus its uncertainty ellipse.
//! Heard-nearby entities are never drawn (they live in the tray), so an omni
//! detection can never read as "the bird is on top of that node".

use crate::map::bindings::{ensure_layer, json_value_to_js, remove_layer, set_layer_data};
use crate::state::entities::{BirdEntity, ENTITY_STATE_DIRECTION_KNOWN, ENTITY_STATE_PINPOINTED};
use crate::state::AppState;
use leptos::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

const AREA_LAYER_ID: &str = "bird-entity-areas";
const POINT_LAYER_ID: &str = "bird-entity-points";
const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Fan length when the range is unobservable (bearing-only solve).
const DEFAULT_FAN_LENGTH_M: f64 = 150.0;
const ARC_STEP_DEG: f64 = 3.0;
const MIN_ARC_STEPS: usize = 6;
const ELLIPSE_VERTEX_COUNT: usize = 36;
/// "Just sang" highlight duration after an absorbed bout.
pub const PULSE_DURATION_MS: f64 = 1_500.0;
const DIRECTION_KNOWN_COLOR: &str = "#f59e0b";
const PINPOINTED_COLOR: &str = "#22c55e";

thread_local! {
    static PULSE_STARTED_MS_BY_ENTITY: RefCell<HashMap<String, f64>> = RefCell::new(HashMap::new());
}

/// Called from the WS handler on an `absorbed` event.
pub fn note_bird_entity_pulse(entity_id: &str) {
    let now_ms = js_sys::Date::now();
    PULSE_STARTED_MS_BY_ENTITY.with(|pulses| {
        let mut pulses = pulses.borrow_mut();
        pulses.retain(|_, started_ms| now_ms - *started_ms < PULSE_DURATION_MS);
        pulses.insert(entity_id.to_string(), now_ms);
    });
}

fn is_pulsing(entity_id: &str) -> bool {
    let now_ms = js_sys::Date::now();
    PULSE_STARTED_MS_BY_ENTITY.with(|pulses| {
        pulses
            .borrow()
            .get(entity_id)
            .is_some_and(|started_ms| now_ms - *started_ms < PULSE_DURATION_MS)
    })
}

/// Destination [lon, lat] from (lat, lon) along a compass bearing (spherical earth).
fn destination_lon_lat(lat: f64, lon: f64, bearing_deg: f64, distance_m: f64) -> [f64; 2] {
    let angular = distance_m / EARTH_RADIUS_M;
    let bearing = bearing_deg.to_radians();
    let lat1 = lat.to_radians();
    let lon1 = lon.to_radians();
    let lat2 = (lat1.sin() * angular.cos() + lat1.cos() * angular.sin() * bearing.cos()).asin();
    let lon2 = lon1
        + (bearing.sin() * angular.sin() * lat1.cos()).atan2(angular.cos() - lat1.sin() * lat2.sin());
    [lon2.to_degrees(), lat2.to_degrees()]
}

fn arc(lat: f64, lon: f64, from_deg: f64, to_deg: f64, radius_m: f64) -> Vec<[f64; 2]> {
    let steps = (((to_deg - from_deg) / ARC_STEP_DEG).ceil() as usize).max(MIN_ARC_STEPS);
    (0..=steps)
        .map(|index| {
            let angle = from_deg + (to_deg - from_deg) * index as f64 / steps as f64;
            destination_lon_lat(lat, lon, angle, radius_m)
        })
        .collect()
}

/// Fan polygon ring; an annular sector when a range band is known.
pub fn bearing_fan_ring(
    lat: f64,
    lon: f64,
    bearing_deg: f64,
    half_width_deg: f64,
    range_min_m: Option<f64>,
    range_max_m: Option<f64>,
) -> Vec<[f64; 2]> {
    let from_deg = bearing_deg - half_width_deg;
    let to_deg = bearing_deg + half_width_deg;
    let outer_m = range_max_m.filter(|r| *r > 0.0).unwrap_or(DEFAULT_FAN_LENGTH_M);
    let mut ring = Vec::new();
    match range_min_m.filter(|r| *r > 0.0 && *r < outer_m) {
        Some(inner_m) => {
            ring.extend(arc(lat, lon, from_deg, to_deg, outer_m));
            let mut inner = arc(lat, lon, from_deg, to_deg, inner_m);
            inner.reverse();
            ring.extend(inner);
        }
        None => {
            ring.push([lon, lat]);
            ring.extend(arc(lat, lon, from_deg, to_deg, outer_m));
        }
    }
    if let Some(first) = ring.first().copied() {
        ring.push(first);
    }
    ring
}

/// Ellipse ring around (lat, lon); orientation is the major axis' compass bearing.
pub fn ellipse_ring(lat: f64, lon: f64, semi_major_m: f64, semi_minor_m: f64, orientation_deg: f64) -> Vec<[f64; 2]> {
    let mut ring: Vec<[f64; 2]> = (0..ELLIPSE_VERTEX_COUNT)
        .map(|index| {
            let t = std::f64::consts::TAU * index as f64 / ELLIPSE_VERTEX_COUNT as f64;
            let along = semi_major_m * t.cos();
            let across = semi_minor_m * t.sin();
            let distance_m = along.hypot(across);
            let angle_deg = orientation_deg + across.atan2(along).to_degrees();
            destination_lon_lat(lat, lon, angle_deg, distance_m)
        })
        .collect();
    if let Some(first) = ring.first().copied() {
        ring.push(first);
    }
    ring
}

fn area_feature(entity: &BirdEntity) -> Option<serde_json::Value> {
    let ring = match entity.state.as_str() {
        ENTITY_STATE_DIRECTION_KNOWN => {
            let bearing = entity.bearing.as_ref()?;
            let origin = bearing.origin_geo.as_ref()?;
            bearing_fan_ring(
                origin.lat,
                origin.lon,
                bearing.bearing_deg,
                bearing.half_width_deg,
                bearing.range_min_m,
                bearing.range_max_m,
            )
        }
        ENTITY_STATE_PINPOINTED => {
            let geo = entity.position_geo.as_ref()?;
            let ellipse = entity.uncertainty_ellipse.as_ref()?;
            ellipse_ring(geo.lat, geo.lon, ellipse.semi_major_m, ellipse.semi_minor_m, ellipse.orientation_deg)
        }
        _ => return None,
    };
    Some(serde_json::json!({
        "type": "Feature",
        "properties": {
            "id": entity.id,
            "state": entity.state,
            "label": entity.species_top_label,
            "color": if entity.state == ENTITY_STATE_PINPOINTED { PINPOINTED_COLOR } else { DIRECTION_KNOWN_COLOR },
        },
        "geometry": { "type": "Polygon", "coordinates": [ring] },
    }))
}

fn point_feature(entity: &BirdEntity) -> Option<serde_json::Value> {
    if entity.state != ENTITY_STATE_PINPOINTED {
        return None;
    }
    let geo = entity.position_geo.as_ref()?;
    Some(serde_json::json!({
        "type": "Feature",
        "properties": {
            "id": entity.id,
            "label": entity.species_top_label,
            "probability": entity.species_top_probability,
            "pulse": is_pulsing(&entity.id),
        },
        "geometry": { "type": "Point", "coordinates": [geo.lon, geo.lat] },
    }))
}

fn area_layer_spec() -> serde_json::Value {
    serde_json::json!({
        "type": "geojson",
        "layer": {
            "type": "fill",
            "paint": {
                "fill-color": ["get", "color"],
                "fill-opacity": 0.22,
                "fill-outline-color": ["get", "color"],
            }
        }
    })
}

fn point_layer_spec() -> serde_json::Value {
    serde_json::json!({
        "type": "geojson",
        "layer": {
            "type": "circle",
            "paint": {
                "circle-color": PINPOINTED_COLOR,
                "circle-radius": ["case", ["==", ["get", "pulse"], true], 11, 6],
                "circle-opacity": 0.9,
                "circle-stroke-color": "#0b1220",
                "circle-stroke-width": ["case", ["==", ["get", "pulse"], true], 3, 1.2],
            }
        }
    })
}

fn publish(layer_id: &str, spec: serde_json::Value, features: Vec<serde_json::Value>) {
    if features.is_empty() {
        remove_layer(layer_id);
        return;
    }
    let collection = serde_json::json!({ "type": "FeatureCollection", "features": features });
    ensure_layer(layer_id, &json_value_to_js(&spec));
    set_layer_data(layer_id, &json_value_to_js(&collection));
}

pub fn mount(state: &AppState) {
    let entities = state.bird_entities;
    let enabled = state.bird_entities_enabled;
    let map_layers = state.map_layers;
    let theme = state.theme;

    Effect::new(move |_| {
        let _ = theme.get();
        let visible = enabled.get() && map_layers.get().bird_entities;
        entities.with(|items| {
            let (areas, points) = if visible {
                (
                    items.iter().filter_map(area_feature).collect(),
                    items.iter().filter_map(point_feature).collect(),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            publish(AREA_LAYER_ID, area_layer_spec(), areas);
            publish(POINT_LAYER_ID, point_layer_spec(), points);
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_without_range_starts_and_ends_at_origin() {
        let ring = bearing_fan_ring(37.0, -122.0, 90.0, 10.0, None, None);
        assert_eq!(ring.first(), Some(&[-122.0, 37.0]));
        assert_eq!(ring.first(), ring.last());
        // Points head east of the origin.
        assert!(ring[1][0] > -122.0);
    }

    #[test]
    fn annular_fan_is_closed_and_ellipse_has_major_axis_north() {
        let ring = bearing_fan_ring(37.0, -122.0, 0.0, 10.0, Some(20.0), Some(60.0));
        assert_eq!(ring.first(), ring.last());
        let ellipse = ellipse_ring(37.0, -122.0, 30.0, 5.0, 0.0);
        let north_extent = ellipse.iter().map(|p| p[1]).fold(f64::MIN, f64::max) - 37.0;
        let east_extent = ellipse.iter().map(|p| p[0]).fold(f64::MIN, f64::max) + 122.0;
        assert!(north_extent > east_extent);
    }
}
