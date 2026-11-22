//! Field Mode bird entities (`GET /api/v1/entities`, `{"type":"entity"}` on
//! `/ws/live`). One bird is one entity that moves Heard nearby -> Direction
//! known -> Pinpointed; the tray lists heard-nearby entities and the map draws
//! only the spatial ones (fans and points), never an omni pin.

use crate::state::GeoPoint;
use serde::Deserialize;

pub const ENTITY_STATE_HEARD_NEARBY: &str = "heard_nearby";
pub const ENTITY_STATE_DIRECTION_KNOWN: &str = "direction_known";
pub const ENTITY_STATE_PINPOINTED: &str = "pinpointed";

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BirdEntityBearing {
    pub origin_node_id: String,
    pub origin_geo: Option<GeoPoint>,
    pub bearing_deg: f64,
    pub half_width_deg: f64,
    pub range_min_m: Option<f64>,
    pub range_max_m: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BirdEntityUncertaintyEllipse {
    pub semi_major_m: f64,
    pub semi_minor_m: f64,
    pub orientation_deg: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct BirdEntity {
    pub id: String,
    pub state: String,
    pub species_top_label: String,
    pub species_top_probability: f64,
    pub reporting_node_id: Option<String>,
    #[serde(default)]
    pub reporting_node_ids: Vec<String>,
    pub position_geo: Option<GeoPoint>,
    pub horizontal_uncertainty_m: Option<f64>,
    pub uncertainty_ellipse: Option<BirdEntityUncertaintyEllipse>,
    pub bearing: Option<BirdEntityBearing>,
    pub track_id: Option<String>,
    pub first_heard_ns: i64,
    pub last_heard_ns: i64,
    pub last_pulse_ns: i64,
    #[serde(default)]
    pub detection_count: u32,
    #[serde(default)]
    pub absorbed_detection_count: u32,
}

impl BirdEntity {
    pub fn is_heard_nearby(&self) -> bool {
        self.state == ENTITY_STATE_HEARD_NEARBY
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct BirdEntityEvent {
    pub event: String,
    pub entity: BirdEntity,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BirdEntityListResponse {
    pub enabled: bool,
    #[serde(default)]
    pub entities: Vec<BirdEntity>,
}

/// Apply one live event to the entity list (upsert, or remove on expiry).
pub fn apply_bird_entity_event(entities: &mut Vec<BirdEntity>, event: BirdEntityEvent) {
    entities.retain(|existing| existing.id != event.entity.id);
    if event.event != "expired" {
        entities.push(event.entity);
    }
    entities.sort_by(|a, b| b.last_heard_ns.cmp(&a.last_heard_ns));
}
