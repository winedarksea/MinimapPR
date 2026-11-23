//! Seeding the registry with entities another host already published: a
//! successor mesh host takes over from its event-log mirror, so the birds the
//! dead host was tracking keep their entities (and wire ids) instead of each
//! next call creating a duplicate. Only what the wire carries is restored:
//! state, position, a covariance rebuilt from the published contour, species
//! belief and when the bird was last heard. Tracks are not: the successor's
//! perched tracker starts empty and binds on the next fix.

use super::model::{BirdEntityRecord, BirdEntitySpatialState, EntityId};
use super::registry::BirdEntityRegistry;
use crate::wildlife_tracking::species_log_odds::{first_maximum, LabelScores};
use crate::wildlife_tracking::{Covariance3, Position3};

/// One published entity as a successor rebuilds it.
#[derive(Debug, Clone, PartialEq)]
pub struct SeededBirdEntity {
    pub state: BirdEntitySpatialState,
    pub position_m: Option<Position3>,
    pub covariance_m2: Option<Covariance3>,
    pub species_belief: LabelScores,
    pub last_heard_ns: i64,
}

impl BirdEntityRegistry {
    /// Insert a restored entity under a fresh local id (returned).
    pub fn seed_entity(&mut self, seed: &SeededBirdEntity) -> EntityId {
        self.last_entity_id += 1;
        let entity_id = self.last_entity_id;
        let (top_label, top_probability) = first_maximum(&seed.species_belief).map_or((String::new(), 0.0), |(label, score)| (label.to_string(), score));
        // A spatial state needs its geometry; without it the bird is only heard.
        let state = if seed.position_m.is_some() && seed.covariance_m2.is_some() { seed.state } else { BirdEntitySpatialState::HeardNearby };
        let spatial = state != BirdEntitySpatialState::HeardNearby;
        let entity = BirdEntityRecord {
            entity_id,
            state,
            species_top_label: top_label,
            species_top_probability: top_probability,
            species_belief: seed.species_belief.clone(),
            first_heard_ns: seed.last_heard_ns,
            last_heard_ns: seed.last_heard_ns,
            reporting_node_id: None,
            reporting_node_position_m: None,
            reporting_node_ids: Vec::new(),
            position_m: spatial.then_some(seed.position_m).flatten(),
            position_covariance_m2: spatial.then_some(seed.covariance_m2).flatten(),
            horizontal_uncertainty_m: spatial.then(|| super::model::horizontal_major_sigma_m(seed.covariance_m2.as_ref())).flatten(),
            bearing: None,
            track_id: None,
            canonical_detection_id: None,
            last_spatial_update_ns: spatial.then_some(seed.last_heard_ns),
            last_state_change_ns: seed.last_heard_ns,
            detection_count: 1,
            absorbed_detection_count: 0,
            last_pulse_ns: seed.last_heard_ns,
            recent_fixes: Vec::new(),
        };
        self.entities.insert(entity_id, entity);
        entity_id
    }

    /// Move every entity's timestamps by `delta_ns` (a successor learned the
    /// seeded times were on another reference clock).
    pub fn shift_entity_times(&mut self, delta_ns: i64) {
        for entity in self.entities.values_mut() {
            entity.first_heard_ns += delta_ns;
            entity.last_heard_ns += delta_ns;
            entity.last_state_change_ns += delta_ns;
            entity.last_pulse_ns += delta_ns;
            entity.last_spatial_update_ns = entity.last_spatial_update_ns.map(|time| time + delta_ns);
            for fix in &mut entity.recent_fixes {
                fix.event_time_ns += delta_ns;
            }
        }
    }
}
