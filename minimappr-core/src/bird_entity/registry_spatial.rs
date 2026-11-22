//! Spatial half of the entity registry: what an observation says about where
//! the bird is, bearing fans, following the bound track, step-back and
//! maintenance (expiry, track loss, uncertainty growth, staleness).

use std::collections::BTreeMap;

use super::model::{
    bearing_estimate_toward, horizontal_major_sigma_m, BearingFanLimits, BirdEntityBearingEstimate, BirdEntityRecord,
    BirdEntitySpatialState, SpatialDisplayMode,
};
use super::registry::{refresh_species_from_track, BirdEntityChange, BirdEntityChangeKind, BirdEntityObservation, BirdEntityRegistry};
use crate::wildlife_tracking::wildlife_track::WildlifeTrack;
use crate::wildlife_tracking::{Covariance3, Position3, TrackId};

#[derive(Debug, Clone, PartialEq)]
pub struct ObservedSpatialEstimate {
    pub state: BirdEntitySpatialState,
    pub position_m: Option<Position3>,
    pub covariance_m2: Option<Covariance3>,
    pub horizontal_uncertainty_m: Option<f64>,
    pub bearing: Option<BirdEntityBearingEstimate>,
}

impl ObservedSpatialEstimate {
    fn heard_nearby() -> Self {
        Self { state: BirdEntitySpatialState::HeardNearby, position_m: None, covariance_m2: None, horizontal_uncertainty_m: None, bearing: None }
    }
}

impl BirdEntityRegistry {
    fn fan_limits(&self) -> BearingFanLimits {
        BearingFanLimits {
            default_half_width_deg: self.config.bearing_default_half_width_deg,
            min_half_width_deg: self.config.bearing_min_half_width_deg,
            max_half_width_deg: self.config.bearing_max_half_width_deg,
            sigma_multiplier: self.config.uncertainty_sigma_multiplier,
        }
    }

    fn bearing(&self, node_id: Option<&str>, node_position_m: &Position3, target_m: &Position3, covariance_m2: Option<&Covariance3>, range_observable: bool) -> BirdEntityBearingEstimate {
        bearing_estimate_toward(node_id.unwrap_or(""), node_position_m, target_m, covariance_m2, range_observable, &self.fan_limits())
    }

    pub(super) fn observed_estimate(&self, observation: &BirdEntityObservation<'_>) -> ObservedSpatialEstimate {
        if let Some(track) = observation.track {
            return self.estimate_from_position(observation, &track.position_m, Some(&track.position_covariance_m2));
        }
        match (observation.spatial_display_mode, observation.detection_position_m) {
            (SpatialDisplayMode::Localized, Some(position_m)) => {
                self.estimate_from_position(observation, &position_m, observation.detection_covariance_m2.as_ref())
            }
            (SpatialDisplayMode::BearingOnly, Some(position_m)) if observation.reporting_node_position_m.is_some() => {
                let node_position_m = observation.reporting_node_position_m.expect("checked");
                ObservedSpatialEstimate {
                    state: BirdEntitySpatialState::DirectionKnown,
                    position_m: Some(position_m),
                    covariance_m2: observation.detection_covariance_m2,
                    horizontal_uncertainty_m: None,
                    bearing: Some(self.bearing(
                        observation.reporting_node_id.as_deref(),
                        &node_position_m,
                        &position_m,
                        observation.detection_covariance_m2.as_ref(),
                        false,
                    )),
                }
            }
            _ => ObservedSpatialEstimate::heard_nearby(),
        }
    }

    fn estimate_from_position(&self, observation: &BirdEntityObservation<'_>, position_m: &Position3, covariance_m2: Option<&Covariance3>) -> ObservedSpatialEstimate {
        let sigma_m = horizontal_major_sigma_m(covariance_m2);
        let too_uncertain = sigma_m.is_some_and(|sigma_m| sigma_m > self.config.pinpointed_max_horizontal_uncertainty_m);
        if let (true, Some(node_position_m)) = (too_uncertain, observation.reporting_node_position_m.as_ref()) {
            return ObservedSpatialEstimate {
                state: BirdEntitySpatialState::DirectionKnown,
                position_m: Some(*position_m),
                covariance_m2: covariance_m2.copied(),
                horizontal_uncertainty_m: sigma_m,
                bearing: Some(self.bearing(observation.reporting_node_id.as_deref(), node_position_m, position_m, covariance_m2, true)),
            };
        }
        ObservedSpatialEstimate {
            state: BirdEntitySpatialState::Pinpointed,
            position_m: Some(*position_m),
            covariance_m2: covariance_m2.copied(),
            horizontal_uncertainty_m: sigma_m,
            bearing: None,
        }
    }

    /// Expiry and step-back. `visible_tracks` None = tracker unknown (skip track checks).
    pub fn maintain(&mut self, now_ns: i64, visible_tracks: Option<&BTreeMap<TrackId, WildlifeTrack>>) -> Vec<BirdEntityChange> {
        let mut changes = Vec::new();
        let entity_ids: Vec<_> = self.entities.keys().copied().collect();
        for entity_id in entity_ids {
            let entity = &self.entities[&entity_id];
            if now_ns - entity.last_heard_ns > self.expiry_ns {
                let state = entity.state;
                self.entities.remove(&entity_id);
                changes.push(BirdEntityChange { kind: BirdEntityChangeKind::Expired, entity_id, state, previous_state: Some(state) });
                continue;
            }
            let previous_state = entity.state;
            let mut entity = self.entities.remove(&entity_id).expect("present");
            if let (Some(track_id), Some(tracks)) = (entity.track_id, visible_tracks) {
                match tracks.get(&track_id).filter(|track| !track.is_dropped()) {
                    Some(track) => self.refresh_from_track(&mut entity, track),
                    None => {
                        entity.track_id = None;
                        self.step_back(&mut entity);
                    }
                }
            }
            let hold_reference_ns = entity.last_spatial_update_ns.unwrap_or(0).max(entity.last_state_change_ns);
            if entity.is_spatial() && now_ns - hold_reference_ns > self.spatial_hold_ns {
                self.step_back(&mut entity);
            }
            if entity.state != previous_state {
                entity.last_state_change_ns = now_ns;
                changes.push(BirdEntityChange { kind: BirdEntityChangeKind::StateChanged, entity_id, state: entity.state, previous_state: Some(previous_state) });
            }
            self.entities.insert(entity_id, entity);
        }
        changes
    }

    /// Follow the bound track; uncertainty past the pinpoint limit demotes to a fan.
    fn refresh_from_track(&self, entity: &mut BirdEntityRecord, track: &WildlifeTrack) {
        refresh_species_from_track(entity, track);
        if !entity.is_spatial() {
            return;
        }
        entity.position_m = Some(track.position_m);
        entity.position_covariance_m2 = Some(track.position_covariance_m2);
        entity.horizontal_uncertainty_m = horizontal_major_sigma_m(Some(&track.position_covariance_m2));
        let too_uncertain = entity.horizontal_uncertainty_m.is_some_and(|sigma_m| sigma_m > self.config.pinpointed_max_horizontal_uncertainty_m);
        if entity.state == BirdEntitySpatialState::Pinpointed && too_uncertain {
            self.step_back(entity);
        }
    }

    fn step_back(&self, entity: &mut BirdEntityRecord) {
        let target = entity.state.one_step_back();
        if target == BirdEntitySpatialState::DirectionKnown {
            if let (Some(position_m), Some(node_position_m)) = (entity.position_m, entity.reporting_node_position_m) {
                entity.bearing = Some(self.bearing(
                    entity.reporting_node_id.as_deref(),
                    &node_position_m,
                    &position_m,
                    entity.position_covariance_m2.as_ref(),
                    true,
                ));
                entity.state = BirdEntitySpatialState::DirectionKnown;
                return;
            }
        }
        // Heard nearby: the map is spatial-only, so the entity drops its geometry.
        entity.state = BirdEntitySpatialState::HeardNearby;
        entity.position_m = None;
        entity.position_covariance_m2 = None;
        entity.horizontal_uncertainty_m = None;
        entity.bearing = None;
    }
}
