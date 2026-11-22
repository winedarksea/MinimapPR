//! In-memory bird-entity registry, the state machine (port of
//! `minimappr/core/bird_entity_registry.py`, S6 + S-fixes tracking).
//!
//! Sits on top of the tracker: tracks and their species belief decide *which
//! bird* a detection is; the registry turns that into one entity per bird
//! with a sliding per-entity absorption window. Spatial estimates, step-back
//! and maintenance live in [`super::registry_spatial`]. Synchronous, IO-free;
//! entity ids come from a counter (creation order = the reference's dict order).

use std::collections::BTreeMap;

use super::absorption_geometry::{concurrent_fix_conflict, remember_fix};
use super::config::BirdEntityConfig;
use super::matching::{spatial_mismatch_score, DetectionGeometry};
use super::model::{BirdEntityRecord, BirdEntitySpatialState, EntityFixSample, EntityId, SpatialDisplayMode};
use crate::wildlife_tracking::species_log_odds::{first_maximum, score_for, set_score, UNKNOWN_LABEL};
use crate::wildlife_tracking::wildlife_track::WildlifeTrack;
use crate::wildlife_tracking::{seconds_to_nanoseconds, Covariance3, Position3, TrackId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BirdEntityChangeKind {
    Created,
    Updated,
    Absorbed,
    StateChanged,
    Expired,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BirdEntityChange {
    pub kind: BirdEntityChangeKind,
    pub entity_id: EntityId,
    pub state: BirdEntitySpatialState,
    pub previous_state: Option<BirdEntitySpatialState>,
}

/// A surfaced detection as the registry sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct BirdEntityObservation<'track> {
    pub event_time_ns: i64,
    pub label: String,
    pub confidence: f64,
    pub spatial_display_mode: SpatialDisplayMode,
    pub reporting_node_id: Option<String>,
    pub reporting_node_position_m: Option<Position3>,
    pub detection_position_m: Option<Position3>,
    pub detection_covariance_m2: Option<Covariance3>,
    pub detection_id: Option<String>,
    pub track: Option<&'track WildlifeTrack>,
}

impl BirdEntityObservation<'_> {
    pub(super) fn geometry(&self) -> DetectionGeometry<'_> {
        DetectionGeometry {
            spatial_display_mode: self.spatial_display_mode,
            reporting_node_id: self.reporting_node_id.as_deref(),
            reporting_node_position_m: self.reporting_node_position_m.as_ref(),
            detection_position_m: self.detection_position_m.as_ref(),
            detection_covariance_m2: self.detection_covariance_m2.as_ref(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BirdEntityRegistry {
    pub(super) config: BirdEntityConfig,
    pub(super) entities: BTreeMap<EntityId, BirdEntityRecord>,
    last_entity_id: EntityId,
    absorption_window_ns: i64,
    pub(super) expiry_ns: i64,
    pub(super) spatial_hold_ns: i64,
}

pub(super) fn fix_sample(event_time_ns: i64, geometry: &DetectionGeometry<'_>) -> Option<EntityFixSample> {
    if geometry.spatial_display_mode != SpatialDisplayMode::Localized {
        return None;
    }
    Some(EntityFixSample {
        event_time_ns,
        position_m: *geometry.detection_position_m?,
        covariance_m2: geometry.detection_covariance_m2.copied(),
    })
}

/// The track's top species label (`species_top_label or label`).
pub(super) fn track_top_label(track: &WildlifeTrack) -> &str {
    track.species_top_label.as_deref().filter(|label| !label.is_empty()).unwrap_or(&track.label)
}

impl BirdEntityRegistry {
    pub fn new(config: BirdEntityConfig) -> Self {
        Self {
            config,
            entities: BTreeMap::new(),
            last_entity_id: 0,
            absorption_window_ns: seconds_to_nanoseconds(config.absorption_window_seconds),
            expiry_ns: seconds_to_nanoseconds(config.expiry_seconds),
            spatial_hold_ns: seconds_to_nanoseconds(config.spatial_state_hold_seconds),
        }
    }

    pub fn config(&self) -> &BirdEntityConfig {
        &self.config
    }

    /// Entities, most recently heard first.
    pub fn entities(&self) -> Vec<&BirdEntityRecord> {
        let mut entities: Vec<&BirdEntityRecord> = self.entities.values().collect();
        entities.sort_by(|first, second| second.last_heard_ns.cmp(&first.last_heard_ns));
        entities
    }

    pub fn get(&self, entity_id: EntityId) -> Option<&BirdEntityRecord> {
        self.entities.get(&entity_id)
    }

    pub fn get_mut(&mut self, entity_id: EntityId) -> Option<&mut BirdEntityRecord> {
        self.entities.get_mut(&entity_id)
    }

    pub fn entity_for_track(&self, track_id: Option<TrackId>) -> Option<EntityId> {
        let track_id = track_id?;
        self.entities.values().find(|entity| entity.track_id == Some(track_id)).map(|entity| entity.entity_id)
    }

    /// The entity this detection belongs to, or None (a new entity). A track
    /// bound to an entity wins outright; otherwise same species, heard within
    /// the window, spatially consistent and not contradicted by a concurrent
    /// fix; the lowest normalised mismatch wins.
    pub fn find_absorbing_entity(&self, label: &str, event_time_ns: i64, geometry: &DetectionGeometry<'_>, track_id: Option<TrackId>) -> Option<EntityId> {
        if let Some(bound) = self.entity_for_track(track_id) {
            return Some(bound);
        }
        let candidate_fix = fix_sample(event_time_ns, geometry);
        let mut best: Option<(EntityId, f64)> = None;
        for entity in self.entities.values() {
            if !entity.believes_label(label, self.config.absorption_min_species_belief)
                || event_time_ns - entity.last_heard_ns > self.absorption_window_ns
                || concurrent_fix_conflict(entity, candidate_fix.as_ref(), &self.config)
            {
                continue;
            }
            if let Some(score) = spatial_mismatch_score(entity, &self.config, geometry) {
                if best.is_none_or(|(_, best_score)| score < best_score) {
                    best = Some((entity.entity_id, score));
                }
            }
        }
        best.map(|(entity_id, _)| entity_id)
    }

    /// A bout that failed to localize joins its entity: recency + pulse, no demotion.
    pub fn absorb_unlocalized(
        &mut self,
        entity_id: EntityId,
        event_time_ns: i64,
        label: &str,
        confidence: f64,
        reporting_node_id: Option<&str>,
    ) -> Option<BirdEntityChange> {
        let entity = self.entities.get_mut(&entity_id)?;
        entity.last_heard_ns = entity.last_heard_ns.max(event_time_ns);
        entity.last_pulse_ns = entity.last_pulse_ns.max(event_time_ns);
        entity.absorbed_detection_count += 1;
        note_reporting_node(entity, reporting_node_id);
        if entity.track_id.is_none() {
            merge_detection_belief(entity, label, confidence);
        }
        Some(BirdEntityChange { kind: BirdEntityChangeKind::Absorbed, entity_id, state: entity.state, previous_state: Some(entity.state) })
    }

    pub fn observe(&mut self, observation: &BirdEntityObservation<'_>, preferred_entity_id: Option<EntityId>) -> BirdEntityChange {
        let mut track = observation.track;
        let geometry = observation.geometry();
        let candidate_fix = fix_sample(observation.event_time_ns, &geometry);
        let mut entity_id = preferred_entity_id.filter(|id| self.entities.contains_key(id));
        if entity_id.is_none() {
            let (bound, refused_concurrently) = self.entity_for_track_if_same_species(observation, candidate_fix.as_ref());
            entity_id = bound;
            if refused_concurrently {
                // The track carries another bird: this one stands on its own fix.
                track = None;
            }
        }
        let observation = BirdEntityObservation { track, ..observation.clone() };
        if entity_id.is_none() {
            entity_id = self.find_absorbing_entity(&observation.label, observation.event_time_ns, &observation.geometry(), None);
        }
        let estimate = self.observed_estimate(&observation);
        let Some(entity_id) = entity_id else {
            let created_id = self.create(&observation, estimate);
            let config = self.config;
            let created = self.entities.get_mut(&created_id).expect("just created");
            if let Some(sample) = candidate_fix {
                remember_fix(created, sample, &config);
            }
            return BirdEntityChange { kind: BirdEntityChangeKind::Created, entity_id: created_id, state: created.state, previous_state: None };
        };
        if let Some(track) = track {
            self.bind_track(entity_id, track.id);
        }
        let config = self.config;
        let entity = self.entities.get_mut(&entity_id).expect("entity exists");
        if let Some(sample) = candidate_fix {
            remember_fix(entity, sample, &config);
        }
        let previous_state = entity.state;
        entity.last_heard_ns = entity.last_heard_ns.max(observation.event_time_ns);
        entity.last_pulse_ns = entity.last_pulse_ns.max(observation.event_time_ns);
        entity.detection_count += 1;
        if observation.reporting_node_id.is_some() {
            entity.reporting_node_id.clone_from(&observation.reporting_node_id);
        }
        if observation.reporting_node_position_m.is_some() {
            entity.reporting_node_position_m = observation.reporting_node_position_m;
        }
        note_reporting_node(entity, observation.reporting_node_id.as_deref());
        if entity.canonical_detection_id.is_none() {
            entity.canonical_detection_id.clone_from(&observation.detection_id);
        }
        apply_species(entity, &observation);
        if estimate.state.rank() >= entity.state.rank() {
            apply_estimate(entity, &estimate, observation.event_time_ns);
        }
        if entity.state != previous_state {
            entity.last_state_change_ns = observation.event_time_ns;
            return BirdEntityChange { kind: BirdEntityChangeKind::StateChanged, entity_id, state: entity.state, previous_state: Some(previous_state) };
        }
        BirdEntityChange { kind: BirdEntityChangeKind::Updated, entity_id, state: entity.state, previous_state: Some(previous_state) }
    }

    /// (the track's entity, refused-for-a-concurrent-fix). A track that merged
    /// two birds must not drag the entity: follow it when the entity believes
    /// the label or the track's top species, never against a moments-old fix.
    fn entity_for_track_if_same_species(&self, observation: &BirdEntityObservation<'_>, candidate_fix: Option<&EntityFixSample>) -> (Option<EntityId>, bool) {
        let Some(track) = observation.track else { return (None, false) };
        let Some(entity_id) = self.entity_for_track(Some(track.id)) else { return (None, false) };
        let entity = &self.entities[&entity_id];
        if concurrent_fix_conflict(entity, candidate_fix, &self.config) {
            return (None, true);
        }
        let floor = self.config.absorption_min_species_belief;
        if entity.believes_label(&observation.label, floor)
            || (!track.species_belief.is_empty() && entity.believes_label(track_top_label(track), floor))
        {
            return (Some(entity_id), false);
        }
        (None, false)
    }

    fn create(&mut self, observation: &BirdEntityObservation<'_>, estimate: super::registry_spatial::ObservedSpatialEstimate) -> EntityId {
        if self.entities.len() >= self.config.max_tracked_entities {
            let oldest = self.entities.values().fold(None::<&BirdEntityRecord>, |oldest, entity| match oldest {
                Some(current) if current.last_heard_ns <= entity.last_heard_ns => Some(current),
                _ => Some(entity),
            });
            if let Some(oldest_id) = oldest.map(|entity| entity.entity_id) {
                self.entities.remove(&oldest_id);
            }
        }
        self.last_entity_id += 1;
        let entity_id = self.last_entity_id;
        let mut entity = BirdEntityRecord {
            entity_id,
            state: BirdEntitySpatialState::HeardNearby,
            species_top_label: observation.label.clone(),
            species_top_probability: observation.confidence,
            species_belief: vec![(observation.label.clone(), observation.confidence)],
            first_heard_ns: observation.event_time_ns,
            last_heard_ns: observation.event_time_ns,
            reporting_node_id: observation.reporting_node_id.clone(),
            reporting_node_position_m: observation.reporting_node_position_m,
            reporting_node_ids: Vec::new(),
            position_m: None,
            position_covariance_m2: None,
            horizontal_uncertainty_m: None,
            bearing: None,
            track_id: None,
            canonical_detection_id: observation.detection_id.clone(),
            last_spatial_update_ns: None,
            last_state_change_ns: observation.event_time_ns,
            detection_count: 1,
            absorbed_detection_count: 0,
            last_pulse_ns: observation.event_time_ns,
            recent_fixes: Vec::new(),
        };
        note_reporting_node(&mut entity, observation.reporting_node_id.as_deref());
        self.entities.insert(entity_id, entity);
        if let Some(track) = observation.track {
            self.bind_track(entity_id, track.id);
        }
        let entity = self.entities.get_mut(&entity_id).expect("just inserted");
        apply_species(entity, observation);
        apply_estimate(entity, &estimate, observation.event_time_ns);
        entity_id
    }

    /// A track belongs to one entity; a stale binding elsewhere is released.
    fn bind_track(&mut self, entity_id: EntityId, track_id: TrackId) {
        if let Some(previous_owner) = self.entity_for_track(Some(track_id)).filter(|owner| *owner != entity_id) {
            self.entities.get_mut(&previous_owner).expect("owner exists").track_id = None;
        }
        if let Some(entity) = self.entities.get_mut(&entity_id) {
            entity.track_id = Some(track_id);
        }
    }
}

pub(super) fn apply_estimate(entity: &mut BirdEntityRecord, estimate: &super::registry_spatial::ObservedSpatialEstimate, event_time_ns: i64) {
    if estimate.state == BirdEntitySpatialState::HeardNearby {
        return;
    }
    entity.state = estimate.state;
    entity.position_m = estimate.position_m;
    entity.position_covariance_m2 = estimate.covariance_m2;
    entity.horizontal_uncertainty_m = estimate.horizontal_uncertainty_m;
    entity.bearing.clone_from(&estimate.bearing);
    entity.last_spatial_update_ns = Some(event_time_ns);
}

fn apply_species(entity: &mut BirdEntityRecord, observation: &BirdEntityObservation<'_>) {
    match observation.track {
        Some(track) if !track.species_belief.is_empty() => refresh_species_from_track(entity, track),
        Some(track) if track.label != UNKNOWN_LABEL => {
            set_score(&mut entity.species_belief, &track.label, track.confidence);
            entity.species_top_label.clone_from(&track.label);
            entity.species_top_probability = track.confidence;
        }
        _ if entity.track_id.is_none() => merge_detection_belief(entity, &observation.label, observation.confidence),
        _ => {}
    }
}

/// Copy a log-odds track's belief onto its entity.
pub fn refresh_species_from_track(entity: &mut BirdEntityRecord, track: &WildlifeTrack) {
    if track.species_belief.is_empty() {
        return;
    }
    entity.species_belief = track.species_belief.clone();
    entity.species_top_label = track_top_label(track).to_string();
    entity.species_top_probability = track.species_top_probability.filter(|probability| *probability != 0.0).unwrap_or(track.confidence);
}

/// Without a tracker belief, keep each species' best score; the top is the max.
fn merge_detection_belief(entity: &mut BirdEntityRecord, label: &str, confidence: f64) {
    let best = score_for(&entity.species_belief, label).unwrap_or(0.0).max(confidence);
    set_score(&mut entity.species_belief, label, best);
    if let Some((top_label, top_probability)) = first_maximum(&entity.species_belief).map(|(l, p)| (l.to_string(), p)) {
        entity.species_top_label = top_label;
        entity.species_top_probability = top_probability;
    }
}

fn note_reporting_node(entity: &mut BirdEntityRecord, node_id: Option<&str>) {
    if let Some(node_id) = node_id.filter(|node_id| !node_id.is_empty()) {
        if !entity.reporting_node_ids.iter().any(|existing| existing == node_id) {
            entity.reporting_node_ids.push(node_id.to_string());
        }
    }
}

