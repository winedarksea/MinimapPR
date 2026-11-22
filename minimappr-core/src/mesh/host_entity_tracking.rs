//! Host-side entity tracking: localized call records → the ported server
//! track (perched tracker + bird-entity registry, `bird_entity::entity_pipeline`)
//! → `BirdEntityUpdate`s with stable ids. One pipeline per island, because
//! each island has its own map frame and reference clock.
//!
//! Time: entity logic runs on the island reference clock (the call centre).
//! Maintenance needs a "now" on that clock, so each island remembers its
//! latest event and the session millisecond it arrived at, and extrapolates
//! (the server coordinator does the same with wall time).

use std::collections::{BTreeMap, VecDeque};

use crate::bird_entity::entity_pipeline::{EntityPipelineDetection, EntityPipelineOutcome, WildlifeEntityPipeline};
use crate::bird_entity::model::{BirdEntityRecord, BirdEntitySpatialState, EntityId, SpatialDisplayMode};
use crate::bird_entity::registry::{BirdEntityChange, BirdEntityChangeKind};
use crate::bird_entity::reporting_decision::EntityReportingAction;
use crate::protocol::detection_messages::{BirdEntityUpdate, EntityBearing};
use crate::wildlife_tracking::perched_relocation::RelocationRevert;
use crate::protocol::wire_codes::{BirdEntityState, SpeciesScore};
use crate::protocol::NodeId;
use crate::wildlife_tracking::species_log_odds::{first_maximum, LabelScores, UNKNOWN_LABEL};
use crate::wildlife_tracking::{Position3, NANOSECONDS_PER_SECOND};

use super::call_pipeline::{CallFix, LocalizedCallRecord};
use super::config::{EntityContourConfig, EntityTrackingConfig, DEVICE_SAMPLE_RATE_HZ};
use super::fix_contour::horizontal_covariance_contour;
use super::MeshMilliseconds;

const NANOSECONDS_PER_MILLISECOND: i64 = 1_000_000;

/// What happened to one call in the entity layer (for assertions and the UI).
#[derive(Debug, Clone, PartialEq)]
pub struct HostEntityCallOutcome {
    pub call_id: u64,
    /// Which source cluster of the call (0 for an unlocalized call).
    pub cluster_index: usize,
    pub island_id: u32,
    pub bird_entity_id: Option<u64>,
    pub action: EntityReportingAction,
    pub change: Option<BirdEntityChangeKind>,
    pub track_id: Option<u64>,
    /// Event time on the island reference clock.
    pub event_time_ns: i64,
    /// Moved off a track whose perched hop was later undone (see `repoint_reverted_calls`).
    pub repointed_by_revert: bool,
}

/// Shared inputs for building wire updates.
pub struct EntityUpdateContext<'host> {
    pub contour: &'host EntityContourConfig,
    pub next_event_log_sequence: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HostEntityTracking {
    pub pipelines: BTreeMap<u32, WildlifeEntityPipeline>,
    wire_ids: BTreeMap<(u32, EntityId), u64>,
    /// Per island: (latest event ns on the reference clock, session ms it arrived).
    clock_anchors: BTreeMap<u32, (i64, MeshMilliseconds)>,
    last_maintenance_ms: Option<MeshMilliseconds>,
    /// Newest last; capped at `EntityTrackingConfig::max_retained_call_outcomes`.
    pub call_outcomes: VecDeque<HostEntityCallOutcome>,
}

/// Where a cluster's detection is reported from: its own loudest node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterReportingNode {
    pub node_id: NodeId,
    pub position_m: Option<Position3>,
}

fn species_evidence(scores: &[SpeciesScore]) -> LabelScores {
    scores.iter().map(|score| (score.label_id.to_string(), f64::from(score.score))).collect()
}

/// The cluster's top label (unknown with no species: the position still feeds the tracker).
fn detection_for(
    detection_id: String,
    event_time_ns: i64,
    evidence: LabelScores,
    fix: Option<&CallFix>,
    reporting_node_id: Option<String>,
    reporting_node_position_m: Option<Position3>,
) -> EntityPipelineDetection {
    let (label, confidence) =
        first_maximum(&evidence).map_or((UNKNOWN_LABEL.to_string(), 0.0), |(label, score)| (label.to_string(), score));
    EntityPipelineDetection {
        detection_id,
        event_time_ns,
        label,
        confidence,
        species_evidence: evidence,
        spatial_display_mode: if fix.is_some() { SpatialDisplayMode::Localized } else { SpatialDisplayMode::NodeOnly },
        reporting_node_id,
        reporting_node_position_m,
        position_m: fix.map(|fix| fix.position_m),
        covariance_m2: fix.map(|fix| fix.covariance_m2),
        range_projection_mode: None,
    }
}

/// Do these node positions lie close to one line? Then a TDOA fix has a mirror
/// twin across that line the solver's local covariance cannot show (the test
/// strategy's collinear trail): minor/major horizontal spread below `max_aspect_ratio`.
pub fn node_set_is_collinear(positions_m: &[Position3], max_aspect_ratio: f64) -> bool {
    if positions_m.len() < 2 {
        return true;
    }
    let count = positions_m.len() as f64;
    let mean = [0, 1].map(|axis| positions_m.iter().map(|p| p[axis]).sum::<f64>() / count);
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for position in positions_m {
        let (dx, dy) = (position[0] - mean[0], position[1] - mean[1]);
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let half_trace = 0.5 * (xx + yy);
    let spread = (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let (major, minor) = (half_trace + spread, (half_trace - spread).max(0.0));
    major <= 0.0 || (minor / major).sqrt() < max_aspect_ratio
}

fn fix_is_finite(fix: &CallFix) -> bool {
    fix.position_m.iter().chain(fix.covariance_m2.iter().flatten()).all(|value| value.is_finite())
}

fn wire_state(state: BirdEntitySpatialState) -> BirdEntityState {
    match state {
        BirdEntitySpatialState::HeardNearby => BirdEntityState::HeardNearby,
        BirdEntitySpatialState::DirectionKnown => BirdEntityState::DirectionKnown,
        BirdEntitySpatialState::Pinpointed => BirdEntityState::Pinpointed,
    }
}

impl HostEntityTracking {
    pub fn wire_id_of(&self, island_id: u32, entity_id: EntityId) -> Option<u64> {
        self.wire_ids.get(&(island_id, entity_id)).copied()
    }

    /// Feed one completed call: each localized source cluster (or the
    /// unlocalized call once) becomes a detection; returns the updates to log.
    pub fn on_call_record(
        &mut self,
        record: &LocalizedCallRecord,
        reporting_node_position_m: Option<Position3>,
        cluster_reporting_nodes: &[ClusterReportingNode],
        mirror_ambiguous_clusters: &[bool],
        config: &EntityTrackingConfig,
        update_context: &EntityUpdateContext<'_>,
        next_bird_entity_id: &mut u64,
        now_ms: MeshMilliseconds,
    ) -> Vec<BirdEntityUpdate> {
        let event_time_ns = (record.centre_reference_s * NANOSECONDS_PER_SECOND as f64).round() as i64;
        let anchor = self.clock_anchors.entry(record.island_id).or_insert((event_time_ns, now_ms));
        if event_time_ns >= anchor.0 {
            *anchor = (event_time_ns, now_ms);
        }
        let reporting_node_id = Some(record.classifying_node.0.to_string());
        let detections: Vec<EntityPipelineDetection> = if record.clusters.is_empty() {
            let evidence = species_evidence(&record.species_belief);
            vec![detection_for(format!("call-{}", record.call_id), event_time_ns, evidence, None, reporting_node_id, reporting_node_position_m)]
        } else {
            record
                .clusters
                .iter()
                .enumerate()
                .map(|(index, cluster)| {
                    let species = cluster.attributed_species.as_deref().unwrap_or(&record.species_belief);
                    // A mirror-ambiguous fix places the bird on one of two sides, and a
                    // non-finite one would poison the tracker's filter: heard, not placed.
                    let placed = !mirror_ambiguous_clusters.get(index).copied().unwrap_or(false) && fix_is_finite(&cluster.fix);
                    // Each cluster is reported from its own loudest node (the server's
                    // `loudest_node_id`), which anchors its bearing fan on step-back.
                    let (cluster_node_id, cluster_node_position_m) = cluster_reporting_nodes
                        .get(index)
                        .map_or((reporting_node_id.clone(), reporting_node_position_m), |node| {
                            (Some(node.node_id.0.to_string()), node.position_m)
                        });
                    detection_for(
                        format!("call-{}-{index}", record.call_id),
                        event_time_ns,
                        species_evidence(species),
                        placed.then_some(&cluster.fix),
                        cluster_node_id,
                        cluster_node_position_m,
                    )
                })
                .collect()
        };
        let mut updates = Vec::new();
        for (cluster_index, detection) in detections.iter().enumerate() {
            if let Some(update) = self.process(record, cluster_index, detection, config, update_context, next_bird_entity_id) {
                updates.push(update);
            }
        }
        updates
    }

    fn process(
        &mut self,
        record: &LocalizedCallRecord,
        cluster_index: usize,
        detection: &EntityPipelineDetection,
        config: &EntityTrackingConfig,
        update_context: &EntityUpdateContext<'_>,
        next_bird_entity_id: &mut u64,
    ) -> Option<BirdEntityUpdate> {
        let pipeline = self
            .pipelines
            .entry(record.island_id)
            .or_insert_with(|| WildlifeEntityPipeline::new(config.tracker, config.entity));
        let outcome: EntityPipelineOutcome = pipeline.process(detection);
        let change = outcome.change.clone();
        let wire_id = change.as_ref().map(|change| self.wire_id_for(record.island_id, change.entity_id, next_bird_entity_id));
        if let Some(revert) = outcome.track_update.and_then(|update| update.relocation_revert) {
            self.repoint_reverted_calls(record.island_id, revert);
        }
        let track_id = outcome.track_update.map(|update| update.track_id);
        if let (Some(track_id), Some(wire_id)) = (track_id, wire_id) {
            self.resolve_repointed_calls(record.island_id, track_id, wire_id);
        }
        // Bounded history: revert re-pointing only needs calls since a hop
        // (the ghost window), and the shell persists outcomes it wants to keep.
        while !self.call_outcomes.is_empty() && self.call_outcomes.len() >= config.max_retained_call_outcomes {
            self.call_outcomes.pop_front();
        }
        self.call_outcomes.push_back(HostEntityCallOutcome {
            call_id: record.call_id,
            cluster_index,
            island_id: record.island_id,
            event_time_ns: detection.event_time_ns,
            bird_entity_id: wire_id,
            action: outcome.decision.action,
            change: change.as_ref().map(|change| change.kind),
            track_id,
            repointed_by_revert: false,
        });
        let change = change?;
        let entity = self.pipelines[&record.island_id].registry.get(change.entity_id)?;
        Some(self.wire_update(record.island_id, entity, wire_id?, None, true, update_context, config))
    }

    /// Periodic maintenance: step-backs and expiries become wire updates.
    pub fn maintain(&mut self, now_ms: MeshMilliseconds, config: &EntityTrackingConfig, update_context: &EntityUpdateContext<'_>) -> Vec<BirdEntityUpdate> {
        let due = self.last_maintenance_ms.is_none_or(|last| now_ms.saturating_sub(last) >= config.maintenance_interval_ms);
        if !due {
            return Vec::new();
        }
        self.last_maintenance_ms = Some(now_ms);
        let mut updates = Vec::new();
        let islands: Vec<u32> = self.pipelines.keys().copied().collect();
        for island_id in islands {
            let Some((anchor_ns, anchor_ms)) = self.clock_anchors.get(&island_id).copied() else { continue };
            let elapsed_ns = i64::try_from(now_ms.saturating_sub(anchor_ms)).unwrap_or(i64::MAX / NANOSECONDS_PER_MILLISECOND) * NANOSECONDS_PER_MILLISECOND;
            let island_now_ns = anchor_ns.saturating_add(elapsed_ns);
            let pipeline = self.pipelines.get_mut(&island_id).expect("listed above");
            let snapshots: BTreeMap<EntityId, BirdEntityRecord> =
                pipeline.registry.entities().into_iter().map(|entity| (entity.entity_id, entity.clone())).collect();
            let changes: Vec<BirdEntityChange> = pipeline.maintain(island_now_ns);
            for change in changes {
                let Some(wire_id) = self.wire_id_of(island_id, change.entity_id) else { continue };
                let expired = change.kind == BirdEntityChangeKind::Expired;
                let entity = if expired {
                    snapshots.get(&change.entity_id).cloned()
                } else {
                    self.pipelines[&island_id].registry.get(change.entity_id).cloned()
                };
                let Some(entity) = entity else { continue };
                let state_override = expired.then_some(BirdEntityState::Expired);
                updates.push(self.wire_update(island_id, &entity, wire_id, state_override, false, update_context, config));
                if expired {
                    self.wire_ids.remove(&(island_id, change.entity_id));
                }
            }
        }
        // Entities the registry evicted at capacity (LRU) never expire: drop
        // their wire ids too, or the map grows with every evicted bird.
        let pipelines = &self.pipelines;
        self.wire_ids.retain(|(island_id, entity_id), _| {
            pipelines.get(island_id).is_some_and(|pipeline| pipeline.registry.get(*entity_id).is_some())
        });
        updates
    }

    /// A perched hop was undone: the calls the track took since the hop were
    /// another bird's (the split track). They wait for that track's entity.
    /// (The server keeps such rows on the old track: S-fixes open issue.)
    fn repoint_reverted_calls(&mut self, island_id: u32, revert: RelocationRevert) {
        for call in self.call_outcomes.iter_mut().filter(|call| {
            call.island_id == island_id && call.track_id == Some(revert.track_id) && call.event_time_ns >= revert.hop_time_ns
        }) {
            call.track_id = Some(revert.split_track_id);
            call.bird_entity_id = None;
            call.repointed_by_revert = true;
        }
    }

    fn resolve_repointed_calls(&mut self, island_id: u32, track_id: u64, wire_id: u64) {
        for call in self.call_outcomes.iter_mut().filter(|call| {
            call.island_id == island_id && call.repointed_by_revert && call.bird_entity_id.is_none() && call.track_id == Some(track_id)
        }) {
            call.bird_entity_id = Some(wire_id);
        }
    }

    fn wire_id_for(&mut self, island_id: u32, entity_id: EntityId, next_bird_entity_id: &mut u64) -> u64 {
        *self.wire_ids.entry((island_id, entity_id)).or_insert_with(|| {
            *next_bird_entity_id += 1;
            *next_bird_entity_id
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn wire_update(
        &self,
        island_id: u32,
        entity: &BirdEntityRecord,
        wire_id: u64,
        state_override: Option<BirdEntityState>,
        just_sang: bool,
        update_context: &EntityUpdateContext<'_>,
        config: &EntityTrackingConfig,
    ) -> BirdEntityUpdate {
        let contour = match (entity.state, entity.position_m, entity.position_covariance_m2) {
            (BirdEntitySpatialState::Pinpointed, Some(position), Some(covariance)) => {
                horizontal_covariance_contour(position, &covariance, update_context.contour)
            }
            _ => Vec::new(),
        };
        let bearing = entity.bearing.as_ref().and_then(|bearing| {
            Some(EntityBearing {
                from_node_id: NodeId(bearing.origin_node_id.parse().ok()?),
                azimuth_deg: bearing.bearing_deg as f32,
                azimuth_standard_deviation_deg: (bearing.half_width_deg / config.entity.uncertainty_sigma_multiplier) as f32,
            })
        });
        BirdEntityUpdate {
            bird_entity_id: wire_id,
            event_log_sequence: update_context.next_event_log_sequence,
            state: state_override.unwrap_or_else(|| wire_state(entity.state)),
            island_id,
            position_east_north_up_m: entity.position_m.map(|position| position.map(|value| value as f32)),
            bearing,
            likelihood_contour_east_north_m: contour,
            species_belief: entity
                .species_belief
                .iter()
                .filter_map(|(label, score)| Some(SpeciesScore { label_id: label.parse().ok()?, score: *score as f32 }))
                .collect(),
            just_sang,
            last_heard_reference_sample_index: (entity.last_heard_ns.max(0) as f64 / NANOSECONDS_PER_SECOND as f64 * DEVICE_SAMPLE_RATE_HZ) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::call_pipeline::LocalizedCluster;

    fn record(call_id: u64, centre_reference_s: f64, position_m: [f64; 3]) -> LocalizedCallRecord {
        let fix = CallFix {
            position_m,
            covariance_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
            horizontal_major_sigma_m: 1.0,
            used_nodes: Vec::new(),
            rejected_nodes: Vec::new(),
        };
        LocalizedCallRecord {
            call_id,
            centre_reference_s,
            classifying_node: NodeId(1),
            species_belief: vec![SpeciesScore { label_id: 7, score: 0.9 }],
            fix: Some(fix.clone()),
            clusters: vec![LocalizedCluster::single(fix, NodeId(1))],
            ..LocalizedCallRecord::default()
        }
    }

    #[test]
    fn call_outcome_history_is_capped_and_non_finite_fixes_are_not_placed() {
        let config = EntityTrackingConfig { max_retained_call_outcomes: 3, ..EntityTrackingConfig::default() };
        let contour = EntityContourConfig::default();
        let context = EntityUpdateContext { contour: &contour, next_event_log_sequence: 1 };
        let mut tracking = HostEntityTracking::default();
        let mut next_id = 0;
        for call_id in 0..5 {
            tracking.on_call_record(&record(call_id, 10.0 * call_id as f64, [5.0, 5.0, 3.0]), None, &[], &[], &config, &context, &mut next_id, 0);
        }
        assert_eq!(tracking.call_outcomes.iter().map(|call| call.call_id).collect::<Vec<_>>(), vec![2, 3, 4]);
        assert!(tracking.call_outcomes.iter().all(|call| call.bird_entity_id == Some(1)), "one bird, one entity");

        let poisoned = record(9, 60.0, [f64::NAN, 5.0, 3.0]);
        let updates = tracking.on_call_record(&poisoned, None, &[], &[], &config, &context, &mut next_id, 0);
        assert!(updates.iter().all(|update| update.position_east_north_up_m.is_none_or(|p| p.iter().all(|v| v.is_finite()))));
        assert!(tracking.pipelines[&0].tracker.tracks().all(|track| track.position_m.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn single_file_nodes_are_collinear_and_a_ring_is_not() {
        let trail: Vec<[f64; 3]> = (0..5).map(|index| [8.0 * f64::from(index), 0.3 * f64::from(index % 2), 1.4]).collect();
        assert!(node_set_is_collinear(&trail, 0.2));
        let ring: Vec<[f64; 3]> = (0..5).map(|index| {
            let angle = std::f64::consts::TAU * f64::from(index) / 5.0;
            [12.0 * angle.cos(), 12.0 * angle.sin(), 1.4]
        }).collect();
        assert!(!node_set_is_collinear(&ring, 0.2));
    }
}
