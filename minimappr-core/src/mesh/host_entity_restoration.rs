//! A successor host seeds its entity layer from the dead host's mirror
//! (closeout C5), so a bird heard before and after a takeover keeps one
//! entity and one wire id. No wire change: each entity comes back from its
//! latest `BirdEntityUpdate`, the covariance rebuilt from the published
//! contour.
//!
//! Caveats, by design: entities are keyed by island id, so a successor whose
//! island tracker numbers the group differently starts that island empty;
//! a bearing-only entity comes back as heard-nearby (the bearing's origin
//! position is not on the wire).

use std::collections::BTreeMap;

use crate::bird_entity::entity_pipeline::WildlifeEntityPipeline;
use crate::bird_entity::model::BirdEntitySpatialState;
use crate::bird_entity::registry_seed::SeededBirdEntity;
use crate::protocol::detection_messages::BirdEntityUpdate;
use crate::protocol::session_messages::ClockModel;
use crate::protocol::wire_codes::BirdEntityState;
use crate::protocol::NodeId;
use crate::wildlife_tracking::{Covariance3, NANOSECONDS_PER_SECOND};

use super::clock_mapping::NodeClockMapping;
use super::config::{EntityContourConfig, EntityTrackingConfig, FixCovarianceCalibrationConfig, DEVICE_SAMPLE_RATE_HZ};
use super::host_entity_tracking::HostEntityTracking;
use super::MeshMilliseconds;

/// Contour vertices are sampled uniformly in the ellipse's parametric
/// angle, so their second moment about the centre is k²/2 · Σ.
const ELLIPSE_SECOND_MOMENT_FACTOR: f64 = 2.0;
const MINIMUM_CONTOUR_VERTICES: usize = 3;

/// The published horizontal covariance, back out of its contour polygon,
/// returned in raw fix units: divided by the inside-layout calibration
/// scale² and with the map anchor's share left in. For an inside-layout
/// entity both err wide, which is right here: the successor's fresh anchor
/// puts new fixes metres off the dead host's frame. An outside-layout entity
/// was published at the smaller outside scale, so its fix share comes back
/// at (outside / inside)² ≈ 0.2× the raw variance (too tight; only the
/// anchor share widens it). The wire does not say which scale was used.
/// Height variance takes the horizontal major axis.
fn covariance_from_contour(
    centre_m: [f64; 2],
    contour_m: &[[f32; 2]],
    contour: &EntityContourConfig,
    calibration: &FixCovarianceCalibrationConfig,
) -> Option<Covariance3> {
    if contour_m.len() < MINIMUM_CONTOUR_VERTICES || contour.contour_sigma_multiplier <= 0.0 {
        return None;
    }
    let count = contour_m.len() as f64;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for vertex in contour_m {
        let (dx, dy) = (f64::from(vertex[0]) - centre_m[0], f64::from(vertex[1]) - centre_m[1]);
        xx += dx * dx / count;
        xy += dx * dy / count;
        yy += dy * dy / count;
    }
    let to_raw = ELLIPSE_SECOND_MOMENT_FACTOR / (contour.contour_sigma_multiplier * calibration.inside_layout_sigma_scale).powi(2);
    let (xx, xy, yy) = (xx * to_raw, xy * to_raw, yy * to_raw);
    let major = 0.5 * (xx + yy) + (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let covariance = [[xx, xy, 0.0], [xy, yy, 0.0], [0.0, 0.0, major]];
    covariance.iter().flatten().all(|value| value.is_finite()).then_some(covariance)
}

fn seeded_from_update(update: &BirdEntityUpdate, contour: &EntityContourConfig, calibration: &FixCovarianceCalibrationConfig) -> SeededBirdEntity {
    let position_m = update.position_east_north_up_m.map(|position| position.map(f64::from));
    let covariance_m2 = position_m.and_then(|position| covariance_from_contour([position[0], position[1]], &update.likelihood_contour_east_north_m, contour, calibration));
    SeededBirdEntity {
        state: match update.state {
            BirdEntityState::Pinpointed => BirdEntitySpatialState::Pinpointed,
            _ => BirdEntitySpatialState::HeardNearby,
        },
        position_m,
        covariance_m2,
        species_belief: update.species_belief.iter().map(|score| (score.label_id.to_string(), f64::from(score.score))).collect(),
        last_heard_ns: (update.last_heard_reference_sample_index as f64 / DEVICE_SAMPLE_RATE_HZ * NANOSECONDS_PER_SECOND as f64).round() as i64,
    }
}

impl HostEntityTracking {
    /// Seed every island's pipeline from the mirror's latest entity updates,
    /// keeping their wire ids. Each island's clock anchor is its latest
    /// last-heard time at `now_ms` (the takeover), which can only make the
    /// first maintenance see the birds as fresher than they are, never stale.
    pub fn restore_entities(
        &mut self,
        updates: &BTreeMap<(u32, u64), BirdEntityUpdate>,
        clock_models: &BTreeMap<u32, ClockModel>,
        config: &EntityTrackingConfig,
        contour: &EntityContourConfig,
        calibration: &FixCovarianceCalibrationConfig,
        now_ms: MeshMilliseconds,
    ) {
        for ((island_id, wire_id), update) in updates {
            let seed = seeded_from_update(update, contour, calibration);
            let pipeline = self.pipelines.entry(*island_id).or_insert_with(|| WildlifeEntityPipeline::new(config.tracker, config.entity));
            let entity_id = pipeline.seed_entity(&seed);
            self.wire_ids.insert((*island_id, entity_id), *wire_id);
            let anchor = self.clock_anchors.entry(*island_id).or_insert((seed.last_heard_ns, now_ms));
            anchor.0 = anchor.0.max(seed.last_heard_ns);
            if let Some(model) = clock_models.get(island_id) {
                self.pending_clock_rebases.insert(*island_id, model.clone());
            }
        }
    }

    /// Call before the island's first call on this host. Entity times were
    /// on the dead host's reference clock (`ClockModel::reference_node_id`);
    /// if this host references another node, map them through that node's
    /// clock in the dead host's model. A reference the old model never saw
    /// cannot be related, and the times are left as they are.
    pub fn rebase_restored_island(&mut self, island_id: u32, reference_node: Option<NodeId>) {
        let (Some(reference_node), Some(model)) = (reference_node, self.pending_clock_rebases.get(&island_id)) else { return };
        let model = model.clone();
        self.pending_clock_rebases.remove(&island_id);
        if model.reference_node_id == reference_node {
            return;
        }
        let Some(estimate) = model.node_clocks.iter().find(|estimate| estimate.node_id == reference_node) else { return };
        let Some(anchor) = self.clock_anchors.get_mut(&island_id) else { return };
        let mapping = NodeClockMapping::from_wire(model.clock_epoch, estimate);
        let old_seconds = anchor.0 as f64 / NANOSECONDS_PER_SECOND as f64;
        let delta_ns = ((mapping.node_seconds_from_reference_seconds(old_seconds) - old_seconds) * NANOSECONDS_PER_SECOND as f64).round() as i64;
        anchor.0 += delta_ns;
        if let Some(pipeline) = self.pipelines.get_mut(&island_id) {
            pipeline.registry.shift_entity_times(delta_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::fix_contour::horizontal_covariance_contour;
    use crate::protocol::session_messages::NodeClockEstimate;
    use crate::protocol::wire_codes::SpeciesScore;

    fn update(wire_id: u64, last_heard_s: f64, covariance: &[[f64; 3]; 3], contour: &EntityContourConfig, calibration: &FixCovarianceCalibrationConfig) -> BirdEntityUpdate {
        let published = covariance.map(|row| row.map(|value| value * calibration.inside_layout_sigma_scale.powi(2)));
        BirdEntityUpdate {
            bird_entity_id: wire_id,
            event_log_sequence: 7,
            state: BirdEntityState::Pinpointed,
            island_id: 1,
            position_east_north_up_m: Some([10.0, -4.0, 6.0]),
            bearing: None,
            likelihood_contour_east_north_m: horizontal_covariance_contour([10.0, -4.0, 6.0], &published, contour),
            species_belief: vec![SpeciesScore { label_id: 103, score: 0.8 }],
            just_sang: true,
            last_heard_reference_sample_index: (last_heard_s * DEVICE_SAMPLE_RATE_HZ) as u64,
        }
    }

    #[test]
    fn seeded_entity_keeps_its_wire_id_position_and_covariance() {
        let (config, contour, calibration) = (EntityTrackingConfig::default(), EntityContourConfig::default(), FixCovarianceCalibrationConfig::default());
        let covariance = [[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 9.0]];
        let updates = BTreeMap::from([((1, 42), update(42, 20.0, &covariance, &contour, &calibration))]);
        let mut tracking = HostEntityTracking::default();
        tracking.restore_entities(&updates, &BTreeMap::new(), &config, &contour, &calibration, 30_000);
        let entity = tracking.pipelines[&1].registry.entities()[0].clone();
        assert_eq!(tracking.wire_id_of(1, entity.entity_id), Some(42));
        assert_eq!(entity.state, BirdEntitySpatialState::Pinpointed);
        assert_eq!(entity.species_top_label, "103");
        let restored = entity.position_covariance_m2.expect("rebuilt from the contour");
        for (row, column) in [(0, 0), (0, 1), (1, 1)] {
            // f32 contour vertices: the round trip holds to ~1e-5 relative.
            assert!((restored[row][column] - covariance[row][column]).abs() < 1e-3, "{restored:?}");
        }
    }

    #[test]
    fn times_move_onto_the_new_reference_clock_at_the_first_call() {
        let (config, contour, calibration) = (EntityTrackingConfig::default(), EntityContourConfig::default(), FixCovarianceCalibrationConfig::default());
        let covariance = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let updates = BTreeMap::from([((1, 5), update(5, 20.0, &covariance, &contour, &calibration))]);
        // Node 3 ran 2.5 s ahead of the dead host's reference (node 1).
        let model = ClockModel {
            island_id: 1,
            clock_epoch: 4,
            reference_node_id: NodeId(1),
            node_clocks: vec![NodeClockEstimate { node_id: NodeId(3), anchor_node_sample_index: 0, offset_samples: 2.5 * DEVICE_SAMPLE_RATE_HZ, rate_offset_ppm: 0.0, offset_standard_deviation_samples: 1.0 }],
        };
        let mut tracking = HostEntityTracking::default();
        tracking.restore_entities(&updates, &BTreeMap::from([(1, model)]), &config, &contour, &calibration, 30_000);
        tracking.rebase_restored_island(1, Some(NodeId(3)));
        let last_heard_ns = tracking.pipelines[&1].registry.entities()[0].last_heard_ns;
        assert!((last_heard_ns - 22_500_000_000).abs() < 1_000, "{last_heard_ns}");
        // Only once: a later call must not shift again.
        tracking.rebase_restored_island(1, Some(NodeId(3)));
        assert_eq!(tracking.pipelines[&1].registry.entities()[0].last_heard_ns, last_heard_ns);
    }
}
