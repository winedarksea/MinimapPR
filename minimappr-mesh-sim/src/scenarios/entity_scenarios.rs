//! Server-track scenarios (W4-port-server-track; test strategy Layer 2 table):
//! the host runs the ported perched tracker, log-odds, entity registry,
//! multi-cluster windows with echo rejection, and S5 attribution. Each checks
//! the common invariants plus the entity metrics: no duplicate entities, no
//! merged birds (swaps), no wrong labels, no echo promotions.

use minimappr_core::bird_entity::reporting_decision::EntityReportingAction;
use minimappr_core::bird_entity::registry::BirdEntityChangeKind;

use crate::acoustic_world::{NoiseInterval, NoiseKind};
use crate::device_spec::DeviceSpec;
use crate::entity_metrics::{entity_invariants, entity_metrics, EntityMetrics, EntityTruth};
use crate::invariants::{ensure, fixes_within_reported_region};
use crate::scenario::{ring_of_phones, ScenarioSpec, DEFAULT_CALL_LEVEL_DB_SPL};
use crate::scenario_outcome::ScenarioOutcome;

use super::call_scenarios::{healthy_path_checks, INSIDE_LAYOUT_GATE_M, REPORTED_REGION_FLOOR_M, REPORTED_REGION_SIGMA_MULTIPLIER};
use super::{report, run_checked, ScenarioReport};

const CARDINAL: u32 = 101;
const CHICKADEE: u32 = 102;
const ROBIN: u32 = 103;
const TANAGER: u32 = 104;
const WREN: u32 = 105;
const CONFIDENT_SCORE: f32 = 0.85;
const PHONE_HEIGHT_M: f64 = 1.4;
/// Trunk reflections: this far behind the perch (image source at twice it), this much quieter.
const TRUNK_DISTANCE_M: f64 = 2.0;
const ECHO_LEVEL_DROP_DB: f64 = 8.0;

/// A scenario spec plus which bird sang each scripted call.
struct EntityScene {
    spec: ScenarioSpec,
    truth: EntityTruth,
}

impl EntityScene {
    fn new(name: &'static str, duration_s: f64, devices: Vec<DeviceSpec>) -> Self {
        Self { spec: ScenarioSpec::new(name, duration_s, devices), truth: EntityTruth::default() }
    }

    fn bird(&mut self, species: u32) -> usize {
        self.truth.add_bird(species)
    }

    fn bout(mut self, bird: usize, time_s: f64, position_m: [f64; 3], score: f32, level_db: f64) -> Self {
        let species = self.truth.bird_species[bird];
        self.spec = self.spec.call_bird(time_s, position_m, species, score, level_db);
        self.truth.note_call(Some(bird));
        self
    }

    /// A trunk reflection of a bout: image source behind the perch, later and quieter.
    fn echo(mut self, bird: usize, time_s: f64, perch_m: [f64; 3], away_from: [f64; 2], level_db: f64) -> Self {
        let (dx, dy) = (perch_m[0] - away_from[0], perch_m[1] - away_from[1]);
        let norm = dx.hypot(dy).max(f64::EPSILON);
        let image = [perch_m[0] + 2.0 * TRUNK_DISTANCE_M * dx / norm, perch_m[1] + 2.0 * TRUNK_DISTANCE_M * dy / norm, perch_m[2]];
        let species = self.truth.bird_species[bird];
        self.spec = self.spec.call_bird(time_s, image, species, CONFIDENT_SCORE, level_db - ECHO_LEVEL_DROP_DB);
        self.truth.note_call(None);
        self
    }
}

/// Developer aid: `MINIMAPPR_SIM_ENTITY_DEBUG=1` prints each call's clusters and entity outcome.
fn debug_dump(outcome: &ScenarioOutcome) {
    if std::env::var_os("MINIMAPPR_SIM_ENTITY_DEBUG").is_none() {
        return;
    }
    for trace in &outcome.traces {
        eprintln!("trace call {} truth {} fix {:?}", trace.host_call_id, trace.true_call_index, trace.fix.as_ref().map(|fix| (fix.position_m, fix.horizontal_major_sigma_m, fix.used_nodes.len())));
    }
    for record in outcome.call_records() {
        let clusters: Vec<_> = record.clusters.iter().map(|c| (c.classifying_node.0, c.attribution_tier, c.attributed_species.clone())).collect();
        eprintln!("record {} t {:.2} belief {:?} clusters {clusters:?} failure {:?}", record.call_id, record.centre_reference_s, record.species_belief, record.failure);
    }
    for call in host_call_outcomes(outcome) {
        eprintln!("outcome {call:?}");
    }
    for record in outcome.host_log_records() {
        if let minimappr_core::protocol::FieldMessage::BirdEntityUpdate(update) = record {
            eprintln!("update {} {:?} {:?} bearing {:?}", update.bird_entity_id, update.state, update.position_east_north_up_m, update.bearing);
        }
    }
}

fn run_scene(scene: EntityScene, seed: u64) -> Result<(ScenarioOutcome, EntityMetrics), String> {
    let EntityScene { spec, truth } = scene;
    let outcome = run_checked(spec, seed)?;
    debug_dump(&outcome);
    let metrics = entity_metrics(&outcome, &truth);
    Ok((outcome, metrics))
}

fn host_call_outcomes(outcome: &ScenarioOutcome) -> Vec<minimappr_core::mesh::host_entity_tracking::HostEntityCallOutcome> {
    outcome.all_hosts().into_iter().flat_map(|(_, host)| host.entities.call_outcomes.clone()).collect()
}

/// Two cardinals 22 m apart answering each other, twice within one window.
pub fn countersinging_pair(seed: u64) -> Result<ScenarioReport, String> {
    let mut scene = EntityScene::new("countersinging_pair", 60.0, ring_of_phones(6, [0.0, 0.0], 16.0, seed));
    let (west, east) = (scene.bird(CARDINAL), scene.bird(CARDINAL));
    let (west_perch, east_perch) = ([-11.0, 2.0, 6.0], [11.0, -2.0, 6.0]);
    for (index, start_s) in [20.0, 26.0, 32.0, 38.0].into_iter().enumerate() {
        let answer_delay_s = if index % 2 == 0 { 0.5 } else { 1.6 };
        scene = scene
            .bout(west, start_s, west_perch, CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL)
            .bout(east, start_s + answer_delay_s, east_perch, CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL);
    }
    report("countersinging_pair", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        ensure(metrics.pinpointed_birds == 2, || format!("pinpointed birds {}: {}", metrics.pinpointed_birds, metrics.summary()))?;
        let worst = healthy_path_checks(&outcome, 4, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("{}, worst {worst:.2} m", metrics.summary()))
    })())
}

/// Four chickadees ~20–28 m apart with trunk echoes 8 dB down behind each perch.
pub fn flock_echo(seed: u64) -> Result<ScenarioReport, String> {
    let mut scene = EntityScene::new("flock_echo", 85.0, ring_of_phones(6, [0.0, 0.0], 22.0, seed));
    let perches = [[14.0, 0.0, 5.0], [0.0, 14.0, 6.0], [-14.0, 0.0, 5.0], [0.0, -14.0, 7.0]];
    let birds: Vec<usize> = perches.iter().map(|_| scene.bird(CHICKADEE)).collect();
    let bout_times = [20.0, 25.0, 31.0, 36.0, 42.0, 48.0, 55.0, 61.0, 66.0, 72.0];
    for (index, time_s) in bout_times.into_iter().enumerate() {
        let bird = index % birds.len();
        scene = scene
            .bout(birds[bird], time_s, perches[bird], CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL)
            .echo(birds[bird], time_s + 0.004, perches[bird], [0.0, 0.0], DEFAULT_CALL_LEVEL_DB_SPL);
    }
    report("flock_echo", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        ensure(metrics.pinpointed_birds == perches.len(), || format!("pinpointed birds {}: {}", metrics.pinpointed_birds, metrics.summary()))?;
        fixes_within_reported_region(&outcome.scored_fixes(), REPORTED_REGION_SIGMA_MULTIPLIER, REPORTED_REGION_FLOOR_M)?;
        Ok(metrics.summary())
    })())
}

/// One bird, two bouts 8 s apart: a weak ID with a clean fix, then a strong
/// ID while two phones have speech nearby. Both fuse into one entity.
pub fn asymmetric_bouts(seed: u64) -> Result<ScenarioReport, String> {
    const WEAK_SCORE: f32 = 0.25;
    const WEAK_LOCALIZE_TO_TRACK_SCORE: f32 = 0.2;
    let mut scene = EntityScene::new("asymmetric_bouts", 50.0, ring_of_phones(5, [0.0, 0.0], 12.0, seed));
    let robin = scene.bird(ROBIN);
    let perch = [3.0, 2.0, 6.0];
    scene = scene.bout(robin, 25.0, perch, WEAK_SCORE, DEFAULT_CALL_LEVEL_DB_SPL).bout(robin, 33.0, perch, CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL);
    scene.spec.mesh.matcher.localize_to_track_score = WEAK_LOCALIZE_TO_TRACK_SCORE;
    for device in [1, 3] {
        scene.spec.noise_intervals.push(NoiseInterval { device, start_s: 31.0, end_s: 36.0, kind: NoiseKind::Speech });
    }
    report("asymmetric_bouts", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        let localized = outcome.call_records().iter().filter(|record| record.fix.is_some()).count();
        ensure(localized == 2, || format!("{localized} of 2 bouts localized"))?;
        ensure(metrics.pinpointed_birds == 1 && metrics.entities_by_bird[robin].len() == 1, || metrics.summary())?;
        Ok(metrics.summary())
    })())
}

/// A loud robin and a quieter tanager 25 m apart inside one window: two
/// clusters, each labelled with its own species (attribution), no swaps.
pub fn same_band_two_species(seed: u64) -> Result<ScenarioReport, String> {
    const ROBIN_LEVEL_DB: f64 = 90.0;
    const TANAGER_LEVEL_DB: f64 = 82.0;
    let mut scene = EntityScene::new("same_band_two_species", 55.0, ring_of_phones(6, [0.0, 0.0], 15.0, seed));
    let (robin, tanager) = (scene.bird(ROBIN), scene.bird(TANAGER));
    for start_s in [22.0, 30.0, 38.0, 46.0] {
        scene = scene
            .bout(robin, start_s, [-10.0, 4.0, 6.0], CONFIDENT_SCORE, ROBIN_LEVEL_DB)
            .bout(tanager, start_s + 0.6, [12.0, -6.0, 6.0], CONFIDENT_SCORE, TANAGER_LEVEL_DB);
    }
    report("same_band_two_species", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        ensure(metrics.pinpointed_birds == 2, || metrics.summary())?;
        let cap = outcome.spec.mesh.attribution.loco_max_reruns_per_window;
        let clusters_per_call = outcome.call_records().iter().map(|record| record.clusters.len()).max().unwrap_or(0);
        ensure(clusters_per_call >= 2, || "no window resolved two clusters".into())?;
        healthy_path_checks(&outcome, 4, Some(INSIDE_LAYOUT_GATE_M))?;
        Ok(format!("{}, LOCO cap {cap}", metrics.summary()))
    })())
}

/// A wren sings 5 times; 2 quiet bouts only its nearest phone hears (no fix).
/// One entity: the quiet bouts update it, no unlocalized duplicate, one creation.
pub fn omni_to_localized_dedup(seed: u64) -> Result<ScenarioReport, String> {
    const QUIET_LEVEL_DB: f64 = 55.0;
    let devices = vec![
        DeviceSpec::android_at([0.0, 0.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([18.0, 4.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([6.0, 19.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-14.0, 12.0, PHONE_HEIGHT_M]),
        DeviceSpec::android_at([-6.0, -16.0, PHONE_HEIGHT_M]),
    ];
    let mut scene = EntityScene::new("omni_to_localized_dedup", 60.0, devices);
    let wren = scene.bird(WREN);
    let perch = [1.5, 1.0, 2.0];
    for (time_s, level_db) in [(22.0, QUIET_LEVEL_DB), (28.0, DEFAULT_CALL_LEVEL_DB_SPL), (34.0, QUIET_LEVEL_DB), (40.0, DEFAULT_CALL_LEVEL_DB_SPL), (46.0, DEFAULT_CALL_LEVEL_DB_SPL)] {
        scene = scene.bout(wren, time_s, perch, CONFIDENT_SCORE, level_db);
    }
    report("omni_to_localized_dedup", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        let calls = host_call_outcomes(&outcome);
        let created = calls.iter().filter(|call| call.change == Some(BirdEntityChangeKind::Created)).count();
        let absorbed = calls.iter().filter(|call| call.action == EntityReportingAction::EnrichExisting).count();
        let unlocalized = outcome.call_records().iter().filter(|record| record.fix.is_none()).count();
        ensure(created == 1, || format!("{created} entity creations (one notification wanted): {}", metrics.summary()))?;
        ensure(unlocalized >= 1 && absorbed + 1 >= unlocalized, || format!("{unlocalized} unlocalized bouts, {absorbed} absorbed"))?;
        ensure(metrics.pinpointed_birds == 1, || metrics.summary())?;
        Ok(format!("{}, {unlocalized} unlocalized bouts, {absorbed} absorbed", metrics.summary()))
    })())
}

/// Five phones in single file, a bird 15 m to one side: never a confident
/// wrong side, never a second entity from a mirror flip.
pub fn collinear_trail(seed: u64) -> Result<ScenarioReport, String> {
    const PHONE_SPACING_M: f64 = 8.0;
    let devices = (0..5).map(|index| DeviceSpec::android_at([PHONE_SPACING_M * index as f64, 0.0, PHONE_HEIGHT_M])).collect();
    let mut scene = EntityScene::new("collinear_trail", 60.0, devices);
    let robin = scene.bird(ROBIN);
    for time_s in [22.0, 30.0, 38.0, 46.0] {
        scene = scene.bout(robin, time_s, [16.0, 15.0, 5.0], CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL);
    }
    report("collinear_trail", seed, (|| {
        let (outcome, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        fixes_within_reported_region(&outcome.scored_fixes(), REPORTED_REGION_SIGMA_MULTIPLIER, REPORTED_REGION_FLOOR_M)?;
        ensure(metrics.pinpointed_birds == 0, || format!("a mirror-ambiguous trail fix pinpointed the bird: {}", metrics.summary()))?;
        let wrong_side = pinpointed_on_wrong_side_of_trail(&outcome);
        ensure(wrong_side == 0, || format!("{wrong_side} pinpointed entity updates on the mirror side of the trail: {}", metrics.summary()))?;
        Ok(format!("{}, {} fixes", metrics.summary(), outcome.scored_fixes().len()))
    })())
}

/// Pinpointed entity updates that land on the far side of the trail (world y < 0,
/// the bird is at y > 0), after moving the host frame onto the phones' true
/// positions with a proper rigid fit from the last solve trace.
fn pinpointed_on_wrong_side_of_trail(outcome: &ScenarioOutcome) -> usize {
    use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
    use minimappr_core::protocol::wire_codes::BirdEntityState;
    use minimappr_core::protocol::FieldMessage;
    let Some(trace) = outcome.traces.iter().rev().find(|trace| trace.fix.is_some()) else { return 0 };
    let call_time_s = outcome.world.calls[trace.true_call_index as usize].time_s;
    let estimated: Vec<[f64; 2]> = trace.estimated_node_positions_m.iter().map(|p| [p[0], p[1]]).collect();
    let truth: Vec<[f64; 2]> = trace.node_ids.iter().map(|node| {
        let p = outcome.world.position(node.0 as usize, call_time_s);
        [p[0], p[1]]
    }).collect();
    let transform = weighted_kabsch_2d(&estimated, &truth, &vec![1.0; estimated.len()], false);
    outcome
        .host_log_records()
        .into_iter()
        .filter_map(|record| match record {
            FieldMessage::BirdEntityUpdate(update) if update.state == BirdEntityState::Pinpointed => update.position_east_north_up_m,
            _ => None,
        })
        .filter(|position| transform.apply([f64::from(position[0]), f64::from(position[1])])[1] < 0.0)
        .count()
}

/// Two phones (a couple) and two birds of different species near each: no
/// fix is possible, so each bird is one heard-nearby entity, never a pin.
pub fn couples(seed: u64) -> Result<ScenarioReport, String> {
    let devices = vec![DeviceSpec::android_at([0.0, 0.0, PHONE_HEIGHT_M]), DeviceSpec::android_at([6.0, 0.0, PHONE_HEIGHT_M])];
    let mut scene = EntityScene::new("couples", 60.0, devices);
    let (robin, tanager) = (scene.bird(ROBIN), scene.bird(TANAGER));
    for start_s in [22.0, 32.0, 42.0] {
        scene = scene
            .bout(robin, start_s, [-4.0, 5.0, 4.0], CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL)
            .bout(tanager, start_s + 4.0, [10.0, -5.0, 4.0], CONFIDENT_SCORE, DEFAULT_CALL_LEVEL_DB_SPL);
    }
    report("couples", seed, (|| {
        let (_, metrics) = run_scene(scene, seed)?;
        entity_invariants(&metrics)?;
        ensure(metrics.pinpointed_birds == 0, || format!("a two-phone couple pinpointed a bird: {}", metrics.summary()))?;
        ensure(metrics.entities_by_bird.iter().all(|entities| entities.len() == 1), || metrics.summary())?;
        Ok(metrics.summary())
    })())
}
