//! Island isolation over randomized multi-clump Layer 2 sessions: 2–3
//! phone clumps beyond chirp range of each other, calls inside each clump
//! and between two of them (heard by both). No call may mix clock domains,
//! no clip may span a clump boundary, and in merge variants (odd seeds: a
//! far clump walks in) no post-merge call may reuse a pre-merge domain.

use std::collections::BTreeSet;

use minimappr_core::mesh::config::MeshConfig;
use minimappr_mesh_sim::device_spec::Trajectory;
use minimappr_mesh_sim::invariants::common_invariants;
use minimappr_mesh_sim::scenario::{ring_of_phones, ScenarioSpec};
use minimappr_mesh_sim::ScenarioRun;

use super::GateRun;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::trial_geometry::{count_in, uniform_in};

const CLUMP_COUNT_RANGE: [usize; 2] = [2, 3];
const PHONES_PER_CLUMP_RANGE: [usize; 2] = [4, 5];
const CLUMP_RADIUS_RANGE_M: [f64; 2] = [5.0, 9.0];
/// Mesh world chirps still close two-way at ~90 m; clumps sit beyond that.
const CLUMP_SEPARATION_RANGE_M: [f64; 2] = [150.0, 300.0];
const FIRST_CALL_S: f64 = 30.0;
const CALL_SPACING_S: f64 = 12.0;
const MERGE_WALK_START_S: f64 = 80.0;
const MERGE_WALK_END_S: f64 = 140.0;
/// After the walk ends, the post-merge call waits one heartbeat round
/// (the latest a merge survey can come) plus this margin.
const POST_MERGE_MARGIN_S: f64 = 15.0;
const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
const SESSION_TAIL_S: f64 = 15.0;
const CALL_HEIGHT_M: f64 = 6.0;

struct IslandTrial {
    problems: Vec<String>,
    calls_localized: usize,
    calls_total: usize,
    is_merge_variant: bool,
    merged_calls: usize,
}

fn run_trial(seed: u64) -> IslandTrial {
    let clump_count = count_in(seed, "island_clump_count", CLUMP_COUNT_RANGE);
    let is_merge_variant = seed % 2 == 1;
    let mut devices = Vec::new();
    let mut clump_of_device = Vec::new();
    let mut centres = Vec::new();
    for clump in 0..clump_count {
        let key = clump as u64;
        let angle = std::f64::consts::TAU * clump as f64 / clump_count as f64;
        let separation = uniform_in(seed, "island_separation", key, CLUMP_SEPARATION_RANGE_M);
        let centre = if clump == 0 { [0.0, 0.0] } else { [separation * angle.cos(), separation * angle.sin()] };
        centres.push(centre);
        let phones = count_in(seed ^ key.wrapping_mul(0x9e37), "island_phone_count", PHONES_PER_CLUMP_RANGE);
        let radius = uniform_in(seed, "island_radius", key, CLUMP_RADIUS_RANGE_M);
        for device in ring_of_phones(phones, centre, radius, seed ^ (key + 1).wrapping_mul(0x5851_f42d)) {
            clump_of_device.push(clump);
            devices.push(device);
        }
    }
    if is_merge_variant {
        // The last clump walks to sit beside clump 0 (offset so phones don't overlap).
        let walker = clump_count - 1;
        let shift = [centres[0][0] - centres[walker][0] + 2.0 * CLUMP_RADIUS_RANGE_M[1], centres[0][1] - centres[walker][1]];
        for (_, spec) in devices.iter_mut().enumerate().filter(|(device, _)| clump_of_device[*device] == walker) {
            let home = spec.trajectory.position_at(0.0);
            let joined = [home[0] + shift[0], home[1] + shift[1], home[2]];
            spec.trajectory = Trajectory::stationary(home).then_move(MERGE_WALK_START_S, MERGE_WALK_END_S, joined);
        }
    }
    let mut calls = Vec::new();
    for (clump, centre) in centres.iter().enumerate() {
        calls.push((FIRST_CALL_S + CALL_SPACING_S * clump as f64, [centre[0] + 1.0, centre[1] - 1.0, CALL_HEIGHT_M]));
    }
    // Between clumps 0 and 1: heard by both domains, must stay single-domain.
    let between = [(centres[0][0] + centres[1][0]) / 2.0, (centres[0][1] + centres[1][1]) / 2.0, CALL_HEIGHT_M];
    calls.push((FIRST_CALL_S + CALL_SPACING_S * clump_count as f64, between));
    let mut duration = calls.last().map_or(0.0, |c| c.0) + SESSION_TAIL_S;
    if is_merge_variant {
        let heartbeat_s = MeshConfig::default().scheduler.heartbeat_round_interval_ms as f64 / MILLISECONDS_PER_SECOND;
        let post_merge_call_s = MERGE_WALK_END_S + heartbeat_s + POST_MERGE_MARGIN_S;
        calls.push((post_merge_call_s, [centres[0][0] + CLUMP_RADIUS_RANGE_M[1], centres[0][1], CALL_HEIGHT_M]));
        duration = post_merge_call_s + SESSION_TAIL_S;
    }
    let mut spec = ScenarioSpec::new("sim_gates_islands", duration, devices);
    for (time, position) in &calls {
        spec = spec.call(*time, *position);
    }
    let outcome = ScenarioRun::new(spec, seed).run();
    let mut problems = Vec::new();
    if let Err(message) = common_invariants(&outcome) {
        problems.push(message);
    }
    let records = outcome.call_records();
    let merge_ms = (MERGE_WALK_START_S * MILLISECONDS_PER_SECOND) as u64;
    let walker = clump_count - 1;
    let mut merged_calls = 0;
    let mut pre_merge_domains = BTreeSet::new();
    for record in &records {
        let clumps: BTreeSet<usize> = record.clip_nodes.iter().map(|node| clump_of_device[node.0 as usize]).collect();
        // Before a merge, a clip set spanning two clumps would be a cross-island TDOA.
        let still_split = !is_merge_variant || record.received_ms < merge_ms;
        if still_split && clumps.len() > 1 {
            problems.push(format!("call {} fetched clips from clumps {clumps:?} while they shared no clock domain", record.call_id));
        }
        if record.received_ms < merge_ms {
            pre_merge_domains.extend(record.clip_domains.iter().map(|d| (d.1, d.2)));
        }
    }
    if is_merge_variant {
        // Only calls whose clips span clump 0 and the walker ran on a merged
        // domain; a call before the merge survey stays island-local (fine).
        let spans_merge = |record: &&&minimappr_core::mesh::call_pipeline::LocalizedCallRecord| {
            let clumps: BTreeSet<usize> = record.clip_nodes.iter().map(|node| clump_of_device[node.0 as usize]).collect();
            clumps.contains(&0) && clumps.contains(&walker)
        };
        for record in records.iter().filter(spans_merge) {
            merged_calls += 1;
            let reused: Vec<_> = record.clip_domains.iter().map(|d| (d.1, d.2)).filter(|d| pre_merge_domains.contains(d)).collect();
            if !reused.is_empty() {
                problems.push(format!("post-merge call {} reused pre-merge clock domains {reused:?}", record.call_id));
            }
        }
    }
    IslandTrial {
        problems,
        calls_localized: records.iter().filter(|r| r.fix.is_some()).count(),
        merged_calls,
        calls_total: calls.len(),
        is_merge_variant,
    }
}

pub fn island_isolation(run: &GateRun) -> GateOutcome {
    let results = run_trials_in_parallel(&run.seeds, run.threads, run_trial);
    let mut outcome = GateOutcome::new("island_isolation", results.len());
    for (seed, trial) in &results {
        for problem in &trial.problems {
            outcome.fail_seed(*seed, problem.clone());
        }
    }
    let localized: usize = results.iter().map(|(_, t)| t.calls_localized).sum();
    let total: usize = results.iter().map(|(_, t)| t.calls_total).sum();
    let merges = results.iter().filter(|(_, t)| t.is_merge_variant).count();
    let merged = results.iter().filter(|(_, t)| t.merged_calls > 0).count();
    outcome.summary(format!("{} sessions ({merges} with a merge, {merged} ran a call on the merged domain), {localized}/{total} scripted calls localized (reported, not gated)", results.len()));
    outcome
}
