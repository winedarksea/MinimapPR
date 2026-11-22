//! Chirality: an MDS layout is only known up to reflection. The GPS fit
//! (6 m σ) picks a side and reports a confidence; a bearing pair from one
//! stereo phone locks it. Gates: no wrong side at confidence ≥ 0.99, the
//! confidence is calibrated (wrong rate per confidence bin within binomial
//! slack of 1 − bin edge), and a bearing-pair lock is never wrong.

use std::collections::BTreeMap;

use minimappr_core::layout::gps_anchor_fit::ChiralityCue;
use minimappr_core::mesh::chirality_cue::chirality_from_bearing_pair;
use minimappr_core::mesh::config::IslandSyncConfig;
use minimappr_core::protocol::mesh_messages::ChirpDirectionReport;
use minimappr_core::protocol::NodeId;

use super::GateRun;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::layout_chain::{apply_all, fit_to_gps, gps_fixes, horizontal, is_reflection_of, solve_layout_from_truth};
use crate::statistics::binomial_standard_error;
use crate::trial_geometry::{azimuth_from, count_in, gaussian, random_phone_layout, uniform_in};

const PHONE_COUNT_RANGE: [usize; 2] = [3, 6];
/// Stereo-phone bearings (mesh world defaults): compass error common to the
/// round, independent DOA noise per bearing.
const HEADING_ERROR_STD_DEG: f64 = 20.0;
const PER_BEARING_NOISE_STD_DEG: f64 = 5.0;
/// `island_sync_engine::DECISIVE_CUE_SIGMA_MULTIPLIER` (private there).
const DECISIVE_CUE_SIGMA_MULTIPLIER: f64 = 3.0;

struct ChiralityTrial {
    confidence: f64,
    gps_side_wrong: bool,
    bearing_lock: Option<bool>,
    failure: Option<String>,
}

fn run_trial(seed: u64, run: &GateRun) -> ChiralityTrial {
    let config = &run.config;
    let phone_count = count_in(seed, "chirality_phone_count", PHONE_COUNT_RANGE);
    let radius = uniform_in(seed, "chirality_radius", 0, config.chirality.layout_radius_range_m);
    let truth = random_phone_layout(seed, phone_count, 2.0 * radius, &config.geometry);
    let truth_xy = horizontal(&truth);
    let mut trial = ChiralityTrial { confidence: 0.5, gps_side_wrong: false, bearing_lock: None, failure: None };
    let layout = match solve_layout_from_truth(seed, "chirality_ranges", &truth, config.chirality.chirp_range_std_m) {
        Ok(layout) => layout,
        Err(error) => {
            trial.failure = Some(format!("{phone_count} phones: layout solve failed {error:?}"));
            return trial;
        }
    };
    let layout_xy = horizontal(&layout.positions_m);
    let gps = gps_fixes(seed, "chirality_gps", 0, &truth, config.perturbation.gps_error_std_m);
    match fit_to_gps(&layout_xy, &gps, ChiralityCue::Unknown) {
        Ok(fit) => {
            trial.confidence = fit.chirality_confidence;
            trial.gps_side_wrong = is_reflection_of(&apply_all(&fit.transform, &layout_xy), &truth_xy);
            if trial.gps_side_wrong && fit.chirality_confidence >= config.chirality.confident_threshold {
                trial.failure = Some(format!(
                    "{phone_count} phones r={radius:.1} m: GPS fit picked the mirror at confidence {:.5}",
                    fit.chirality_confidence
                ));
            }
        }
        Err(error) => trial.failure = Some(format!("anchor fit failed {error:?}")),
    }
    if phone_count >= 3 {
        let local: BTreeMap<NodeId, [f64; 3]> = layout.positions_m.iter().enumerate().map(|(i, p)| (NodeId(i as u32), *p)).collect();
        let heading_error = HEADING_ERROR_STD_DEG * gaussian(seed, "chirality_heading", 0, 0);
        let report = |emitter: usize| ChirpDirectionReport {
            round_id: 1,
            reporting_node_id: NodeId(0),
            emitter_node_id: NodeId(emitter as u32),
            world_azimuth_deg: (azimuth_from([truth[0][0], truth[0][1]], truth[emitter]).to_degrees()
                + heading_error
                + PER_BEARING_NOISE_STD_DEG * gaussian(seed, "chirality_bearing", emitter as u64, 0)) as f32,
            azimuth_standard_deviation_deg: PER_BEARING_NOISE_STD_DEG as f32,
        };
        let relative_noise = IslandSyncConfig::default().relative_bearing_noise_std_deg.to_radians();
        if let Some(ChiralityCue::Locked { mirrored }) =
            chirality_from_bearing_pair(&local, &report(1), &report(2), relative_noise, DECISIVE_CUE_SIGMA_MULTIPLIER)
        {
            let truly_mirrored = is_reflection_of(&layout_xy, &truth_xy);
            trial.bearing_lock = Some(mirrored == truly_mirrored);
            if mirrored != truly_mirrored {
                trial.failure = Some(format!("bearing pair locked the wrong chirality ({phone_count} phones, heading error {heading_error:.1} deg)"));
            }
        }
    }
    trial
}

pub fn chirality(run: &GateRun) -> GateOutcome {
    let config = &run.config.chirality;
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| run_trial(seed, run));
    let mut outcome = GateOutcome::new("chirality", results.len());
    for (seed, trial) in &results {
        if let Some(message) = &trial.failure {
            outcome.fail_seed(*seed, message.clone());
        }
    }
    let enough = results.len() >= run.config.coverage.minimum_trials_for_aggregate_gate;
    let edges = config.calibration_bin_lower_edges;
    for (index, &low) in edges.iter().enumerate() {
        let high = edges.get(index + 1).copied().unwrap_or(f64::INFINITY);
        let in_bin: Vec<_> = results.iter().filter(|(_, t)| t.confidence >= low && t.confidence < high).collect();
        let wrong = in_bin.iter().filter(|(_, t)| t.gps_side_wrong).count();
        let rate = wrong as f64 / in_bin.len().max(1) as f64;
        let allowed_rate = 1.0 - low;
        let allowed = allowed_rate + config.calibration_slack_standard_errors * binomial_standard_error(allowed_rate.max(1.0 / in_bin.len().max(1) as f64), in_bin.len());
        outcome.summary(format!("confidence [{low}, {high}): {wrong}/{} wrong side ({rate:.3}, allowed {allowed:.3})", in_bin.len()));
        outcome.check_aggregate(rate <= allowed || in_bin.is_empty(), enough, || format!("chirality confidence bin [{low}, {high}) overconfident: {rate:.3} wrong"));
    }
    let below = results.iter().filter(|(_, t)| t.confidence < edges[0]).count();
    let locks = results.iter().filter(|(_, t)| t.bearing_lock.is_some()).count();
    let wrong_locks = results.iter().filter(|(_, t)| t.bearing_lock == Some(false)).count();
    outcome.summary(format!("{below} fits below {} (reported low); bearing pairs locked {locks}/{} ({wrong_locks} wrong)", edges[0], results.len()));
    outcome
}
