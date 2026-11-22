//! Resource bounds: wall time and peak heap per solve (arrival LM, layout
//! solve, audio SRP) and per session (a dense "dawn chorus" Layer 2 session
//! at two lengths: doubling the length may not more than double peak heap,
//! the pending-call queue stays within capacity and no detection is lost
//! silently). Budgets are ~5× measured M2 costs (see `ResourceGateConfig`).
//! Wall-time checks are skipped in unoptimised builds, where they mean
//! nothing; heap checks need `heap_accounting::CountingAllocator` installed.

use std::time::Instant;

use minimappr_core::mesh::arrival_time_localizer::ArrivalTimeLocalizer;
use minimappr_mesh_sim::invariants::no_silent_detection_loss;
use minimappr_mesh_sim::scenario::{ring_of_phones, ScenarioSpec};
use minimappr_mesh_sim::ScenarioRun;

use super::GateRun;
use crate::arrival_trials::{solve_trial, ArrivalTrial};
use crate::gate_seeds::GateOutcome;
use crate::heap_accounting::{is_installed, measure_peak_heap};
use crate::layout_chain::solve_layout_from_truth;
use crate::srp_trials::{render_srp_trial, SrpTrialConfig};
use crate::trial_geometry::{inside_source, random_phone_layout};

const ARRIVAL_NODE_COUNT: usize = 7;
const ARRIVAL_APERTURE_M: f64 = 25.0;
const CHORUS_PHONE_COUNT: usize = 6;
const CHORUS_RING_RADIUS_M: f64 = 10.0;
/// Calls every few seconds overlap classification windows and pending calls.
const CHORUS_CALL_INTERVAL_S: f64 = 3.0;
const CHORUS_FIRST_CALL_S: f64 = 20.0;
const CHORUS_BASE_DURATION_S: f64 = 120.0;
const CHORUS_CALL_RADIUS_M: f64 = 25.0;
const CHORUS_CALL_HEIGHT_M: f64 = 6.0;

fn chorus_spec(seed: u64, duration_s: f64) -> ScenarioSpec {
    let mut spec = ScenarioSpec::new("sim_gates_dawn_chorus", duration_s, ring_of_phones(CHORUS_PHONE_COUNT, [0.0, 0.0], CHORUS_RING_RADIUS_M, seed));
    let mut time = CHORUS_FIRST_CALL_S;
    let mut index = 0.0_f64;
    while time < duration_s - CHORUS_CALL_INTERVAL_S {
        let angle = index * 2.399; // golden-angle spiral of call positions
        let radius = CHORUS_CALL_RADIUS_M * ((index % 7.0) + 1.0) / 7.0;
        spec = spec.call(time, [radius * angle.cos(), radius * angle.sin(), CHORUS_CALL_HEIGHT_M]);
        time += CHORUS_CALL_INTERVAL_S;
        index += 1.0;
    }
    spec
}

pub fn resource_bounds(run: &GateRun) -> GateOutcome {
    let config = &run.config;
    let budget = &config.resources;
    let timing_meaningful = !cfg!(debug_assertions);
    let heap_measured = is_installed();
    let mut outcome = GateOutcome::new("resource_bounds", run.seeds.len());
    outcome.summary(format!("wall-time checks {}, heap checks {}", if timing_meaningful { "on" } else { "off (debug build)" }, if heap_measured { "on" } else { "off (CountingAllocator not installed)" }));
    let check = |outcome: &mut GateOutcome, seed: u64, label: &str, seconds: f64, max_seconds: f64, heap: usize, max_heap: usize| {
        outcome.summary(format!("seed {seed} {label}: {:.4} s (budget {max_seconds} s), peak heap {:.1} KiB (budget {})", seconds, heap as f64 / 1024.0, if max_heap == usize::MAX { "none".to_string() } else { format!("{} KiB", max_heap / 1024) }));
        if timing_meaningful && seconds > max_seconds {
            outcome.fail_seed(seed, format!("{label} took {seconds:.3} s > {max_seconds} s"));
        }
        if heap_measured && heap > max_heap {
            outcome.fail_seed(seed, format!("{label} peak heap {heap} B > {max_heap} B"));
        }
    };
    let localizer = ArrivalTimeLocalizer::default();
    for &seed in &run.seeds {
        let layout = random_phone_layout(seed, ARRIVAL_NODE_COUNT, ARRIVAL_APERTURE_M, &config.geometry);
        let trial = ArrivalTrial::new(seed, layout.clone(), inside_source(seed, &layout, ARRIVAL_APERTURE_M, &config.geometry));
        let started = Instant::now();
        let (_, heap) = measure_peak_heap(|| solve_trial(&trial, &trial.all_node_indices(), &localizer, &config.perturbation));
        check(&mut outcome, seed, "arrival LM solve (7 phones)", started.elapsed().as_secs_f64(), budget.maximum_arrival_solve_s, heap, budget.maximum_arrival_peak_heap_bytes);

        let started = Instant::now();
        let _ = solve_layout_from_truth(seed, "resource_ranges", &layout, config.chirality.chirp_range_std_m);
        check(&mut outcome, seed, "layout solve (7 phones)", started.elapsed().as_secs_f64(), budget.maximum_layout_solve_s, 0, usize::MAX);

        match render_srp_trial(seed, &SrpTrialConfig::default(), &config.geometry, &config.perturbation) {
            Ok(srp) => {
                let base = srp.base_config();
                let started = Instant::now();
                let (result, heap) = measure_peak_heap(|| srp.localize(&base));
                check(&mut outcome, seed, "SRP localize (5 phones, contour)", started.elapsed().as_secs_f64(), budget.maximum_srp_solve_s, heap, budget.maximum_srp_peak_heap_bytes);
                if let Err(error) = result {
                    outcome.fail_seed(seed, format!("SRP localize failed: {error}"));
                }
            }
            Err(error) => outcome.fail_seed(seed, format!("SRP render failed: {error}")),
        }

        let mut session_peaks = Vec::new();
        for multiple in [1.0, 2.0] {
            let duration = CHORUS_BASE_DURATION_S * multiple;
            let started = Instant::now();
            let (session, heap) = measure_peak_heap(|| ScenarioRun::new(chorus_spec(seed, duration), seed).run());
            let label = format!("dawn chorus session {duration:.0} s");
            check(&mut outcome, seed, &label, started.elapsed().as_secs_f64(), budget.maximum_session_wall_s * multiple, heap, budget.maximum_session_peak_heap_bytes);
            session_peaks.push(heap);
            if let Err(message) = no_silent_detection_loss(&session) {
                outcome.fail_seed(seed, format!("{label}: {message}"));
            }
            for (_, host) in session.all_hosts() {
                let pending = host.calls.pending_call_count();
                let capacity = session.spec.mesh.matcher.pending_call_capacity;
                if pending > capacity {
                    outcome.fail_seed(seed, format!("{label}: {pending} pending calls over capacity {capacity}"));
                }
                let counters = &host.calls.counters;
                outcome.summary(format!(
                    "seed {seed} {label}: {} calls completed, {} localized, {} abandoned, {} dropped by backpressure (counted), {} skipped for clip budget",
                    host.calls.completed.len(), counters.calls_localized, counters.calls_abandoned, counters.calls_dropped_backpressure, counters.calls_skipped_for_clip_budget
                ));
            }
        }
        if heap_measured && session_peaks[0] > 0 {
            let growth = session_peaks[1] as f64 / session_peaks[0] as f64;
            outcome.summary(format!("seed {seed}: session peak heap growth for 2x length {growth:.2}x"));
            if growth > budget.maximum_session_heap_growth_factor {
                outcome.fail_seed(seed, format!("doubling the session grew peak heap {growth:.2}x (> {})", budget.maximum_session_heap_growth_factor));
            }
        }
    }
    outcome
}
