//! Inspect one SRP gate trial: accuracy and drawn-region coverage with and
//! without the residual perturbation and the random trunks, so a failing
//! seed can be attributed to multipath, layout/clock residuals or neither.
//!
//! cargo run --release -p minimappr-sim-gates --example inspect_srp_trial -- SEED [TEMPERATURE]

use minimappr_core::srp_layout::LayoutSrpConfig;
use minimappr_sim_gates::gate_config::{RealisticPerturbationConfig, SimGateConfig};
use minimappr_sim_gates::srp_trials::{render_srp_trial, SrpTrialConfig};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let seed: u64 = arguments.next().expect("SEED").parse().expect("integer seed");
    let temperature: f64 =
        arguments.next().map_or(LayoutSrpConfig::default().scoring.likelihood_temperature, |t| t.parse().expect("temperature"));
    let gate_config = SimGateConfig::default();
    let clean = RealisticPerturbationConfig { clock_residual_std_s: 0.0, layout_position_std_m: 0.0, phone_height_std_m: 0.0, ..Default::default() };
    for trunks in [true, false] {
        for (label, perturbation) in [("perturbed", &gate_config.perturbation), ("true inputs", &clean)] {
            let mut trial_config = SrpTrialConfig::default();
            trial_config.scene.random_trunks = trunks;
            let trial = render_srp_trial(seed, &trial_config, &gate_config.geometry, perturbation).expect("render");
            let mut config = trial.base_config();
            config.scoring.likelihood_temperature = temperature;
            let started = std::time::Instant::now();
            let result = trial.localize(&config).expect("localize");
            let outcome = trial.score(&result, started.elapsed().as_secs_f64());
            println!(
                "seed {seed} trunks {trunks} {label}: source {:?} inside {} -> estimate {:?}; err {:.2} m (LM {:?}), bearing {:.2} deg, covered {}, area {:.1} m2, fan {}, certified {}, cells {}, region complete {}, {:.2} s",
                trial.source_m.map(|v| (v * 10.0).round() / 10.0),
                trial.is_inside,
                result.estimate.position_m.map(|v| (v * 10.0).round() / 10.0),
                outcome.horizontal_error_m,
                outcome.lm_refined_horizontal_error_m.map(|e| (e * 100.0).round() / 100.0),
                outcome.bearing_error_deg,
                outcome.drawn_region_covers_truth,
                outcome.region_area_m2,
                outcome.drew_bearing_fan,
                result.estimate.is_certified_within_tolerance,
                outcome.cells_evaluated,
                outcome.region_refinement_complete,
                outcome.solve_seconds
            );
        }
    }
}
