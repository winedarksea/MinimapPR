//! Geometric invariance of the audio chain: render a scene and its rigidly
//! moved (optionally mirrored) twin, run layout SRP on both, polish each
//! `LmSeed` with the shipping `JointLmSeedRefiner`, and require the
//! refined estimates to agree within 1 mm after the motion (strategy note:
//! compare after LM, the coarse grid is not exactly invariant).
//!
//! Trunks are off (their random placement would not move with the world);
//! ambient noise is drawn per device, so both renders hear the same noise.

use minimappr_core::mesh::srp_seed_refiner::{JointLmSeedRefiner, SeedRefinerConfig};
use minimappr_core::srp_layout::{LayoutSrpSearcher, NodeClip, TdoaSeedRefiner};

use super::GateRun;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::srp_scene::{build_scene, call_window, render, srp_config_for, true_layout_nodes, SrpSceneConfig};
use crate::trial_geometry::{gaussian, inside_source, random_phone_layout, source_at_range, uniform_in, PlanarRigidMotion};

const PHONE_COUNT: usize = 5;
const APERTURE_RANGE_M: [f64; 2] = [18.0, 30.0];
const OUTSIDE_RANGE_M: f64 = 50.0;

/// SRP + LM on a (possibly moved) world; returns the refined position.
fn refined_estimate(seed: u64, layout: &[[f64; 3]], source: [f64; 3], motion: Option<&PlanarRigidMotion>, run: &GateRun) -> Result<[f64; 3], String> {
    let place = |p: [f64; 3]| motion.map_or(p, |m| m.apply(p));
    let moved_layout: Vec<[f64; 3]> = layout.iter().map(|p| place(*p)).collect();
    let scene_config = SrpSceneConfig { random_trunks: false, ..SrpSceneConfig::default() };
    let rendered = render(&build_scene(seed, &moved_layout, place(source), &scene_config))?;
    let perturbation = &run.config.perturbation;
    let mut nodes = true_layout_nodes(&rendered.plan);
    for (index, node) in nodes.iter_mut().enumerate() {
        let key = index as u64;
        let error = [
            perturbation.layout_position_std_m * gaussian(seed, "srp_invariance_layout", key, 0),
            perturbation.layout_position_std_m * gaussian(seed, "srp_invariance_layout", key, 1),
        ];
        let moved_error = motion.map_or(error, |m| m.rotate_vector(error));
        node.position_m[0] += moved_error[0];
        node.position_m[1] += moved_error[1];
        node.clock.anchor_island_time_s += perturbation.clock_residual_std_s * gaussian(seed, "srp_invariance_clock", key, 0);
    }
    let clips: Vec<NodeClip<'_>> = rendered
        .channels_per_device
        .iter()
        .map(|channels| NodeClip { channels: channels.iter().map(Vec::as_slice).collect(), first_sample_index: 0 })
        .collect();
    let result = LayoutSrpSearcher::new(&nodes, &srp_config_for(&rendered.plan))
        .and_then(|searcher| searcher.localize(&clips, call_window(&rendered.plan, scene_config.window_padding_s)))
        .map_err(|error| format!("srp: {error:?}"))?;
    let refiner = JointLmSeedRefiner {
        config: SeedRefinerConfig::default(),
        node_position_std_m: vec![perturbation.layout_position_std_m; nodes.len()],
    };
    refiner.refine(&result.lm_seed).map(|fix| fix.position_m).map_err(|error| format!("lm: {error:?}"))
}

pub fn srp_lm_invariance(run: &GateRun) -> GateOutcome {
    let config = &run.config;
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| {
        let aperture = uniform_in(seed, "srp_invariance_aperture", 0, APERTURE_RANGE_M);
        let layout = random_phone_layout(seed, PHONE_COUNT, aperture, &config.geometry);
        let source = if seed % 2 == 0 {
            inside_source(seed, &layout, aperture, &config.geometry)
        } else {
            source_at_range(seed, &layout, OUTSIDE_RANGE_M, &config.geometry)
        };
        let motion = PlanarRigidMotion::random(seed, config.invariance.maximum_translation_m, seed % 3 == 0);
        let original = refined_estimate(seed, &layout, source, None, run)?;
        let moved = refined_estimate(seed, &layout, source, Some(&motion), run)?;
        let expected = motion.apply(original);
        let discrepancy =
            (expected[0] - moved[0]).hypot(expected[1] - moved[1]).hypot(expected[2] - moved[2]);
        Ok::<_, String>((motion, discrepancy))
    });
    let mut outcome = GateOutcome::new("srp_lm_invariance", results.len());
    let mut worst: f64 = 0.0;
    for (seed, result) in &results {
        match result {
            Ok((motion, discrepancy)) => {
                worst = worst.max(*discrepancy);
                if *discrepancy > config.invariance.maximum_position_discrepancy_m {
                    outcome.fail_seed(*seed, format!("{motion:?}: refined estimate off its transformed twin by {discrepancy:.4} m"));
                }
            }
            Err(error) => outcome.fail_seed(*seed, error.clone()),
        }
    }
    outcome.summary(format!("worst SRP+LM discrepancy {:.2e} m (gate {} m)", worst, config.invariance.maximum_position_discrepancy_m));
    outcome
}
