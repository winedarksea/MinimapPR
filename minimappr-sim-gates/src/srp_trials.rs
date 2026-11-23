//! Audio-rendered SRP trials for the likelihood-contour calibration.
//!
//! A trial renders a random 5-phone layout and one call, then hands the
//! searcher node positions and clock models carrying realistic residuals
//! (±10 cm layout, ±10 µs clock). The *drawn* region (contour, or bearing
//! fan when range is unobservable) is checked against the truth, mapped
//! into the reported layout's frame by a proper rigid fit (as the mesh
//! scoring does). Rendering dominates the cost, so one render is reused
//! across every temperature a calibration tries.

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::mesh::srp_seed_refiner::{JointLmSeedRefiner, SeedRefinerConfig};
use minimappr_core::srp_layout::{TdoaSeedRefiner, LayoutNode, LayoutSrpConfig, LayoutSrpResult, LayoutSrpSearcher, NodeClip, UncertaintyDisplay};

use crate::gate_config::{RealisticPerturbationConfig, TrialGeometryConfig};
use crate::srp_scene::{build_scene, call_window, render, srp_config_for, true_layout_nodes, RenderedScene, SrpSceneConfig};
use crate::trial_geometry::{
    azimuth_from, gaussian, horizontal_centroid, inside_source, random_phone_layout, source_at_range, uniform_in,
    wrapped_angle_difference_rad,
};

/// SRP trial population (the core SRP tests' scale: 5 phones ~24 m).
#[derive(Debug, Clone, PartialEq)]
pub struct SrpTrialConfig {
    pub phone_count: usize,
    pub aperture_range_m: [f64; 2],
    /// Every other trial is outside the layout at a range in this band.
    pub outside_range_m: [f64; 2],
    pub scene: SrpSceneConfig,
}

impl Default for SrpTrialConfig {
    fn default() -> Self {
        Self { phone_count: 5, aperture_range_m: [18.0, 30.0], outside_range_m: [40.0, 150.0], scene: SrpSceneConfig::default() }
    }
}

pub struct RenderedSrpTrial {
    pub seed: u64,
    pub is_inside: bool,
    pub source_m: [f64; 3],
    pub true_nodes: Vec<LayoutNode>,
    /// What the searcher is told (residual-perturbed).
    pub reported_nodes: Vec<LayoutNode>,
    pub rendered: RenderedScene,
}

/// Build and render one trial.
pub fn render_srp_trial(
    seed: u64,
    config: &SrpTrialConfig,
    geometry: &TrialGeometryConfig,
    perturbation: &RealisticPerturbationConfig,
) -> Result<RenderedSrpTrial, String> {
    let aperture = uniform_in(seed, "srp_aperture", 0, config.aperture_range_m);
    let layout = random_phone_layout(seed, config.phone_count, aperture, geometry);
    let is_inside = seed.is_multiple_of(2);
    let source_m = if is_inside {
        inside_source(seed, &layout, aperture, geometry)
    } else {
        source_at_range(seed, &layout, uniform_in(seed, "srp_outside_range", 0, config.outside_range_m), geometry)
    };
    let rendered = render(&build_scene(seed, &layout, source_m, &config.scene))?;
    let true_nodes = true_layout_nodes(&rendered.plan);
    let reported_nodes = true_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let key = index as u64;
            let mut reported = node.clone();
            reported.position_m[0] += perturbation.layout_position_std_m * gaussian(seed, "srp_layout_error", key, 0);
            reported.position_m[1] += perturbation.layout_position_std_m * gaussian(seed, "srp_layout_error", key, 1);
            reported.position_m[2] += perturbation.phone_height_std_m * gaussian(seed, "srp_layout_error", key, 2);
            reported.clock.anchor_island_time_s += perturbation.clock_residual_std_s * gaussian(seed, "srp_clock_residual", key, 0);
            // The searcher is told the σ it suffers, as a host is told its layout and clock σ.
            reported.position_std_m = perturbation.layout_position_std_m.max(perturbation.phone_height_std_m);
            reported.clock_offset_std_s = perturbation.clock_residual_std_s;
            reported
        })
        .collect();
    Ok(RenderedSrpTrial { seed, is_inside, source_m, true_nodes, reported_nodes, rendered })
}

/// One localization of a rendered trial at one configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SrpTrialOutcome {
    pub horizontal_error_m: f64,
    pub bearing_error_deg: f64,
    pub drawn_region_covers_truth: bool,
    pub drew_bearing_fan: bool,
    /// Horizontal area of the near-field region columns (sharpness).
    pub region_area_m2: f64,
    pub cells_evaluated: usize,
    pub region_refinement_complete: bool,
    pub solve_seconds: f64,
    /// The pipeline's final estimate: the SRP `LmSeed` polished by the
    /// shipping `JointLmSeedRefiner` (None when the refiner refuses).
    pub lm_refined_horizontal_error_m: Option<f64>,
}

impl RenderedSrpTrial {
    pub fn base_config(&self) -> LayoutSrpConfig {
        srp_config_for(&self.rendered.plan)
    }

    pub fn localize(&self, config: &LayoutSrpConfig) -> Result<LayoutSrpResult, String> {
        let clips: Vec<NodeClip<'_>> = self
            .rendered
            .channels_per_device
            .iter()
            .map(|channels| NodeClip { channels: channels.iter().map(Vec::as_slice).collect(), first_sample_index: 0 })
            .collect();
        let window = call_window(&self.rendered.plan, SrpSceneConfig::default().window_padding_s);
        LayoutSrpSearcher::new(&self.reported_nodes, config)
            .and_then(|searcher| searcher.localize(&clips, window))
            .map_err(|error| format!("{error:?}"))
    }

    /// Truth expressed in the reported layout's frame (proper rigid fit of
    /// reported → true node positions, inverted).
    pub fn truth_in_reported_frame(&self) -> [f64; 2] {
        let reported: Vec<[f64; 2]> = self.reported_nodes.iter().map(|n| [n.position_m[0], n.position_m[1]]).collect();
        let truth: Vec<[f64; 2]> = self.true_nodes.iter().map(|n| [n.position_m[0], n.position_m[1]]).collect();
        let to_truth = weighted_kabsch_2d(&reported, &truth, &vec![1.0; reported.len()], false);
        let (sin, cos) = to_truth.rotation_rad.sin_cos();
        let (qx, qy) = (self.source_m[0] - to_truth.translation_m[0], self.source_m[1] - to_truth.translation_m[1]);
        [cos * qx + sin * qy, -sin * qx + cos * qy]
    }

    pub fn score(&self, result: &LayoutSrpResult, solve_seconds: f64) -> SrpTrialOutcome {
        let truth = self.truth_in_reported_frame();
        let estimate = result.estimate.position_m;
        let reported_positions: Vec<[f64; 3]> = self.reported_nodes.iter().map(|n| n.position_m).collect();
        let centroid = horizontal_centroid(&reported_positions);
        let truth_3d = [truth[0], truth[1], self.source_m[2]];
        let drawn_region_covers_truth = match result.display {
            UncertaintyDisplay::Contour => result.region.contains_xy(truth),
            UncertaintyDisplay::BearingFan(fan) => {
                let off_axis = wrapped_angle_difference_rad(azimuth_from(fan.origin_xy_m, truth_3d), fan.world_azimuth_rad).abs();
                let range = (truth[0] - fan.origin_xy_m[0]).hypot(truth[1] - fan.origin_xy_m[1]);
                off_axis <= fan.half_width_rad && range >= fan.minimum_range_m
            }
        };
        SrpTrialOutcome {
            horizontal_error_m: (estimate[0] - truth[0]).hypot(estimate[1] - truth[1]),
            bearing_error_deg: wrapped_angle_difference_rad(azimuth_from(centroid, estimate), azimuth_from(centroid, truth_3d))
                .abs()
                .to_degrees(),
            drawn_region_covers_truth,
            drew_bearing_fan: matches!(result.display, UncertaintyDisplay::BearingFan(_)),
            region_area_m2: result
                .region
                .columns
                .iter()
                .filter(|column| !column.is_far_field)
                .map(|column| 0.5 * (column.world_azimuth_rad[1] - column.world_azimuth_rad[0]) * (column.radial_m[1].powi(2) - column.radial_m[0].powi(2)))
                .sum(),
            cells_evaluated: result.estimate.cells_evaluated + result.region.refinement_cells_evaluated,
            region_refinement_complete: result.region.is_refinement_complete,
            solve_seconds,
            lm_refined_horizontal_error_m: self.lm_refined_position(result).map(|p| (p[0] - truth[0]).hypot(p[1] - truth[1])),
        }
    }

    /// LM polish of the SRP seed with each node's declared layout σ.
    pub fn lm_refined_position(&self, result: &LayoutSrpResult) -> Option<[f64; 3]> {
        let perturbation = RealisticPerturbationConfig::default();
        let refiner = JointLmSeedRefiner {
            config: SeedRefinerConfig::default(),
            node_position_std_m: vec![perturbation.layout_position_std_m.max(perturbation.phone_height_std_m); self.reported_nodes.len()],
        };
        refiner.refine(&result.lm_seed).ok().map(|fix| fix.position_m)
    }

    /// Localize at `temperature` (everything else shipping defaults) and score.
    pub fn evaluate_at_temperature(&self, temperature: f64) -> Result<SrpTrialOutcome, String> {
        let mut config = self.base_config();
        config.scoring.likelihood_temperature = temperature;
        let started = std::time::Instant::now();
        let result = self.localize(&config)?;
        Ok(self.score(&result, started.elapsed().as_secs_f64()))
    }

    /// Smallest temperature on `ladder` (ascending) whose drawn region
    /// covers the truth; ∞ if even the top rung does not. Binary search over
    /// the rungs: coverage is monotone in T (a hotter likelihood only widens
    /// the highest-density set) up to region-refinement granularity.
    pub fn minimum_covering_temperature(&self, ladder: &[f64]) -> Result<f64, String> {
        let Some(&top) = ladder.last() else { return Ok(f64::INFINITY) };
        if !self.evaluate_at_temperature(top)?.drawn_region_covers_truth {
            return Ok(f64::INFINITY);
        }
        let (mut uncovered_below, mut covered_at) = (None::<usize>, ladder.len() - 1);
        while covered_at > uncovered_below.map_or(0, |index| index + 1) {
            let low = uncovered_below.map_or(0, |index| index + 1);
            let middle = low + (covered_at - low) / 2;
            if self.evaluate_at_temperature(ladder[middle])?.drawn_region_covers_truth {
                covered_at = middle;
            } else {
                uncovered_below = Some(middle);
            }
        }
        Ok(ladder[covered_at])
    }
}

/// Log-spaced temperature ladder from `low` to `high` with `steps` rungs.
pub fn temperature_ladder(low: f64, high: f64, steps: usize) -> Vec<f64> {
    let ratio = (high / low).powf(1.0 / (steps.max(2) - 1) as f64);
    (0..steps).map(|step| low * ratio.powi(step as i32)).collect()
}
