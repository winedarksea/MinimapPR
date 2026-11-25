//! The production point-network localizer, as it runs today.
//!
//! 1. Event window: the node-local window grown by the cross-node lag beyond
//!    half of it on each side, the lag capped at the coverage-wait cap
//!    (`localization_event_span.rs::event_solve_window_seconds`).
//! 2. Sidecar front-end and SRP: `estimate_tetrahedral_steering_with_solve_constraints`
//!    over every node, every pair admitted (same rate, long baselines, no
//!    timeline residual), clock offsets carried into its covariance
//!    (`actors/dsp_compute_srp_solve.rs`).
//! 3. Server re-solve (default `localization_single_node_solver =
//!    python_cartesian`): the pair TDOAs, inverse-variance weighted by pair
//!    clock uncertainty, into `solve_cartesian_tdoa` with the SRP steering
//!    direction as a bearing prior (`minimappr/core/rust_tdoa_solve.py`).
//!    No re-solve (fewer than 4 sensors or 3 pairs, or a solver error) falls
//!    back to the sidecar estimate, as `localized_handoff_solution.py` does.
//!
//! The server's reportability gates (range envelope, unobservable range,
//! covariance ceiling) are applied by the harness to every candidate alike.
//! Not modelled: the low-dominant-frequency lateral widening (it never fires
//! at point-network baselines, whose aliasing cutoff is far below bird
//! calls) and the confidence/GDOP gates, which ship disabled.

use std::collections::BTreeMap;
use std::time::Instant;

use minimappr_core::srp_phat::{
    estimate_tetrahedral_steering_with_solve_constraints, ChannelClockOffsets, NoiseFloorRelativeReference, SolveChannelConstraints,
    SolverReferenceChannelFloor, SrpPhatConfig, SrpPhatEvaluation, DEFAULT_ABSOLUTE_REFERENCE_CHANNEL_RMS,
};
use minimappr_core::tdoa_solver::cartesian_tdoa_solve::{solve_cartesian_tdoa, CartesianTdoaSolveRequest};
use minimappr_core::tdoa_solver::measurement_types::{normalized_pair_quality, BearingPrior, PairTdoaMeasurement, SensorPositions};
use minimappr_core::tdoa_solver::nalgebra::{DMatrix, DVector, Vector3};

use crate::bakeoff_candidate_fix::{matrix3_to_array, CandidateFix, CandidateSolveOutcome};
use crate::bakeoff_config::ProductionCandidateConfig;
use crate::bakeoff_scenes::BakeoffScene;
use crate::bakeoff_windows::{cut_window, BakeoffTrigger};

/// `localization_event_span.rs`: the far sensor may hear the event the max
/// lag before or after the trigger; a lag up to half the node-local window
/// is already tolerated.
const LAG_EXTENT_SIDES: f64 = 2.0;
const NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION: f64 = 0.5;
/// `rust_tdoa_solve.py::_bearing_prior_from_direction` strength floor.
const BEARING_PRIOR_MINIMUM_STRENGTH: f64 = 0.01;
const DIRECTION_NORM_EPSILON: f64 = 1.0e-9;
const LEAST_SQUARES_SINGULAR_VALUE_EPSILON: f64 = 1.0e-12;
const SRP_RESOLVED_ALGORITHM: &str = "srp_phat";

/// Both outputs of one production solve (they share the front-end).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProductionSolveOutcomes {
    /// What the server stores by default: the re-solve, else the sidecar estimate.
    pub deployed: CandidateSolveOutcome,
    /// The sidecar's own SRP estimate (`localization_single_node_solver = rust`).
    pub sidecar_srp: CandidateSolveOutcome,
}

pub fn sensor_id(node: usize) -> String {
    format!("node-{node:02}")
}

/// Seconds of audio the event solve reads; `lag_cap_s = None` lifts the cap.
pub fn production_event_window_seconds(aperture_m: f64, sound_speed_mps: f64, config: &ProductionCandidateConfig, lag_cap_s: Option<f64>) -> f64 {
    let node_local_window_s = config.node_local_window_seconds;
    let max_lag_s = lag_cap_s.map_or(aperture_m / sound_speed_mps, |cap| (aperture_m / sound_speed_mps).min(cap));
    let excess_lag_s = (max_lag_s - NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION * node_local_window_s).max(0.0);
    (node_local_window_s + LAG_EXTENT_SIDES * excess_lag_s).max(node_local_window_s)
}

fn srp_config(config: &ProductionCandidateConfig, quietest_node_floor_rms: f64, band_hz: [f32; 2]) -> SrpPhatConfig {
    SrpPhatConfig {
        localization_band_hz: band_hz,
        grid_resolution_m: config.grid_resolution_m,
        search_padding_m: config.search_padding_m,
        interp_factor: config.interpolation_factor as usize,
        far_field_default_range_m: config.far_field_default_range_m as f32,
        far_field_max_range_m: config.far_field_max_range_m,
        reference_channel_floor: SolverReferenceChannelFloor {
            absolute_rms: DEFAULT_ABSOLUTE_REFERENCE_CHANNEL_RMS,
            noise_floor_relative: Some(NoiseFloorRelativeReference {
                noise_floor_rms: quietest_node_floor_rms as f32,
                relative_db: config.solver_floor_relative_db,
            }),
        },
        ..SrpPhatConfig::default()
    }
}

/// `pair_timing_uncertainty.py::inverse_variance_pair_weight`.
fn inverse_variance_pair_weight(measurement_quality: f64, timing_uncertainty_s: f64, reference_time_std_s: f64) -> f64 {
    let timing_variance_ratio = (timing_uncertainty_s / reference_time_std_s).powi(2);
    1.0 / ((1.0 / measurement_quality) + timing_variance_ratio)
}

/// `tdoa_measurements.py::solve_reference_tdoas`: weighted least-squares
/// reference-relative delays from the pairwise TDOAs.
fn solve_reference_tdoas(measurements: &[PairTdoaMeasurement], sensor_ids: &[String], reference: &str) -> BTreeMap<String, f64> {
    let measured: Vec<&String> = sensor_ids.iter().filter(|id| id.as_str() != reference).collect();
    if measurements.is_empty() || measured.is_empty() {
        return BTreeMap::new();
    }
    let column_of = |id: &str| measured.iter().position(|candidate| candidate.as_str() == id);
    let mut design = DMatrix::<f64>::zeros(measurements.len(), measured.len());
    let mut targets = DVector::<f64>::zeros(measurements.len());
    for (row, measurement) in measurements.iter().enumerate() {
        let weight = measurement.weight.sqrt();
        if let Some(column) = column_of(&measurement.sensor_a) {
            design[(row, column)] = weight;
        }
        if let Some(column) = column_of(&measurement.sensor_b) {
            design[(row, column)] = -weight;
        }
        targets[row] = measurement.tdoa_seconds * weight;
    }
    let Ok(solution) = design.svd(true, true).solve(&targets, LEAST_SQUARES_SINGULAR_VALUE_EPSILON) else {
        return BTreeMap::new();
    };
    measured.iter().enumerate().map(|(column, id)| ((*id).clone(), solution[column])).collect()
}

struct ResolveInputs<'a> {
    scene: &'a BakeoffScene,
    evaluation: &'a SrpPhatEvaluation,
    config: &'a ProductionCandidateConfig,
}

/// `rust_tdoa_solve.py::solve_localization_from_rust_tdoas`; `None` = no re-solve.
fn python_cartesian_resolve(inputs: &ResolveInputs<'_>) -> Option<CandidateFix> {
    let ResolveInputs { scene, evaluation, config } = inputs;
    let sensor_ids: Vec<String> = (0..scene.nodes.len()).map(sensor_id).collect();
    if sensor_ids.len() < config.resolve_minimum_sensor_count {
        return None;
    }
    let reference_time_std_s = 1.0 / f64::from(scene.sample_rate_hz * config.interpolation_factor.max(1));
    let mut measurements = Vec::new();
    let mut quality_by_sensor: BTreeMap<String, f64> = sensor_ids.iter().map(|id| (id.clone(), 0.0)).collect();
    for pair in &evaluation.pair_tdoas {
        let lag_s = f64::from(pair.tdoa.lag_seconds);
        if pair.ch_a == pair.ch_b || !lag_s.is_finite() {
            continue;
        }
        let confidence = f64::from(pair.tdoa.confidence);
        let quality = normalized_pair_quality(confidence);
        // Every pair here is cross-node: independent clocks add in quadrature.
        let timing_uncertainty_s = scene.nodes[pair.ch_a].clock_class.declared_offset_std_s().hypot(scene.nodes[pair.ch_b].clock_class.declared_offset_std_s());
        measurements.push(PairTdoaMeasurement {
            sensor_a: sensor_ids[pair.ch_a].clone(),
            sensor_b: sensor_ids[pair.ch_b].clone(),
            tdoa_seconds: lag_s,
            correlation_peak: confidence,
            weight: inverse_variance_pair_weight(quality.max(config.minimum_pair_measurement_quality), timing_uncertainty_s, reference_time_std_s),
        });
        *quality_by_sensor.get_mut(&sensor_ids[pair.ch_a])? += quality;
        *quality_by_sensor.get_mut(&sensor_ids[pair.ch_b])? += quality;
    }
    if measurements.len() < config.resolve_minimum_pair_count {
        return None;
    }
    let reference = sensor_ids.iter().max_by(|a, b| quality_by_sensor[*a].total_cmp(&quality_by_sensor[*b]))?.clone();
    let reference_tdoa_s = solve_reference_tdoas(&measurements, &sensor_ids, &reference);
    let mean_confidence = measurements.iter().map(|measurement| measurement.correlation_peak).sum::<f64>() / measurements.len() as f64;
    let positions: SensorPositions =
        scene.nodes.iter().enumerate().map(|(index, node)| (sensor_ids[index].clone(), Vector3::from(node.measured_position_m))).collect();
    let mut request = CartesianTdoaSolveRequest::with_defaults(
        positions,
        measurements,
        scene.sound_speed_mps,
        scene.sample_rate_hz,
        config.interpolation_factor,
        reference,
        reference_tdoa_s,
    );
    request.bearing_prior = bearing_prior(evaluation.localization.steering_direction, mean_confidence, config.node_bearing_strength);
    request.far_field_initial_range_m = config.far_field_default_range_m;
    request.bearing_observed_obs_floor = config.bearing_observed_obs_floor;
    let solve = solve_cartesian_tdoa(&request).ok()?;
    let position = [solve.position_m[0], solve.position_m[1], solve.position_m[2]];
    Some(CandidateFix {
        position_m: position,
        covariance_m2: Some(matrix3_to_array(&solve.covariance_m2)),
        range_projection_mode: Some(solve.range_projection_mode.to_string()),
        attribution_cues: Vec::new(),
    })
}

fn bearing_prior(steering_direction: [f32; 3], quality: f64, bearing_strength: f64) -> Option<BearingPrior> {
    let direction = Vector3::new(f64::from(steering_direction[0]), f64::from(steering_direction[1]), f64::from(steering_direction[2]));
    let norm = direction.norm();
    if !norm.is_finite() || norm <= DIRECTION_NORM_EPSILON {
        return None;
    }
    let clipped_quality = quality.clamp(0.0, 1.0);
    Some(BearingPrior {
        direction: direction / norm,
        strength: (bearing_strength * clipped_quality).max(BEARING_PRIOR_MINIMUM_STRENGTH),
        quality: clipped_quality,
    })
}

fn sidecar_fix(evaluation: &SrpPhatEvaluation, cue: (usize, f64)) -> Option<CandidateFix> {
    let localization = &evaluation.localization;
    if localization.resolved_algorithm != SRP_RESOLVED_ALGORITHM {
        return None;
    }
    let position = localization.position_m?;
    Some(CandidateFix {
        position_m: position.map(f64::from),
        covariance_m2: localization.position_covariance_m2.map(|matrix| matrix.map(|row| row.map(f64::from))),
        range_projection_mode: localization.range_projection_mode.clone(),
        attribution_cues: vec![cue],
    })
}

/// How the front-end cuts and filters the event window. Production is the
/// capped lag, no taper and its shipped band; the diagnostics change these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductionFrontEndVariant {
    pub lag_cap_s: Option<f64>,
    /// Tukey taper: this fraction of the window is cosine-ramped (half at each end).
    pub taper_fraction: Option<f64>,
    /// GCC-PHAT band; `None` keeps `localization_band_hz`.
    pub band_hz: Option<[f32; 2]>,
}

/// A rectangular cut through a loud call leaves the same broadband step at
/// the same sample index on every channel; PHAT whitening then favours zero
/// lag. A Tukey taper removes the step (diagnostic only: production does not taper).
fn apply_tukey_taper(window: &mut [f32], taper_fraction: f64) {
    let ramp_samples = ((0.5 * taper_fraction * window.len() as f64) as usize).min(window.len() / 2);
    if ramp_samples == 0 {
        return;
    }
    let length = window.len();
    for index in 0..ramp_samples {
        let gain = 0.5 * (1.0 - (std::f64::consts::PI * index as f64 / ramp_samples as f64).cos());
        window[index] *= gain as f32;
        window[length - 1 - index] *= gain as f32;
    }
}

/// One production solve for `trigger`, its front-end set as `variant` says.
pub fn run_production_candidate(
    scene: &BakeoffScene,
    trigger: &BakeoffTrigger,
    config: &ProductionCandidateConfig,
    quietest_node_floor_rms: f64,
    variant: ProductionFrontEndVariant,
) -> ProductionSolveOutcomes {
    let span_s = production_event_window_seconds(scene.aperture_m, scene.sound_speed_mps, config, variant.lag_cap_s);
    let start_s = trigger.stream_time_s - 0.5 * span_s;
    let mut windows: Vec<Vec<f32>> = scene.streams.iter().map(|stream| cut_window(stream, scene.sample_rate_hz, start_s, start_s + span_s)).collect();
    if let Some(taper_fraction) = variant.taper_fraction {
        windows.iter_mut().for_each(|window| apply_tukey_taper(window, taper_fraction));
    }
    let positions_m: Vec<[f32; 3]> = scene.nodes.iter().map(|node| node.measured_position_m.map(|value| value as f32)).collect();
    let active_channels: Vec<usize> = (0..scene.nodes.len()).collect();
    let clock_std_s: Vec<f64> = scene.nodes.iter().map(|node| node.clock_class.declared_offset_std_s()).collect();
    let started = Instant::now();
    let evaluation = estimate_tetrahedral_steering_with_solve_constraints(
        &windows,
        &active_channels,
        &positions_m,
        scene.sample_rate_hz,
        scene.sound_speed_mps as f32,
        srp_config(config, quietest_node_floor_rms, variant.band_hz.unwrap_or(config.localization_band_hz)),
        SolveChannelConstraints {
            admitted_pair_filter: None,
            channel_clock_offsets: Some(ChannelClockOffsets { clock_index_by_channel: &active_channels, clock_offset_std_seconds: &clock_std_s }),
        },
    );
    let sidecar_wall_time_s = started.elapsed().as_secs_f64();
    let cue = (trigger.node, trigger.stream_time_s);
    let sidecar = sidecar_fix(&evaluation, cue);
    let resolve_started = Instant::now();
    let resolved = python_cartesian_resolve(&ResolveInputs { scene, evaluation: &evaluation, config });
    let deployed_wall_time_s = sidecar_wall_time_s + resolve_started.elapsed().as_secs_f64();
    let deployed_fix = match resolved {
        Some(fix) => Some(CandidateFix { attribution_cues: vec![cue], ..fix }),
        None => sidecar.clone(),
    };
    let failure = |fix: &Option<CandidateFix>| fix.is_none().then(|| evaluation.localization.resolved_algorithm.clone());
    ProductionSolveOutcomes {
        deployed: CandidateSolveOutcome {
            failure_reason: failure(&deployed_fix),
            fixes: deployed_fix.into_iter().collect(),
            wall_time_s: deployed_wall_time_s,
            gate_rejections: Vec::new(),
        },
        sidecar_srp: CandidateSolveOutcome {
            failure_reason: failure(&sidecar),
            fixes: sidecar.into_iter().collect(),
            wall_time_s: sidecar_wall_time_s,
            gate_rejections: Vec::new(),
        },
    }
}
