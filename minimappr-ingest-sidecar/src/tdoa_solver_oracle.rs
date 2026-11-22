//! Hidden `tdoa-solver-oracle` subcommand: runs `minimappr_core::tdoa_solver`
//! on a batch of cases supplied as JSON on stdin and prints one JSON result per
//! case. Used by `tests/test_tdoa_solver_cross_language_parity.py` to compare
//! the Rust port against Python `cartesian_tdoa.solve_cartesian_tdoa` and
//! `multi_node_bearing_fusion.fuse_bearings` goldens.
//!
//! Non-finite floats (e.g. `gdop = inf`) are emitted as JSON `null`.

use std::collections::BTreeMap;

use minimappr_core::tdoa_solver::bearing_fusion::{
    fuse_bearings, BearingFusionConfig, BearingObservation,
};
use minimappr_core::tdoa_solver::cartesian_tdoa_solve::{
    solve_cartesian_tdoa, CartesianRefinementEngine, CartesianTdoaSolveRequest,
    BEARING_OBSERVED_OBS_FACTOR_FLOOR, DEFAULT_FAR_FIELD_INITIAL_RANGE_M,
};
use minimappr_core::tdoa_solver::measurement_types::{
    AmplitudeRatioConstraint, BearingPrior, PairTdoaMeasurement, SpatialBearingConstraint,
};
use minimappr_core::tdoa_solver::nalgebra::{Matrix3, Vector3};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct BearingPriorInput {
    direction: [f64; 3],
    strength: f64,
    quality: f64,
}

#[derive(Deserialize)]
struct BearingConstraintInput {
    node_id: String,
    origin_m: [f64; 3],
    direction: [f64; 3],
    strength: f64,
    quality: f64,
}

#[derive(Deserialize)]
struct AmplitudeConstraintInput {
    node_a: String,
    node_b: String,
    origin_a_m: [f64; 3],
    origin_b_m: [f64; 3],
    measured_log_amplitude_ratio: f64,
    strength: f64,
}

fn default_far_field_initial_range_m() -> f64 {
    DEFAULT_FAR_FIELD_INITIAL_RANGE_M
}
fn default_true() -> bool {
    true
}
fn default_obs_floor() -> f64 {
    BEARING_OBSERVED_OBS_FACTOR_FLOOR
}

#[derive(Deserialize)]
struct CartesianCaseInput {
    sensor_positions: BTreeMap<String, [f64; 3]>,
    measurements: Vec<PairTdoaMeasurement>,
    sound_speed_mps: f64,
    sample_rate_hz: u32,
    interpolation_factor: u32,
    reference_sensor: String,
    reference_tdoa_s: BTreeMap<String, f64>,
    #[serde(default)]
    bearing_prior: Option<BearingPriorInput>,
    #[serde(default)]
    bearing_constraints: Vec<BearingConstraintInput>,
    #[serde(default)]
    amplitude_constraints: Vec<AmplitudeConstraintInput>,
    #[serde(default = "default_far_field_initial_range_m")]
    far_field_initial_range_m: f64,
    #[serde(default)]
    far_field_prior_radial_std_m: Option<f64>,
    #[serde(default = "default_true")]
    radial_refinement_enabled: bool,
    #[serde(default = "default_obs_floor")]
    bearing_observed_obs_floor: f64,
}

#[derive(Deserialize)]
struct BearingFusionCaseInput {
    observations: Vec<BearingObservation>,
    #[serde(default)]
    config: Option<BearingFusionConfig>,
}

fn vector(values: [f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

fn finite_or_null(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn matrix_rows(matrix: &Matrix3<f64>) -> Value {
    json!((0..3)
        .map(|row| (0..3)
            .map(|column| matrix[(row, column)])
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

fn run_cartesian_case(input: CartesianCaseInput) -> Value {
    let request = CartesianTdoaSolveRequest {
        sensor_positions: input
            .sensor_positions
            .into_iter()
            .map(|(sensor_id, position)| (sensor_id, vector(position)))
            .collect(),
        measurements: input.measurements,
        sound_speed_mps: input.sound_speed_mps,
        sample_rate_hz: input.sample_rate_hz,
        interpolation_factor: input.interpolation_factor,
        reference_sensor: input.reference_sensor,
        reference_tdoa_s: input.reference_tdoa_s,
        bearing_prior: input.bearing_prior.map(|prior| BearingPrior {
            direction: vector(prior.direction),
            strength: prior.strength,
            quality: prior.quality,
        }),
        bearing_constraints: input
            .bearing_constraints
            .into_iter()
            .map(|constraint| SpatialBearingConstraint {
                node_id: constraint.node_id,
                origin_m: vector(constraint.origin_m),
                direction: vector(constraint.direction),
                strength: constraint.strength,
                quality: constraint.quality,
            })
            .collect(),
        amplitude_constraints: input
            .amplitude_constraints
            .into_iter()
            .map(|constraint| AmplitudeRatioConstraint {
                node_a: constraint.node_a,
                node_b: constraint.node_b,
                origin_a_m: vector(constraint.origin_a_m),
                origin_b_m: vector(constraint.origin_b_m),
                measured_log_amplitude_ratio: constraint.measured_log_amplitude_ratio,
                strength: constraint.strength,
            })
            .collect(),
        far_field_initial_range_m: input.far_field_initial_range_m,
        far_field_prior_radial_std_m: input.far_field_prior_radial_std_m,
        radial_refinement_enabled: input.radial_refinement_enabled,
        bearing_observed_obs_floor: input.bearing_observed_obs_floor,
        canopy_height_prior: None,
        refinement_engine: CartesianRefinementEngine::ScipyTrfParity,
    };
    match solve_cartesian_tdoa(&request) {
        Ok(solve) => json!({
            "position_m": [solve.position_m.x, solve.position_m.y, solve.position_m.z],
            "covariance_m2": matrix_rows(&solve.covariance_m2),
            "confidence": solve.confidence,
            "gdop": finite_or_null(solve.gdop),
            "residual_rms_seconds": solve.residual_rms_seconds,
            "radial_observability": solve.radial_observability,
            "range_projection_mode": solve.range_projection_mode,
            "condition_number": finite_or_null(solve.condition_number),
        }),
        Err(error) => json!({ "error": error.to_string() }),
    }
}

fn run_bearing_fusion_case(input: BearingFusionCaseInput) -> Value {
    let config = input.config.unwrap_or_default();
    match fuse_bearings(&input.observations, &config) {
        Ok(fused) => json!({
            "position_m": [fused.position_m.x, fused.position_m.y, fused.position_m.z],
            "covariance_m2": matrix_rows(&fused.covariance_m2),
            "range_m": fused.range_m,
            "confidence": fused.confidence,
            "contributor_node_ids": fused.contributor_node_ids,
        }),
        Err(rejection) => json!({ "rejection": rejection.reason() }),
    }
}

/// Input: `{"cases": [{"kind": "cartesian" | "bearing_fusion", ...}, ...]}`.
/// Output: `{"results": [...]}` in case order.
pub fn run_tdoa_solver_oracle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let value: Value = serde_json::from_str(&input)?;
    let mut results = Vec::new();
    for case in value["cases"].as_array().ok_or("cases")? {
        let result = match case["kind"].as_str().ok_or("kind")? {
            "cartesian" => run_cartesian_case(serde_json::from_value(case.clone())?),
            "bearing_fusion" => run_bearing_fusion_case(serde_json::from_value(case.clone())?),
            other => return Err(format!("unknown case kind {other:?}").into()),
        };
        results.push(result);
    }
    println!("{}", serde_json::to_string(&json!({ "results": results }))?);
    Ok(())
}
