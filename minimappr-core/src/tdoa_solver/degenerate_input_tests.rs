//! Panic-freedom on hostile inputs (NaN, zero sound speed, coincident or
//! self-paired sensors) and the compact-array far-field guard. Each solver
//! must return `Ok` or `Err`, never abort a phone or WASM worker.

use nalgebra::Vector3;

use super::cartesian_tdoa_solve::{solve_cartesian_tdoa, CartesianTdoaSolveRequest};
use super::joint_objective::{solve_joint_position, JointSolverConfig};
use super::joint_residual_terms::{JointObservations, MicPairTdoaObservation};
use super::loop_consistency::{filter_loop_consistent_pairs, LoopConsistencyConfig};
use super::node_descriptor::{
    ClockSource, NodeClock, NodeDescriptor, NodeKind, NodeOrientation, OrientationSource,
    PositionSource,
};
use super::test_support::*;
use crate::range_projection::RANGE_REFINED;

const TETRA_OFFSETS_M: [[f64; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];

fn tetra_layout() -> super::measurement_types::SensorPositions {
    let entries: Vec<(String, [f64; 3])> = TETRA_OFFSETS_M
        .iter()
        .enumerate()
        .map(|(index, offset)| {
            (
                format!("t:ch{index}"),
                [offset[0], offset[1], offset[2] + 1.5],
            )
        })
        .collect();
    let borrowed: Vec<(&str, [f64; 3])> = entries
        .iter()
        .map(|(sensor_id, position)| (sensor_id.as_str(), *position))
        .collect();
    positions(&borrowed)
}

fn distributed_request(source_m: Vector3<f64>) -> CartesianTdoaSolveRequest {
    let layout = positions(&[
        ("n0", [0.0, 0.0, 1.5]),
        ("n1", [18.0, 1.0, 1.2]),
        ("n2", [17.0, 19.0, 1.8]),
        ("n3", [-1.0, 17.5, 1.0]),
    ]);
    let measurements =
        synthetic_measurements(&layout, &source_m, 0.0, &mut DeterministicGaussian::new(11));
    request_for(layout, measurements)
}

#[test]
fn cartesian_solve_never_panics_on_hostile_inputs() {
    let source_m = Vector3::new(7.0, 9.0, 6.0);
    let mut variants: Vec<CartesianTdoaSolveRequest> = Vec::new();

    let mut nan_tdoa = distributed_request(source_m);
    nan_tdoa.measurements[0].tdoa_seconds = f64::NAN;
    variants.push(nan_tdoa);

    let mut nan_sensor = distributed_request(source_m);
    nan_sensor
        .sensor_positions
        .insert("n1".into(), Vector3::new(f64::NAN, 1.0, 1.2));
    variants.push(nan_sensor);

    for sound_speed_mps in [0.0, -343.0, f64::NAN, f64::INFINITY] {
        let mut request = distributed_request(source_m);
        request.sound_speed_mps = sound_speed_mps;
        variants.push(request);
    }

    let mut zero_rate = distributed_request(source_m);
    zero_rate.sample_rate_hz = 0;
    zero_rate.interpolation_factor = 0;
    variants.push(zero_rate);

    let mut nan_weight = distributed_request(source_m);
    nan_weight.measurements[1].weight = f64::NAN;
    nan_weight.measurements[2].weight = -1.0;
    variants.push(nan_weight);

    let mut self_pair = distributed_request(source_m);
    self_pair.measurements[0].sensor_b = self_pair.measurements[0].sensor_a.clone();
    variants.push(self_pair);

    let mut nan_reference = distributed_request(source_m);
    if let Some(delay) = nan_reference.reference_tdoa_s.values_mut().next() {
        *delay = f64::NAN;
    }
    variants.push(nan_reference);

    let mut nan_far_field = distributed_request(source_m);
    nan_far_field.far_field_initial_range_m = f64::NAN;
    variants.push(nan_far_field);

    let single = positions(&[("solo", [0.0, 0.0, 0.0])]);
    variants.push(request_for(single, Vec::new()));

    for request in &variants {
        if let Ok(solve) = solve_cartesian_tdoa(request) {
            // A returned fix must at least be a number, not silent NaN.
            assert!(solve.position_m.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn compact_array_far_source_does_not_collapse_onto_the_node() {
    // Ideal TDOAs from a 400 m source: before the far-field guard the solve
    // collapsed to a few centimetres from the tetra.
    let layout = tetra_layout();
    let source_m = Vector3::new(400.0, 100.0, 30.0);
    let measurements =
        synthetic_measurements(&layout, &source_m, 0.0, &mut DeterministicGaussian::new(5));
    let request = request_for(layout.clone(), measurements);
    let solve = solve_cartesian_tdoa(&request).expect("tetra solve");
    let centroid_m = layout.values().fold(Vector3::zeros(), |sum, p| sum + p) / 4.0;
    let offset = solve.position_m - centroid_m;
    let range_m = offset.norm();
    assert!(range_m > 1.0, "collapsed to {range_m} m");
    assert_ne!(solve.range_projection_mode, RANGE_REFINED);
    let true_bearing = (source_m - centroid_m).normalize();
    let bearing_error_rad = (offset / range_m)
        .dot(&true_bearing)
        .clamp(-1.0, 1.0)
        .acos();
    assert!(
        bearing_error_rad < 0.1,
        "bearing off by {bearing_error_rad} rad"
    );
}

fn mono_phone(node_id: &str, position_m: [f64; 3]) -> NodeDescriptor {
    NodeDescriptor {
        node_id: node_id.into(),
        kind: NodeKind::PhoneMono,
        position_m,
        position_source: PositionSource::AcousticMesh,
        position_std_m: 0.0,
        mic_offsets_m: vec![[0.0; 3]],
        orientation: NodeOrientation::identity(OrientationSource::Unknown),
        clock: NodeClock {
            source: ClockSource::AcousticChirp,
            clock_domain_id: 1,
            clock_offset_std_s: 0.0,
        },
        has_speaker: true,
    }
}

#[test]
fn joint_solve_never_panics_on_hostile_inputs() {
    let nodes = vec![
        mono_phone("a", [0.0, 0.0, 1.0]),
        mono_phone("b", [10.0, 0.0, 1.0]),
        mono_phone("c", [0.0, 10.0, 1.0]),
        mono_phone("d", [10.0, 10.0, 3.0]),
    ];
    let pair =
        |node_a: &str, node_b: &str, tdoa_seconds: f64, timing_std_s: f64| MicPairTdoaObservation {
            node_a: node_a.into(),
            mic_a: 0,
            node_b: node_b.into(),
            mic_b: 0,
            tdoa_seconds,
            timing_std_s,
        };
    let hostile_pairs = vec![
        vec![pair("a", "b", f64::NAN, 1e-5)],
        vec![pair("a", "b", 1e-3, 0.0)],
        vec![pair("a", "b", 1e-3, f64::NAN)],
        vec![pair("a", "a", 0.0, 1e-5)],
        vec![pair("a", "b", 1e-3, 1e-5), pair("a", "c", 1e9, 1e-5)],
    ];
    for mic_pair_tdoas in hostile_pairs {
        let observations = JointObservations {
            mic_pair_tdoas,
            ..JointObservations::default()
        };
        let _ = solve_joint_position(&nodes, &observations, &JointSolverConfig::default());
    }
    for sound_speed_mps in [0.0, f64::NAN] {
        let config = JointSolverConfig {
            sound_speed_mps,
            ..JointSolverConfig::default()
        };
        let observations = JointObservations {
            mic_pair_tdoas: vec![pair("a", "b", 1e-3, 1e-5), pair("a", "c", 2e-3, 1e-5)],
            ..JointObservations::default()
        };
        let _ = solve_joint_position(&nodes, &observations, &config);
    }
    let coincident = vec![mono_phone("a", [0.0; 3]), mono_phone("b", [0.0; 3])];
    let observations = JointObservations {
        mic_pair_tdoas: vec![pair("a", "b", 0.0, 1e-5)],
        ..JointObservations::default()
    };
    assert!(
        solve_joint_position(&coincident, &observations, &JointSolverConfig::default()).is_err()
    );
}

#[test]
fn loop_consistency_never_panics_on_hostile_config() {
    let layout = tetra_layout();
    let measurements = synthetic_measurements(
        &layout,
        &Vector3::new(3.0, 2.0, 1.0),
        0.0,
        &mut DeterministicGaussian::new(9),
    );
    for pair_timing_std_s in [0.0, -1.0, f64::NAN] {
        let config = LoopConsistencyConfig {
            pair_timing_std_s,
            closure_tolerance_sigma: 3.0,
        };
        let report = filter_loop_consistent_pairs(&measurements, &config);
        assert_eq!(
            report.kept.len() + report.rejected.len(),
            measurements.len()
        );
    }
}

#[test]
fn bearing_fusion_extreme_timestamps_are_stale_not_overflow() {
    use super::bearing_fusion::{
        fuse_bearings, BearingFusionConfig, BearingFusionRejection, BearingObservation,
    };
    let observation = |node_id: &str, origin_m: [f64; 3], event_time_ns: i64| BearingObservation {
        node_id: node_id.into(),
        origin_m,
        direction: [0.0, 1.0, 0.0],
        lateral_std_m: 0.5,
        confidence: 0.8,
        event_time_ns,
    };
    let mut observations = vec![
        observation("a", [0.0, 0.0, 0.0], i64::MIN),
        observation("b", [40.0, 0.0, 0.0], i64::MAX),
    ];
    observations[1].direction = [-0.6, 0.8, 0.0];
    assert_eq!(
        fuse_bearings(&observations, &BearingFusionConfig::default()),
        Err(BearingFusionRejection::Stale)
    );
}

#[test]
fn scipy_trf_with_zero_evaluation_budget_terminates() {
    use super::cartesian_tdoa_solve::CartesianResidualModel;
    use super::scipy_trf_compatible::{scipy_trf_least_squares, TrfSettings};
    let request = distributed_request(Vector3::new(7.0, 9.0, 6.0));
    let model = CartesianResidualModel {
        request: &request,
        centroid_m: Vector3::new(8.5, 9.4, 1.4),
        sample_time_std_s: 1.0 / 192_000.0,
    };
    let settings = TrfSettings {
        ftol: 0.0,
        xtol: 0.0,
        gtol: 0.0,
        max_function_evaluations: 0,
    };
    let outcome = scipy_trf_least_squares(&model, Vector3::new(1.0, 1.0, 1.0), &settings)
        .expect("finite start");
    assert!(!outcome.success);
}
