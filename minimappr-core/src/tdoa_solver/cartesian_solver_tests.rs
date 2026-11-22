//! Synthetic-geometry tests for the Cartesian TDOA solve.

use nalgebra::Vector3;

use super::canopy_height_prior::CanopyHeightPrior;
use super::cartesian_tdoa_solve::{solve_cartesian_tdoa, CartesianRefinementEngine};
use super::test_support::*;
use super::TdoaSolverError;
use crate::range_projection::RANGE_REFINED;

const MONTE_CARLO_NOISE_SAMPLE_PERIODS: f64 = 50.0;

fn distributed_layout() -> super::measurement_types::SensorPositions {
    positions(&[
        ("n0", [0.0, 0.0, 1.5]),
        ("n1", [18.0, 1.0, 1.2]),
        ("n2", [17.0, 19.0, 1.8]),
        ("n3", [-1.0, 17.5, 1.0]),
        ("n4", [8.0, -6.0, 2.5]),
    ])
}

#[test]
fn noise_free_distributed_layout_recovers_source_with_both_engines() {
    let source = Vector3::new(7.0, 9.0, 6.0);
    let layout = distributed_layout();
    let measurements =
        synthetic_measurements(&layout, &source, 0.0, &mut DeterministicGaussian::new(1));
    for engine in [
        CartesianRefinementEngine::ScipyTrfParity,
        CartesianRefinementEngine::LevenbergMarquardt,
    ] {
        let mut request = request_for(layout.clone(), measurements.clone());
        request.refinement_engine = engine;
        let solve = solve_cartesian_tdoa(&request).unwrap();
        assert!(
            (solve.position_m - source).norm() < 1e-3,
            "{engine:?}: {:?}",
            solve.position_m
        );
        assert_eq!(solve.range_projection_mode, RANGE_REFINED);
        assert!(solve.gdop.is_finite());
    }
}

#[test]
fn engines_agree_on_noisy_well_conditioned_layout() {
    let source = Vector3::new(-30.0, 25.0, 12.0);
    let layout = distributed_layout();
    let one_sample_s = 1.0 / (SAMPLE_RATE_HZ * INTERPOLATION_FACTOR) as f64;
    let measurements = synthetic_measurements(
        &layout,
        &source,
        2.0 * one_sample_s,
        &mut DeterministicGaussian::new(2),
    );
    let mut trf_request = request_for(layout, measurements);
    let trf = solve_cartesian_tdoa(&trf_request).unwrap();
    trf_request.refinement_engine = CartesianRefinementEngine::LevenbergMarquardt;
    let lm = solve_cartesian_tdoa(&trf_request).unwrap();
    assert!((trf.position_m - lm.position_m).norm() / lm.position_m.norm() < 1e-5);
    assert!((trf.position_m - source).norm() < 1.0);
}

#[test]
fn collinear_layout_is_flagged_not_refined_and_does_not_panic() {
    let layout = positions(&[
        ("l0", [0.0, 0.0, 1.5]),
        ("l1", [6.0, 0.0, 1.5]),
        ("l2", [12.0, 0.0, 1.5]),
        ("l3", [18.0, 0.0, 1.5]),
    ]);
    let source = Vector3::new(9.0, 14.0, 3.0);
    let measurements =
        synthetic_measurements(&layout, &source, 5.0e-6, &mut DeterministicGaussian::new(8));
    for engine in [
        CartesianRefinementEngine::ScipyTrfParity,
        CartesianRefinementEngine::LevenbergMarquardt,
    ] {
        let mut request = request_for(layout.clone(), measurements.clone());
        request.refinement_engine = engine;
        let solve = solve_cartesian_tdoa(&request).unwrap();
        assert_ne!(
            solve.range_projection_mode, RANGE_REFINED,
            "{engine:?} {solve:?}"
        );
        assert!(solve.condition_number > 1e6 || !solve.condition_number.is_finite());
        // The along-axis coordinate is still observed.
        assert!((solve.position_m.x - source.x).abs() < 0.5);
    }
}

#[test]
fn degenerate_inputs_return_errors() {
    let coincident = positions(&[
        ("a", [1.0, 1.0, 1.0]),
        ("b", [1.0, 1.0, 1.0]),
        ("c", [1.0, 1.0, 1.0]),
    ]);
    let measurements = synthetic_measurements(
        &coincident,
        &Vector3::new(5.0, 0.0, 0.0),
        0.0,
        &mut DeterministicGaussian::new(3),
    );
    assert!(solve_cartesian_tdoa(&request_for(coincident, measurements)).is_err());

    let layout = distributed_layout();
    let mut measurements = synthetic_measurements(
        &layout,
        &Vector3::new(3.0, 3.0, 3.0),
        0.0,
        &mut DeterministicGaussian::new(4),
    );
    measurements.truncate(2);
    assert!(matches!(
        solve_cartesian_tdoa(&request_for(layout.clone(), measurements.clone())),
        Err(TdoaSolverError::InsufficientMeasurements {
            needed: 3,
            available: 2
        })
    ));
    measurements[0].sensor_a = "ghost".into();
    assert!(matches!(
        solve_cartesian_tdoa(&request_for(layout, measurements)),
        Err(TdoaSolverError::UnknownSensor(_))
    ));
}

#[test]
fn canopy_prior_selects_the_above_ground_mirror_on_a_flat_layout() {
    // Coplanar ground layout: the TDOAs of (x, y, z) and (x, y, −z) are identical.
    let layout = positions(&[
        ("g0", [0.0, 0.0, 0.0]),
        ("g1", [15.0, 0.0, 0.0]),
        ("g2", [15.0, 15.0, 0.0]),
        ("g3", [0.0, 15.0, 0.0]),
        ("g4", [7.0, -4.0, 0.0]),
    ]);
    let source = Vector3::new(6.0, 8.0, 12.0);
    let measurements =
        synthetic_measurements(&layout, &source, 0.0, &mut DeterministicGaussian::new(5));
    for engine in [
        CartesianRefinementEngine::ScipyTrfParity,
        CartesianRefinementEngine::LevenbergMarquardt,
    ] {
        let mut request = request_for(layout.clone(), measurements.clone());
        request.refinement_engine = engine;
        request.canopy_height_prior = Some(CanopyHeightPrior::default());
        let solve = solve_cartesian_tdoa(&request).unwrap();
        assert!(
            solve.position_m.z > 0.0,
            "{engine:?}: {:?}",
            solve.position_m
        );
        assert!(
            (solve.position_m - source).norm() < 0.05,
            "{engine:?}: {:?}",
            solve.position_m
        );
    }
}

#[test]
fn cartesian_covariance_matches_monte_carlo_scatter() {
    let layout = distributed_layout();
    let source = Vector3::new(9.0, 8.0, 7.0);
    // Noise well above the solver's 5 cm position-std floor (≈ 50 samples ≈
    // 9 cm of path) so the comparison tests the SVD covariance, not the floor.
    let noise_std_s =
        MONTE_CARLO_NOISE_SAMPLE_PERIODS / (SAMPLE_RATE_HZ * INTERPOLATION_FACTOR) as f64;
    let mut noise = DeterministicGaussian::new(77);
    let trial_count = 300;
    let mut solved = Vec::with_capacity(trial_count);
    let mut predicted_trace_sum = 0.0;
    for _ in 0..trial_count {
        let measurements = synthetic_measurements(&layout, &source, noise_std_s, &mut noise);
        let mut request = request_for(layout.clone(), measurements);
        request.refinement_engine = CartesianRefinementEngine::LevenbergMarquardt;
        let solve = solve_cartesian_tdoa(&request).unwrap();
        predicted_trace_sum += solve.covariance_m2.trace();
        solved.push(solve.position_m);
    }
    let empirical_trace = sample_covariance(&solved).trace();
    let predicted_trace = predicted_trace_sum / trial_count as f64;
    let ratio = empirical_trace / predicted_trace;
    // The solver scales by the residual RMS, which underestimates σ by
    // √((m − 3)/m) for m pair rows, so empirical ≳ predicted; bound both ways.
    assert!(
        (0.7..1.6).contains(&ratio),
        "empirical/predicted trace = {ratio}"
    );
}
