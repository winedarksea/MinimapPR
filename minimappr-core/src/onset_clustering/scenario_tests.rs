//! End-to-end scenarios for the onset-clustering port on analytically delayed
//! linear-FM calls (exact fractional delays: the call is evaluated at
//! `t - r / c`), deterministic noise and 5 cm tetrahedral nodes. Python
//! parity lives in `tests/test_onset_clustering_cross_language_parity.py`;
//! these check the behaviour the reference was built for.

use std::collections::BTreeMap;

use super::bounded_trf::{bounded_trf_least_squares, VerticalBounds};
use super::pipeline::{localize_onset_clusters, OnsetClusteringInput};
use super::refinement::ScaledTdoaResiduals;
use super::srp_grid::{distance, Point3};
use super::types::ClusterStatus;
use super::OnsetClusteringConfig;
use crate::tdoa_solver::nalgebra::Vector3;
use crate::tdoa_solver::scipy_trf_compatible::{scipy_trf_least_squares, TrfSettings};

const SAMPLE_RATE_HZ: u32 = 12_000;
const SOUND_SPEED_MPS: f64 = 343.2;
const WINDOW_SECONDS: f64 = 0.9;
const CALL_START_S: f64 = 0.1;
const CALL_DURATION_S: f64 = 0.5;
const SWEEP_HZ: (f64, f64) = (2_000.0, 2_600.0);
const NOISE_RMS: f64 = 0.002;
const SOURCE_AMPLITUDE: f64 = 2.0;
const TETRA_EDGE_M: f64 = 0.05;
const NODE_ORIGINS_M: [(&str, Point3); 3] = [
    ("node-a", [0.0, 0.0, 0.0]),
    ("node-b", [20.0, 0.0, 0.0]),
    ("node-c", [10.0, 18.0, 0.0]),
];
const BIRD_M: Point3 = [6.0, 12.0, 2.0];

fn tetra_offsets() -> [Point3; 4] {
    let a = TETRA_EDGE_M;
    let vertices = [
        [0.0, 0.0, 0.0],
        [a, 0.0, 0.0],
        [0.5 * a, 3f64.sqrt() / 2.0 * a, 0.0],
        [0.5 * a, 3f64.sqrt() / 6.0 * a, (2.0 / 3.0f64).sqrt() * a],
    ];
    let mut centroid = [0.0; 3];
    for vertex in &vertices {
        for axis in 0..3 {
            centroid[axis] += vertex[axis] / 4.0;
        }
    }
    vertices.map(|v| [v[0] - centroid[0], v[1] - centroid[1], v[2] - centroid[2]])
}

/// Deterministic approximately Gaussian noise (sum of uniforms from an LCG).
struct NoiseSource(u64);

impl NoiseSource {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

fn call_sample(time_s: f64) -> f64 {
    if !(0.0..CALL_DURATION_S).contains(&time_s) {
        return 0.0;
    }
    let sweep_rate = (SWEEP_HZ.1 - SWEEP_HZ.0) / CALL_DURATION_S;
    let phase = std::f64::consts::TAU * (SWEEP_HZ.0 * time_s + 0.5 * sweep_rate * time_s * time_s);
    let fade = 0.1 * CALL_DURATION_S;
    let envelope = if time_s < fade {
        0.5 - 0.5 * (std::f64::consts::PI * time_s / fade).cos()
    } else if time_s > CALL_DURATION_S - fade {
        0.5 - 0.5 * (std::f64::consts::PI * (CALL_DURATION_S - time_s) / fade).cos()
    } else {
        1.0
    };
    phase.sin() * envelope
}

fn window(nodes: &[(&str, Point3)], bird_m: Point3) -> OnsetClusteringInput {
    let samples = (WINDOW_SECONDS * SAMPLE_RATE_HZ as f64) as usize;
    let mut noise = NoiseSource(7);
    let mut input = OnsetClusteringInput {
        sensor_signals: BTreeMap::new(),
        sensor_positions_m: BTreeMap::new(),
        sensor_node_ids: BTreeMap::new(),
        sample_rate_hz: SAMPLE_RATE_HZ,
        sound_speed_mps: SOUND_SPEED_MPS,
        interpolation_factor: 4,
        far_field_default_range_m: 50.0,
        consumed_onset_samples_by_node: BTreeMap::new(),
        attribution_audible_snr_db: None,
    };
    for (node, origin) in nodes {
        for (channel, offset) in tetra_offsets().iter().enumerate() {
            let position = [
                origin[0] + offset[0],
                origin[1] + offset[1],
                origin[2] + offset[2],
            ];
            let range = distance(&position, &bird_m);
            let gain = SOURCE_AMPLITUDE / range.max(1.0);
            let delay = CALL_START_S + range / SOUND_SPEED_MPS;
            let signal = (0..samples)
                .map(|index| {
                    gain * call_sample(index as f64 / SAMPLE_RATE_HZ as f64 - delay)
                        + NOISE_RMS * noise.gaussian()
                })
                .collect();
            let sensor = format!("{node}:ch{channel}");
            input.sensor_signals.insert(sensor.clone(), signal);
            input.sensor_positions_m.insert(sensor.clone(), position);
            input.sensor_node_ids.insert(sensor, node.to_string());
        }
    }
    input
}

#[test]
fn multi_node_fix_lands_on_the_bird() {
    let result = localize_onset_clusters(
        &window(&NODE_ORIGINS_M, BIRD_M),
        &OnsetClusteringConfig::default(),
    );
    let accepted: Vec<_> = result
        .clusters
        .iter()
        .filter(|c| c.status == ClusterStatus::Accepted)
        .collect();
    assert_eq!(
        accepted.len(),
        1,
        "{:?}",
        result
            .clusters
            .iter()
            .map(|c| (c.status, c.position_m))
            .collect::<Vec<_>>()
    );
    let fix = accepted[0];
    let horizontal_error =
        ((fix.position_m[0] - BIRD_M[0]).powi(2) + (fix.position_m[1] - BIRD_M[1]).powi(2)).sqrt();
    assert!(horizontal_error < 1.5, "fix {:?}", fix.position_m);
    assert!(
        fix.position_m[2] >= -1.5,
        "below ground: {:?}",
        fix.position_m
    );
    assert_eq!(fix.range_projection_mode.as_deref(), Some("range_refined"));
    assert!(fix.range_interval_m.is_some());
}

#[test]
fn single_tetra_gives_a_bearing_with_unobservable_range() {
    let result = localize_onset_clusters(
        &window(&NODE_ORIGINS_M[..1], BIRD_M),
        &OnsetClusteringConfig::default(),
    );
    let primary = result
        .clusters
        .iter()
        .find(|c| c.status == ClusterStatus::Accepted)
        .expect("an accepted cluster");
    let unit = |p: &Point3| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / length, p[1] / length, p[2] / length]
    };
    let (estimate, truth) = (unit(&primary.position_m), unit(&BIRD_M));
    let cosine = estimate[0] * truth[0] + estimate[1] * truth[1] + estimate[2] * truth[2];
    assert!(
        cosine.clamp(-1.0, 1.0).acos().to_degrees() < 10.0,
        "bearing {:?}",
        primary.position_m
    );
    assert_eq!(
        primary.range_projection_mode.as_deref(),
        Some("range_bearing_projected")
    );
}

#[test]
fn consumed_onsets_are_not_relocalized() {
    let config = OnsetClusteringConfig::default();
    let mut input = window(&NODE_ORIGINS_M, BIRD_M);
    let first = localize_onset_clusters(&input, &config);
    assert!(first.frame_count > 0);
    input.consumed_onset_samples_by_node = first.consumed_onset_samples_by_node;
    let second = localize_onset_clusters(&input, &config);
    assert_eq!(second.frame_count, 0);
    assert!(second.clusters.is_empty());
}

#[test]
fn bounded_trf_matches_unbounded_when_the_solution_is_interior_and_respects_bounds() {
    let sensors = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(20.0, 0.0, 0.0),
        Vector3::new(10.0, 18.0, 0.0),
        Vector3::new(10.0, 6.0, 7.0),
    ];
    let truth = Vector3::new(6.0, 12.0, 3.0);
    let sample_time_s = 1.0 / 48_000.0;
    let mut rows = Vec::new();
    for first in 0..sensors.len() {
        for second in first + 1..sensors.len() {
            let tdoa = ((truth - sensors[first]).norm() - (truth - sensors[second]).norm())
                / SOUND_SPEED_MPS;
            rows.push((sensors[first], sensors[second], tdoa, 1.0));
        }
    }
    let model = ScaledTdoaResiduals {
        rows,
        sound_speed_mps: SOUND_SPEED_MPS,
        residual_scale_s: sample_time_s,
    };
    let settings = TrfSettings {
        ftol: 1e-8,
        xtol: 1e-8,
        gtol: 1e-8 / (sample_time_s * sample_time_s),
        max_function_evaluations: 300,
    };
    let start = Vector3::new(7.0, 11.0, 2.0);
    let wide = VerticalBounds {
        lower_z_m: -1.5,
        upper_z_m: 48.5,
    };
    let bounded = bounded_trf_least_squares(
        &model,
        start,
        &wide,
        &settings,
        sample_time_s * sample_time_s,
    )
    .expect("bounded solve");
    let unbounded = scipy_trf_least_squares(&model, start, &settings).expect("unbounded solve");
    assert!(bounded.success && unbounded.success);
    assert!(
        (bounded.position_m - truth).norm() < 1e-3,
        "{:?}",
        bounded.position_m
    );
    assert!((unbounded.position_m - truth).norm() < 1e-3);
    let tight = VerticalBounds {
        lower_z_m: -1.5,
        upper_z_m: 1.0,
    };
    let clamped = bounded_trf_least_squares(
        &model,
        Vector3::new(7.0, 11.0, 0.5),
        &tight,
        &settings,
        sample_time_s * sample_time_s,
    )
    .expect("tight solve");
    assert!(
        clamped.position_m[2] <= 1.0 && clamped.position_m[2] >= -1.5,
        "{:?}",
        clamped.position_m
    );
}
