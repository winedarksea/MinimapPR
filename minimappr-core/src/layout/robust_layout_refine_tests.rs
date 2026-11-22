use super::*;
use crate::layout::gps_anchor_fit::weighted_kabsch_2d;
use crate::sync::chirp_codes::SplitMix64;

const TRUE_POSITIONS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [14.0, 2.0, 1.5],
    [6.0, 11.0, 3.0],
    [-5.0, 8.0, 0.5],
    [9.0, -9.0, -1.0],
    [20.0, 12.0, 2.0],
];
const RANGE_SIGMA_M: f64 = 0.03;

fn gaussian(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn true_distance(i: usize, j: usize) -> f64 {
    let (a, b) = (TRUE_POSITIONS[i], TRUE_POSITIONS[j]);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn measurements(seed: u64) -> Vec<RangeMeasurement> {
    let mut rng = SplitMix64::new(seed);
    let mut list = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            list.push(RangeMeasurement {
                node_i: i,
                node_j: j,
                distance_m: true_distance(i, j) + RANGE_SIGMA_M * gaussian(&mut rng),
                sigma_m: RANGE_SIGMA_M,
            });
        }
    }
    list
}

fn heights() -> Vec<f64> {
    TRUE_POSITIONS.iter().map(|p| p[2]).collect()
}

/// Worst horizontal error after the best rigid alignment (mirror allowed).
fn aligned_worst_error(solution: &LayoutSolution) -> f64 {
    let source: Vec<[f64; 2]> = solution.positions_m.iter().map(|p| [p[0], p[1]]).collect();
    let target: Vec<[f64; 2]> = TRUE_POSITIONS.iter().map(|p| [p[0], p[1]]).collect();
    [false, true]
        .iter()
        .map(|&mirrored| {
            let transform = weighted_kabsch_2d(&source, &target, &[1.0; 6], mirrored);
            source
                .iter()
                .zip(&target)
                .map(|(s, t)| {
                    let p = transform.apply(*s);
                    ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn recovers_layout_with_heights() {
    let solution = solve_layout(
        6,
        &heights(),
        &measurements(1),
        &LayoutRefineConfig::default(),
    )
    .unwrap();
    assert!(solution.is_rigid);
    assert!(solution.rejected_measurements.is_empty());
    assert!(
        aligned_worst_error(&solution) < 0.08,
        "{}",
        aligned_worst_error(&solution)
    );
    assert!(solution
        .horizontal_sigmas_m
        .iter()
        .all(|s| *s > 0.0 && *s < 0.1));
}

#[test]
fn ignoring_heights_distorts_the_layout() {
    let flat = vec![0.0; 6];
    let solution =
        solve_layout(6, &flat, &measurements(1), &LayoutRefineConfig::default()).unwrap();
    assert!(aligned_worst_error(&solution) > 0.15);
}

#[test]
fn reflection_locked_range_is_rejected() {
    let mut list = measurements(2);
    list[3].distance_m += 4.0; // a trunk echo picked instead of the direct path
    let solution = solve_layout(6, &heights(), &list, &LayoutRefineConfig::default()).unwrap();
    assert_eq!(solution.rejected_measurements, vec![3]);
    assert!(aligned_worst_error(&solution) < 0.1);
}

#[test]
fn dangling_node_is_not_rigid() {
    // Node 5 measured only to node 0: it can swing around node 0.
    let list: Vec<RangeMeasurement> = measurements(3)
        .into_iter()
        .filter(|m| m.node_j != 5 || m.node_i == 0)
        .collect();
    let solution = solve_layout(6, &heights(), &list, &LayoutRefineConfig::default()).unwrap();
    assert!(!solution.is_rigid);
}

#[test]
fn height_count_mismatch_is_an_error() {
    let result = solve_layout(
        6,
        &[0.0; 5],
        &measurements(1),
        &LayoutRefineConfig::default(),
    );
    assert_eq!(result, Err(LayoutError::HeightCountMismatch));
}
