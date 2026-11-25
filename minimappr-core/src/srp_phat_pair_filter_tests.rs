//! Admitted-pair filter of the SRP-PHAT solve (`estimate_tetrahedral_steering_with_pair_filter`).

use super::*;

const SAMPLE_RATE_HZ: u32 = 16_000;
const WINDOW_SAMPLES: usize = 4_096;
const SOUND_SPEED_MPS: f32 = 343.2;
const SOURCE_POSITION_M: [f32; 3] = [-2.0, 5.0, 2.0];
/// Sirith tetra offsets about the node origin.
const TETRA_OFFSETS_M: [[f32; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];

/// Tetra at the origin (channels 0-3) plus two point nodes 1 cm apart
/// (channels 4 and 5): the 4-5 pair is the one admission rejects.
fn three_node_mics() -> Vec<[f32; 3]> {
    let mut mics = TETRA_OFFSETS_M.to_vec();
    mics.push([4.0, 0.0, 0.0]);
    mics.push([4.0, 0.01, 0.0]);
    mics
}

fn noise(len: usize) -> Vec<f32> {
    let mut seed = 0x2468_ace0_u32;
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

/// One white-noise source heard at every mic with its true propagation delay.
fn point_source_windows(mics: &[[f32; 3]]) -> Vec<Vec<f32>> {
    let delays: Vec<f32> = mics
        .iter()
        .map(|mic| {
            magnitude(sub(SOURCE_POSITION_M, *mic)) / SOUND_SPEED_MPS * SAMPLE_RATE_HZ as f32
        })
        .collect();
    let earliest = delays.iter().copied().fold(f32::INFINITY, f32::min);
    let latest = delays.iter().copied().fold(0.0_f32, f32::max);
    let source = noise(WINDOW_SAMPLES + (latest - earliest).ceil() as usize + 2);
    delays
        .iter()
        .map(|delay| {
            let shift = delay - earliest;
            (0..WINDOW_SAMPLES)
                .map(|index| {
                    let position = index as f32 - shift;
                    if position < 0.0 {
                        return 0.0;
                    }
                    let lower = position.floor() as usize;
                    let fraction = position - lower as f32;
                    source[lower] * (1.0 - fraction) + source[lower + 1] * fraction
                })
                .collect()
        })
        .collect()
}

fn config() -> SrpPhatConfig {
    SrpPhatConfig {
        grid_resolution_m: 0.25,
        ..SrpPhatConfig::default()
    }
}

fn solve(filter: Option<AdmittedChannelPairFilter<'_>>) -> SrpPhatEvaluation {
    let mics = three_node_mics();
    let windows = point_source_windows(&mics);
    estimate_tetrahedral_steering_with_pair_filter(
        &windows,
        &[0, 1, 2, 3, 4, 5],
        &mics,
        SAMPLE_RATE_HZ,
        SOUND_SPEED_MPS,
        config(),
        filter,
    )
}

fn pair_keys(evaluation: &SrpPhatEvaluation) -> Vec<(usize, usize)> {
    evaluation
        .pair_tdoas
        .iter()
        .map(|pair| (pair.ch_a, pair.ch_b))
        .collect()
}

#[test]
fn no_filter_correlates_every_pair_exactly_like_the_unfiltered_entry_point() {
    let mics = three_node_mics();
    let windows = point_source_windows(&mics);
    let unfiltered = estimate_tetrahedral_steering(
        &windows,
        &[0, 1, 2, 3, 4, 5],
        &mics,
        SAMPLE_RATE_HZ,
        SOUND_SPEED_MPS,
        config(),
    );
    let no_filter = solve(None);
    assert_eq!(pair_keys(&no_filter).len(), 15);
    assert_eq!(pair_keys(&no_filter), pair_keys(&unfiltered));
    assert_eq!(
        no_filter.localization.position_m,
        unfiltered.localization.position_m
    );
    assert_eq!(
        no_filter.localization.position_covariance_m2,
        unfiltered.localization.position_covariance_m2
    );
}

#[test]
fn a_rejected_pair_is_never_correlated_and_the_rest_still_solve() {
    let admits_pair = |ch_a: usize, ch_b: usize| (ch_a, ch_b) != (4, 5);
    let evaluation = solve(Some(&admits_pair));
    let pairs = pair_keys(&evaluation);
    assert_eq!(pairs.len(), 14);
    assert!(!pairs.contains(&(4, 5)));
    assert_eq!(evaluation.localization.resolved_algorithm, "srp_phat");
    let position_m = evaluation.localization.position_m.expect("solved position");
    let bearing_dot = dot(
        normalize_or_zero(position_m),
        normalize_or_zero(SOURCE_POSITION_M),
    );
    assert!(bearing_dot > 0.95, "bearing off: {position_m:?}");
}

#[test]
fn a_channel_without_a_reference_pair_does_not_void_the_far_field_direction() {
    // Admitted: the tetra's own pairs, tetra ch0 - point ch4, and ch4 - ch5.
    // Channel 0 anchors the fit (most partners) and has no pair with ch5, so
    // ch5 has no relative TDOA; that used to abort the far-field direction.
    let admits_pair = |ch_a: usize, ch_b: usize| ch_b < 4 || (ch_a, ch_b) == (0, 4) || ch_a == 4;
    let evaluation = solve(Some(&admits_pair));
    assert_eq!(pair_keys(&evaluation).len(), 8);
    let mics = three_node_mics();
    let windows = point_source_windows(&mics);
    let pair_front_ends = ChannelPairFrontEnds::resolve(
        config().band_policy,
        &[0, 1, 2, 3, 4, 5],
        &mics,
        Some(&[0, 0, 0, 0, 1, 2]),
        SOUND_SPEED_MPS,
    );
    let observations = build_admitted_pair_observations(
        &windows,
        &[0, 1, 2, 3, 4, 5],
        &mics,
        SAMPLE_RATE_HZ,
        SOUND_SPEED_MPS,
        &pair_front_ends,
        config().interp_factor,
        Some(&admits_pair),
    );
    let measurements = build_reference_measurements(0, &observations);
    assert!(measurements.iter().all(|m| m.channel_index != 5));
    let (direction, _fit_error) = pairwise_far_field_direction(
        &[0, 1, 2, 3, 4, 5],
        0,
        &measurements,
        &mics,
        SOUND_SPEED_MPS,
    )
    .expect("direction from the channels that have a reference TDOA");
    assert!(
        dot(direction, normalize_or_zero(SOURCE_POSITION_M)) > 0.9,
        "bearing off: {direction:?}"
    );
}

fn solve_with_node_position_std(node_position_std_m: Option<&[f64]>) -> SrpPhatEvaluation {
    const NODE_INDEX_BY_CHANNEL: [usize; 6] = [0, 0, 0, 0, 1, 2];
    let mics = three_node_mics();
    let windows = point_source_windows(&mics);
    estimate_tetrahedral_steering_with_solve_constraints(
        &windows,
        &[0, 1, 2, 3, 4, 5],
        &mics,
        SAMPLE_RATE_HZ,
        SOUND_SPEED_MPS,
        config(),
        SolveChannelConstraints {
            channel_node_position_std: node_position_std_m.map(|stds| SensorNodePositionStd {
                node_index_by_sensor: &NODE_INDEX_BY_CHANNEL,
                node_position_std_m: stds,
            }),
            ..SolveChannelConstraints::default()
        },
    )
}

fn covariance_trace_m2(evaluation: &SrpPhatEvaluation) -> f32 {
    let covariance = evaluation
        .localization
        .position_covariance_m2
        .expect("solved covariance");
    covariance[0][0] + covariance[1][1] + covariance[2][2]
}

#[test]
fn a_hand_measured_node_widens_the_published_covariance_and_unstated_ones_do_not() {
    let without = solve_with_node_position_std(None);
    let unstated = solve_with_node_position_std(Some(&[0.0, 0.0, 0.0]));
    assert_eq!(
        unstated.localization.position_covariance_m2, without.localization.position_covariance_m2,
        "no stated accuracy keeps today's covariance"
    );
    let hand_measured_point = solve_with_node_position_std(Some(&[0.0, 0.3, 0.0]));
    assert_eq!(
        hand_measured_point.localization.position_m, without.localization.position_m,
        "a stated accuracy widens the error, it never moves the fix"
    );
    // A 0.3 m rigid error on the 4 m point node, with the source ~5 m out,
    // moves the fix by a sizeable fraction of that; the bare solve states
    // centimetres.
    let added_variance_m2 =
        covariance_trace_m2(&hand_measured_point) - covariance_trace_m2(&without);
    assert!(
        added_variance_m2 > 0.01,
        "added variance {added_variance_m2} m^2"
    );
}
