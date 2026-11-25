use super::super::{predicted_reference_tau_s, refine_reference_position};
use super::*;

const SOUND_SPEED_MPS: f32 = 343.0;
const NODE_SPREAD_M: f32 = 100.0;
const NTP_CLOCK_STD_SECONDS: f64 = 2.0e-3;
const GPS_CLOCK_STD_SECONDS: f64 = 1.0e-6;
/// Sirith tetra offsets about the node origin.
const TETRA_OFFSETS_M: [[f32; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];
/// Channels 0-3: tetra A at the origin (GPS); 4-7: tetra B 100 m east (NTP).
const CLOCK_INDEX_BY_CHANNEL: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

fn two_tetra_mics() -> Vec<[f32; 3]> {
    let mut mics = TETRA_OFFSETS_M.to_vec();
    mics.extend(
        TETRA_OFFSETS_M
            .iter()
            .map(|offset| [offset[0] + NODE_SPREAD_M, offset[1], offset[2]]),
    );
    mics
}

/// Broadside source at `range_m` from the midpoint between the two nodes.
fn broadside_source_m(range_m: f32) -> [f32; 3] {
    [NODE_SPREAD_M * 0.5, range_m, 0.0]
}

/// Exact reference-channel TDOAs (reference channel 0) of a source.
fn exact_measurements(source_m: [f32; 3], mics: &[[f32; 3]]) -> Vec<ReferenceMeasurement> {
    (1..mics.len())
        .map(|channel_index| ReferenceMeasurement {
            channel_index,
            tdoa_seconds: predicted_reference_tau_s(
                source_m,
                mics[channel_index],
                mics[0],
                SOUND_SPEED_MPS,
            ),
            peak_value: 1.0,
        })
        .collect()
}

fn clock_covariance_at(range_m: f32, clock_std_seconds: [f64; 2]) -> [[f32; 3]; 3] {
    let mics = two_tetra_mics();
    let source_m = broadside_source_m(range_m);
    clock_offset_position_covariance_m2(
        source_m,
        0,
        &exact_measurements(source_m, &mics),
        &mics,
        SOUND_SPEED_MPS,
        ChannelClockOffsets {
            clock_index_by_channel: &CLOCK_INDEX_BY_CHANNEL,
            clock_offset_std_seconds: &clock_std_seconds,
        },
    )
    .expect("a cross-node solve has a clock term")
}

fn trace(covariance_m2: [[f32; 3]; 3]) -> f32 {
    covariance_m2[0][0] + covariance_m2[1][1] + covariance_m2[2][2]
}

#[test]
fn one_clock_has_no_offset_term() {
    let mics = TETRA_OFFSETS_M.to_vec();
    let source_m = [3.0, 4.0, 0.5];
    let covariance = clock_offset_position_covariance_m2(
        source_m,
        0,
        &exact_measurements(source_m, &mics),
        &mics,
        SOUND_SPEED_MPS,
        ChannelClockOffsets {
            clock_index_by_channel: &[0, 0, 0, 0],
            clock_offset_std_seconds: &[NTP_CLOCK_STD_SECONDS],
        },
    );
    assert!(covariance.is_none());
}

#[test]
fn weak_clock_covariance_at_one_km_dwarfs_the_same_pair_at_fifty_m() {
    let clocks = [GPS_CLOCK_STD_SECONDS, NTP_CLOCK_STD_SECONDS];
    let near_trace_m2 = trace(clock_covariance_at(50.0, clocks));
    let far_trace_m2 = trace(clock_covariance_at(1_000.0, clocks));
    // Geometric dilution: lateral ~ r/b, range ~ r^2/b, so 20x the range is
    // well over 400x the variance.
    assert!(
        far_trace_m2 > 400.0 * near_trace_m2,
        "1 km {far_trace_m2} m^2 vs 50 m {near_trace_m2} m^2"
    );
    // The retired flat inflation added (c * u)^2, ~0.7 m of std, at any range.
    let flat_std_m = f64::from(SOUND_SPEED_MPS) * NTP_CLOCK_STD_SECONDS;
    let far_std_m = f64::from(far_trace_m2).sqrt();
    assert!(
        far_std_m > 10.0 * flat_std_m,
        "1 km std {far_std_m} m is not tens of metres"
    );
    // A GPS-class pair at the same range stays small.
    let gps_trace_m2 = trace(clock_covariance_at(
        1_000.0,
        [GPS_CLOCK_STD_SECONDS, GPS_CLOCK_STD_SECONDS],
    ));
    assert!(gps_trace_m2 < far_trace_m2 * 1.0e-4);
}

#[test]
fn clock_term_matches_the_refit_shift_of_an_offset_node() {
    // With the reference node's clock exact, the term is sigma_B^2 g g^T,
    // where g is how far the unweighted refit moves per second of node B
    // offset. Shift B's TDOAs and refit to measure g directly. Broadside, the
    // shift is along the node baseline (x); the weakly seen vertical axis is
    // left out because the damped refit barely moves along it.
    const OFFSET_SECONDS: f32 = 1.0e-4;
    let mics = two_tetra_mics();
    let source_m = broadside_source_m(50.0);
    let mut shifted = exact_measurements(source_m, &mics);
    for measurement in &mut shifted {
        if CLOCK_INDEX_BY_CHANNEL[measurement.channel_index] == 1 {
            measurement.tdoa_seconds += OFFSET_SECONDS;
        }
    }
    let refit_m = refine_reference_position(source_m, 0, &shifted, &mics, SOUND_SPEED_MPS);
    let measured_baseline_shift_m = f64::from((refit_m[0] - source_m[0]).abs());

    let covariance = clock_covariance_at(50.0, [0.0, NTP_CLOCK_STD_SECONDS]);
    let predicted_baseline_shift_m =
        f64::from(covariance[0][0]).sqrt() / NTP_CLOCK_STD_SECONDS * f64::from(OFFSET_SECONDS);
    assert!(
        (measured_baseline_shift_m - predicted_baseline_shift_m).abs()
            < 0.02 * predicted_baseline_shift_m,
        "refit moved {measured_baseline_shift_m} m, covariance predicts {predicted_baseline_shift_m} m"
    );
}

#[test]
fn a_point_nodes_position_error_is_a_clock_offset_of_sigma_over_c() {
    // One sensor per node: a rigid position error d moves that node's only
    // arrival by u . d / c, a scalar with std sigma / c shared by all of its
    // TDOAs, which is exactly how a clock offset enters.
    use crate::tdoa_node_position_covariance::SensorNodePositionStd;
    const HAND_MEASURED_STD_M: f64 = 0.3;
    let mics: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 1.5],
        [60.0, 0.0, 2.0],
        [0.0, 60.0, 1.0],
        [60.0, 60.0, 3.0],
        [30.0, 30.0, 6.0],
    ];
    let source_m = [35.0, 20.0, 4.0];
    let measurements = exact_measurements(source_m, &mics);
    let node_by_channel = [0, 1, 2, 3, 4];
    let position_std_m = [0.0, HAND_MEASURED_STD_M, 0.0, 0.0, 0.0];
    let position_term = super::super::reference_tdoa_node_position_covariance_m2(
        source_m,
        0,
        &measurements,
        &mics,
        SOUND_SPEED_MPS,
        SensorNodePositionStd {
            node_index_by_sensor: &node_by_channel,
            node_position_std_m: &position_std_m,
        },
    )
    .expect("stated error");
    let clock_term = clock_offset_position_covariance_m2(
        source_m,
        0,
        &measurements,
        &mics,
        SOUND_SPEED_MPS,
        ChannelClockOffsets {
            clock_index_by_channel: &node_by_channel,
            clock_offset_std_seconds: &[
                0.0,
                HAND_MEASURED_STD_M / f64::from(SOUND_SPEED_MPS),
                0.0,
                0.0,
                0.0,
            ],
        },
    )
    .expect("clock term");
    for row in 0..3 {
        for column in 0..3 {
            let scale = clock_term[row][row]
                .abs()
                .max(clock_term[column][column].abs());
            assert!(
                (position_term[row][column] - clock_term[row][column]).abs() <= 1.0e-4 * scale,
                "[{row}][{column}]: position {} vs clock {}",
                position_term[row][column],
                clock_term[row][column]
            );
        }
    }
}
