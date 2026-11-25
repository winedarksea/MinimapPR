use std::collections::BTreeMap;

use super::*;
use crate::dsp_worker::{sensor_id_for_node_channel, NodeClockAssessment, SolveSensorLayoutParts};
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, NodeTimeQuality, SensorPositionSource, SolveBufferScope,
    SolveCoordinateFrame, SolveSensorEntry,
};

const RATE_HZ: u32 = 16_000;
const SOUND_SPEED_MPS: f32 = 343.0;

fn sensor(node_id: &str, channel: usize, position_m: [f64; 3], rate_hz: u32) -> SolveSensorEntry {
    SolveSensorEntry {
        sensor_id: sensor_id_for_node_channel(node_id, channel),
        node_id: node_id.to_string(),
        node_channel_index: channel,
        position_m,
        sample_rate_hz: rate_hz,
        position_source: SensorPositionSource::ExplicitClusterManifest,
        position_std_m: None,
        clock_source: NodeTimeQuality::Unknown,
        clock_domain: "utc".to_string(),
        clock_offset_std_s: None,
    }
}

/// tetra-a ch0..1 (0, 1), point-b (2), point-c 1 cm from point-b (3),
/// point-d at 48 kHz (4).
fn layout(reporting_node_id: &str) -> SolveSensorLayout {
    SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors: vec![
            sensor("tetra-a", 0, [0.0, 0.05, 0.0], RATE_HZ),
            sensor("tetra-a", 1, [0.04, 0.0, 0.0], RATE_HZ),
            sensor("point-b", 0, [4.0, 0.0, 0.0], RATE_HZ),
            sensor("point-c", 0, [4.0, 0.01, 0.0], RATE_HZ),
            sensor("point-d", 0, [0.0, 4.0, 0.0], 48_000),
        ],
        layout_revision: 1,
        geometry_push_revision: None,
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::ExplicitClusterPositions,
            origin: None,
            projection: None,
        },
        buffer_scope: SolveBufferScope::ExplicitCluster,
        cluster_id: Some("cluster".to_string()),
        reporting_node_id: reporting_node_id.to_string(),
        node_local_routing_reason: None,
        solve_sample_rate_hz: RATE_HZ,
        half_space_by_node: BTreeMap::new(),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    })
}

const ALL_RATE_MATCHED: [usize; 4] = [0, 1, 2, 3];

fn assessment(
    node_id: &str,
    time_quality: NodeTimeQuality,
    timing_uncertainty_seconds: f64,
    residual_over_hard_limit: bool,
) -> NodeClockAssessment {
    NodeClockAssessment {
        node_id: node_id.to_string(),
        time_quality,
        timing_uncertainty_seconds,
        timeline_residual_seconds: None,
        last_timestamp_step_ns: None,
        timestamp_step_count: 0,
        residual_over_hard_limit,
    }
}

fn clocks(entries: &[NodeClockAssessment]) -> NodeClockAssessments {
    entries
        .iter()
        .map(|entry| (entry.node_id.clone(), entry.clone()))
        .collect()
}

fn gps_locked_clocks() -> NodeClockAssessments {
    clocks(&[
        assessment("tetra-a", NodeTimeQuality::GpsLocked, 1e-6, false),
        assessment("point-b", NodeTimeQuality::GpsLocked, 1e-6, false),
        assessment("point-c", NodeTimeQuality::GpsLocked, 1e-6, false),
        assessment("point-d", NodeTimeQuality::GpsLocked, 1e-6, false),
    ])
}

fn exclusions(admission: &SolvePairAdmission) -> Vec<(usize, usize, SolveExclusionReason)> {
    admission
        .excluded_pairs()
        .iter()
        .map(|pair| (pair.sensor_index_a, pair.sensor_index_b, pair.reason))
        .collect()
}

#[test]
fn short_baseline_pair_is_excluded_and_rate_mismatched_sensor_contributes_no_pairs() {
    let admission = SolvePairAdmission::compute(
        &layout("tetra-a"),
        &ALL_RATE_MATCHED,
        &gps_locked_clocks(),
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(
        exclusions(&admission),
        [(2, 3, SolveExclusionReason::PairBaselineBelowMinimum)]
    );
    assert!(!admission.admits_pair(3, 2));
    assert!(admission.admits_pair(0, 3) && admission.admits_pair(0, 1));
}

#[test]
fn inactive_sensor_pairs_are_excluded_for_coverage() {
    let admission = SolvePairAdmission::compute(
        &layout("tetra-a"),
        &[0, 1, 3],
        &gps_locked_clocks(),
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(
        exclusions(&admission),
        [
            (0, 2, SolveExclusionReason::CoverageBelowThreshold),
            (1, 2, SolveExclusionReason::CoverageBelowThreshold),
            (2, 3, SolveExclusionReason::CoverageBelowThreshold),
        ]
    );
}

#[test]
fn residual_over_limit_drops_only_that_nodes_cross_node_pairs() {
    let mut node_clock = gps_locked_clocks();
    node_clock.insert(
        "tetra-a".to_string(),
        assessment("tetra-a", NodeTimeQuality::NtpDisciplined, 0.02, true),
    );
    let layout = layout("point-b");
    let admission = SolvePairAdmission::compute(
        &layout,
        &ALL_RATE_MATCHED,
        &node_clock,
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(
        exclusions(&admission),
        [
            (0, 2, SolveExclusionReason::TimelineResidualOverLimit),
            (0, 3, SolveExclusionReason::TimelineResidualOverLimit),
            (1, 2, SolveExclusionReason::TimelineResidualOverLimit),
            (1, 3, SolveExclusionReason::TimelineResidualOverLimit),
            (2, 3, SolveExclusionReason::PairBaselineBelowMinimum),
        ]
    );
    assert!(
        admission.admits_pair(0, 1),
        "own intra-node pair stays admitted"
    );
    assert_eq!(admission.render_excluded_sensor_indices(&layout), [0, 1]);
}

#[test]
fn a_timeline_rejected_reporter_renders_only_its_own_channels() {
    let mut node_clock = gps_locked_clocks();
    node_clock.insert(
        "tetra-a".to_string(),
        assessment("tetra-a", NodeTimeQuality::NtpDisciplined, 0.02, true),
    );
    let layout = layout("tetra-a");
    let admission = SolvePairAdmission::compute(
        &layout,
        &ALL_RATE_MATCHED,
        &node_clock,
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(admission.render_excluded_sensor_indices(&layout), [2, 3, 4]);
}

#[test]
fn weak_clock_widens_the_pair_uncertainty_but_never_excludes() {
    let free_running_seconds = 0.05;
    let mut node_clock = gps_locked_clocks();
    node_clock.insert(
        "point-b".to_string(),
        assessment(
            "point-b",
            NodeTimeQuality::FreeRunning,
            free_running_seconds,
            false,
        ),
    );
    let layout = layout("tetra-a");
    let admission = SolvePairAdmission::compute(
        &layout,
        &ALL_RATE_MATCHED,
        &node_clock,
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(
        exclusions(&admission),
        [(2, 3, SolveExclusionReason::PairBaselineBelowMinimum)],
        "only the geometric exclusion"
    );
    assert!(admission.admits_pair(0, 2));
    assert_eq!(admission.pair_timing_uncertainty_seconds(0, 1), 0.0);
    let expected = (free_running_seconds.powi(2) + 1e-6_f64.powi(2)).sqrt();
    assert!((admission.pair_timing_uncertainty_seconds(0, 2) - expected).abs() < 1e-12);
    assert!(
        (admission.pair_timing_uncertainty_seconds(0, 3) - 2e-6_f64.sqrt() * 1e-3).abs() < 1e-12
    );
    assert!(admission.render_excluded_sensor_indices(&layout).is_empty());
}

#[test]
fn a_node_without_clock_evidence_is_timed_as_an_unknown_clock() {
    let config = LocalizationSolveConfig::default();
    let admission = SolvePairAdmission::compute(
        &layout("tetra-a"),
        &ALL_RATE_MATCHED,
        &NodeClockAssessments::new(),
        &config,
    );
    let unknown = config.localization_clock_unknown_uncertainty_seconds;
    assert!(
        (admission.pair_timing_uncertainty_seconds(1, 2) - unknown * 2.0_f64.sqrt()).abs() < 1e-12
    );
    assert_eq!(exclusions(&admission).len(), 1, "unknown never excludes");
}

#[test]
fn per_channel_clock_offsets_follow_the_layout_nodes() {
    let mut node_clock = gps_locked_clocks();
    node_clock.insert(
        "point-c".to_string(),
        assessment("point-c", NodeTimeQuality::NtpDisciplined, 0.002, false),
    );
    let admission = SolvePairAdmission::compute(
        &layout("tetra-a"),
        &ALL_RATE_MATCHED,
        &node_clock,
        &LocalizationSolveConfig::default(),
    );
    let clocks = admission.channel_clock_offsets();
    assert_eq!(clocks.clock_index_by_channel, [0, 0, 1, 2, 3]);
    assert_eq!(clocks.clock_offset_std_seconds, [1e-6, 1e-6, 0.002, 1e-6]);
}

#[test]
fn per_channel_node_position_std_follows_the_layout_nodes_and_unstated_is_zero() {
    let mut operator_placed = layout("tetra-a");
    // tetra-a hand-measured (one of its sensors states it), point-c a
    // stationary estimate; point-b and point-d state nothing.
    operator_placed.sensors[1].position_std_m = Some(0.3);
    operator_placed.sensors[3].position_std_m = Some(0.05);
    let admission = SolvePairAdmission::compute(
        &operator_placed,
        &ALL_RATE_MATCHED,
        &gps_locked_clocks(),
        &LocalizationSolveConfig::default(),
    );
    let nodes = admission.channel_node_position_std();
    assert_eq!(nodes.node_index_by_sensor, [0, 0, 1, 2, 3]);
    assert_eq!(nodes.node_position_std_m, [0.3, 0.0, 0.05, 0.0]);
}

const FAR_NODE_SPREAD_M: f64 = 100.0;
/// Sirith tetra offsets about the node origin.
const TETRA_OFFSETS_M: [[f64; 3]; 4] = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
];

/// Reporter tetra-a at the origin (channels 0-3), tetra-b 100 m east (4-7).
fn far_two_tetra_layout(reporting_node_id: &str) -> SolveSensorLayout {
    let tetra = |node_id: &str, origin_x_m: f64| {
        TETRA_OFFSETS_M
            .iter()
            .enumerate()
            .map(|(channel, offset)| {
                sensor(
                    node_id,
                    channel,
                    [offset[0] + origin_x_m, offset[1], offset[2]],
                    RATE_HZ,
                )
            })
            .collect::<Vec<_>>()
    };
    let mut sensors = tetra("tetra-a", 0.0);
    sensors.extend(tetra("tetra-b", FAR_NODE_SPREAD_M));
    SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors,
        layout_revision: 1,
        geometry_push_revision: None,
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::ExplicitClusterPositions,
            origin: None,
            projection: None,
        },
        buffer_scope: SolveBufferScope::ExplicitCluster,
        cluster_id: Some("cluster".to_string()),
        reporting_node_id: reporting_node_id.to_string(),
        node_local_routing_reason: None,
        solve_sample_rate_hz: RATE_HZ,
        half_space_by_node: BTreeMap::new(),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    })
}

const BOTH_TETRAS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

fn far_tetra_b_over_the_limit() -> NodeClockAssessments {
    clocks(&[
        assessment("tetra-a", NodeTimeQuality::GpsLocked, 1e-6, false),
        assessment("tetra-b", NodeTimeQuality::NtpDisciplined, 0.03, true),
    ])
}

#[test]
fn a_timeline_rejected_far_node_leaves_the_solve_geometry() {
    let config = LocalizationSolveConfig::default();
    let admission = SolvePairAdmission::compute(
        &far_two_tetra_layout("tetra-a"),
        &BOTH_TETRAS,
        &far_tetra_b_over_the_limit(),
        &config,
    );
    assert_eq!(admission.solve_sensor_indices(), [0, 1, 2, 3]);
    let timeline_exclusions = exclusions(&admission)
        .into_iter()
        .filter(|(_, _, reason)| *reason == SolveExclusionReason::TimelineResidualOverLimit)
        .count();
    assert_eq!(timeline_exclusions, 16, "provenance keeps every a-b pair");
    assert!(admission.admits_pair(4, 5), "b's own pairs stay admitted");

    let admission_reported_by_b = SolvePairAdmission::compute(
        &far_two_tetra_layout("tetra-b"),
        &BOTH_TETRAS,
        &far_tetra_b_over_the_limit(),
        &config,
    );
    assert_eq!(admission_reported_by_b.solve_sensor_indices(), [4, 5, 6, 7]);

    let healthy = SolvePairAdmission::compute(
        &far_two_tetra_layout("tetra-a"),
        &BOTH_TETRAS,
        &NodeClockAssessments::new(),
        &config,
    );
    assert_eq!(healthy.solve_sensor_indices(), BOTH_TETRAS);
}

#[test]
fn a_reporter_without_coverage_solves_on_the_largest_connected_group() {
    let admission = SolvePairAdmission::compute(
        &far_two_tetra_layout("tetra-a"),
        &[4, 5, 6, 7],
        &far_tetra_b_over_the_limit(),
        &LocalizationSolveConfig::default(),
    );
    assert_eq!(admission.solve_sensor_indices(), [4, 5, 6, 7]);
}

/// White noise from `source_m`, delayed to every channel of `positions_m`.
fn point_source_windows(source_m: [f32; 3], positions_m: &[[f32; 3]]) -> Vec<Vec<f32>> {
    const WINDOW_SAMPLES: usize = 4_096;
    let delays: Vec<f32> = positions_m
        .iter()
        .map(|position| {
            let distance_m = (0..3)
                .map(|axis| (source_m[axis] - position[axis]).powi(2))
                .sum::<f32>()
                .sqrt();
            distance_m / SOUND_SPEED_MPS * RATE_HZ as f32
        })
        .collect();
    let earliest = delays.iter().copied().fold(f32::INFINITY, f32::min);
    let latest = delays.iter().copied().fold(0.0_f32, f32::max);
    let mut seed = 0x1357_9bdf_u32;
    let source: Vec<f32> = (0..WINDOW_SAMPLES + (latest - earliest).ceil() as usize + 2)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as i32 as f32) / (i32::MAX as f32)
        })
        .collect();
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

#[test]
fn srp_on_the_reporter_component_solves_around_the_reporter_not_the_node_midpoint() {
    use crate::srp_phat::{
        estimate_tetrahedral_steering_with_solve_constraints, SolveChannelConstraints,
        SrpPhatConfig,
    };
    let layout = far_two_tetra_layout("tetra-a");
    let admission = SolvePairAdmission::compute(
        &layout,
        &BOTH_TETRAS,
        &far_tetra_b_over_the_limit(),
        &LocalizationSolveConfig::default(),
    );
    let source_m = [-3.0_f32, 4.0, 0.5];
    // Tetra b's audio is irrelevant to the solve; it hears the same source
    // on a clock that is known to be wrong.
    let windows = point_source_windows(source_m, &layout.positions_m);
    let admits_pair = |a: usize, b: usize| admission.admits_pair(a, b);
    let evaluation = estimate_tetrahedral_steering_with_solve_constraints(
        &windows,
        admission.solve_sensor_indices(),
        &layout.positions_m,
        RATE_HZ,
        SOUND_SPEED_MPS,
        SrpPhatConfig {
            half_space: layout.solve_half_space(admission.solve_sensor_indices()),
            // Pinned: this 5 m white-noise source sits on the single tetra's
            // near/far knife edge, which flips with the band edge alone
            // (probed 2026-11-25: range-projected at 3000/3200/3266/3300 Hz,
            // near-field boundary at 3100 and 3350-3500 Hz). The projected
            // answer's range-sized covariance would hide what this checks
            // (the aperture no longer spanning both nodes).
            band_policy: minimappr_core::localization_band_policy::LocalizationBandPolicy::with_intra_array_band(
                [50.0, 3430.0],
            ),
            ..SrpPhatConfig::default()
        },
        SolveChannelConstraints {
            admitted_pair_filter: Some(&admits_pair),
            channel_clock_offsets: Some(admission.channel_clock_offsets()),
            channel_node_position_std: Some(admission.channel_node_position_std()),
            array_index_by_channel: Some(admission.sensor_node_indices()),
        },
    );
    assert!(
        evaluation
            .pair_tdoas
            .iter()
            .all(|pair| pair.ch_a < 4 && pair.ch_b < 4),
        "only the reporter's pairs are correlated"
    );
    let localization = evaluation.localization;
    let position_m = localization
        .position_m
        .expect("the reporter's array solves");
    let source_norm = (source_m.iter().map(|axis| axis * axis).sum::<f32>()).sqrt();
    let bearing_cosine = (0..3)
        .map(|axis| localization.steering_direction[axis] * source_m[axis] / source_norm)
        .sum::<f32>();
    assert!(
        bearing_cosine > 0.95,
        "steering measured from the reporter, got {:?}",
        localization.steering_direction
    );
    let covariance = localization.position_covariance_m2.expect("covariance");
    let lateral_std_m = covariance[0][0].max(covariance[1][1]).sqrt();
    assert!(
        lateral_std_m < 25.0,
        "aperture no longer spans the 100 m node spread: {lateral_std_m} m at {position_m:?}"
    );
}
