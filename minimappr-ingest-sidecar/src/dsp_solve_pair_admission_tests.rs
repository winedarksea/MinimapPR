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

fn pair_tdoa(ch_a: usize, ch_b: usize) -> PairTdoa {
    PairTdoa {
        ch_a,
        ch_b,
        tdoa: minimappr_core::gcc_phat::TdoaResult {
            delay_samples: 0.0,
            lag_seconds: 0.0,
            peak_value: 1.0,
            confidence: 1.0,
        },
    }
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
        "own intra-node pair still solves"
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
fn covariance_inflation_uses_the_worst_correlated_cross_node_pair() {
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
    assert_eq!(
        admission.cross_node_timing_variance_m2(&[pair_tdoa(0, 1)], SOUND_SPEED_MPS),
        0.0,
        "intra-node pairs share one clock"
    );
    let worst_seconds = admission.pair_timing_uncertainty_seconds(0, 3);
    let variance_m2 = admission.cross_node_timing_variance_m2(
        &[pair_tdoa(0, 1), pair_tdoa(0, 2), pair_tdoa(0, 3)],
        SOUND_SPEED_MPS,
    );
    let expected_m2 = (f64::from(SOUND_SPEED_MPS) * worst_seconds).powi(2);
    assert!((variance_m2 - expected_m2).abs() < 1e-9);
    assert!(
        variance_m2 > 0.4,
        "2 ms of NTP error is ~0.7 m of range difference"
    );
}
