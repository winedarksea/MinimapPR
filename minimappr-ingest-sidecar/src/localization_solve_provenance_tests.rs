use std::collections::BTreeMap;

use super::*;
use crate::dsp_worker::{sensor_id_for_node_channel, SolveSensorLayoutParts};
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, NodeLocalRoutingReason, NodeTimeQuality, SensorPositionSource,
    SolveAttemptKind, SolveBufferScope, SolveCoordinateFrame, SolveSensorEntry,
};

const RATE_HZ: u32 = 16_000;

fn sensor(node_id: &str, channel: usize, sample_rate_hz: u32) -> SolveSensorEntry {
    SolveSensorEntry {
        sensor_id: sensor_id_for_node_channel(node_id, channel),
        node_id: node_id.to_string(),
        node_channel_index: channel,
        position_m: [channel as f64, 2.0, 1.5],
        sample_rate_hz,
        position_source: SensorPositionSource::OperatorOverride,
        position_std_m: Some(0.05),
        clock_source: NodeTimeQuality::Unknown,
        clock_domain: "utc".to_string(),
        clock_offset_std_s: None,
    }
}

/// planar-a ch0..2 at the solve rate, point-b ch0 at 48 kHz.
fn site_layout(node_local_routing_reason: Option<NodeLocalRoutingReason>) -> SolveSensorLayout {
    SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors: vec![
            sensor("planar-a", 0, RATE_HZ),
            sensor("planar-a", 1, RATE_HZ),
            sensor("planar-a", 2, RATE_HZ),
            sensor("point-b", 0, 48_000),
        ],
        layout_revision: 9,
        geometry_push_revision: Some(4),
        coordinate_frame: SolveCoordinateFrame {
            frame_id: Some("site-7".to_string()),
            kind: CoordinateFrameKind::SiteEnu,
            origin: None,
            projection: None,
        },
        buffer_scope: SolveBufferScope::AutomaticSiteCluster,
        cluster_id: Some("auto-gps".to_string()),
        reporting_node_id: "planar-a".to_string(),
        node_local_routing_reason,
        solve_sample_rate_hz: RATE_HZ,
        half_space_by_node: BTreeMap::new(),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    })
}

#[test]
fn provenance_reports_the_snapshot_and_a_reason_for_every_inactive_sensor() {
    let layout = site_layout(None);
    let provenance = build_solve_provenance(
        &layout,
        SolveProvenanceInputs {
            solve_id: "solve-1",
            attempt_kind: SolveAttemptKind::EventTriggerAbsoluteRms,
            active_sensor_indices: &[0, 2],
            excluded_pairs: &[ExcludedSensorPair {
                sensor_index_a: 0,
                sensor_index_b: 1,
                reason: SolveExclusionReason::CoverageBelowThreshold,
            }],
            window_center_time_ns: 1_500_000_000,
            window_seconds: 0.064,
            node_clock_assessments: &NodeClockAssessments::new(),
        },
    );
    assert_eq!(provenance.solve_id, "solve-1");
    assert_eq!(provenance.layout_revision, 9);
    assert_eq!(provenance.geometry_push_revision, Some(4));
    assert_eq!(
        provenance.coordinate_frame.frame_id.as_deref(),
        Some("site-7")
    );
    assert_eq!(
        provenance.sensors, layout.sensors,
        "positions are the snapshot"
    );
    assert_eq!(provenance.active_sensor_indices, [0, 2]);
    let exclusions: Vec<(usize, SolveExclusionReason)> = provenance
        .excluded_sensors
        .iter()
        .map(|excluded| (excluded.sensor_index, excluded.reason))
        .collect();
    assert_eq!(
        exclusions,
        [
            (1, SolveExclusionReason::CoverageBelowThreshold),
            (3, SolveExclusionReason::SampleRateMismatch),
        ]
    );
    assert_eq!(provenance.node_local_routing_reason, None);
    let wire = serde_json::to_value(&provenance).unwrap();
    assert_eq!(
        wire["excluded_pairs"],
        serde_json::json!([
            {"sensor_index_a": 0, "sensor_index_b": 1, "reason": "coverage_below_threshold"}
        ])
    );
    assert!(provenance.node_timing.is_empty());
    assert_eq!(provenance.window_center_time_ns, 1_500_000_000);
    assert_eq!(provenance.solve_sample_rate_hz, RATE_HZ);
}

#[test]
fn node_timing_and_sensor_clock_fields_come_from_the_payload_snapshot() {
    use crate::dsp_worker::NodeClockAssessment;
    let layout = site_layout(None);
    let assessment = |node_id: &str, time_quality, uncertainty, residual, step, count| {
        (
            node_id.to_string(),
            NodeClockAssessment {
                node_id: node_id.to_string(),
                time_quality,
                timing_uncertainty_seconds: uncertainty,
                timeline_residual_seconds: residual,
                last_timestamp_step_ns: step,
                timestamp_step_count: count,
                residual_over_hard_limit: false,
            },
        )
    };
    let snapshot = NodeClockAssessments::from([
        assessment(
            "planar-a",
            NodeTimeQuality::GpsLocked,
            1e-6,
            Some(1.2e-6),
            None,
            0,
        ),
        assessment(
            "point-b",
            NodeTimeQuality::NtpDisciplined,
            0.021,
            Some(0.015),
            Some(-21_000_000),
            1,
        ),
    ]);
    let provenance = build_solve_provenance(
        &layout,
        SolveProvenanceInputs {
            solve_id: "solve-3",
            attempt_kind: SolveAttemptKind::EventTriggerAbsoluteRms,
            active_sensor_indices: &[0, 1, 2],
            excluded_pairs: &[],
            window_center_time_ns: 0,
            window_seconds: 0.064,
            node_clock_assessments: &snapshot,
        },
    );
    let wire = serde_json::to_value(&provenance).unwrap();
    assert_eq!(
        wire["node_timing"],
        serde_json::json!([
            {"node_id": "planar-a", "time_quality": "gps_locked", "clock_offset_std_s": 1e-6,
             "timeline_residual_ns": 1200, "last_timestamp_step_ns": null, "timestamp_step_count": 0},
            {"node_id": "point-b", "time_quality": "ntp_disciplined", "clock_offset_std_s": 0.021,
             "timeline_residual_ns": 15_000_000, "last_timestamp_step_ns": -21_000_000,
             "timestamp_step_count": 1}
        ])
    );
    assert_eq!(wire["sensors"][0]["clock_source"], "gps_locked");
    assert_eq!(wire["sensors"][0]["clock_domain"], "utc");
    assert_eq!(wire["sensors"][0]["clock_offset_std_s"], 1e-6);
    assert_eq!(wire["sensors"][3]["clock_source"], "ntp_disciplined");
    assert_eq!(wire["sensors"][3]["clock_offset_std_s"], 0.021);
    assert_eq!(
        provenance.sensors[3].position_m, layout.sensors[3].position_m,
        "positions stay the snapshot"
    );
}

#[test]
fn node_local_routing_reason_is_published() {
    let layout = site_layout(Some(
        NodeLocalRoutingReason::NodePositionUntrustedNoOverride,
    ));
    let provenance = build_solve_provenance(
        &layout,
        SolveProvenanceInputs {
            solve_id: "solve-2",
            attempt_kind: SolveAttemptKind::NotAttempted,
            active_sensor_indices: &[],
            excluded_pairs: &[],
            window_center_time_ns: 0,
            window_seconds: 0.064,
            node_clock_assessments: &NodeClockAssessments::new(),
        },
    );
    let wire = serde_json::to_value(&provenance).unwrap();
    assert_eq!(
        wire["node_local_routing_reason"],
        "node_position_untrusted_no_override"
    );
    assert_eq!(provenance.excluded_sensors.len(), 4);
}
