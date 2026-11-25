use std::collections::BTreeMap;

use super::*;
use crate::dsp_worker::{sensor_id_for_node_channel, SolveSensorLayoutParts};
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, SensorPositionSource, SolveBufferScope, SolveCoordinateFrame,
    SolveSensorEntry,
};

const RATE_HZ: u32 = 16_000;
const FRAME_SAMPLES: usize = 512;
const FRAME_NS: i128 = 32_000_000;
const START_NS: i128 = 1_764_000_000_000_000_000;
const STEP_NS: i128 = 21_000_000;
const NOW_NS: u128 = 5_000_000_000;
const NODE: &str = "node-a";
const BOOT: u32 = 3;

fn config() -> LocalizationSolveConfig {
    LocalizationSolveConfig::default()
}

/// One upload of `frame_count` contiguous frames starting at `first_frame`;
/// `extra_ns(frame)` is added to each frame's nominal packet time.
struct Upload {
    first_frame: i64,
    frame_count: usize,
    time_quality: NodeTimeQuality,
    boot_session: u32,
    holdover_error_ns: Option<u64>,
}

impl Upload {
    fn gps(first_frame: i64, frame_count: usize) -> Self {
        Self {
            first_frame,
            frame_count,
            time_quality: NodeTimeQuality::GpsLocked,
            boot_session: BOOT,
            holdover_error_ns: None,
        }
    }

    fn frame_timing(&self, extra_ns: impl Fn(i64) -> i128) -> Vec<PacketFrameTiming> {
        (0..self.frame_count)
            .map(|offset| {
                let frame = self.first_frame + offset as i64;
                PacketFrameTiming {
                    sample_offset: offset * FRAME_SAMPLES,
                    packet_timestamp_ns: Some(
                        START_NS + i128::from(frame) * FRAME_NS + extra_ns(frame),
                    ),
                    time_quality: self.time_quality,
                    reported_holdover_error_ns: self.holdover_error_ns,
                    boot_session: Some(self.boot_session),
                }
            })
            .collect()
    }
}

/// Buffer anchored at frame `anchor_frame`'s packet time (plus `anchor_extra_ns`).
fn timeline_anchored_at(anchor_frame: i64, anchor_extra_ns: i128) -> NominalSampleTimeline {
    NominalSampleTimeline {
        origin_time_ns: START_NS + i128::from(anchor_frame) * FRAME_NS + anchor_extra_ns,
        origin_sample_index: anchor_frame * FRAME_SAMPLES as i64,
        sample_rate_hz: RATE_HZ,
    }
}

fn observe(
    tracker: &mut NodeClockQualityTracker,
    upload: &Upload,
    extra_ns: impl Fn(i64) -> i128,
    timeline: Option<NominalSampleTimeline>,
) {
    observe_segment(tracker, upload, extra_ns, timeline, true);
}

fn observe_segment(
    tracker: &mut NodeClockQualityTracker,
    upload: &Upload,
    extra_ns: impl Fn(i64) -> i128,
    timeline: Option<NominalSampleTimeline>,
    segment_on_packet_time: bool,
) {
    let frame_timing = upload.frame_timing(extra_ns);
    tracker.observe(NodeFrameTimingObservation {
        node_id: NODE,
        frame_timing: &frame_timing,
        segment_start_sample_index: Some(upload.first_frame * FRAME_SAMPLES as i64),
        sample_rate_hz: RATE_HZ,
        segment_on_packet_time,
        buffer_timeline: timeline,
        now_ns: NOW_NS,
    });
}

fn residual_window_frames() -> usize {
    config().localization_timeline_residual_window_frames
}

fn on_time(_: i64) -> i128 {
    0
}

/// Frames needed to span `seconds` of node sample time.
fn frames_for_seconds(seconds: f64) -> usize {
    (seconds * f64::from(RATE_HZ) / FRAME_SAMPLES as f64).ceil() as usize
}

#[test]
fn class_default_sets_the_uncertainty_of_a_clean_clock() {
    let mut tracker = NodeClockQualityTracker::new(&config());
    observe(
        &mut tracker,
        &Upload::gps(0, 10),
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    let assessment = tracker.assessment(NODE);
    assert_eq!(assessment.time_quality, NodeTimeQuality::GpsLocked);
    assert_eq!(assessment.timing_uncertainty_seconds, 1e-6);
    assert_eq!(assessment.timeline_residual_seconds, Some(0.0));
    assert_eq!(assessment.timestamp_step_count, 0);
    assert!(!assessment.residual_over_hard_limit);

    let ntp = Upload {
        time_quality: NodeTimeQuality::NtpDisciplined,
        ..Upload::gps(10, 1)
    };
    observe(
        &mut tracker,
        &ntp,
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    assert_eq!(tracker.assessment(NODE).timing_uncertainty_seconds, 0.002);
}

#[test]
fn unknown_or_unmapped_class_takes_the_unknown_default_and_is_never_excluded() {
    let defaults = config();
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    let unknown = Upload {
        time_quality: NodeTimeQuality::Unknown,
        ..Upload::gps(0, 4)
    };
    observe(&mut tracker, &unknown, on_time, None);
    let assessment = tracker.assessment(NODE);
    assert_eq!(
        assessment.timing_uncertainty_seconds,
        defaults.localization_clock_unknown_uncertainty_seconds
    );
    assert!(!assessment.residual_over_hard_limit);
    assert_eq!(
        tracker.assessment("never-heard").timing_uncertainty_seconds,
        defaults.localization_clock_unknown_uncertainty_seconds
    );

    let mut gps_only_map = config();
    gps_only_map.localization_clock_class_uncertainty_seconds.0 =
        BTreeMap::from([("gps_locked".to_string(), 1e-6)]);
    let mut tracker = NodeClockQualityTracker::new(&gps_only_map);
    let free_running = Upload {
        time_quality: NodeTimeQuality::FreeRunning,
        ..Upload::gps(0, 1)
    };
    observe(&mut tracker, &free_running, on_time, None);
    assert_eq!(
        tracker.assessment(NODE).timing_uncertainty_seconds,
        defaults.localization_clock_unknown_uncertainty_seconds
    );
}

#[test]
fn reported_holdover_error_widens_past_the_class_default() {
    let mut tracker = NodeClockQualityTracker::new(&config());
    let holdover = Upload {
        time_quality: NodeTimeQuality::GpsHoldover,
        holdover_error_ns: Some(400_000),
        ..Upload::gps(0, 2)
    };
    observe(
        &mut tracker,
        &holdover,
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    assert_eq!(tracker.assessment(NODE).timing_uncertainty_seconds, 4e-4);
}

#[test]
fn measured_residual_against_the_buffer_timeline_widens_and_is_reported() {
    let mut tracker = NodeClockQualityTracker::new(&config());
    // The buffer was anchored 3 ms early relative to these packets.
    observe(
        &mut tracker,
        &Upload::gps(0, 4),
        on_time,
        Some(timeline_anchored_at(0, -3_000_000)),
    );
    let assessment = tracker.assessment(NODE);
    let residual = assessment.timeline_residual_seconds.unwrap();
    assert!((residual - 0.003).abs() < 1e-12, "{residual}");
    assert!((assessment.timing_uncertainty_seconds - 0.003).abs() < 1e-12);
    assert!(!assessment.residual_over_hard_limit);
}

#[test]
fn residual_over_the_hard_limit_is_flagged_only_when_measured_over_it() {
    let defaults = config();
    let limit_ns =
        (defaults.localization_cross_node_max_timeline_residual_seconds * 1e9).round() as i128;
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    let window = residual_window_frames();
    observe(
        &mut tracker,
        &Upload::gps(0, window),
        on_time,
        Some(timeline_anchored_at(0, -(limit_ns - 1_000_000))),
    );
    assert!(!tracker.assessment(NODE).residual_over_hard_limit);
    observe(
        &mut tracker,
        &Upload::gps(window as i64, window),
        on_time,
        Some(timeline_anchored_at(0, -(limit_ns + 2_000_000))),
    );
    assert!(tracker.assessment(NODE).residual_over_hard_limit);
    // A huge class default alone (no measurement) never sets the flag.
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    let free_running = Upload {
        time_quality: NodeTimeQuality::FreeRunning,
        ..Upload::gps(0, 2)
    };
    observe(&mut tracker, &free_running, on_time, None);
    let assessment = tracker.assessment(NODE);
    assert!(
        assessment.timing_uncertainty_seconds
            > defaults.localization_cross_node_max_timeline_residual_seconds
    );
    assert!(!assessment.residual_over_hard_limit);
}

#[test]
fn a_step_inside_one_upload_on_an_unmoved_buffer_timeline_stays_over_the_limit() {
    let mut tracker = NodeClockQualityTracker::new(&config());
    observe(
        &mut tracker,
        &Upload::gps(0, 20),
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    // 21 ms step at frame 25 of one upload; sample indices stay contiguous,
    // and the upload runs on long enough to fill the residual window.
    let step_at_25 = |frame: i64| if frame >= 25 { STEP_NS } else { 0 };
    observe(
        &mut tracker,
        &Upload::gps(20, 5 + residual_window_frames()),
        step_at_25,
        Some(timeline_anchored_at(0, 0)),
    );
    let assessment = tracker.assessment(NODE);
    assert_eq!(assessment.timestamp_step_count, 1);
    assert_eq!(assessment.last_timestamp_step_ns, Some(21_000_000));
    assert!((assessment.timing_uncertainty_seconds - 0.021).abs() < 1e-12);
    assert!(
        assessment.residual_over_hard_limit,
        "buffer audio sits 21 ms off its packets"
    );
}

#[test]
fn step_penalty_holds_until_the_residual_settles_for_the_holdoff() {
    let defaults = config();
    let holdoff_frames =
        frames_for_seconds(defaults.localization_timestamp_step_recovery_holdoff_seconds);
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    observe(
        &mut tracker,
        &Upload::gps(0, 10),
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    // The buffer re-anchors on the stepped timeline, so the residual is ~0
    // right away: only the holdoff keeps the step's uncertainty.
    let stepped = |_: i64| STEP_NS;
    let stepped_timeline = Some(timeline_anchored_at(10, STEP_NS));
    observe(
        &mut tracker,
        &Upload::gps(10, holdoff_frames - 1),
        stepped,
        stepped_timeline,
    );
    let held = tracker.assessment(NODE);
    assert_eq!(held.timestamp_step_count, 1);
    assert!((held.timing_uncertainty_seconds - 0.021).abs() < 1e-12);
    assert!(!held.residual_over_hard_limit);

    observe(
        &mut tracker,
        &Upload::gps(10 + holdoff_frames as i64 - 1, 2),
        stepped,
        stepped_timeline,
    );
    let recovered = tracker.assessment(NODE);
    assert_eq!(recovered.timing_uncertainty_seconds, 1e-6);
    assert_eq!(recovered.timestamp_step_count, 1, "history is kept");
    assert_eq!(recovered.last_timestamp_step_ns, Some(21_000_000));
}

#[test]
fn a_residual_excursion_restarts_the_holdoff() {
    const RAMP_FRAMES: i64 = 30;
    const RAMP_STEP_NS: i128 = 100_000;
    let defaults = config();
    let holdoff_frames =
        frames_for_seconds(defaults.localization_timestamp_step_recovery_holdoff_seconds) as i64;
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    observe(
        &mut tracker,
        &Upload::gps(0, 2),
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    // Step at frame 2, then mid-holdoff a slow drift out to 3 ms and back:
    // 0.1 ms per frame is never a step, but the residual leaves the threshold.
    let ramp_start = 2 + holdoff_frames / 2;
    let excursion_ns = move |frame: i64| {
        let into_ramp = frame - ramp_start;
        let ramp_ns = if (0..=2 * RAMP_FRAMES).contains(&into_ramp) {
            i128::from(RAMP_FRAMES - (into_ramp - RAMP_FRAMES).abs()) * RAMP_STEP_NS
        } else {
            0
        };
        STEP_NS + ramp_ns
    };
    // Long enough that the penalty would have cleared without the excursion.
    let after_step = Upload::gps(2, (holdoff_frames + 2 * RAMP_FRAMES + 10) as usize);
    observe(
        &mut tracker,
        &after_step,
        excursion_ns,
        Some(timeline_anchored_at(2, STEP_NS)),
    );
    let assessment = tracker.assessment(NODE);
    assert_eq!(
        assessment.timestamp_step_count, 1,
        "the drift is not a step"
    );
    assert!((assessment.timing_uncertainty_seconds - 0.021).abs() < 1e-12);
}

#[test]
fn boot_session_change_raises_the_unknown_default_until_the_holdoff() {
    let defaults = config();
    let holdoff_frames =
        frames_for_seconds(defaults.localization_timestamp_step_recovery_holdoff_seconds);
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    observe(
        &mut tracker,
        &Upload::gps(400, 4),
        on_time,
        Some(timeline_anchored_at(400, 0)),
    );
    // Reboot: sample indices restart at 0, packet time keeps going.
    let rebooted = Upload {
        boot_session: BOOT + 1,
        ..Upload::gps(0, holdoff_frames - 1)
    };
    let after_reboot = |_: i64| 1_000 * FRAME_NS;
    let new_timeline = Some(timeline_anchored_at(0, 1_000 * FRAME_NS));
    observe(&mut tracker, &rebooted, after_reboot, new_timeline);
    let held = tracker.assessment(NODE);
    assert_eq!(
        held.timestamp_step_count, 0,
        "a reboot has no measured step size"
    );
    assert_eq!(held.last_timestamp_step_ns, None);
    assert_eq!(
        held.timing_uncertainty_seconds,
        defaults.localization_clock_unknown_uncertainty_seconds
    );
    let settled = Upload {
        boot_session: BOOT + 1,
        ..Upload::gps(holdoff_frames as i64 - 1, 2)
    };
    observe(&mut tracker, &settled, after_reboot, new_timeline);
    assert_eq!(tracker.assessment(NODE).timing_uncertainty_seconds, 1e-6);
}

fn sensor(node_id: &str, channel: usize) -> SolveSensorEntry {
    SolveSensorEntry {
        sensor_id: sensor_id_for_node_channel(node_id, channel),
        node_id: node_id.to_string(),
        node_channel_index: channel,
        position_m: [channel as f64, 0.0, 1.5],
        sample_rate_hz: RATE_HZ,
        position_source: SensorPositionSource::OperatorOverride,
        position_std_m: None,
        clock_source: NodeTimeQuality::Unknown,
        clock_domain: "utc".to_string(),
        clock_offset_std_s: None,
    }
}

#[test]
fn snapshot_covers_every_layout_node_and_stale_nodes_are_evicted() {
    let mut tracker = NodeClockQualityTracker::new(&config());
    observe(&mut tracker, &Upload::gps(0, 2), on_time, None);
    let layout = SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors: vec![sensor(NODE, 0), sensor(NODE, 1), sensor("node-b", 0)],
        layout_revision: 1,
        geometry_push_revision: None,
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::SiteEnu,
            origin: None,
            projection: None,
        },
        buffer_scope: SolveBufferScope::AutomaticSiteCluster,
        cluster_id: Some("auto-gps".to_string()),
        reporting_node_id: NODE.to_string(),
        node_local_routing_reason: None,
        solve_sample_rate_hz: RATE_HZ,
        half_space_by_node: BTreeMap::new(),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    });
    let snapshot = tracker.snapshot_for_layout(&layout);
    assert_eq!(snapshot.keys().collect::<Vec<_>>(), [NODE, "node-b"]);
    assert_eq!(snapshot[NODE].time_quality, NodeTimeQuality::GpsLocked);
    assert_eq!(snapshot["node-b"].time_quality, NodeTimeQuality::Unknown);

    let stale_after_ns = (config().localization_sensor_registry_stale_seconds * 1e9) as u128;
    tracker.evict_stale(NOW_NS + stale_after_ns);
    assert_eq!(tracker.tracked_node_count(), 1);
    tracker.evict_stale(NOW_NS + stale_after_ns + 1);
    assert_eq!(tracker.tracked_node_count(), 0);
}

#[test]
fn a_single_late_frame_does_not_flip_the_hard_limit_gate() {
    let defaults = config();
    let late_ns =
        (defaults.localization_cross_node_max_timeline_residual_seconds * 1.5e9).round() as i128;
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    observe(
        &mut tracker,
        &Upload::gps(0, 20),
        on_time,
        Some(timeline_anchored_at(0, 0)),
    );
    // Frame 20 is stamped 15 ms late (network or firmware jitter); the
    // solve snapshotting right after it must not lose this node's pairs.
    let one_late_frame = move |frame: i64| if frame == 20 { late_ns } else { 0 };
    observe(
        &mut tracker,
        &Upload::gps(20, 1),
        one_late_frame,
        Some(timeline_anchored_at(0, 0)),
    );
    let assessment = tracker.assessment(NODE);
    assert!(!assessment.residual_over_hard_limit);
    assert_eq!(assessment.timeline_residual_seconds, Some(0.0));
}

#[test]
fn the_hard_limit_flag_clears_only_under_the_hysteresis_ratio() {
    let defaults = config();
    let limit_seconds = defaults.localization_cross_node_max_timeline_residual_seconds;
    let clear_seconds =
        limit_seconds * defaults.localization_timeline_residual_over_limit_clear_ratio;
    let residual_ns = |seconds: f64| (seconds * 1e9).round() as i128;
    let window = residual_window_frames();
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    let mut next_frame = 0_i64;
    let mut fill_window_at = |tracker: &mut NodeClockQualityTracker, residual_seconds: f64| {
        observe(
            tracker,
            &Upload::gps(next_frame, window),
            on_time,
            Some(timeline_anchored_at(0, -residual_ns(residual_seconds))),
        );
        next_frame += window as i64;
        tracker.assessment(NODE).residual_over_hard_limit
    };
    assert!(fill_window_at(&mut tracker, limit_seconds * 1.2));
    let between_clear_and_limit_seconds = (clear_seconds + limit_seconds) * 0.5;
    assert!(
        fill_window_at(&mut tracker, between_clear_and_limit_seconds),
        "under the limit but over the clear level: still flagged"
    );
    assert!(!fill_window_at(&mut tracker, clear_seconds * 0.5));
    assert!(
        !fill_window_at(&mut tracker, between_clear_and_limit_seconds),
        "rising back under the limit does not set it"
    );
}

#[test]
fn a_segment_off_packet_time_is_assessed_as_an_unknown_clock() {
    let defaults = config();
    let window = residual_window_frames();
    let mut tracker = NodeClockQualityTracker::new(&defaults);
    observe(
        &mut tracker,
        &Upload::gps(0, window),
        on_time,
        Some(timeline_anchored_at(0, -2 * STEP_NS)),
    );
    assert!(tracker.assessment(NODE).residual_over_hard_limit);
    // No node timestamp: the audio sits on server/receipt time, whatever
    // class (binary code 0 = gps_locked) the frames still claim.
    observe_segment(
        &mut tracker,
        &Upload::gps(window as i64, 2),
        on_time,
        None,
        false,
    );
    let assessment = tracker.assessment(NODE);
    assert_eq!(assessment.time_quality, NodeTimeQuality::Unknown);
    assert_eq!(
        assessment.timing_uncertainty_seconds,
        defaults.localization_clock_unknown_uncertainty_seconds
    );
    assert_eq!(assessment.timeline_residual_seconds, None);
    assert!(!assessment.residual_over_hard_limit);

    observe(
        &mut tracker,
        &Upload::gps(window as i64 + 2, 2),
        on_time,
        Some(timeline_anchored_at(window as i64 + 2, 0)),
    );
    let back_on_packet_time = tracker.assessment(NODE);
    assert_eq!(back_on_packet_time.time_quality, NodeTimeQuality::GpsLocked);
    assert_eq!(back_on_packet_time.timeline_residual_seconds, Some(0.0));
}
