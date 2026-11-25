//! Tracker-level timeline discipline: the tracker's corrections are applied
//! back to the simulated buffer timeline exactly as the worker does
//! (`dsp_manifest_buffering.rs`), so these runs show the closed loop.

use super::*;

const RATE_HZ: u32 = 16_000;
const FRAME_SAMPLES: usize = 512;
const FRAME_NS: i128 = 32_000_000;
const START_NS: i128 = 1_764_000_000_000_000_000;
const STEP_NS: i128 = 21_000_000;
const NOW_NS: u128 = 5_000_000_000;
const NODE: &str = "node-a";
const BOOT: u32 = 3;
const FRAMES_PER_UPLOAD: i64 = 25;
const DRIFT_PARTS_PER_MILLION: f64 = 20.0;
const DRIFT_RUN_SECONDS: f64 = 15.0 * 60.0;
const SECONDS_PER_MINUTE: f64 = 60.0;

fn frames_for_seconds(seconds: f64) -> i64 {
    (seconds * f64::from(RATE_HZ) / FRAME_SAMPLES as f64).ceil() as i64
}

/// One node: its tracker plus the buffer timeline the worker would keep.
struct DisciplinedNode {
    tracker: NodeClockQualityTracker,
    /// Packet time the buffer maps sample index 0 to.
    buffer_origin_ns: i128,
    corrections: NodeTimelineCorrections,
    max_abs_windowed_residual_seconds: f64,
    ever_over_hard_limit: bool,
}

impl DisciplinedNode {
    /// Anchored at frame 0's packet time, like a buffer's first append.
    fn new(config: &LocalizationSolveConfig, extra_ns: &impl Fn(i64) -> i128) -> Self {
        Self {
            tracker: NodeClockQualityTracker::new(config),
            buffer_origin_ns: START_NS + extra_ns(0),
            corrections: NodeTimelineCorrections::default(),
            max_abs_windowed_residual_seconds: 0.0,
            ever_over_hard_limit: false,
        }
    }

    /// Delivers frames `[first, end)` as one segment.
    fn deliver(&mut self, first: i64, end: i64, extra_ns: &impl Fn(i64) -> i128) {
        let frame_timing: Vec<PacketFrameTiming> = (first..end)
            .map(|frame| PacketFrameTiming {
                sample_offset: (frame - first) as usize * FRAME_SAMPLES,
                packet_timestamp_ns: Some(
                    START_NS + i128::from(frame) * FRAME_NS + extra_ns(frame),
                ),
                time_quality: NodeTimeQuality::GpsLocked,
                reported_holdover_error_ns: None,
                boot_session: Some(BOOT),
            })
            .collect();
        let corrections = self.tracker.observe(NodeFrameTimingObservation {
            node_id: NODE,
            frame_timing: &frame_timing,
            segment_start_sample_index: Some(first * FRAME_SAMPLES as i64),
            sample_rate_hz: RATE_HZ,
            segment_on_packet_time: true,
            buffer_timeline: Some(NominalSampleTimeline {
                origin_time_ns: self.buffer_origin_ns,
                origin_sample_index: 0,
                sample_rate_hz: RATE_HZ,
            }),
            now_ns: NOW_NS,
        });
        self.buffer_origin_ns += corrections.origin_shift_ns;
        self.corrections.origin_shift_ns += corrections.origin_shift_ns;
        self.corrections.step_reanchors += corrections.step_reanchors;
        self.corrections.drift_slews += corrections.drift_slews;
        let assessment = self.assessment();
        if let Some(residual_seconds) = assessment.timeline_residual_seconds {
            self.max_abs_windowed_residual_seconds = self
                .max_abs_windowed_residual_seconds
                .max(residual_seconds.abs());
        }
        self.ever_over_hard_limit |= assessment.residual_over_hard_limit;
    }

    fn deliver_uploads(&mut self, first: i64, end: i64, extra_ns: &impl Fn(i64) -> i128) {
        let mut upload_start = first;
        while upload_start < end {
            let upload_end = (upload_start + FRAMES_PER_UPLOAD).min(end);
            self.deliver(upload_start, upload_end, extra_ns);
            upload_start = upload_end;
        }
    }

    fn assessment(&self) -> NodeClockAssessment {
        self.tracker.assessment(NODE)
    }
}

/// Packet clock running fast of the sample clock by `ppm`.
fn drifting(ppm: f64) -> impl Fn(i64) -> i128 {
    move |frame: i64| (i128::from(frame) as f64 * FRAME_NS as f64 * ppm * 1e-6).round() as i128
}

#[test]
fn a_step_inside_one_upload_reanchors_and_the_penalty_holds_for_the_holdoff() {
    const STEP_FRAME: i64 = 22;
    let config = LocalizationSolveConfig::default();
    let stepped = |frame: i64| if frame >= STEP_FRAME { STEP_NS } else { 0 };
    let mut node = DisciplinedNode::new(&config, &stepped);
    node.deliver(0, 30, &stepped);

    // Frames 22..24 confirm the step: one re-anchor by exactly the step.
    assert_eq!(node.corrections.step_reanchors, 1);
    assert_eq!(node.corrections.drift_slews, 0);
    assert_eq!(node.corrections.origin_shift_ns, STEP_NS);
    let held = node.assessment();
    assert_eq!(held.timestamp_step_count, 1);
    assert_eq!(held.timeline_step_reanchor_count, 1);
    assert_eq!(held.last_timestamp_step_ns, Some(STEP_NS as i64));
    assert_eq!(held.timeline_residual_seconds, Some(0.0));
    assert!(!held.residual_over_hard_limit);
    assert!(!node.ever_over_hard_limit, "the gate never saw the step");
    // Back in the solve, but with the step's honest, widened uncertainty.
    assert!((held.timing_uncertainty_seconds - 0.021).abs() < 1e-12);

    let holdoff_frames =
        frames_for_seconds(config.localization_timestamp_step_recovery_holdoff_seconds);
    node.deliver_uploads(30, STEP_FRAME + 3 + holdoff_frames + 1, &stepped);
    let recovered = node.assessment();
    assert_eq!(
        recovered.timing_uncertainty_seconds, 1e-6,
        "class default after the holdoff"
    );
    assert_eq!(recovered.timeline_step_reanchor_count, 1);
    assert_eq!(recovered.timestamp_step_count, 1);
}

#[test]
fn a_negative_step_across_uploads_reanchors_too() {
    const STEP_FRAME: i64 = FRAMES_PER_UPLOAD;
    let config = LocalizationSolveConfig::default();
    let stepped = |frame: i64| if frame >= STEP_FRAME { -STEP_NS } else { 0 };
    let mut node = DisciplinedNode::new(&config, &stepped);
    node.deliver_uploads(0, 3 * FRAMES_PER_UPLOAD, &stepped);
    assert_eq!(node.corrections.step_reanchors, 1);
    assert_eq!(node.corrections.origin_shift_ns, -STEP_NS);
    assert_eq!(node.assessment().timeline_residual_seconds, Some(0.0));
}

#[test]
fn slow_drift_at_20_ppm_for_15_minutes_never_reaches_the_hard_limit() {
    let config = LocalizationSolveConfig::default();
    let hard_limit_seconds = config.localization_cross_node_max_timeline_residual_seconds;
    let slew_seconds = hard_limit_seconds * config.localization_timeline_drift_slew_fraction;
    let total_frames = frames_for_seconds(DRIFT_RUN_SECONDS);
    let drift = drifting(DRIFT_PARTS_PER_MILLION);

    // Without the discipline the node is gated out after about 8 minutes.
    let undisciplined = LocalizationSolveConfig {
        localization_timeline_step_reanchor_enabled: false,
        localization_timeline_drift_slew_fraction: 0.0,
        ..config.clone()
    };
    let mut gated = DisciplinedNode::new(&undisciplined, &drift);
    let mut gated_after_frame = None;
    let mut upload_start = 0;
    while upload_start < total_frames && gated_after_frame.is_none() {
        let upload_end = (upload_start + FRAMES_PER_UPLOAD).min(total_frames);
        gated.deliver(upload_start, upload_end, &drift);
        if gated.assessment().residual_over_hard_limit {
            gated_after_frame = Some(upload_end);
        }
        upload_start = upload_end;
    }
    let gated_after_minutes =
        gated_after_frame.expect("undisciplined drift reaches the limit") as f64 * FRAME_NS as f64
            / 1e9
            / SECONDS_PER_MINUTE;
    assert!(
        (8.0..9.0).contains(&gated_after_minutes),
        "{gated_after_minutes} min"
    );

    let mut node = DisciplinedNode::new(&config, &drift);
    node.deliver_uploads(0, total_frames, &drift);
    let accumulated_drift_seconds = DRIFT_RUN_SECONDS * DRIFT_PARTS_PER_MILLION * 1e-6;
    assert!(
        !node.ever_over_hard_limit,
        "cross-node pairs kept for 15 minutes"
    );
    assert!(
        node.max_abs_windowed_residual_seconds <= slew_seconds + 1e-5,
        "residual {} never far past the slew threshold",
        node.max_abs_windowed_residual_seconds
    );
    assert_eq!(node.corrections.step_reanchors, 0);
    assert_eq!(
        node.assessment().timestamp_step_count,
        0,
        "drift is never a step"
    );
    assert_eq!(
        node.corrections.drift_slews,
        (accumulated_drift_seconds / slew_seconds).floor() as u32
    );
    // Everything slewed plus what is still measured is the whole drift.
    let absorbed_seconds = node.corrections.origin_shift_ns as f64 / 1e9;
    let residual_seconds = node.assessment().timeline_residual_seconds.unwrap();
    let delivered_drift_seconds = drift(total_frames - 1) as f64 / 1e9;
    assert!(
        (absorbed_seconds + residual_seconds - delivered_drift_seconds).abs() < 1e-5,
        "{absorbed_seconds} + {residual_seconds} != {delivered_drift_seconds}"
    );
}

/// Deterministic jitter in [-amplitude, amplitude] ns.
fn jitter(frame: i64, amplitude_ns: i128) -> i128 {
    let hashed = (frame as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let unit = (hashed >> 11) as f64 / (1_u64 << 53) as f64;
    ((unit * 2.0 - 1.0) * amplitude_ns as f64).round() as i128
}

#[test]
fn jitter_a_late_frame_and_a_packet_gap_never_reanchor_or_slew() {
    const JITTER_AMPLITUDE_NS: i128 = 900_000;
    const LATE_FRAME: i64 = 40;
    const LATE_NS: i128 = 6_000_000;
    const GAP_START: i64 = 70;
    const GAP_FRAMES: i64 = 12;
    const RUN_FRAMES: i64 = 400;
    let config = LocalizationSolveConfig::default();
    let noisy = |frame: i64| {
        jitter(frame, JITTER_AMPLITUDE_NS) + if frame == LATE_FRAME { LATE_NS } else { 0 }
    };
    let mut node = DisciplinedNode::new(&config, &noisy);
    node.deliver_uploads(0, GAP_START, &noisy);
    // Twelve frames lost in transit: sample indices and packet times skip together.
    node.deliver_uploads(GAP_START + GAP_FRAMES, RUN_FRAMES, &noisy);
    assert_eq!(node.corrections, NodeTimelineCorrections::default());
    let assessment = node.assessment();
    assert_eq!(assessment.timeline_step_reanchor_count, 0);
    assert_eq!(assessment.timeline_drift_slew_count, 0);
    assert!(!node.ever_over_hard_limit);
    // The late frame is flagged as two steps (out and back) and only widens.
    assert_eq!(assessment.timestamp_step_count, 2);
}

#[test]
fn with_the_discipline_off_the_tracker_never_corrects() {
    let config = LocalizationSolveConfig {
        localization_timeline_step_reanchor_enabled: false,
        localization_timeline_drift_slew_fraction: 0.0,
        ..LocalizationSolveConfig::default()
    };
    let stepped_and_drifting =
        |frame: i64| drifting(2_000.0)(frame) + if frame >= 30 { STEP_NS } else { 0 };
    let mut node = DisciplinedNode::new(&config, &stepped_and_drifting);
    node.deliver_uploads(0, 300, &stepped_and_drifting);
    assert_eq!(node.corrections, NodeTimelineCorrections::default());
    assert!(node.assessment().residual_over_hard_limit);
}
