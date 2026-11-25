//! Per-node clock evidence tracker (localization overhaul R5).
//!
//! Fed every buffered segment's per-frame timing, it turns packet clock
//! classes, holdover reports, timestamp steps and the measured timeline
//! residual into one `NodeClockAssessment` per node. A solve snapshots the
//! assessments of its layout's nodes at trigger time (`ComputePayload::
//! node_clock_assessments`); R7 admits pairs and widens covariance from it.
//!
//! Holdoff time runs on the node's own sample clock (sample index / rate), so
//! it is immune to the timestamp steps it is recovering from and to replay
//! speed. Entries are evicted with the sensor registry's stale window.
//!
//! The reported residual is a percentile over the last few frames, and the
//! hard-limit flag has hysteresis: a single late-stamped frame (network or
//! firmware jitter) must not drop a node's cross-node pairs for whichever
//! solve happens to snapshot at that moment.
//!
//! The tracker also disciplines the node's buffer timeline
//! (`minimappr_core::timeline_discipline`): a confirmed timestamp step
//! re-anchors it, and a windowed residual past the slew fraction of the hard
//! limit slews it. `observe` returns the summed origin shift; the worker
//! applies it to every buffer channel of the node, and the tracker shifts its
//! own copy of the timeline (later frames of the same segment) and its
//! recorded residuals to match. The step penalty is independent of the
//! re-anchor: a re-anchored node's pairs come back with the step's widened
//! uncertainty until the holdoff clears it.
//!
//! A node whose latest segment is not on packet time (no node timestamp, so
//! its audio sits on server or receipt time) is assessed as an unknown clock:
//! its frames' claimed class says nothing about where its audio landed.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use minimappr_core::{
    localization_sensor_entry::NodeTimeQuality,
    packet_frame_timing::PacketFrameTiming,
    timeline_discipline::{TimelineDiscipline, TimelineDisciplinePolicy},
    timestamp_step_detector::{
        FrameTimelinePoint, NominalSampleTimeline, TimestampStepDetector, TimingDiscontinuity,
    },
};

use super::{
    NodeClockAssessment, NodeClockAssessments, NodeTimelineCorrections, SolveSensorLayout,
};
use crate::{dsp::SensorStreamBuffer, localization_solve_config::LocalizationSolveConfig};

const NANOSECONDS_PER_SECOND_F64: f64 = 1.0e9;

#[derive(Clone, Debug)]
struct NodeClockPolicy {
    clock_class_uncertainty_seconds: BTreeMap<String, f64>,
    unknown_uncertainty_seconds: f64,
    step_threshold_seconds: f64,
    step_recovery_holdoff_seconds: f64,
    max_timeline_residual_seconds: f64,
    residual_window_frames: usize,
    residual_percentile: f64,
    residual_over_limit_clear_ratio: f64,
    stale_after_ns: u128,
    timeline_discipline: TimelineDisciplinePolicy,
}

impl NodeClockPolicy {
    fn from_config(config: &LocalizationSolveConfig) -> Self {
        Self {
            clock_class_uncertainty_seconds: config
                .localization_clock_class_uncertainty_seconds
                .0
                .clone(),
            unknown_uncertainty_seconds: config.localization_clock_unknown_uncertainty_seconds,
            step_threshold_seconds: config.localization_timestamp_step_threshold_seconds,
            step_recovery_holdoff_seconds: config
                .localization_timestamp_step_recovery_holdoff_seconds,
            max_timeline_residual_seconds: config
                .localization_cross_node_max_timeline_residual_seconds,
            residual_window_frames: config.localization_timeline_residual_window_frames.max(1),
            residual_percentile: config
                .localization_timeline_residual_percentile
                .clamp(0.0, 100.0),
            residual_over_limit_clear_ratio: config
                .localization_timeline_residual_over_limit_clear_ratio
                .clamp(0.0, 1.0),
            stale_after_ns: (config.localization_sensor_registry_stale_seconds.max(0.0)
                * NANOSECONDS_PER_SECOND_F64) as u128,
            timeline_discipline: TimelineDisciplinePolicy::from_seconds(
                config.localization_timeline_step_reanchor_enabled,
                config.localization_timestamp_step_threshold_seconds,
                config.localization_timestamp_step_confirm_frames,
                config.localization_timeline_drift_slew_fraction,
                config.localization_cross_node_max_timeline_residual_seconds,
            ),
        }
    }

    /// Unknown class, or a class the configured map leaves out, takes the
    /// conservative unknown default: widened, never excluded.
    fn class_default_seconds(&self, time_quality: NodeTimeQuality) -> f64 {
        if time_quality == NodeTimeQuality::Unknown {
            return self.unknown_uncertainty_seconds;
        }
        // The map is keyed by the same spellings the enum serializes to.
        serde_json::to_value(time_quality)
            .ok()
            .and_then(|class| {
                self.clock_class_uncertainty_seconds
                    .get(class.as_str()?)
                    .copied()
            })
            .unwrap_or(self.unknown_uncertainty_seconds)
    }
}

/// Raised uncertainty after a step or reboot. Cleared once the residual has
/// stayed under the step threshold for the holdoff.
#[derive(Clone, Copy, Debug)]
struct StepPenalty {
    uncertainty_seconds: f64,
    residual_settled_since_sample_index: Option<i64>,
}

#[derive(Clone, Debug)]
struct NodeClockState {
    detector: TimestampStepDetector,
    time_quality: NodeTimeQuality,
    reported_holdover_error_seconds: Option<f64>,
    /// Signed per-frame residuals, newest last, at most the policy window.
    recent_residuals_seconds: VecDeque<f64>,
    /// Robust aggregate of `recent_residuals_seconds` (`None` when empty).
    timeline_residual_seconds: Option<f64>,
    residual_over_hard_limit: bool,
    last_timestamp_step_ns: Option<i64>,
    timestamp_step_count: u32,
    step_penalty: Option<StepPenalty>,
    timeline_discipline: TimelineDiscipline,
    timeline_step_reanchor_count: u32,
    timeline_drift_slew_count: u32,
    latest_segment_on_packet_time: bool,
    last_heard_ns: u128,
}

impl NodeClockState {
    fn new(policy: &NodeClockPolicy) -> Self {
        Self {
            detector: TimestampStepDetector::new(policy.step_threshold_seconds),
            time_quality: NodeTimeQuality::Unknown,
            reported_holdover_error_seconds: None,
            recent_residuals_seconds: VecDeque::new(),
            timeline_residual_seconds: None,
            residual_over_hard_limit: false,
            last_timestamp_step_ns: None,
            timestamp_step_count: 0,
            step_penalty: None,
            timeline_discipline: TimelineDiscipline::new(policy.timeline_discipline),
            timeline_step_reanchor_count: 0,
            timeline_drift_slew_count: 0,
            latest_segment_on_packet_time: true,
            last_heard_ns: 0,
        }
    }

    /// Forgets the measured residual: it was measured against a timeline
    /// that no longer describes this node's audio.
    fn clear_residual(&mut self) {
        self.recent_residuals_seconds.clear();
        self.timeline_residual_seconds = None;
        self.residual_over_hard_limit = false;
    }

    /// Adds one frame's residual and re-evaluates the aggregate and the
    /// flag: it sets above the hard limit and clears only once the aggregate
    /// is back under `limit * clear_ratio`.
    fn record_residual(&mut self, residual_seconds: f64, policy: &NodeClockPolicy) {
        if self.recent_residuals_seconds.len() == policy.residual_window_frames {
            self.recent_residuals_seconds.pop_front();
        }
        self.recent_residuals_seconds.push_back(residual_seconds);
        self.reevaluate_residual(policy);
    }

    /// A slew moved the timeline: every recorded residual drops by the shift.
    fn shift_recorded_residuals(&mut self, origin_shift_ns: i128, policy: &NodeClockPolicy) {
        let shift_seconds = origin_shift_ns as f64 / NANOSECONDS_PER_SECOND_F64;
        for residual_seconds in &mut self.recent_residuals_seconds {
            *residual_seconds -= shift_seconds;
        }
        self.reevaluate_residual(policy);
    }

    fn reevaluate_residual(&mut self, policy: &NodeClockPolicy) {
        let aggregate_seconds =
            residual_at_percentile(&self.recent_residuals_seconds, policy.residual_percentile);
        self.timeline_residual_seconds = aggregate_seconds;
        let Some(aggregate_seconds) = aggregate_seconds else {
            return;
        };
        let magnitude_seconds = aggregate_seconds.abs();
        if magnitude_seconds > policy.max_timeline_residual_seconds {
            self.residual_over_hard_limit = true;
        } else if magnitude_seconds
            <= policy.max_timeline_residual_seconds * policy.residual_over_limit_clear_ratio
        {
            self.residual_over_hard_limit = false;
        }
    }

    fn raise_step_penalty(&mut self, uncertainty_seconds: f64) {
        let uncertainty_seconds = self.step_penalty.map_or(uncertainty_seconds, |active| {
            active.uncertainty_seconds.max(uncertainty_seconds)
        });
        self.step_penalty = Some(StepPenalty {
            uncertainty_seconds,
            residual_settled_since_sample_index: None,
        });
    }

    fn advance_step_recovery(
        &mut self,
        residual_seconds: f64,
        sample_index: i64,
        sample_rate_hz: u32,
        policy: &NodeClockPolicy,
    ) {
        let Some(penalty) = self.step_penalty.as_mut() else {
            return;
        };
        if residual_seconds.abs() > policy.step_threshold_seconds {
            penalty.residual_settled_since_sample_index = None;
            return;
        }
        let settled_since = *penalty
            .residual_settled_since_sample_index
            .get_or_insert(sample_index);
        let settled_seconds =
            (sample_index - settled_since) as f64 / f64::from(sample_rate_hz.max(1));
        if settled_seconds >= policy.step_recovery_holdoff_seconds {
            self.step_penalty = None;
        }
    }

    fn observe_frame(
        &mut self,
        frame: &PacketFrameTiming,
        point: Option<FrameTimelinePoint>,
        observation: &NodeFrameTimingObservation<'_>,
        timeline: &mut Option<NominalSampleTimeline>,
        policy: &NodeClockPolicy,
        corrections: &mut NodeTimelineCorrections,
    ) {
        self.time_quality = frame.time_quality;
        self.reported_holdover_error_seconds = frame
            .reported_holdover_error_ns
            .map(|error_ns| error_ns as f64 / NANOSECONDS_PER_SECOND_F64);
        let Some(point) = point else {
            return;
        };
        let mut detected_step_ns = None;
        match self.detector.observe(point, observation.sample_rate_hz) {
            Some(TimingDiscontinuity::TimestampStep { step_ns }) => {
                self.timestamp_step_count = self.timestamp_step_count.saturating_add(1);
                self.last_timestamp_step_ns =
                    Some(step_ns.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64);
                self.raise_step_penalty(step_ns.unsigned_abs() as f64 / NANOSECONDS_PER_SECOND_F64);
                detected_step_ns = Some(step_ns);
            }
            Some(TimingDiscontinuity::BootSessionChange) => {
                // The old session's residual says nothing about the new clock,
                // and a reboot has no step size: hold the unknown default.
                self.clear_residual();
                self.timeline_discipline.cancel_pending_step();
                self.raise_step_penalty(policy.unknown_uncertainty_seconds);
            }
            None => {}
        }
        let Some(timeline) = timeline.as_mut() else {
            return;
        };
        if let Some(correction) = self
            .timeline_discipline
            .observe_frame(timeline.residual_ns(&point), detected_step_ns)
        {
            // The residuals recorded before the step were on the old clock.
            timeline.origin_time_ns += correction.origin_shift_ns;
            self.clear_residual();
            self.timeline_step_reanchor_count = self.timeline_step_reanchor_count.saturating_add(1);
            corrections.add(correction);
        }
        self.record_residual(
            timeline.residual_ns(&point) as f64 / NANOSECONDS_PER_SECOND_F64,
            policy,
        );
        let windowed_residual_ns = self
            .timeline_residual_seconds
            .map(|seconds| (seconds * NANOSECONDS_PER_SECOND_F64).round() as i128);
        if let Some(correction) = self.timeline_discipline.drift_slew(windowed_residual_ns) {
            timeline.origin_time_ns += correction.origin_shift_ns;
            self.shift_recorded_residuals(correction.origin_shift_ns, policy);
            self.timeline_drift_slew_count = self.timeline_drift_slew_count.saturating_add(1);
            corrections.add(correction);
        }
        self.advance_step_recovery(
            timeline.residual_ns(&point) as f64 / NANOSECONDS_PER_SECOND_F64,
            point.start_sample_index,
            observation.sample_rate_hz,
            policy,
        );
    }

    fn assessment(&self, node_id: &str, policy: &NodeClockPolicy) -> NodeClockAssessment {
        if !self.latest_segment_on_packet_time {
            return NodeClockAssessment {
                node_id: node_id.to_string(),
                time_quality: NodeTimeQuality::Unknown,
                timing_uncertainty_seconds: policy.unknown_uncertainty_seconds,
                timeline_residual_seconds: None,
                last_timestamp_step_ns: self.last_timestamp_step_ns,
                timestamp_step_count: self.timestamp_step_count,
                residual_over_hard_limit: false,
                timeline_step_reanchor_count: self.timeline_step_reanchor_count,
                timeline_drift_slew_count: self.timeline_drift_slew_count,
            };
        }
        let measured_residual_seconds = self.timeline_residual_seconds.map(f64::abs);
        let timing_uncertainty_seconds = [
            Some(policy.class_default_seconds(self.time_quality)),
            self.reported_holdover_error_seconds,
            measured_residual_seconds,
            self.step_penalty.map(|penalty| penalty.uncertainty_seconds),
        ]
        .into_iter()
        .flatten()
        .fold(0.0_f64, f64::max);
        NodeClockAssessment {
            node_id: node_id.to_string(),
            time_quality: self.time_quality,
            timing_uncertainty_seconds,
            timeline_residual_seconds: self.timeline_residual_seconds,
            last_timestamp_step_ns: self.last_timestamp_step_ns,
            timestamp_step_count: self.timestamp_step_count,
            residual_over_hard_limit: self.residual_over_hard_limit,
            timeline_step_reanchor_count: self.timeline_step_reanchor_count,
            timeline_drift_slew_count: self.timeline_drift_slew_count,
        }
    }
}

/// The residual whose magnitude sits at `percentile` (nearest rank) of the
/// window, sign kept: 50 is the median, so one outlier frame never decides it.
fn residual_at_percentile(residuals_seconds: &VecDeque<f64>, percentile: f64) -> Option<f64> {
    let mut by_magnitude: Vec<f64> = residuals_seconds.iter().copied().collect();
    if by_magnitude.is_empty() {
        return None;
    }
    by_magnitude.sort_by(|left, right| left.abs().total_cmp(&right.abs()));
    let rank = ((percentile / 100.0) * (by_magnitude.len() - 1) as f64).round() as usize;
    Some(by_magnitude[rank.min(by_magnitude.len() - 1)])
}

/// One buffered segment of one node.
pub(crate) struct NodeFrameTimingObservation<'a> {
    pub(crate) node_id: &'a str,
    pub(crate) frame_timing: &'a [PacketFrameTiming],
    /// `None` when the segment's sample indices are incomplete: steps need them.
    pub(crate) segment_start_sample_index: Option<i64>,
    pub(crate) sample_rate_hz: u32,
    /// Whether this segment's audio was placed by node packet time. When it
    /// was not (no node timestamp: server or receipt time), the node is
    /// assessed as an unknown clock until a segment is on packet time again.
    pub(crate) segment_on_packet_time: bool,
    /// The node's buffer timeline after this segment was appended (before
    /// this segment's own corrections, which `observe` returns); `None` when
    /// that buffer is not on packet time (receipt-time alignment).
    pub(crate) buffer_timeline: Option<NominalSampleTimeline>,
    pub(crate) now_ns: u128,
}

/// The nominal-rate timeline `buffer` placed a just-appended segment on (its
/// own anchor, extrapolated at the nominal rate). `None` when the buffer is
/// not on packet time, so a residual against it would be meaningless.
pub(crate) fn packet_time_buffer_timeline(
    buffer: &SensorStreamBuffer,
    segment_start_sample_index: Option<i64>,
    buffer_is_on_packet_time: bool,
) -> Option<NominalSampleTimeline> {
    if !buffer_is_on_packet_time {
        return None;
    }
    let origin_sample_index = segment_start_sample_index?;
    Some(NominalSampleTimeline {
        origin_time_ns: buffer.time_for_sample_index(origin_sample_index)?,
        origin_sample_index,
        sample_rate_hz: buffer.sample_rate_hz_value(),
    })
}

pub(crate) struct NodeClockQualityTracker {
    policy: NodeClockPolicy,
    nodes: HashMap<String, NodeClockState>,
}

impl NodeClockQualityTracker {
    pub(crate) fn new(config: &LocalizationSolveConfig) -> Self {
        Self {
            policy: NodeClockPolicy::from_config(config),
            nodes: HashMap::new(),
        }
    }

    /// Returns the timeline corrections the worker must apply to every buffer
    /// channel of `observation.node_id`.
    pub(crate) fn observe(
        &mut self,
        observation: NodeFrameTimingObservation<'_>,
    ) -> NodeTimelineCorrections {
        let policy = &self.policy;
        let state = self
            .nodes
            .entry(observation.node_id.to_string())
            .or_insert_with(|| NodeClockState::new(policy));
        state.last_heard_ns = observation.now_ns;
        if !observation.segment_on_packet_time {
            state.clear_residual();
            state.timeline_discipline.cancel_pending_step();
        }
        state.latest_segment_on_packet_time = observation.segment_on_packet_time;
        let mut timeline = observation.buffer_timeline;
        let mut corrections = NodeTimelineCorrections::default();
        for frame in observation.frame_timing {
            let point = observation
                .segment_start_sample_index
                .zip(frame.packet_timestamp_ns)
                .map(
                    |(segment_start_sample_index, packet_timestamp_ns)| FrameTimelinePoint {
                        start_sample_index: frame.start_sample_index(segment_start_sample_index),
                        packet_timestamp_ns,
                        boot_session: frame.boot_session,
                    },
                );
            state.observe_frame(
                frame,
                point,
                &observation,
                &mut timeline,
                policy,
                &mut corrections,
            );
        }
        corrections
    }

    /// A node never heard (a declared member that has not streamed) gets the
    /// unknown-class default: its pairs are widened, not excluded.
    pub(crate) fn assessment(&self, node_id: &str) -> NodeClockAssessment {
        match self.nodes.get(node_id) {
            Some(state) => state.assessment(node_id, &self.policy),
            None => NodeClockState::new(&self.policy).assessment(node_id, &self.policy),
        }
    }

    /// Every node of the layout, taken when the solve is triggered.
    pub(crate) fn snapshot_for_layout(
        &self,
        layout: &SolveSensorLayout,
    ) -> Arc<NodeClockAssessments> {
        let mut assessments = NodeClockAssessments::new();
        for sensor in &layout.sensors {
            if !assessments.contains_key(&sensor.node_id) {
                assessments.insert(sensor.node_id.clone(), self.assessment(&sensor.node_id));
            }
        }
        Arc::new(assessments)
    }

    pub(crate) fn evict_stale(&mut self, now_ns: u128) {
        let stale_after_ns = self.policy.stale_after_ns;
        self.nodes
            .retain(|_, state| now_ns.saturating_sub(state.last_heard_ns) <= stale_after_ns);
    }

    #[cfg(test)]
    pub(crate) fn tracked_node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
#[path = "dsp_node_clock_quality_tests.rs"]
mod dsp_node_clock_quality_tests;

#[cfg(test)]
#[path = "dsp_node_timeline_discipline_tests.rs"]
mod dsp_node_timeline_discipline_tests;
