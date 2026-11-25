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

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use minimappr_core::{
    localization_sensor_entry::NodeTimeQuality,
    packet_frame_timing::PacketFrameTiming,
    timestamp_step_detector::{
        FrameTimelinePoint, NominalSampleTimeline, TimestampStepDetector, TimingDiscontinuity,
    },
};

use super::{NodeClockAssessment, NodeClockAssessments, SolveSensorLayout};
use crate::{dsp::SensorStreamBuffer, localization_solve_config::LocalizationSolveConfig};

const NANOSECONDS_PER_SECOND_F64: f64 = 1.0e9;

#[derive(Clone, Debug)]
struct NodeClockPolicy {
    clock_class_uncertainty_seconds: BTreeMap<String, f64>,
    unknown_uncertainty_seconds: f64,
    step_threshold_seconds: f64,
    step_recovery_holdoff_seconds: f64,
    max_timeline_residual_seconds: f64,
    stale_after_ns: u128,
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
            stale_after_ns: (config.localization_sensor_registry_stale_seconds.max(0.0)
                * NANOSECONDS_PER_SECOND_F64) as u128,
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
    timeline_residual_seconds: Option<f64>,
    last_timestamp_step_ns: Option<i64>,
    timestamp_step_count: u32,
    step_penalty: Option<StepPenalty>,
    last_heard_ns: u128,
}

impl NodeClockState {
    fn new(policy: &NodeClockPolicy) -> Self {
        Self {
            detector: TimestampStepDetector::new(policy.step_threshold_seconds),
            time_quality: NodeTimeQuality::Unknown,
            reported_holdover_error_seconds: None,
            timeline_residual_seconds: None,
            last_timestamp_step_ns: None,
            timestamp_step_count: 0,
            step_penalty: None,
            last_heard_ns: 0,
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
        policy: &NodeClockPolicy,
    ) {
        self.time_quality = frame.time_quality;
        self.reported_holdover_error_seconds = frame
            .reported_holdover_error_ns
            .map(|error_ns| error_ns as f64 / NANOSECONDS_PER_SECOND_F64);
        let Some(point) = point else {
            return;
        };
        match self.detector.observe(point, observation.sample_rate_hz) {
            Some(TimingDiscontinuity::TimestampStep { step_ns }) => {
                self.timestamp_step_count = self.timestamp_step_count.saturating_add(1);
                self.last_timestamp_step_ns =
                    Some(step_ns.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64);
                self.raise_step_penalty(step_ns.unsigned_abs() as f64 / NANOSECONDS_PER_SECOND_F64);
            }
            Some(TimingDiscontinuity::BootSessionChange) => {
                // The old session's residual says nothing about the new clock,
                // and a reboot has no step size: hold the unknown default.
                self.timeline_residual_seconds = None;
                self.raise_step_penalty(policy.unknown_uncertainty_seconds);
            }
            None => {}
        }
        if let Some(timeline) = observation.buffer_timeline {
            let residual_seconds = timeline.residual_ns(&point) as f64 / NANOSECONDS_PER_SECOND_F64;
            self.timeline_residual_seconds = Some(residual_seconds);
            self.advance_step_recovery(
                residual_seconds,
                point.start_sample_index,
                observation.sample_rate_hz,
                policy,
            );
        }
    }

    fn assessment(&self, node_id: &str, policy: &NodeClockPolicy) -> NodeClockAssessment {
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
            residual_over_hard_limit: measured_residual_seconds
                .is_some_and(|residual| residual > policy.max_timeline_residual_seconds),
        }
    }
}

/// One buffered segment of one node.
pub(crate) struct NodeFrameTimingObservation<'a> {
    pub(crate) node_id: &'a str,
    pub(crate) frame_timing: &'a [PacketFrameTiming],
    /// `None` when the segment's sample indices are incomplete: steps need them.
    pub(crate) segment_start_sample_index: Option<i64>,
    pub(crate) sample_rate_hz: u32,
    /// The node's buffer timeline after this segment was appended; `None`
    /// when that buffer is not on packet time (receipt-time alignment).
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

    pub(crate) fn observe(&mut self, observation: NodeFrameTimingObservation<'_>) {
        let policy = &self.policy;
        let state = self
            .nodes
            .entry(observation.node_id.to_string())
            .or_insert_with(|| NodeClockState::new(policy));
        state.last_heard_ns = observation.now_ns;
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
            state.observe_frame(frame, point, &observation, policy);
        }
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
