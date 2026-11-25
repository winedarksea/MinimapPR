//! Per-node clock evidence one solve uses (shared contract between R5 clock
//! evidence, which produces it, and R7 pair admission, which consumes it).
//!
//! Policy (owner decision): weak clocks never exclude a pair by class; they
//! widen `timing_uncertainty_seconds`. Only a *measured* timeline residual over
//! `localization_cross_node_max_timeline_residual_seconds` drops that node's
//! cross-node pairs (`residual_over_hard_limit`); its own channels still solve
//! its own triggers. The flag comes from a windowed residual with hysteresis;
//! with the timeline discipline on, steps are re-anchored and drift slewed
//! before the residual gets there (`NodeTimelineCorrections` below).

use std::collections::BTreeMap;

use minimappr_core::{
    localization_sensor_entry::NodeTimeQuality,
    timeline_discipline::{TimelineCorrection, TimelineCorrectionKind},
};

/// Clock assessment of one node at the moment a solve is triggered.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeClockAssessment {
    pub(crate) node_id: String,
    pub(crate) time_quality: NodeTimeQuality,
    /// Largest of: clock-class default, reported holdover error, measured
    /// timeline residual, and an active post-step penalty. Never unknown: an
    /// unknown class takes the configured conservative default.
    pub(crate) timing_uncertainty_seconds: f64,
    /// Measured deviation of packet timestamps from the node's own nominal-rate
    /// timeline, as a percentile over its recent frames; `None` until measured
    /// or while its audio is not on packet time.
    pub(crate) timeline_residual_seconds: Option<f64>,
    pub(crate) last_timestamp_step_ns: Option<i64>,
    pub(crate) timestamp_step_count: u32,
    /// Set when the windowed residual exceeds the hard limit; cleared only
    /// below the limit times the configured clear ratio (hysteresis).
    pub(crate) residual_over_hard_limit: bool,
    /// Confirmed timestamp steps the node's buffer timeline was re-anchored on.
    pub(crate) timeline_step_reanchor_count: u32,
    /// Drift slews of the node's buffer timeline origin.
    pub(crate) timeline_drift_slew_count: u32,
}

/// Snapshot of every node's assessment, keyed by node id, taken at trigger and
/// carried with the solve (like the sensor layout snapshot).
pub(crate) type NodeClockAssessments = BTreeMap<String, NodeClockAssessment>;

/// Origin shift the worker applies to every buffer channel of the node after
/// one segment (summed: every correction is an origin shift of one mapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeTimelineCorrections {
    pub(crate) origin_shift_ns: i128,
    pub(crate) step_reanchors: u32,
    pub(crate) drift_slews: u32,
}

impl NodeTimelineCorrections {
    pub(crate) fn add(&mut self, correction: TimelineCorrection) {
        self.origin_shift_ns += correction.origin_shift_ns;
        match correction.kind {
            TimelineCorrectionKind::StepReanchor => self.step_reanchors += 1,
            TimelineCorrectionKind::DriftSlew => self.drift_slews += 1,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.step_reanchors == 0 && self.drift_slews == 0
    }
}
