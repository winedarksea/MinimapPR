//! Per-node clock evidence one solve uses (shared contract between R5 clock
//! evidence, which produces it, and R7 pair admission, which consumes it).
//!
//! Policy (owner decision): weak clocks never exclude a pair by class; they
//! widen `timing_uncertainty_seconds`. Only a *measured* timeline residual over
//! `localization_cross_node_max_timeline_residual_seconds` drops that node's
//! cross-node pairs (`residual_over_hard_limit`); its own channels still solve.

use std::collections::BTreeMap;

use minimappr_core::localization_sensor_entry::NodeTimeQuality;

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
    /// timeline; `None` until enough frames have been seen to measure it.
    pub(crate) timeline_residual_seconds: Option<f64>,
    pub(crate) last_timestamp_step_ns: Option<i64>,
    pub(crate) timestamp_step_count: u32,
    /// True only when the measured residual exceeds the hard limit.
    pub(crate) residual_over_hard_limit: bool,
}

/// Snapshot of every node's assessment, keyed by node id, taken at trigger and
/// carried with the solve (like the sensor layout snapshot).
pub(crate) type NodeClockAssessments = BTreeMap<String, NodeClockAssessment>;
