//! Per-node clock evidence one solve uses (shared contract between R5 clock
//! evidence, which produces it, and R7 pair admission, which consumes it).
//!
//! Policy (owner decision): weak clocks never exclude a pair by class; they
//! widen `timing_uncertainty_seconds`. Only a *measured* timeline residual over
//! `localization_cross_node_max_timeline_residual_seconds` drops that node's
//! cross-node pairs (`residual_over_hard_limit`); its own channels still solve
//! its own triggers. The flag comes from a windowed residual with hysteresis.

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
    /// timeline, as a percentile over its recent frames; `None` until measured
    /// or while its audio is not on packet time.
    pub(crate) timeline_residual_seconds: Option<f64>,
    pub(crate) last_timestamp_step_ns: Option<i64>,
    pub(crate) timestamp_step_count: u32,
    /// Set when the windowed residual exceeds the hard limit; cleared only
    /// below the limit times the configured clear ratio (hysteresis).
    pub(crate) residual_over_hard_limit: bool,
}

/// Snapshot of every node's assessment, keyed by node id, taken at trigger and
/// carried with the solve (like the sensor layout snapshot).
pub(crate) type NodeClockAssessments = BTreeMap<String, NodeClockAssessment>;
