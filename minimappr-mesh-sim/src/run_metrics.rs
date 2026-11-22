//! What a run records beyond the state machines' own state.

use std::collections::BTreeMap;

use minimappr_core::mesh::host_state_machine::FieldHostStateMachine;
use minimappr_core::mesh::node_state_machine::NodeUserNotice;
use minimappr_core::mesh::DetectionKey;

use crate::simulated_network::ActiveTransport;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostTenure {
    pub device: usize,
    pub host_term: u32,
    pub started_s: f64,
}

#[derive(Default)]
pub struct RunMetrics {
    /// Every detection a node produced, with the device and time.
    pub generated_detections: Vec<(DetectionKey, usize, f64)>,
    /// Which scripted call each detection came from (entity scoring).
    pub call_index_by_detection: BTreeMap<DetectionKey, usize>,
    /// Per call: devices whose Stage-1 triggered / whose classifier ran /
    /// that produced a detection.
    pub stage1_triggers: BTreeMap<usize, Vec<usize>>,
    pub classifier_runs: BTreeMap<usize, Vec<usize>>,
    pub detections_by_call: BTreeMap<usize, Vec<usize>>,
    pub notices: Vec<(f64, usize, NodeUserNotice)>,
    pub transport_changes: Vec<(f64, ActiveTransport)>,
    pub host_tenures: Vec<HostTenure>,
    pub killed_devices: Vec<(f64, usize)>,
    pub stalls_applied: Vec<(f64, usize, bool)>,
    /// (time, host log head) sampled every tick, for the convergence lag.
    pub host_log_head_history: Vec<(f64, u64)>,
    pub retired_hosts: Vec<(usize, FieldHostStateMachine)>,
    pub relay_cap_reached_s: Option<f64>,
    pub mic_only_since: BTreeMap<usize, f64>,
}

impl RunMetrics {
    pub fn host_log_head_at(&self, time_s: f64) -> u64 {
        self.host_log_head_history
            .iter()
            .take_while(|(t, _)| *t <= time_s)
            .last()
            .map_or(0, |(_, head)| *head)
    }

    pub fn first_notice(&self, device: usize, notice: NodeUserNotice) -> Option<f64> {
        self.notices
            .iter()
            .find(|(_, d, n)| *d == device && *n == notice)
            .map(|(t, _, _)| *t)
    }
}
