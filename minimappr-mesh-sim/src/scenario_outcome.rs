//! Final state of a run plus scoring against simulator truth.
//!
//! Fixes are scored in the frame of the phones themselves: the host's node
//! positions are aligned to the true positions with a proper rigid fit
//! (rotation + translation, no reflection), and the fix is moved with them.
//! A map-anchor offset therefore does not count as localization error (it
//! shifts every bird equally), while a mirrored layout does: a reflection
//! cannot be undone by a proper fit, so a mirrored bird shows up as error
//! and is also flagged directly.

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::mesh::call_pipeline::LocalizedCallRecord;
use minimappr_core::mesh::event_log::decode_entry;
use minimappr_core::mesh::host_state_machine::FieldHostStateMachine;
use minimappr_core::protocol::{FieldMessage, NodeId};

use crate::acoustic_world::AcousticWorld;
use crate::device_harness::SimulatedDevice;
use crate::run_metrics::RunMetrics;
use crate::scenario::ScenarioSpec;
use crate::scenario_runner::ScenarioRun;
use crate::simulated_network::{NetworkStats, SimulatedNetwork};
use crate::synthetic_clip::SolveTrace;

const MINIMUM_NODES_FOR_CHIRALITY_TEST: usize = 3;
/// A layout is called mirrored when the reflected fit is this much better.
const MIRROR_RESIDUAL_RATIO: f64 = 0.5;
const MIRROR_MINIMUM_PROPER_RMS_M: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredFix {
    pub host_call_id: u64,
    pub call_index: usize,
    pub horizontal_error_m: f64,
    pub reported_major_sigma_m: f64,
    pub layout_mirrored: Option<bool>,
    pub node_ids: Vec<NodeId>,
}

pub struct ScenarioOutcome {
    pub seed: u64,
    pub spec: ScenarioSpec,
    pub world: AcousticWorld,
    pub devices: Vec<SimulatedDevice>,
    pub network: SimulatedNetwork,
    pub metrics: RunMetrics,
    pub traces: Vec<SolveTrace>,
    pub end_s: f64,
}

fn rms_residual(source: &[[f64; 2]], target: &[[f64; 2]], mirrored: bool) -> f64 {
    let weights = vec![1.0; source.len()];
    let transform = weighted_kabsch_2d(source, target, &weights, mirrored);
    let sum: f64 = source
        .iter()
        .zip(target)
        .map(|(s, t)| {
            let p = transform.apply(*s);
            (p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2)
        })
        .sum();
    (sum / source.len().max(1) as f64).sqrt()
}

impl ScenarioOutcome {
    pub fn from_run(run: ScenarioRun) -> Self {
        let traces = run.localizer.traces.take();
        Self {
            seed: run.seed,
            end_s: run.now_s,
            spec: run.spec,
            world: run.world,
            devices: run.devices,
            network: run.network,
            metrics: run.metrics,
            traces,
        }
    }

    pub fn network_stats(&self) -> &NetworkStats {
        &self.network.stats
    }

    pub fn active_host(&self) -> Option<(usize, &FieldHostStateMachine)> {
        self.devices
            .iter()
            .filter(|device| self.network.alive[device.index])
            .find_map(|device| device.host.as_ref().map(|host| (device.index, host)))
    }

    /// Every host that ever ran, retired ones first.
    pub fn all_hosts(&self) -> Vec<(usize, &FieldHostStateMachine)> {
        let mut hosts: Vec<(usize, &FieldHostStateMachine)> =
            self.metrics.retired_hosts.iter().map(|(device, host)| (*device, host)).collect();
        hosts.extend(self.active_host());
        hosts
    }

    pub fn call_records(&self) -> Vec<&LocalizedCallRecord> {
        self.all_hosts().into_iter().flat_map(|(_, host)| host.calls.completed.iter()).collect()
    }

    pub fn host_log_records(&self) -> Vec<FieldMessage> {
        self.active_host()
            .map(|(_, host)| host.event_log.entries().iter().filter_map(decode_entry).collect())
            .unwrap_or_default()
    }

    pub fn scored_fixes(&self) -> Vec<ScoredFix> {
        self.traces
            .iter()
            .filter_map(|trace| {
                let fix = trace.fix.as_ref()?;
                let call_index = trace.true_call_index as usize;
                let call = &self.world.calls[call_index];
                let estimated: Vec<[f64; 2]> = trace.estimated_node_positions_m.iter().map(|p| [p[0], p[1]]).collect();
                let truth: Vec<[f64; 2]> = trace
                    .node_ids
                    .iter()
                    .map(|node| {
                        let p = self.world.position(node.0 as usize, call.time_s);
                        [p[0], p[1]]
                    })
                    .collect();
                let transform = weighted_kabsch_2d(&estimated, &truth, &vec![1.0; estimated.len()], false);
                let aligned = transform.apply([fix.position_m[0], fix.position_m[1]]);
                let horizontal_error_m =
                    ((aligned[0] - call.position_m[0]).powi(2) + (aligned[1] - call.position_m[1]).powi(2)).sqrt();
                let layout_mirrored = (estimated.len() >= MINIMUM_NODES_FOR_CHIRALITY_TEST).then(|| {
                    let proper = rms_residual(&estimated, &truth, false);
                    let mirrored = rms_residual(&estimated, &truth, true);
                    proper > MIRROR_MINIMUM_PROPER_RMS_M && mirrored < MIRROR_RESIDUAL_RATIO * proper
                });
                Some(ScoredFix {
                    host_call_id: trace.host_call_id,
                    call_index,
                    horizontal_error_m,
                    reported_major_sigma_m: fix.horizontal_major_sigma_m,
                    layout_mirrored,
                    node_ids: trace.node_ids.clone(),
                })
            })
            .collect()
    }

    pub fn alive(&self, device: usize) -> bool {
        self.network.alive[device]
    }
}
