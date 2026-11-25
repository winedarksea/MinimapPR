//! Which sensor pairs one solve may correlate, and how well each admitted
//! pair is timed. Computed once per solve from its layout snapshot, active
//! sensors and clock snapshot, and honoured by SRP pair observations, the
//! position covariance (per-clock timing noise), render channel selection,
//! the deferred path and the published `excluded_pairs` / `pair_tdoas`.
//!
//! SRP solves only the active sensors joined to the reporting node through
//! admitted pairs (`solve_sensor_indices`): its reference fit, centroid,
//! aperture and grid assume one connected array, so a timeline-rejected
//! node (all its cross-node pairs dropped) must leave the geometry too.
//!
//! Policy (owner decision): a weak clock class never excludes a pair, it only
//! widens `timing_uncertainty_seconds`. A pair is dropped only when it cannot
//! be measured (missing coverage), cannot carry information (baseline under
//! the minimum), or a node's measured timeline is known to be wrong (residual
//! over the hard limit; that node's own intra-node pairs stay admitted).
//! Sensors the layout already excluded (rate, geometry, cap) contribute no
//! pairs at all: their exclusion is reported once, on the sensor.

use std::collections::BTreeSet;

use minimappr_core::localization_sensor_entry::{ExcludedSensorPair, SolveExclusionReason};

use crate::{
    dsp_worker::{NodeClockAssessments, SolveSensorLayout},
    localization_solve_config::LocalizationSolveConfig,
    srp_phat::{ChannelClockOffsets, SensorNodePositionStd},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SolvePairAdmission {
    excluded_pairs: Vec<ExcludedSensorPair>,
    /// Per layout sensor: index into `node_timing_uncertainty_seconds`.
    sensor_node_indices: Vec<usize>,
    /// Per node of the layout (first-seen order): 1-sigma timing uncertainty.
    node_timing_uncertainty_seconds: Vec<f64>,
    /// Per node: the 1-sigma position error its placement states (an
    /// operator's entered accuracy, a stationary estimate's spread); 0 when
    /// none is stated, which adds nothing to the covariance.
    node_position_std_m: Vec<f64>,
    /// Nodes whose cross-node pairs were dropped for a timeline residual over
    /// the hard limit; their audio cannot be time-aligned with other nodes.
    timeline_rejected_node_ids: BTreeSet<String>,
    reporting_node_id: String,
    /// Active sensors connected to the reporting node through admitted pairs,
    /// ascending: the channel set SRP solves on.
    solve_sensor_indices: Vec<usize>,
}

impl SolvePairAdmission {
    /// Admission for one solve. `node_clock` is the snapshot taken at trigger;
    /// a node missing from it is timed as an unknown clock
    /// (`localization_clock_unknown_uncertainty_seconds`), never excluded.
    pub(crate) fn compute(
        layout: &SolveSensorLayout,
        active_sensor_indices: &[usize],
        node_clock: &NodeClockAssessments,
        config: &LocalizationSolveConfig,
    ) -> SolvePairAdmission {
        let mut node_ids: Vec<&str> = Vec::new();
        let sensor_node_indices: Vec<usize> = layout
            .sensors
            .iter()
            .map(|sensor| {
                node_ids
                    .iter()
                    .position(|node_id| *node_id == sensor.node_id)
                    .unwrap_or_else(|| {
                        node_ids.push(&sensor.node_id);
                        node_ids.len() - 1
                    })
            })
            .collect();
        let node_timing_uncertainty_seconds = node_ids
            .iter()
            .map(|node_id| {
                node_clock.get(*node_id).map_or(
                    config.localization_clock_unknown_uncertainty_seconds,
                    |assessment| assessment.timing_uncertainty_seconds,
                )
            })
            .collect();
        // Every sensor of a node carries its node's placement accuracy; the
        // largest is kept so a mixed snapshot never understates it.
        let mut node_position_std_m = vec![0.0_f64; node_ids.len()];
        for (sensor, &node_index) in layout.sensors.iter().zip(&sensor_node_indices) {
            if let Some(position_std_m) = sensor.position_std_m {
                node_position_std_m[node_index] =
                    node_position_std_m[node_index].max(position_std_m);
            }
        }
        let residual_over_hard_limit = |node_id: &str| {
            node_clock
                .get(node_id)
                .is_some_and(|assessment| assessment.residual_over_hard_limit)
        };

        let pair_eligible_sensor_indices: Vec<usize> = (0..layout.sensor_count())
            .filter(|sensor_index| {
                !layout
                    .excluded_sensors
                    .iter()
                    .any(|excluded| excluded.sensor_index == *sensor_index)
            })
            .collect();
        let mut excluded_pairs = Vec::new();
        let mut timeline_rejected_node_ids = BTreeSet::new();
        for (left_position, &sensor_index_a) in pair_eligible_sensor_indices.iter().enumerate() {
            for &sensor_index_b in &pair_eligible_sensor_indices[left_position + 1..] {
                let node_a = layout.sensors[sensor_index_a].node_id.as_str();
                let node_b = layout.sensors[sensor_index_b].node_id.as_str();
                let timeline_rejected_nodes: Vec<&str> = if node_a == node_b {
                    Vec::new()
                } else {
                    [node_a, node_b]
                        .into_iter()
                        .filter(|node_id| residual_over_hard_limit(node_id))
                        .collect()
                };
                timeline_rejected_node_ids
                    .extend(timeline_rejected_nodes.iter().map(|id| id.to_string()));
                let reason = if !active_sensor_indices.contains(&sensor_index_a)
                    || !active_sensor_indices.contains(&sensor_index_b)
                {
                    Some(SolveExclusionReason::CoverageBelowThreshold)
                } else if sensor_separation_m(layout, sensor_index_a, sensor_index_b)
                    < f64::from(config.localization_min_pair_baseline_m)
                {
                    Some(SolveExclusionReason::PairBaselineBelowMinimum)
                } else if !timeline_rejected_nodes.is_empty() {
                    Some(SolveExclusionReason::TimelineResidualOverLimit)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    excluded_pairs.push(ExcludedSensorPair {
                        sensor_index_a,
                        sensor_index_b,
                        reason,
                    });
                }
            }
        }
        let admitted_active_sensor_indices: Vec<usize> = pair_eligible_sensor_indices
            .iter()
            .copied()
            .filter(|sensor_index| active_sensor_indices.contains(sensor_index))
            .collect();
        let solve_sensor_indices = reporter_connected_sensor_indices(
            layout,
            &admitted_active_sensor_indices,
            &excluded_pairs,
        );
        SolvePairAdmission {
            excluded_pairs,
            sensor_node_indices,
            node_timing_uncertainty_seconds,
            node_position_std_m,
            timeline_rejected_node_ids,
            reporting_node_id: layout.reporting_node_id.clone(),
            solve_sensor_indices,
        }
    }

    /// The channels SRP solves on (see the module docs). Provenance keeps
    /// listing every covered sensor as active and every rejected pair.
    pub(crate) fn solve_sensor_indices(&self) -> &[usize] {
        &self.solve_sensor_indices
    }

    pub(crate) fn admits_pair(&self, sensor_index_a: usize, sensor_index_b: usize) -> bool {
        !self.excluded_pairs.iter().any(|pair| {
            (pair.sensor_index_a, pair.sensor_index_b) == (sensor_index_a, sensor_index_b)
                || (pair.sensor_index_a, pair.sensor_index_b) == (sensor_index_b, sensor_index_a)
        })
    }

    pub(crate) fn excluded_pairs(&self) -> &[ExcludedSensorPair] {
        &self.excluded_pairs
    }

    /// 1-sigma timing uncertainty of a pair: 0 for two sensors of one node
    /// (one clock, so no relative offset); otherwise the two nodes'
    /// independent clock uncertainties in quadrature.
    pub(crate) fn pair_timing_uncertainty_seconds(
        &self,
        sensor_index_a: usize,
        sensor_index_b: usize,
    ) -> f64 {
        let node_a = self.sensor_node_indices[sensor_index_a];
        let node_b = self.sensor_node_indices[sensor_index_b];
        if node_a == node_b {
            return 0.0;
        }
        self.node_timing_uncertainty_seconds[node_a]
            .hypot(self.node_timing_uncertainty_seconds[node_b])
    }

    /// Each buffer channel's clock (its node) and every clock's 1-sigma
    /// offset, for the solver's measurement-noise term: a node's offset
    /// shifts all of its cross-node TDOAs together, so the noise is
    /// correlated per clock, not independent per pair.
    pub(crate) fn channel_clock_offsets(&self) -> ChannelClockOffsets<'_> {
        ChannelClockOffsets {
            clock_index_by_channel: &self.sensor_node_indices,
            clock_offset_std_seconds: &self.node_timing_uncertainty_seconds,
        }
    }

    /// Each buffer channel's node and every node's stated position error,
    /// for the solver's node-position covariance term (owner decision: a
    /// stated accuracy widens the error, it never excludes the node).
    pub(crate) fn channel_node_position_std(&self) -> SensorNodePositionStd<'_> {
        SensorNodePositionStd {
            node_index_by_sensor: &self.sensor_node_indices,
            node_position_std_m: &self.node_position_std_m,
        }
    }

    /// Buffer channels the beamformed render must leave out: a node whose
    /// timeline is over the limit cannot be delay-aligned with other nodes.
    /// When that node is the reporter, its own channels are the render (as
    /// its own pairs are the solve); otherwise the rejected node drops out.
    pub(crate) fn render_excluded_sensor_indices(&self, layout: &SolveSensorLayout) -> Vec<usize> {
        if self.timeline_rejected_node_ids.is_empty() {
            return Vec::new();
        }
        let reporter_rejected = self
            .timeline_rejected_node_ids
            .contains(&self.reporting_node_id);
        layout
            .sensors
            .iter()
            .enumerate()
            .filter(|(_, sensor)| {
                if reporter_rejected {
                    sensor.node_id != self.reporting_node_id
                } else {
                    self.timeline_rejected_node_ids.contains(&sensor.node_id)
                }
            })
            .map(|(sensor_index, _)| sensor_index)
            .collect()
    }
}

/// Walks admitted pairs outward from the reporter's active
/// sensors. A reporter with no covered sensor (a parked solve released after
/// its own coverage lapsed) anchors on the largest connected group instead,
/// so the solve still describes one array.
fn reporter_connected_sensor_indices(
    layout: &SolveSensorLayout,
    admitted_active_sensor_indices: &[usize],
    excluded_pairs: &[ExcludedSensorPair],
) -> Vec<usize> {
    let pair_admitted = |a: usize, b: usize| {
        !excluded_pairs
            .iter()
            .any(|pair| (pair.sensor_index_a, pair.sensor_index_b) == (a.min(b), a.max(b)))
    };
    let connected_from = |seeds: Vec<usize>| {
        let mut connected = seeds.clone();
        let mut frontier = seeds;
        while let Some(sensor_index) = frontier.pop() {
            for &candidate in admitted_active_sensor_indices {
                if !connected.contains(&candidate) && pair_admitted(sensor_index, candidate) {
                    connected.push(candidate);
                    frontier.push(candidate);
                }
            }
        }
        connected.sort_unstable();
        connected
    };
    let reporter_sensor_indices: Vec<usize> = admitted_active_sensor_indices
        .iter()
        .copied()
        .filter(|index| layout.sensors[*index].node_id == layout.reporting_node_id)
        .collect();
    if !reporter_sensor_indices.is_empty() {
        return connected_from(reporter_sensor_indices);
    }
    let mut largest_group: Vec<usize> = Vec::new();
    for &sensor_index in admitted_active_sensor_indices {
        if largest_group.contains(&sensor_index) {
            continue;
        }
        let group = connected_from(vec![sensor_index]);
        if group.len() > largest_group.len() {
            largest_group = group;
        }
    }
    largest_group
}

fn sensor_separation_m(
    layout: &SolveSensorLayout,
    sensor_index_a: usize,
    sensor_index_b: usize,
) -> f64 {
    let position_a = layout.sensors[sensor_index_a].position_m;
    let position_b = layout.sensors[sensor_index_b].position_m;
    position_a
        .iter()
        .zip(position_b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
#[path = "dsp_solve_pair_admission_tests.rs"]
mod dsp_solve_pair_admission_tests;
