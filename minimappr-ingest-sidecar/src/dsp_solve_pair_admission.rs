//! Which sensor pairs one solve may correlate, and how well each admitted
//! pair is timed. Computed once per solve from its layout snapshot, active
//! sensors and clock snapshot, and honoured by SRP pair observations (and so
//! the covariance), render channel selection, the deferred path and the
//! published `excluded_pairs` / `pair_tdoas`.
//!
//! Policy (owner decision): a weak clock class never excludes a pair, it only
//! widens `timing_uncertainty_seconds`. A pair is dropped only when it cannot
//! be measured (missing coverage), cannot carry information (baseline under
//! the minimum), or a node's measured timeline is known to be wrong (residual
//! over the hard limit; that node's own intra-node pairs still solve).
//! Sensors the layout already excluded (rate, geometry, cap) contribute no
//! pairs at all: their exclusion is reported once, on the sensor.

use std::collections::BTreeSet;

use minimappr_core::localization_sensor_entry::{ExcludedSensorPair, SolveExclusionReason};

use crate::{
    dsp_worker::{NodeClockAssessments, PairTdoa, SolveSensorLayout},
    localization_solve_config::LocalizationSolveConfig,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SolvePairAdmission {
    excluded_pairs: Vec<ExcludedSensorPair>,
    /// Per layout sensor: index into `node_timing_uncertainty_seconds`.
    sensor_node_indices: Vec<usize>,
    /// Per node of the layout (first-seen order): 1-sigma timing uncertainty.
    node_timing_uncertainty_seconds: Vec<f64>,
    /// Nodes whose cross-node pairs were dropped for a timeline residual over
    /// the hard limit; their audio cannot be time-aligned with other nodes.
    timeline_rejected_node_ids: BTreeSet<String>,
    reporting_node_id: String,
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
        SolvePairAdmission {
            excluded_pairs,
            sensor_node_indices,
            node_timing_uncertainty_seconds,
            timeline_rejected_node_ids,
            reporting_node_id: layout.reporting_node_id.clone(),
        }
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

    /// Variance (m^2) to add on every axis of the published position
    /// covariance for the clock error of the pairs the solve correlated.
    ///
    /// Why the worst admitted cross-node pair, isotropically: the SRP solver
    /// weights every pair equally, so it cannot down-weight a poorly timed
    /// node; and a node's clock offset shifts all of its cross-node TDOAs by
    /// the same amount, so the error does not average out across its pairs.
    /// The largest pair uncertainty times sound speed is therefore the range
    /// difference error the fix absorbs. The solver's covariance does not say
    /// which direction it lands in, so it widens every axis (honest coverage
    /// over a tight ellipse: "weak clocks widen the error").
    pub(crate) fn cross_node_timing_variance_m2(
        &self,
        correlated_pairs: &[PairTdoa],
        sound_speed_mps: f32,
    ) -> f64 {
        let worst_pair_timing_uncertainty_seconds = correlated_pairs
            .iter()
            .map(|pair| self.pair_timing_uncertainty_seconds(pair.ch_a, pair.ch_b))
            .fold(0.0_f64, f64::max);
        (f64::from(sound_speed_mps) * worst_pair_timing_uncertainty_seconds).powi(2)
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
