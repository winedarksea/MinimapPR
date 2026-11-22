//! Cluster SRP scoring and per-pair TDOA votes (port of `cluster_srp_score(s)`,
//! `vote_pair_tdoas` and `solve_reference_tdoas`). Each cluster's frames vote
//! for the curve maximum nearest the lag predicted at the cluster's SRP seed,
//! aggregated by weighted median. Split from `refinement.rs`.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use super::srp_grid::{distance, Point3};
use super::types::{OnsetCluster, OnsetFrame};
use crate::tdoa_solver::measurement_types::PairTdoaMeasurement;
use crate::tdoa_solver::robust_linear_algebra::minimum_norm_least_squares;

/// Minimum per-pair search half-width around the seed's predicted lag, in samples.
pub const MIN_LAG_SEARCH_HALF_WIDTH_SAMPLES: f64 = 2.0;

/// Correlation-peak floor for measurement weights.
pub const MIN_MEASUREMENT_WEIGHT: f64 = 0.05;

/// Python divides by `sum(weight_sum) or 1.0`.
pub const EMPTY_FRAME_WEIGHT: f64 = 1.0;

/// Shared geometry for scoring and voting.
pub struct ClusterGeometry<'window> {
    pub frames: &'window BTreeMap<usize, OnsetFrame>,
    pub sensor_positions: &'window BTreeMap<String, Point3>,
    pub sound_speed_mps: f64,
}

impl ClusterGeometry<'_> {
    pub fn predicted_lag_s(&self, position_m: &Point3, sensor_a: &str, sensor_b: &str) -> f64 {
        (distance(position_m, &self.sensor_positions[sensor_b])
            - distance(position_m, &self.sensor_positions[sensor_a]))
            / self.sound_speed_mps
    }

    /// The cluster's summed, weight-shared SRP at one point.
    pub fn cluster_srp_score(&self, position_m: &Point3, cluster: &OnsetCluster) -> f64 {
        let mut score = 0.0;
        for peak in &cluster.peaks {
            let frame = &self.frames[&peak.frame_index];
            let frame_total: f64 = frame.curves.iter().map(|curve| curve.weight_sum).sum();
            let total = if frame_total == 0.0 {
                EMPTY_FRAME_WEIGHT
            } else {
                frame_total
            };
            for curve in &frame.curves {
                let lag = self.predicted_lag_s(position_m, &curve.sensor_a, &curve.sensor_b);
                score += peak.weight * curve.value_at(lag) * curve.weight_sum / total;
            }
        }
        score
    }

    pub fn cluster_srp_scores(&self, points_m: &[Point3], cluster: &OnsetCluster) -> Vec<f64> {
        let mut scores = vec![0.0; points_m.len()];
        for peak in &cluster.peaks {
            let frame = &self.frames[&peak.frame_index];
            let frame_total: f64 = frame.curves.iter().map(|curve| curve.weight_sum).sum();
            let total = if frame_total == 0.0 {
                EMPTY_FRAME_WEIGHT
            } else {
                frame_total
            };
            for curve in &frame.curves {
                let position_a = self.sensor_positions[&curve.sensor_a];
                let position_b = self.sensor_positions[&curve.sensor_b];
                for (score, point) in scores.iter_mut().zip(points_m) {
                    let lag = (distance(point, &position_b) - distance(point, &position_a))
                        / self.sound_speed_mps;
                    *score += peak.weight * curve.value_at(lag) * curve.weight_sum / total;
                }
            }
        }
        scores
    }
}

fn weighted_median(values: &[f64], weights: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut cumulative = Vec::with_capacity(order.len());
    let mut running = 0.0;
    for index in &order {
        running += weights[*index];
        cumulative.push(running);
    }
    let half = 0.5 * running;
    let position = cumulative.partition_point(|value| *value < half);
    values[order[position.min(order.len() - 1)]]
}

/// Per-pair TDOA votes near the seed's predicted lags (weighted median).
pub fn vote_pair_tdoas(
    seed_m: &Point3,
    cluster: &OnsetCluster,
    geometry: &ClusterGeometry<'_>,
    sample_rate_hz: u32,
) -> Vec<PairTdoaMeasurement> {
    let mut spread_s: f64 = 0.0;
    for peak in &cluster.peaks {
        for curve in &geometry.frames[&peak.frame_index].curves {
            let at_peak =
                geometry.predicted_lag_s(&peak.position_m, &curve.sensor_a, &curve.sensor_b);
            let at_seed = geometry.predicted_lag_s(seed_m, &curve.sensor_a, &curve.sensor_b);
            spread_s = spread_s.max((at_peak - at_seed).abs());
        }
    }
    let half_width_s = spread_s + MIN_LAG_SEARCH_HALF_WIDTH_SAMPLES / sample_rate_hz as f64;
    // Insertion-ordered like the Python dict.
    let mut votes: Vec<((String, String), Vec<(f64, f64, f64)>)> = Vec::new();
    for peak in &cluster.peaks {
        for curve in &geometry.frames[&peak.frame_index].curves {
            let predicted = geometry.predicted_lag_s(seed_m, &curve.sensor_a, &curve.sensor_b);
            let (lag, value) = curve.fine_peak_lag_near(predicted, half_width_s);
            if value <= 0.0 {
                continue;
            }
            let (key, sign) = if curve.sensor_b < curve.sensor_a {
                ((curve.sensor_b.clone(), curve.sensor_a.clone()), -1.0)
            } else {
                ((curve.sensor_a.clone(), curve.sensor_b.clone()), 1.0)
            };
            let row = (sign * lag, peak.weight * value, value);
            match votes.iter_mut().find(|(existing, _)| *existing == key) {
                Some((_, rows)) => rows.push(row),
                None => votes.push((key, vec![row])),
            }
        }
    }
    votes
        .into_iter()
        .map(|((sensor_a, sensor_b), rows)| {
            let lags: Vec<f64> = rows.iter().map(|row| row.0).collect();
            let weights: Vec<f64> = rows.iter().map(|row| row.1).collect();
            let mean_value = rows.iter().map(|row| row.2).sum::<f64>() / rows.len() as f64;
            // PairTdoaMeasurement predicts (d_a - d_b) / c: swap so it is a behind b.
            PairTdoaMeasurement {
                sensor_a: sensor_b,
                sensor_b: sensor_a,
                tdoa_seconds: weighted_median(&lags, &weights),
                correlation_peak: mean_value,
                weight: mean_value.clamp(MIN_MEASUREMENT_WEIGHT, 1.0),
            }
        })
        .collect()
}

/// Least-squares reference-relative delays from pairwise TDOAs (`solve_reference_tdoas`).
pub fn solve_reference_tdoas(
    measurements: &[PairTdoaMeasurement],
    sensor_ids: &[String],
    reference_sensor: &str,
) -> BTreeMap<String, f64> {
    let measured: Vec<&String> = sensor_ids
        .iter()
        .filter(|id| id.as_str() != reference_sensor)
        .collect();
    if measurements.is_empty() {
        return BTreeMap::new();
    }
    let column_of = |id: &str| {
        measured
            .iter()
            .position(|candidate| candidate.as_str() == id)
    };
    let mut matrix = DMatrix::zeros(measurements.len(), measured.len());
    let mut target = DVector::zeros(measurements.len());
    for (row, measurement) in measurements.iter().enumerate() {
        let weight = measurement.weight.sqrt();
        if let Some(column) = column_of(&measurement.sensor_a) {
            matrix[(row, column)] = weight;
        }
        if let Some(column) = column_of(&measurement.sensor_b) {
            matrix[(row, column)] = -weight;
        }
        target[row] = measurement.tdoa_seconds * weight;
    }
    match minimum_norm_least_squares(&matrix, &target) {
        Ok(solution) => measured
            .iter()
            .enumerate()
            .map(|(column, id)| ((*id).clone(), solution[column]))
            .collect(),
        Err(_) => BTreeMap::new(),
    }
}
