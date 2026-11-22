//! Measurement and constraint types shared by the Cartesian solver, mirroring
//! the Python dataclasses (`PairTdoaMeasurement`, `BearingPrior`,
//! `SpatialBearingConstraint`, `AmplitudeRatioConstraint`).

use std::collections::BTreeMap;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Sensor id → position. A `BTreeMap` so iteration is sorted by id, matching
/// the Python solver's `sorted(sensor_positions)` (centroid summation order).
pub type SensorPositions = BTreeMap<String, Vector3<f64>>;

/// Lower clip of the pair quality score, as in Python `normalized_pair_quality`.
pub const MINIMUM_NORMALIZED_PAIR_QUALITY: f64 = 0.05;
/// Upper clip of the pair quality score.
pub const MAXIMUM_NORMALIZED_PAIR_QUALITY: f64 = 1.0;

/// One GCC-PHAT pair: `tdoa_seconds` = arrival(A) − arrival(B).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairTdoaMeasurement {
    pub sensor_a: String,
    pub sensor_b: String,
    pub tdoa_seconds: f64,
    pub correlation_peak: f64,
    /// Residual weight; the residual row is multiplied by `sqrt(weight)`.
    pub weight: f64,
}

/// Correlation-peak quality in `[0.05, 1]`; non-finite peaks score zero.
pub fn normalized_pair_quality(correlation_peak: f64) -> f64 {
    if !correlation_peak.is_finite() {
        return 0.0;
    }
    correlation_peak.abs().clamp(
        MINIMUM_NORMALIZED_PAIR_QUALITY,
        MAXIMUM_NORMALIZED_PAIR_QUALITY,
    )
}

/// Array-level bearing prior (subspace / beamformer direction) about the centroid.
#[derive(Debug, Clone, PartialEq)]
pub struct BearingPrior {
    pub direction: Vector3<f64>,
    pub strength: f64,
    pub quality: f64,
}

/// A node-local bearing ray from `origin_m` toward the source.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialBearingConstraint {
    pub node_id: String,
    pub origin_m: Vector3<f64>,
    pub direction: Vector3<f64>,
    pub strength: f64,
    pub quality: f64,
}

/// Cross-node log-amplitude ratio `ln(A_a / A_b)` under 1/r spreading.
#[derive(Debug, Clone, PartialEq)]
pub struct AmplitudeRatioConstraint {
    pub node_a: String,
    pub node_b: String,
    pub origin_a_m: Vector3<f64>,
    pub origin_b_m: Vector3<f64>,
    pub measured_log_amplitude_ratio: f64,
    pub strength: f64,
}
