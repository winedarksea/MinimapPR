//! Deterministic helpers shared by the solver unit tests (no `rand` dep).

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};

use super::cartesian_tdoa_solve::CartesianTdoaSolveRequest;
use super::measurement_types::{PairTdoaMeasurement, SensorPositions};

pub const SOUND_SPEED_MPS: f64 = 343.0;
pub const SAMPLE_RATE_HZ: u32 = 48_000;
pub const INTERPOLATION_FACTOR: u32 = 4;

/// SplitMix64 + Box–Muller: reproducible Gaussian noise for Monte Carlo tests.
pub struct DeterministicGaussian {
    state: u64,
}

impl DeterministicGaussian {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        // 53 random mantissa bits in (0, 1).
        ((mixed >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    pub fn sample(&mut self, standard_deviation: f64) -> f64 {
        let radius = (-2.0 * self.next_unit().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.next_unit();
        standard_deviation * radius * angle.cos()
    }
}

pub fn positions(entries: &[(&str, [f64; 3])]) -> SensorPositions {
    entries
        .iter()
        .map(|(sensor_id, position)| {
            (
                sensor_id.to_string(),
                Vector3::new(position[0], position[1], position[2]),
            )
        })
        .collect()
}

/// All pairs, `tdoa = (|x−a| − |x−b|)/c + noise`, unit weight.
pub fn synthetic_measurements(
    sensor_positions: &SensorPositions,
    source_m: &Vector3<f64>,
    noise_std_s: f64,
    noise: &mut DeterministicGaussian,
) -> Vec<PairTdoaMeasurement> {
    let entries: Vec<(&String, &Vector3<f64>)> = sensor_positions.iter().collect();
    let mut measurements = Vec::new();
    for (index, (sensor_a, position_a)) in entries.iter().enumerate() {
        for (sensor_b, position_b) in &entries[index + 1..] {
            let ideal = ((source_m - *position_a).norm() - (source_m - *position_b).norm())
                / SOUND_SPEED_MPS;
            measurements.push(PairTdoaMeasurement {
                sensor_a: (*sensor_a).clone(),
                sensor_b: (*sensor_b).clone(),
                tdoa_seconds: ideal + noise.sample(noise_std_s),
                correlation_peak: 0.8,
                weight: 1.0,
            });
        }
    }
    measurements
}

/// Reference-relative delays read straight off the pair list (reference = first id).
pub fn reference_tdoas(
    sensor_positions: &SensorPositions,
    measurements: &[PairTdoaMeasurement],
) -> (String, BTreeMap<String, f64>) {
    let reference = sensor_positions.keys().next().cloned().unwrap_or_default();
    let delays = measurements
        .iter()
        .filter(|measurement| measurement.sensor_a == reference)
        .map(|measurement| (measurement.sensor_b.clone(), -measurement.tdoa_seconds))
        .collect();
    (reference, delays)
}

pub fn request_for(
    sensor_positions: SensorPositions,
    measurements: Vec<PairTdoaMeasurement>,
) -> CartesianTdoaSolveRequest {
    let (reference_sensor, reference_tdoa_s) = reference_tdoas(&sensor_positions, &measurements);
    CartesianTdoaSolveRequest::with_defaults(
        sensor_positions,
        measurements,
        SOUND_SPEED_MPS,
        SAMPLE_RATE_HZ,
        INTERPOLATION_FACTOR,
        reference_sensor,
        reference_tdoa_s,
    )
}

/// Sample covariance of 3D points about their mean.
pub fn sample_covariance(points: &[Vector3<f64>]) -> Matrix3<f64> {
    let mean = points
        .iter()
        .fold(Vector3::zeros(), |sum, point| sum + point)
        / points.len() as f64;
    points.iter().fold(Matrix3::zeros(), |sum, point| {
        sum + (point - mean) * (point - mean).transpose()
    }) / (points.len() as f64 - 1.0)
}
