//! Relative phone heights from barometers, for the layout solve.
//!
//! Phone barometers carry per-unit biases of about ±1 hPa (≈ ±8 m), so a
//! phone's pressure offset is locked only when it is known to stand next
//! to another phone: at a QR join (the joiner stands next to the host), or
//! when chirp ranging puts two phones under 2 m apart. GPS altitude (often
//! 15–30 m off, mixed ellipsoid/geoid heights) is a wide-σ fallback for a
//! phone with no barometer.

/// Dry-air gas constant, J/(kg·K).
pub const DRY_AIR_GAS_CONSTANT_J_PER_KG_K: f64 = 287.05;
pub const STANDARD_GRAVITY_MPS2: f64 = 9.80665;
pub const CELSIUS_TO_KELVIN_OFFSET: f64 = 273.15;
/// Chirp range under which two phones count as side by side for a lock.
pub const COLOCATION_LOCK_MAXIMUM_DISTANCE_M: f64 = 2.0;

#[derive(Clone, Debug, PartialEq)]
pub struct BarometricHeightConfig {
    /// σ of a locked barometric relative height (hand-height differences, drift).
    pub locked_barometer_sigma_m: f64,
    pub gps_altitude_sigma_m: f64,
    pub unknown_height_sigma_m: f64,
}

impl Default for BarometricHeightConfig {
    fn default() -> Self {
        Self {
            locked_barometer_sigma_m: 0.5,
            gps_altitude_sigma_m: 25.0,
            unknown_height_sigma_m: 30.0,
        }
    }
}

/// Height of the point at `pressure_hpa` above the point at
/// `reference_pressure_hpa` (hypsometric equation, isothermal layer).
pub fn hypsometric_height_difference_m(
    pressure_hpa: f64,
    reference_pressure_hpa: f64,
    air_temperature_c: f64,
) -> f64 {
    let temperature_k = air_temperature_c + CELSIUS_TO_KELVIN_OFFSET;
    DRY_AIR_GAS_CONSTANT_J_PER_KG_K * temperature_k / STANDARD_GRAVITY_MPS2
        * (reference_pressure_hpa / pressure_hpa).ln()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightSource {
    Reference,
    LockedBarometer,
    GpsAltitude,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightEstimate {
    /// Height relative to the reference node, m.
    pub height_m: f64,
    pub sigma_m: f64,
    pub source: HeightSource,
}

/// Latest sensor values for one node.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeightSensorReading {
    pub pressure_hpa: Option<f64>,
    pub gps_altitude_m: Option<f64>,
}

/// Per-island pressure-bias bookkeeping relative to a reference node.
#[derive(Clone, Debug)]
pub struct BarometricHeightEstimator {
    pub config: BarometricHeightConfig,
    reference_node: usize,
    /// Pressure bias (hPa) of each node relative to the reference node's sensor.
    pressure_bias_hpa: Vec<Option<f64>>,
}

impl BarometricHeightEstimator {
    pub fn new(node_count: usize, reference_node: usize, config: BarometricHeightConfig) -> Self {
        let mut pressure_bias_hpa = vec![None; node_count];
        if reference_node < node_count {
            pressure_bias_hpa[reference_node] = Some(0.0);
        }
        Self {
            config,
            reference_node,
            pressure_bias_hpa,
        }
    }

    pub fn is_locked(&self, node: usize) -> bool {
        self.pressure_bias_hpa
            .get(node)
            .copied()
            .flatten()
            .is_some()
    }

    /// `node` stands next to `locked_neighbour` (same height): lock its bias.
    /// Returns false if the neighbour is not locked yet.
    pub fn lock_colocated(
        &mut self,
        node: usize,
        locked_neighbour: usize,
        node_pressure_hpa: f64,
        neighbour_pressure_hpa: f64,
    ) -> bool {
        let Some(neighbour_bias) = self
            .pressure_bias_hpa
            .get(locked_neighbour)
            .copied()
            .flatten()
        else {
            return false;
        };
        if node >= self.pressure_bias_hpa.len() || node == self.reference_node {
            return false;
        }
        self.pressure_bias_hpa[node] =
            Some(neighbour_bias + node_pressure_hpa - neighbour_pressure_hpa);
        true
    }

    /// Lock from a chirp range when the pair is close enough to count as side by side.
    pub fn lock_from_range(
        &mut self,
        node: usize,
        locked_neighbour: usize,
        range_m: f64,
        node_pressure_hpa: f64,
        neighbour_pressure_hpa: f64,
    ) -> bool {
        range_m < COLOCATION_LOCK_MAXIMUM_DISTANCE_M
            && !self.is_locked(node)
            && self.lock_colocated(
                node,
                locked_neighbour,
                node_pressure_hpa,
                neighbour_pressure_hpa,
            )
    }

    /// Heights relative to the reference node.
    pub fn heights(
        &self,
        readings: &[HeightSensorReading],
        air_temperature_c: f64,
    ) -> Vec<HeightEstimate> {
        let reference = readings
            .get(self.reference_node)
            .copied()
            .unwrap_or_default();
        readings
            .iter()
            .enumerate()
            .map(|(node, reading)| {
                if node == self.reference_node {
                    return HeightEstimate {
                        height_m: 0.0,
                        sigma_m: 0.0,
                        source: HeightSource::Reference,
                    };
                }
                let bias = self.pressure_bias_hpa.get(node).copied().flatten();
                if let (Some(bias), Some(pressure), Some(reference_pressure)) =
                    (bias, reading.pressure_hpa, reference.pressure_hpa)
                {
                    return HeightEstimate {
                        height_m: hypsometric_height_difference_m(
                            pressure - bias,
                            reference_pressure,
                            air_temperature_c,
                        ),
                        sigma_m: self.config.locked_barometer_sigma_m,
                        source: HeightSource::LockedBarometer,
                    };
                }
                if let (Some(altitude), Some(reference_altitude)) =
                    (reading.gps_altitude_m, reference.gps_altitude_m)
                {
                    return HeightEstimate {
                        height_m: altitude - reference_altitude,
                        sigma_m: self.config.gps_altitude_sigma_m,
                        source: HeightSource::GpsAltitude,
                    };
                }
                HeightEstimate {
                    height_m: 0.0,
                    sigma_m: self.config.unknown_height_sigma_m,
                    source: HeightSource::Unknown,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEA_LEVEL_HPA: f64 = 1013.25;

    #[test]
    fn hypsometric_gives_about_8_m_per_hpa_near_sea_level() {
        let height = hypsometric_height_difference_m(SEA_LEVEL_HPA - 1.0, SEA_LEVEL_HPA, 15.0);
        assert!((height - 8.2).abs() < 0.3, "{height}");
    }

    #[test]
    fn locked_bias_removes_sensor_offset() {
        let mut estimator = BarometricHeightEstimator::new(3, 0, BarometricHeightConfig::default());
        // Node 1 reads 0.9 hPa high; at the QR join it stands next to node 0.
        assert!(estimator.lock_colocated(1, 0, SEA_LEVEL_HPA + 0.9, SEA_LEVEL_HPA));
        // Later node 1 climbs 5 m: true pressure drops by ~0.6 hPa.
        let climb_drop_hpa = SEA_LEVEL_HPA
            - SEA_LEVEL_HPA
                * (-5.0 * STANDARD_GRAVITY_MPS2 / (DRY_AIR_GAS_CONSTANT_J_PER_KG_K * 288.15)).exp();
        let readings = [
            HeightSensorReading {
                pressure_hpa: Some(SEA_LEVEL_HPA),
                gps_altitude_m: Some(100.0),
            },
            HeightSensorReading {
                pressure_hpa: Some(SEA_LEVEL_HPA + 0.9 - climb_drop_hpa),
                gps_altitude_m: Some(130.0),
            },
            HeightSensorReading {
                pressure_hpa: Some(SEA_LEVEL_HPA - 0.4),
                gps_altitude_m: Some(110.0),
            },
        ];
        let heights = estimator.heights(&readings, 15.0);
        assert_eq!(heights[0].source, HeightSource::Reference);
        assert_eq!(heights[1].source, HeightSource::LockedBarometer);
        assert!((heights[1].height_m - 5.0).abs() < 0.05, "{:?}", heights[1]);
        // Node 2 is unlocked: GPS altitude with a wide σ.
        assert_eq!(heights[2].source, HeightSource::GpsAltitude);
        assert_eq!(heights[2].height_m, 10.0);
        assert!(heights[2].sigma_m > 10.0);
    }

    #[test]
    fn range_lock_needs_close_pair_and_locked_neighbour() {
        let mut estimator = BarometricHeightEstimator::new(3, 0, BarometricHeightConfig::default());
        assert!(
            !estimator.lock_from_range(2, 1, 1.0, 1000.0, 1000.0),
            "neighbour not locked"
        );
        assert!(
            !estimator.lock_from_range(1, 0, 5.0, 1000.0, 1000.0),
            "too far apart"
        );
        assert!(estimator.lock_from_range(1, 0, 1.2, 1000.3, 1000.0));
        assert!(estimator.lock_from_range(2, 1, 0.8, 999.9, 1000.3));
        assert!(estimator.is_locked(2));
        let heights = estimator.heights(&[HeightSensorReading::default(); 3], 15.0);
        assert_eq!(
            heights[1].source,
            HeightSource::Unknown,
            "no current pressure reading"
        );
    }
}
