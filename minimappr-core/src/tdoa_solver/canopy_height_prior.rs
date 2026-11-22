//! Soft prior on source height above local ground (`z ∈ [0, 35] m` by default).
//!
//! Why: a near-flat ground array sees height poorly and has an exact
//! above/below mirror. A 2D solve squeezes canopy birds toward the centroid;
//! a free 3D solve can land underground. A one-sided hinge penalty inside the
//! band costs nothing (so it never biases a valid canopy fix) and grows
//! linearly outside it, which removes the below-ground mirror. Solvers also
//! restart from the z-mirror of their best candidate so the hinge gets the
//! chance to choose between the two basins.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Default band top for birds in canopy (plan §3D with a canopy-height prior).
pub const DEFAULT_MAXIMUM_SOURCE_HEIGHT_ABOVE_GROUND_M: f64 = 35.0;
/// Distance outside the band that costs as much as one residual unit.
pub const DEFAULT_BOUNDARY_SOFTNESS_M: f64 = 0.5;
/// Fractions of the band height used for lifted multi-start probes.
pub const BAND_PROBE_HEIGHT_FRACTIONS: [f64; 2] = [0.25, 0.75];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanopyHeightPrior {
    /// Ground elevation (solver-frame z) under the array.
    pub ground_elevation_m: f64,
    pub maximum_height_above_ground_m: f64,
    /// Metres of violation that cost one residual unit (see [`Self::residual`]).
    pub boundary_softness_m: f64,
}

impl Default for CanopyHeightPrior {
    fn default() -> Self {
        Self {
            ground_elevation_m: 0.0,
            maximum_height_above_ground_m: DEFAULT_MAXIMUM_SOURCE_HEIGHT_ABOVE_GROUND_M,
            boundary_softness_m: DEFAULT_BOUNDARY_SOFTNESS_M,
        }
    }
}

impl CanopyHeightPrior {
    fn band_top_m(&self) -> f64 {
        self.ground_elevation_m + self.maximum_height_above_ground_m
    }

    /// Signed metres outside the band: negative below ground, positive above
    /// the canopy top, zero inside.
    pub fn band_violation_m(&self, position_m: &Vector3<f64>) -> f64 {
        let height_m = position_m.z;
        if height_m < self.ground_elevation_m {
            height_m - self.ground_elevation_m
        } else if height_m > self.band_top_m() {
            height_m - self.band_top_m()
        } else {
            0.0
        }
    }

    /// One residual row, `violation / softness · residual_unit` (residual_unit
    /// is seconds in the Cartesian solver, 1.0 in the whitened joint solver).
    pub fn residual(&self, position_m: &Vector3<f64>, residual_unit: f64) -> f64 {
        self.band_violation_m(position_m) / self.boundary_softness_m * residual_unit
    }

    /// Jacobian of [`Self::residual`] (only the z component is non-zero).
    pub fn jacobian(&self, position_m: &Vector3<f64>, residual_unit: f64) -> Vector3<f64> {
        let slope = if self.band_violation_m(position_m) == 0.0 {
            0.0
        } else {
            residual_unit / self.boundary_softness_m
        };
        Vector3::new(0.0, 0.0, slope)
    }

    /// Reflect `position_m` through the horizontal plane at `mirror_plane_z_m`
    /// (normally the sensor centroid height, where the TDOA mirror lives).
    pub fn mirrored_start(position_m: &Vector3<f64>, mirror_plane_z_m: f64) -> Vector3<f64> {
        Vector3::new(
            position_m.x,
            position_m.y,
            2.0 * mirror_plane_z_m - position_m.z,
        )
    }

    /// Extra multi-start points around a best candidate: its z-mirror, plus
    /// the same (x, y) lifted into the band. The lifted points matter on a
    /// coplanar layout, where ∂r/∂z = 0 in the layout plane and a solve that
    /// starts there can never climb to a canopy source.
    pub fn band_probe_starts(
        &self,
        position_m: &Vector3<f64>,
        mirror_plane_z_m: f64,
    ) -> Vec<Vector3<f64>> {
        let mut starts = vec![Self::mirrored_start(position_m, mirror_plane_z_m)];
        for fraction in BAND_PROBE_HEIGHT_FRACTIONS {
            starts.push(Vector3::new(
                position_m.x,
                position_m.y,
                self.ground_elevation_m + fraction * self.maximum_height_above_ground_m,
            ));
        }
        starts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hinge_is_free_inside_band_and_linear_outside() {
        let prior = CanopyHeightPrior::default();
        assert_eq!(prior.residual(&Vector3::new(0.0, 0.0, 12.0), 1.0), 0.0);
        assert!((prior.residual(&Vector3::new(0.0, 0.0, -1.0), 1.0) + 2.0).abs() < 1e-12);
        assert!((prior.residual(&Vector3::new(0.0, 0.0, 36.0), 1.0) - 2.0).abs() < 1e-12);
        assert_eq!(prior.jacobian(&Vector3::new(0.0, 0.0, 3.0), 1.0).z, 0.0);
    }
}
