//! Nearest-neighbour association (port of the server's
//! `NearestNeighborAssociator`, restricted to the wildlife path: one label
//! category, no classifier fingerprint term).
//!
//! A fix within the plain association distance is always eligible; its score
//! is expressed in the same χ² units as the covariance-gated path (calibrated
//! so `distance == association_distance_m` sits on the χ² gate), so a close
//! plain match and a farther covariance-gated match compare fairly.

use nalgebra::Matrix3;

use super::covariance_math::{
    eigenvalue_floored, finite_symmetrized, largest_eigenvalue, mahalanobis_squared, vector_from_position,
};
use super::{Covariance3, Position3, TrackId};

/// Eigenvalue floor (m²) keeping combined covariances positive definite.
const POSITIVE_DEFINITE_EIGENVALUE_FLOOR_M2: f64 = 1.0e-6;
/// Physical gate radius in standard deviations of the combined covariance.
const PHYSICAL_GATE_SIGMA_MULTIPLIER: f64 = 3.0;
/// Default max gate when none is configured: this × the association distance.
const DEFAULT_MAX_GATE_DISTANCE_MULTIPLIER: f64 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestNeighborAssociationConfig {
    pub association_distance_m: f64,
    /// <= 0 means `DEFAULT_MAX_GATE_DISTANCE_MULTIPLIER × association_distance_m`.
    pub max_gate_m: f64,
    pub chi2_gate: f64,
}

impl Default for NearestNeighborAssociationConfig {
    fn default() -> Self {
        Self { association_distance_m: 8.0, max_gate_m: 32.0, chi2_gate: 9.0 }
    }
}

impl NearestNeighborAssociationConfig {
    fn effective_max_gate_m(&self) -> f64 {
        let configured = if self.max_gate_m > 0.0 {
            self.max_gate_m
        } else {
            self.association_distance_m * DEFAULT_MAX_GATE_DISTANCE_MULTIPLIER
        };
        configured.max(self.association_distance_m)
    }
}

/// A predicted track as the associator sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssociationCandidate {
    pub track_id: TrackId,
    pub position_m: Position3,
    pub covariance_m2: Option<Covariance3>,
}

fn positive_definite(covariance_m2: Option<&Covariance3>) -> Option<Matrix3<f64>> {
    finite_symmetrized(covariance_m2).and_then(|matrix| eigenvalue_floored(&matrix, POSITIVE_DEFINITE_EIGENVALUE_FLOOR_M2))
}

/// The track this fix belongs to, or None (start a new track).
pub fn associate_nearest_neighbor(
    position_m: &Position3,
    measurement_covariance_m2: Option<&Covariance3>,
    candidates: &[AssociationCandidate],
    config: &NearestNeighborAssociationConfig,
) -> Option<TrackId> {
    let chi2_gate = if config.chi2_gate > 0.0 { config.chi2_gate } else { 9.0 };
    let fallback_variance = config.association_distance_m.powi(2) / chi2_gate;
    let max_gate_m = config.effective_max_gate_m();
    let measurement = vector_from_position(position_m);
    let measurement_covariance = positive_definite(measurement_covariance_m2);
    let mut best: Option<(TrackId, f64)> = None;
    for candidate in candidates {
        let residual = measurement - vector_from_position(&candidate.position_m);
        let distance_m = residual.norm();
        let score = if distance_m < config.association_distance_m {
            distance_m * distance_m / fallback_variance
        } else {
            let track_covariance = positive_definite(candidate.covariance_m2.as_ref());
            let combined = match (measurement_covariance, track_covariance) {
                (Some(measurement), Some(track)) => measurement + track,
                (Some(only), None) | (None, Some(only)) => only,
                (None, None) => continue,
            };
            let Some(covariance) = eigenvalue_floored(&combined, POSITIVE_DEFINITE_EIGENVALUE_FLOOR_M2) else { continue };
            let largest_variance = largest_eigenvalue(&covariance);
            if !largest_variance.is_finite() || largest_variance <= 0.0 {
                continue;
            }
            let physical_gate_radius_m = (PHYSICAL_GATE_SIGMA_MULTIPLIER * largest_variance.sqrt())
                .clamp(config.association_distance_m, max_gate_m);
            if distance_m > physical_gate_radius_m {
                continue;
            }
            match mahalanobis_squared(&residual, &covariance) {
                Some(score) if score <= chi2_gate => score,
                _ => continue,
            }
        };
        if best.is_none_or(|(_, best_score)| score < best_score) {
            best = Some((candidate.track_id, score));
        }
    }
    best.map(|(track_id, _)| track_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(track_id: TrackId, x: f64, variance: f64) -> AssociationCandidate {
        let mut covariance = [[0.0; 3]; 3];
        (0..3).for_each(|axis| covariance[axis][axis] = variance);
        AssociationCandidate { track_id, position_m: [x, 0.0, 0.0], covariance_m2: Some(covariance) }
    }

    #[test]
    fn nearest_inside_the_plain_gate_wins_and_far_tight_tracks_are_refused() {
        let config = NearestNeighborAssociationConfig::default();
        let candidates = [candidate(1, 5.0, 1.0), candidate(2, 2.0, 1.0)];
        assert_eq!(associate_nearest_neighbor(&[0.0; 3], None, &candidates, &config), Some(2));
        assert_eq!(associate_nearest_neighbor(&[0.0; 3], None, &[candidate(3, 12.0, 1.0)], &config), None);
        // A loose track admits the same 12 m fix through the covariance gate.
        assert_eq!(associate_nearest_neighbor(&[0.0; 3], None, &[candidate(4, 12.0, 30.0)], &config), Some(4));
    }
}
