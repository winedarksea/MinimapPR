//! Perched random-walk track filter (port of
//! `minimappr/core/track_filters_perched.py`, server track S3).
//!
//! A singing songbird sits still between bouts, then may hop. A
//! constant-velocity Kalman reads fix jitter as velocity and coasts the bird
//! away during silence; this filter's state is position only and its variance
//! grows linearly with the gap. A fix beyond the snap distance that is also
//! χ²-inconsistent with the perch is a hop: the filter restarts at the fix
//! instead of smearing the track halfway between two perches.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Matrix3, Vector3};

use super::covariance_math::{
    covariance_from_matrix, eigenvalue_floored, finite_symmetrized, position_from_vector,
    symmetrized, vector_from_position,
};
use super::{Covariance3, Position3, TrackId};

/// Smallest dt (s) used for variance growth (the Kalman filter's guard).
const MINIMUM_DT_SECONDS: f64 = 1e-3;
/// χ² 99 % quantile, 3 dof: a jump past the snap distance is a new perch only
/// when it is also inconsistent with the perch.
pub const DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED: f64 = 11.34;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerchedFilterConfig {
    pub position_diffusion_m2_per_second: f64,
    pub measurement_noise_m2: f64,
    pub initial_position_variance_m2: f64,
    /// <= 0 disables snapping (every fix is fused).
    pub relocation_snap_distance_m: f64,
    pub relocation_snap_min_mahalanobis_squared: f64,
}

impl Default for PerchedFilterConfig {
    fn default() -> Self {
        Self {
            position_diffusion_m2_per_second: 0.05,
            measurement_noise_m2: 1.5,
            initial_position_variance_m2: 4.0,
            relocation_snap_distance_m: 6.0,
            relocation_snap_min_mahalanobis_squared: DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerchedEstimate {
    pub position_m: Position3,
    pub covariance_m2: Covariance3,
}

#[derive(Debug, Clone, PartialEq)]
struct PerchedInternalState {
    position: Vector3<f64>,
    covariance: Matrix3<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerchedRandomWalkFilter {
    config: PerchedFilterConfig,
    states: BTreeMap<TrackId, PerchedInternalState>,
    /// Tracks whose next fix restarts the filter (a relocation-gate hop).
    restart_at_next_fix_track_ids: BTreeSet<TrackId>,
    pub relocation_count: u64,
}

impl PerchedRandomWalkFilter {
    pub fn new(config: PerchedFilterConfig) -> Self {
        Self { config, states: BTreeMap::new(), restart_at_next_fix_track_ids: BTreeSet::new(), relocation_count: 0 }
    }

    pub fn has_track(&self, track_id: TrackId) -> bool {
        self.states.contains_key(&track_id)
    }

    /// Predicted perch after `dt_s` of silence; None when the track has no state.
    pub fn predict(&self, track_id: TrackId, dt_s: f64) -> Option<PerchedEstimate> {
        let internal = self.states.get(&track_id)?;
        Some(estimate(&internal.position, &(internal.covariance + self.diffusion(dt_s))))
    }

    /// Fuse a fix (or restart on a hop). `fallback_position_m` seeds a track
    /// with no state yet, as the reference does with the track's position.
    pub fn update(
        &mut self,
        track_id: TrackId,
        fallback_position_m: &Position3,
        measurement_m: &Position3,
        dt_s: f64,
        measurement_covariance_m2: Option<&Covariance3>,
    ) -> PerchedEstimate {
        if !self.states.contains_key(&track_id) {
            self.initialize_track(track_id, fallback_position_m);
        }
        let gate_restart = self.restart_at_next_fix_track_ids.remove(&track_id);
        let diffusion = self.diffusion(dt_s);
        let measurement_covariance = self.measurement_covariance(measurement_covariance_m2);
        let snap_distance_m = self.config.relocation_snap_distance_m;
        let snap_threshold = self.config.relocation_snap_min_mahalanobis_squared;
        let internal = self.states.get_mut(&track_id).expect("initialized above");
        let measured = vector_from_position(measurement_m);
        let innovation = measured - internal.position;
        let prior_covariance = internal.covariance + diffusion;
        let innovation_covariance = prior_covariance + measurement_covariance;
        if gate_restart || is_relocation(&innovation, &innovation_covariance, snap_distance_m, snap_threshold) {
            // A gate restart was counted by the relocation gate already.
            if !gate_restart {
                self.relocation_count += 1;
            }
            internal.position = measured;
            internal.covariance = measurement_covariance;
            return estimate(&internal.position, &internal.covariance);
        }
        // gain = P · S⁻¹, solved as Sᵀ · gainᵀ = Pᵀ.
        let Some(gain_transposed) = innovation_covariance.transpose().lu().solve(&prior_covariance.transpose()) else {
            // Singular innovation: keep the prior rather than inventing a fix.
            internal.covariance = prior_covariance;
            return estimate(&internal.position, &internal.covariance);
        };
        let gain = gain_transposed.transpose();
        let residual_projection = Matrix3::identity() - gain;
        internal.position += gain * innovation;
        let posterior = residual_projection * prior_covariance * residual_projection.transpose()
            + gain * measurement_covariance * gain.transpose();
        internal.covariance = symmetrized(&posterior);
        estimate(&internal.position, &internal.covariance)
    }

    pub fn initialize_track(&mut self, track_id: TrackId, position_m: &Position3) {
        self.restart_at_next_fix_track_ids.remove(&track_id);
        self.states.insert(
            track_id,
            PerchedInternalState {
                position: vector_from_position(position_m),
                covariance: Matrix3::identity() * self.config.initial_position_variance_m2,
            },
        );
    }

    /// The track's next `update` restarts the filter at that fix (a
    /// relocation-gate hop). Initialising at the fix and then fusing the same
    /// fix would count it twice and halve its covariance.
    pub fn restart_at_next_fix(&mut self, track_id: TrackId) {
        self.restart_at_next_fix_track_ids.insert(track_id);
    }

    pub fn remove_track(&mut self, track_id: TrackId) {
        self.states.remove(&track_id);
        self.restart_at_next_fix_track_ids.remove(&track_id);
    }

    fn diffusion(&self, dt_s: f64) -> Matrix3<f64> {
        Matrix3::identity() * (self.config.position_diffusion_m2_per_second * dt_s.max(MINIMUM_DT_SECONDS))
    }

    /// The fix covariance with eigenvalues floored at the measurement noise;
    /// isotropic measurement noise when absent or unusable.
    fn measurement_covariance(&self, value: Option<&Covariance3>) -> Matrix3<f64> {
        let default = Matrix3::identity() * self.config.measurement_noise_m2;
        finite_symmetrized(value)
            .and_then(|covariance| eigenvalue_floored(&covariance, self.config.measurement_noise_m2))
            .unwrap_or(default)
    }
}

fn is_relocation(
    innovation: &Vector3<f64>,
    innovation_covariance: &Matrix3<f64>,
    snap_distance_m: f64,
    snap_threshold: f64,
) -> bool {
    if !(0.0 < snap_distance_m && snap_distance_m < innovation.norm()) {
        return false;
    }
    match innovation_covariance.lu().solve(innovation) {
        Some(solved) => innovation.dot(&solved) > snap_threshold,
        None => true,
    }
}

fn estimate(position: &Vector3<f64>, covariance: &Matrix3<f64>) -> PerchedEstimate {
    PerchedEstimate { position_m: position_from_vector(position), covariance_m2: covariance_from_matrix(covariance) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_averages_and_a_far_confident_fix_relocates() {
        let mut filter = PerchedRandomWalkFilter::new(PerchedFilterConfig::default());
        filter.initialize_track(1, &[0.0, 0.0, 3.0]);
        let fused = filter.update(1, &[0.0; 3], &[1.0, 0.0, 3.0], 5.0, None);
        assert!(fused.position_m[0] > 0.0 && fused.position_m[0] < 1.0);
        let hopped = filter.update(1, &[0.0; 3], &[15.0, 0.0, 3.0], 5.0, None);
        assert_eq!(hopped.position_m, [15.0, 0.0, 3.0]);
        assert_eq!(filter.relocation_count, 1);
    }

    #[test]
    fn gate_restart_takes_the_fix_once_not_twice() {
        let mut filter = PerchedRandomWalkFilter::new(PerchedFilterConfig::default());
        filter.initialize_track(1, &[0.0; 3]);
        filter.restart_at_next_fix(1);
        let fix_covariance: Covariance3 = [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 211.0]];
        let restarted = filter.update(1, &[0.0; 3], &[3.0, 0.0, 0.0], 8.0, Some(&fix_covariance));
        assert_eq!(restarted.position_m, [3.0, 0.0, 0.0]);
        assert!((restarted.covariance_m2[2][2] - 211.0).abs() < 1e-9);
        assert_eq!(filter.relocation_count, 0);
    }

    #[test]
    fn variance_grows_with_silence() {
        let mut filter = PerchedRandomWalkFilter::new(PerchedFilterConfig::default());
        filter.initialize_track(7, &[0.0; 3]);
        let short = filter.predict(7, 1.0).unwrap().covariance_m2[0][0];
        let long = filter.predict(7, 100.0).unwrap().covariance_m2[0][0];
        assert!(long > short);
        assert!(filter.predict(8, 1.0).is_none());
    }
}
