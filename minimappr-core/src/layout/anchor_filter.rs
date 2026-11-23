//! Slow filter over the three map-anchor parameters (east, north, rotation).
//!
//! Multipath does not average out while the group stands still: trees and
//! satellites barely move in 30 s. So while every phone is still (step
//! detector and GNSS Doppler speed agree), process noise drops to near zero
//! and a sudden GPS jump cannot drag the map; fixes taken in clearings
//! tighten the anchor and the group coasts on them through dense woods.
//! When the group walks, process noise opens up so the anchor follows.
//!
//! Contract: every fit must use the *same* local layout frame. A fresh
//! `solve_layout` comes out of MDS in an arbitrary rotation and handedness,
//! so align each new layout onto the previous one first (e.g.
//! `weighted_kabsch_2d` against the last layout, mirror allowed); otherwise
//! frame changes look like anchor rotations and chirality flips, and a
//! still group's near-zero process noise would refuse to follow them.

use std::f64::consts::{PI, TAU};

use super::gps_anchor_fit::{AnchorFit, RigidTransform2d};

#[derive(Clone, Debug, PartialEq)]
pub struct AnchorFilterConfig {
    /// Random-walk density of the translation while the group moves, m/√s.
    pub moving_translation_walk_m_per_sqrt_s: f64,
    pub moving_rotation_walk_rad_per_sqrt_s: f64,
    /// Fraction of the moving process noise kept while everyone is still.
    pub stillness_process_noise_fraction: f64,
    /// Innovations beyond this many σ are softly down-weighted.
    pub innovation_gate_sigma: f64,
    /// A fit must be this sure of the opposite chirality to reset the filter.
    pub chirality_flip_confidence: f64,
}

impl Default for AnchorFilterConfig {
    fn default() -> Self {
        Self {
            moving_translation_walk_m_per_sqrt_s: 0.8,
            moving_rotation_walk_rad_per_sqrt_s: 0.02,
            stillness_process_noise_fraction: 1e-4,
            innovation_gate_sigma: 3.0,
            chirality_flip_confidence: 0.95,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarKalman {
    value: f64,
    variance: f64,
}

impl ScalarKalman {
    fn predict(&mut self, process_variance: f64) {
        self.variance += process_variance;
    }

    /// Update with innovation already computed (lets rotation wrap).
    fn update(&mut self, innovation: f64, measurement_variance: f64, gate_sigma: f64) {
        let innovation_sigma = (self.variance + measurement_variance).sqrt();
        let normalized = innovation.abs() / innovation_sigma.max(f64::MIN_POSITIVE);
        // Soft gate: inflate R so a jump moves the estimate by at most ~gate σ.
        let inflated = if normalized > gate_sigma {
            measurement_variance * (normalized / gate_sigma).powi(2)
        } else {
            measurement_variance
        };
        let gain = self.variance / (self.variance + inflated);
        self.value += gain * innovation;
        self.variance *= 1.0 - gain;
    }
}

fn wrap_angle(angle: f64) -> f64 {
    let wrapped = (angle + PI).rem_euclid(TAU) - PI;
    if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorEstimate {
    pub transform: RigidTransform2d,
    pub translation_sigma_m: [f64; 2],
    pub rotation_sigma_rad: f64,
}

#[derive(Clone, Debug)]
pub struct MapAnchorFilter {
    pub config: AnchorFilterConfig,
    state: Option<(ScalarKalman, ScalarKalman, ScalarKalman, bool, f64)>,
}

impl MapAnchorFilter {
    pub fn new(config: AnchorFilterConfig) -> Self {
        Self {
            config,
            state: None,
        }
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Fuse one anchor fit taken at `time_s`. `group_is_still` = every
    /// phone's step detector and GNSS speed say still since the last fit.
    pub fn update(&mut self, time_s: f64, fit: &AnchorFit, group_is_still: bool) {
        let mirrored = fit.transform.mirrored;
        let translation_variance = fit.translation_sigma_m.powi(2);
        let rotation_variance = fit.rotation_sigma_rad.powi(2);
        let fresh = |fit: &AnchorFit| {
            (
                ScalarKalman {
                    value: fit.transform.translation_m[0],
                    variance: translation_variance,
                },
                ScalarKalman {
                    value: fit.transform.translation_m[1],
                    variance: translation_variance,
                },
                ScalarKalman {
                    value: fit.transform.rotation_rad,
                    variance: rotation_variance,
                },
                mirrored,
                time_s,
            )
        };
        let Some((east, north, rotation, state_mirrored, last_time)) = self.state.as_mut() else {
            self.state = Some(fresh(fit));
            return;
        };
        if *state_mirrored != mirrored {
            if fit.chirality_confidence >= self.config.chirality_flip_confidence {
                self.state = Some(fresh(fit));
            }
            return;
        }
        let elapsed = (time_s - *last_time).max(0.0);
        let fraction = if group_is_still {
            self.config.stillness_process_noise_fraction
        } else {
            1.0
        };
        let translation_process =
            fraction * self.config.moving_translation_walk_m_per_sqrt_s.powi(2) * elapsed;
        let rotation_process =
            fraction * self.config.moving_rotation_walk_rad_per_sqrt_s.powi(2) * elapsed;
        east.predict(translation_process);
        north.predict(translation_process);
        rotation.predict(rotation_process);
        let gate = self.config.innovation_gate_sigma;
        east.update(
            fit.transform.translation_m[0] - east.value,
            translation_variance,
            gate,
        );
        north.update(
            fit.transform.translation_m[1] - north.value,
            translation_variance,
            gate,
        );
        rotation.update(
            wrap_angle(fit.transform.rotation_rad - rotation.value),
            rotation_variance,
            gate,
        );
        rotation.value = wrap_angle(rotation.value);
        *last_time = time_s;
    }

    /// Carry the state into a fresh layout frame (`previous_to_new` maps
    /// old-frame points to new-frame points): T_new = T_old ∘ F⁻¹. The
    /// variances are kept: MDS frames are centred on the layout, so the
    /// origin shift that would couple rotation into translation is small.
    pub fn reframe(&mut self, previous_to_new: &RigidTransform2d) {
        let Some(estimate) = self.estimate() else { return };
        let moved = estimate.transform.compose(&previous_to_new.inverse());
        if let Some((east, north, rotation, mirrored, _)) = self.state.as_mut() {
            east.value = moved.translation_m[0];
            north.value = moved.translation_m[1];
            rotation.value = wrap_angle(moved.rotation_rad);
            *mirrored = moved.mirrored;
        }
    }

    pub fn estimate(&self) -> Option<AnchorEstimate> {
        let (east, north, rotation, mirrored, _) = self.state.as_ref()?;
        Some(AnchorEstimate {
            transform: RigidTransform2d {
                rotation_rad: rotation.value,
                translation_m: [east.value, north.value],
                mirrored: *mirrored,
            },
            translation_sigma_m: [east.variance.sqrt(), north.variance.sqrt()],
            rotation_sigma_rad: rotation.variance.sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(east: f64, north: f64, rotation: f64, mirrored: bool, confidence: f64) -> AnchorFit {
        AnchorFit {
            transform: RigidTransform2d {
                rotation_rad: rotation,
                translation_m: [east, north],
                mirrored,
            },
            chirality_confidence: confidence,
            huber_loss: 0.0,
            alternative_huber_loss: 0.0,
            normalized_residuals: vec![],
            outliers: vec![],
            rotation_sigma_rad: 0.05,
            translation_sigma_m: 2.0,
        }
    }

    #[test]
    fn still_group_ignores_a_sudden_gps_jump() {
        let mut filter = MapAnchorFilter::new(AnchorFilterConfig::default());
        for step in 0..10 {
            filter.update(step as f64 * 5.0, &fit(100.0, 50.0, 0.3, false, 0.99), true);
        }
        filter.update(55.0, &fit(112.0, 41.0, 0.3, false, 0.99), true);
        let estimate = filter.estimate().unwrap();
        assert!(
            (estimate.transform.translation_m[0] - 100.0).abs() < 1.0,
            "{estimate:?}"
        );
    }

    #[test]
    fn moving_group_follows_the_anchor() {
        let mut filter = MapAnchorFilter::new(AnchorFilterConfig::default());
        for step in 0..10 {
            filter.update(step as f64 * 5.0, &fit(100.0, 50.0, 0.3, false, 0.99), true);
        }
        for step in 10..20 {
            filter.update(
                step as f64 * 5.0,
                &fit(100.0 + 6.0 * (step - 9) as f64, 50.0, 0.3, false, 0.99),
                false,
            );
        }
        let estimate = filter.estimate().unwrap();
        assert!(
            (estimate.transform.translation_m[0] - 160.0).abs() < 8.0,
            "{estimate:?}"
        );
    }

    #[test]
    fn rotation_wraps_across_pi() {
        let mut filter = MapAnchorFilter::new(AnchorFilterConfig::default());
        filter.update(0.0, &fit(0.0, 0.0, PI - 0.02, false, 0.99), false);
        filter.update(5.0, &fit(0.0, 0.0, -PI + 0.02, false, 0.99), false);
        let rotation = filter.estimate().unwrap().transform.rotation_rad;
        assert!(rotation.abs() > PI - 0.05, "{rotation}");
    }

    #[test]
    fn chirality_flip_needs_high_confidence() {
        let mut filter = MapAnchorFilter::new(AnchorFilterConfig::default());
        filter.update(0.0, &fit(10.0, 0.0, 0.0, false, 0.7), false);
        filter.update(5.0, &fit(30.0, 0.0, 1.0, true, 0.6), false);
        assert!(!filter.estimate().unwrap().transform.mirrored);
        filter.update(10.0, &fit(30.0, 0.0, 1.0, true, 0.99), false);
        assert!(filter.estimate().unwrap().transform.mirrored);
    }
}
