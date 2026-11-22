//! Covariance-aware entity absorption (port of
//! `minimappr/core/bird_entity_absorption_geometry.py`, S-fixes tracking).
//!
//! A fix absorbs into a pinpointed entity when its horizontal Mahalanobis
//! distance (fix + entity covariance, 2 dof) is within a χ² gate, with a small
//! distance floor so overconfident covariances do not split one bird on
//! jitter (a flock 10 m apart stays separate; a far bird's fix 300 m out
//! along the right bearing joins). A fix too uncertain to pinpoint anything
//! is *bearing evidence*: it may join a better-located entity but not move
//! it. Two fixes inconsistent beyond their covariances inside a short window
//! are two birds, whatever the tracker did with them.

use nalgebra::{Matrix2, Vector2};

use super::config::BirdEntityConfig;
use super::model::{horizontal_major_sigma_m, BirdEntityRecord, EntityFixSample};
use crate::range_projection::RANGE_BEARING_PROJECTED;
use crate::wildlife_tracking::covariance_math::{horizontal_block, horizontal_distance_m};
use crate::wildlife_tracking::{seconds_to_nanoseconds, Covariance3, Position3};

/// Variance floor (m²) on the summed horizontal covariance (keeps the solve regular).
const MINIMUM_HORIZONTAL_VARIANCE_M2: f64 = 1.0e-6;
/// Recent fixes are kept for this many concurrency windows (late arrivals can be out of order).
const FIX_MEMORY_WINDOWS: f64 = 2.0;

/// Horizontal Mahalanobis distance squared; None when neither side has a covariance.
pub fn horizontal_mahalanobis_squared(
    first_m: &Position3,
    first_covariance_m2: Option<&Covariance3>,
    second_m: &Position3,
    second_covariance_m2: Option<&Covariance3>,
) -> Option<f64> {
    let covariances: Vec<Matrix2<f64>> =
        [horizontal_block(first_covariance_m2), horizontal_block(second_covariance_m2)].into_iter().flatten().collect();
    if covariances.is_empty() {
        return None;
    }
    let combined = covariances.iter().fold(Matrix2::zeros(), |sum, covariance| sum + covariance)
        + Matrix2::identity() * MINIMUM_HORIZONTAL_VARIANCE_M2;
    let residual = Vector2::new(first_m[0] - second_m[0], first_m[1] - second_m[1]);
    let solved = combined.lu().solve(&residual)?;
    let value = residual.dot(&solved);
    // An indefinite covariance (bad upstream Jacobian) can give a negative
    // "distance" that would pass every gate and win absorption: unusable, as in the reference.
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// A fix that could not pinpoint a bird on its own (range unobserved or too uncertain).
pub fn fix_is_bearing_evidence(covariance_m2: Option<&Covariance3>, range_projection_mode: Option<&str>, config: &BirdEntityConfig) -> bool {
    if range_projection_mode == Some(RANGE_BEARING_PROJECTED) {
        return true;
    }
    horizontal_major_sigma_m(covariance_m2).is_some_and(|sigma_m| sigma_m > config.pinpointed_max_horizontal_uncertainty_m)
}

/// Bearing evidence joining an entity located at least as well: absorb, don't move it.
pub fn should_absorb_as_bearing_evidence(
    entity: &BirdEntityRecord,
    covariance_m2: Option<&Covariance3>,
    range_projection_mode: Option<&str>,
    config: &BirdEntityConfig,
) -> bool {
    if !config.bearing_evidence_absorption_enabled || !entity.is_spatial() {
        return false;
    }
    if !fix_is_bearing_evidence(covariance_m2, range_projection_mode, config) {
        return false;
    }
    match (horizontal_major_sigma_m(covariance_m2), entity.horizontal_uncertainty_m) {
        (Some(fix_sigma_m), Some(entity_sigma_m)) => entity_sigma_m <= fix_sigma_m,
        _ => true,
    }
}

/// Normalised mismatch in [0, 1] for a fix against a pinpointed entity, None = another bird.
/// Falls back to the fixed W3 radius when the fix carries no covariance.
pub fn covariance_gated_fix_score(
    entity_position_m: &Position3,
    entity_covariance_m2: Option<&Covariance3>,
    detection_position_m: &Position3,
    detection_covariance_m2: Option<&Covariance3>,
    config: &BirdEntityConfig,
) -> Option<f64> {
    let distance_m = horizontal_distance_m(detection_position_m, entity_position_m);
    if detection_covariance_m2.is_none() {
        return (distance_m <= config.absorption_max_fix_distance_m).then(|| distance_m / config.absorption_max_fix_distance_m);
    }
    if distance_m <= config.absorption_min_fix_distance_m {
        return Some(0.0);
    }
    let mahalanobis_squared =
        horizontal_mahalanobis_squared(detection_position_m, detection_covariance_m2, entity_position_m, entity_covariance_m2)?;
    (mahalanobis_squared <= config.absorption_chi2_gate).then(|| mahalanobis_squared / config.absorption_chi2_gate)
}

/// Concurrent (within the window) and inconsistent beyond their covariances.
pub fn fixes_are_two_birds(first: &EntityFixSample, second: &EntityFixSample, config: &BirdEntityConfig) -> bool {
    let window_ns = seconds_to_nanoseconds(config.concurrent_fix_window_seconds);
    if (first.event_time_ns - second.event_time_ns).abs() >= window_ns {
        return false;
    }
    let distance_m = horizontal_distance_m(&first.position_m, &second.position_m);
    if distance_m <= config.absorption_min_fix_distance_m {
        return false;
    }
    match horizontal_mahalanobis_squared(
        &first.position_m,
        first.covariance_m2.as_ref(),
        &second.position_m,
        second.covariance_m2.as_ref(),
    ) {
        Some(value) => value > config.absorption_chi2_gate,
        None => distance_m > config.absorption_max_fix_distance_m,
    }
}

/// Does `candidate` contradict a fix this entity took moments ago?
pub fn concurrent_fix_conflict(entity: &BirdEntityRecord, candidate: Option<&EntityFixSample>, config: &BirdEntityConfig) -> bool {
    let Some(candidate) = candidate else { return false };
    config.absorption_covariance_gate_enabled && entity.recent_fixes.iter().any(|recent| fixes_are_two_birds(candidate, recent, config))
}

/// Keep the entity's recent raw fixes for the concurrency check (bounded by time).
pub fn remember_fix(entity: &mut BirdEntityRecord, sample: EntityFixSample, config: &BirdEntityConfig) {
    let keep_ns = seconds_to_nanoseconds(FIX_MEMORY_WINDOWS * config.concurrent_fix_window_seconds);
    let newest_ns = entity.recent_fixes.iter().map(|fix| fix.event_time_ns).fold(sample.event_time_ns, i64::max);
    entity.recent_fixes.push(sample);
    entity.recent_fixes.retain(|fix| newest_ns - fix.event_time_ns <= keep_ns);
}

/// Drop the entity's recent fixes at or after `since_ns` (a perched hop the
/// entity followed was undone: those fixes were another bird's). Returns how many.
pub fn forget_fixes_since(entity: &mut BirdEntityRecord, since_ns: i64) -> usize {
    let before = entity.recent_fixes.len();
    entity.recent_fixes.retain(|fix| fix.event_time_ns < since_ns);
    before - entity.recent_fixes.len()
}

#[cfg(test)]
mod tests {
    use super::horizontal_mahalanobis_squared;

    #[test]
    fn indefinite_covariance_gives_no_distance_instead_of_a_negative_one() {
        let indefinite = [[4.0, 0.0, 0.0], [0.0, -9.0, 0.0], [0.0, 0.0, 1.0]];
        let value = horizontal_mahalanobis_squared(&[10.0, 4.0, 0.0], Some(&indefinite), &[12.0, 9.0, 0.0], Some(&indefinite));
        assert_eq!(value, None);
        let nan = [[f64::NAN; 3]; 3];
        assert_eq!(horizontal_mahalanobis_squared(&[0.0; 3], Some(&nan), &[1.0, 0.0, 0.0], None), None);
    }
}
