//! The bird-entity record and its spatial-state geometry (port of
//! `minimappr/core/bird_entity_model.py`, server track S6).
//!
//! One bird is one entity moving `HeardNearby → DirectionKnown → Pinpointed`
//! as evidence arrives. An observation never lowers the state; only
//! maintenance steps an entity back (track expiry, uncertainty growth,
//! spatial staleness). Pure data + math.

use crate::wildlife_tracking::covariance_math::{horizontal_block, symmetric_eigen_2x2, symmetric_eigenvalues_2x2};
use crate::wildlife_tracking::species_log_odds::{score_for, LabelScores};
use crate::wildlife_tracking::{Covariance3, Position3, TrackId};

pub type EntityId = u64;

const FULL_CIRCLE_DEG: f64 = 360.0;
const HALF_CIRCLE_DEG: f64 = 180.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BirdEntitySpatialState {
    HeardNearby = 0,
    DirectionKnown = 1,
    Pinpointed = 2,
}

impl BirdEntitySpatialState {
    pub fn rank(self) -> u8 {
        self as u8
    }

    pub fn one_step_back(self) -> Self {
        match self {
            Self::Pinpointed => Self::DirectionKnown,
            Self::DirectionKnown | Self::HeardNearby => Self::HeardNearby,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HeardNearby => "heard_nearby",
            Self::DirectionKnown => "direction_known",
            Self::Pinpointed => "pinpointed",
        }
    }
}

/// How a detection may be drawn (the server's `spatial_display_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDisplayMode {
    Localized,
    /// Range unobservable: a bearing from the reporting node (far field).
    BearingOnly,
    /// No position: only the reporting node heard it.
    NodeOnly,
}

/// A bearing fan from a node (compass degrees, clockwise from north).
#[derive(Debug, Clone, PartialEq)]
pub struct BirdEntityBearingEstimate {
    pub origin_node_id: String,
    pub origin_position_m: Position3,
    pub bearing_deg: f64,
    pub half_width_deg: f64,
    /// None when range is unobservable (bearing-only solve).
    pub range_min_m: Option<f64>,
    pub range_max_m: Option<f64>,
}

/// One raw localized fix observed into an entity (for the concurrency check).
#[derive(Debug, Clone, PartialEq)]
pub struct EntityFixSample {
    pub event_time_ns: i64,
    pub position_m: Position3,
    pub covariance_m2: Option<Covariance3>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BirdEntityRecord {
    pub entity_id: EntityId,
    pub state: BirdEntitySpatialState,
    pub species_top_label: String,
    pub species_top_probability: f64,
    pub species_belief: LabelScores,
    pub first_heard_ns: i64,
    pub last_heard_ns: i64,
    pub reporting_node_id: Option<String>,
    pub reporting_node_position_m: Option<Position3>,
    pub reporting_node_ids: Vec<String>,
    pub position_m: Option<Position3>,
    pub position_covariance_m2: Option<Covariance3>,
    pub horizontal_uncertainty_m: Option<f64>,
    pub bearing: Option<BirdEntityBearingEstimate>,
    pub track_id: Option<TrackId>,
    pub canonical_detection_id: Option<String>,
    pub last_spatial_update_ns: Option<i64>,
    /// Restarts on every state change so the hold timer measures time *in* a state.
    pub last_state_change_ns: i64,
    pub detection_count: u32,
    pub absorbed_detection_count: u32,
    /// Last "just sang" pulse.
    pub last_pulse_ns: i64,
    pub recent_fixes: Vec<EntityFixSample>,
}

impl BirdEntityRecord {
    pub fn is_spatial(&self) -> bool {
        self.state != BirdEntitySpatialState::HeardNearby
    }

    /// Top species, or a belief of at least `min_probability`.
    pub fn believes_label(&self, label: &str, min_probability: f64) -> bool {
        label == self.species_top_label || score_for(&self.species_belief, label).is_some_and(|value| value >= min_probability)
    }
}

/// 1σ length of the horizontal (east/north) covariance major axis.
pub fn horizontal_major_sigma_m(covariance_m2: Option<&Covariance3>) -> Option<f64> {
    let eigenvalues = symmetric_eigenvalues_2x2(&horizontal_block(covariance_m2)?);
    eigenvalues.iter().all(|value| value.is_finite()).then(|| eigenvalues[1].max(0.0).sqrt())
}

/// Clockwise-from-north bearing in the local ENU frame (x east, y north).
pub fn compass_bearing_deg(origin_m: &Position3, target_m: &Position3) -> f64 {
    python_modulo((target_m[0] - origin_m[0]).atan2(target_m[1] - origin_m[1]).to_degrees(), FULL_CIRCLE_DEG)
}

pub fn angular_difference_deg(first_deg: f64, second_deg: f64) -> f64 {
    let difference = python_modulo(first_deg - second_deg, FULL_CIRCLE_DEG).abs();
    difference.min(FULL_CIRCLE_DEG - difference)
}

/// Python's `%`: the result takes the divisor's sign.
fn python_modulo(value: f64, divisor: f64) -> f64 {
    let remainder = value % divisor;
    if remainder != 0.0 && (remainder < 0.0) != (divisor < 0.0) { remainder + divisor } else { remainder }
}

/// Fan settings for [`bearing_estimate_toward`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingFanLimits {
    pub default_half_width_deg: f64,
    pub min_half_width_deg: f64,
    pub max_half_width_deg: f64,
    pub sigma_multiplier: f64,
}

/// Bearing fan from a node toward `target`; half width `atan(k·σ_tangential / range)`,
/// so an elongated radial ellipse (good bearing, poor range) gives a narrow fan
/// with a wide range band rather than a misleading blob.
pub fn bearing_estimate_toward(
    origin_node_id: &str,
    origin_position_m: &Position3,
    target_position_m: &Position3,
    covariance_m2: Option<&Covariance3>,
    range_observable: bool,
    limits: &BearingFanLimits,
) -> BirdEntityBearingEstimate {
    let bearing_deg = compass_bearing_deg(origin_position_m, target_position_m);
    let range_m = (target_position_m[0] - origin_position_m[0]).hypot(target_position_m[1] - origin_position_m[1]);
    let mut half_width_deg = limits.default_half_width_deg;
    let (mut range_min_m, mut range_max_m) = (None, None);
    let bearing_rad = bearing_deg.to_radians();
    let radial = nalgebra::Vector2::new(bearing_rad.sin(), bearing_rad.cos());
    let tangential = nalgebra::Vector2::new(bearing_rad.cos(), -bearing_rad.sin());
    let horizontal = horizontal_block(covariance_m2);
    match horizontal {
        Some(horizontal) if range_m > 0.0 => {
            let tangential_sigma_m = tangential.dot(&(horizontal * tangential)).max(0.0).sqrt();
            half_width_deg = (limits.sigma_multiplier * tangential_sigma_m).atan2(range_m).to_degrees();
            if range_observable {
                let radial_sigma_m = radial.dot(&(horizontal * radial)).max(0.0).sqrt();
                range_min_m = Some((range_m - limits.sigma_multiplier * radial_sigma_m).max(0.0));
                range_max_m = Some(range_m + limits.sigma_multiplier * radial_sigma_m);
            }
        }
        _ if range_observable => {
            range_min_m = Some(range_m);
            range_max_m = Some(range_m);
        }
        _ => {}
    }
    BirdEntityBearingEstimate {
        origin_node_id: origin_node_id.to_string(),
        origin_position_m: *origin_position_m,
        bearing_deg,
        half_width_deg: half_width_deg.max(limits.min_half_width_deg).min(limits.max_half_width_deg),
        range_min_m,
        range_max_m,
    }
}

/// `(semi_major_m, semi_minor_m, major_axis_compass_deg)` at `sigma_multiplier`.
pub fn uncertainty_ellipse(covariance_m2: Option<&Covariance3>, sigma_multiplier: f64) -> Option<(f64, f64, f64)> {
    let (eigenvalues, eigenvectors) = symmetric_eigen_2x2(&horizontal_block(covariance_m2)?);
    if !eigenvalues.iter().all(|value| value.is_finite()) {
        return None;
    }
    let major_axis = eigenvectors[1];
    let orientation_deg = python_modulo(major_axis[0].atan2(major_axis[1]).to_degrees(), HALF_CIRCLE_DEG);
    Some((
        sigma_multiplier * eigenvalues[1].max(0.0).sqrt(),
        sigma_multiplier * eigenvalues[0].max(0.0).sqrt(),
        orientation_deg,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearings_are_compass_degrees_and_differences_wrap() {
        assert!((compass_bearing_deg(&[0.0; 3], &[1.0, 0.0, 0.0]) - 90.0).abs() < 1e-12);
        assert!((compass_bearing_deg(&[0.0; 3], &[-1.0, 0.0, 0.0]) - 270.0).abs() < 1e-12);
        assert!((angular_difference_deg(350.0, 10.0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn radial_ellipse_gives_narrow_fan_and_wide_range_band() {
        let limits = BearingFanLimits { default_half_width_deg: 15.0, min_half_width_deg: 5.0, max_half_width_deg: 45.0, sigma_multiplier: 2.0 };
        let covariance = [[1.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 1.0]];
        let fan = bearing_estimate_toward("n", &[0.0; 3], &[0.0, 100.0, 0.0], Some(&covariance), true, &limits);
        assert!((fan.half_width_deg - 5.0).abs() < 1e-12);
        assert!((fan.range_max_m.unwrap() - 140.0).abs() < 1e-9);
    }
}
