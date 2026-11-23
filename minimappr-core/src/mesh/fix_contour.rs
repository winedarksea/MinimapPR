//! The likelihood contour a host publishes for a fix: a polygon on the
//! horizontal covariance ellipse scaled by `EntityContourConfig`'s σ
//! multiplier. Factored out of the host so the statistical coverage gate
//! (minimappr-sim-gates) scores exactly the polygon the map draws.
//!
//! What is drawn is the *published* covariance: the fix's LM covariance
//! scaled by its geometry class's calibration (`FixCovarianceCalibrationConfig`,
//! closeout C3: the mesh LM σ still overstates the error, ~1.4× inside the
//! layout and ~3× outside), plus the map anchor's own error (closeout C2):
//! the polygon lives in the GPS-anchored ENU frame, so on a basemap, in an
//! export or next to another island the anchor's error moves it by metres.
//! `CallFix::covariance_m2` itself stays the raw LM covariance.

use super::config::{EntityContourConfig, FixCovarianceCalibrationConfig};

/// The filtered map anchor's error when a fix was made: translation σ per
/// axis, rotation σ, and the point the rotation turns about (the anchored
/// layout centroid; the anchor fit's rotation is estimated about it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorUncertainty {
    pub translation_sigma_m: f64,
    pub rotation_sigma_rad: f64,
    pub rotation_centre_m: [f64; 2],
}

/// An island's layout extent (in the frame fixes are solved in) and its
/// anchor error (None while unanchored).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionContext {
    pub layout_centre_m: [f64; 2],
    pub layout_radius_m: f64,
    pub anchor: Option<AnchorUncertainty>,
}

impl RegionContext {
    /// Inside = within the layout's circumradius of its centroid (the
    /// geometry split the calibration is fitted on).
    pub fn contains(&self, position_m: [f64; 3]) -> bool {
        (position_m[0] - self.layout_centre_m[0]).hypot(position_m[1] - self.layout_centre_m[1]) <= self.layout_radius_m
    }

    pub fn without_anchor(&self) -> Self {
        Self { anchor: None, ..*self }
    }
}

/// Σ_world = Σ_fix + σ_t²·I + σ_θ²·r⊥r⊥ᵀ (horizontal block), r the fix's
/// offset from the rotation centre: a small anchor rotation δθ moves the
/// point by δθ·r⊥.
pub fn world_frame_covariance(position_m: [f64; 3], covariance_m2: &[[f64; 3]; 3], anchor: &AnchorUncertainty) -> [[f64; 3]; 3] {
    let lever = [position_m[0] - anchor.rotation_centre_m[0], position_m[1] - anchor.rotation_centre_m[1]];
    let perpendicular = [-lever[1], lever[0]];
    let (translation_variance, rotation_variance) = (anchor.translation_sigma_m.powi(2), anchor.rotation_sigma_rad.powi(2));
    let mut world = *covariance_m2;
    for row in 0..2 {
        world[row][row] += translation_variance;
        for column in 0..2 {
            world[row][column] += rotation_variance * perpendicular[row] * perpendicular[column];
        }
    }
    world
}

/// The covariance the host draws: calibrated Σ_fix, plus the anchor terms
/// when the island is anchored. Without a context (no layout) the inside
/// scale, the larger one, applies.
pub fn published_covariance(
    position_m: [f64; 3],
    covariance_m2: &[[f64; 3]; 3],
    context: Option<&RegionContext>,
    calibration: &FixCovarianceCalibrationConfig,
) -> [[f64; 3]; 3] {
    let scale = match context {
        Some(context) if !context.contains(position_m) => calibration.outside_layout_sigma_scale,
        _ => calibration.inside_layout_sigma_scale,
    };
    let calibrated = covariance_m2.map(|row| row.map(|value| value * scale * scale));
    match context.and_then(|context| context.anchor.as_ref()) {
        Some(anchor) => world_frame_covariance(position_m, &calibrated, anchor),
        None => calibrated,
    }
}

/// What the host publishes for a fix or an entity.
pub fn published_contour(
    position_m: [f64; 3],
    covariance_m2: &[[f64; 3]; 3],
    context: Option<&RegionContext>,
    contour: &EntityContourConfig,
    calibration: &FixCovarianceCalibrationConfig,
) -> Vec<[f32; 2]> {
    horizontal_covariance_contour(position_m, &published_covariance(position_m, covariance_m2, context, calibration), contour)
}

/// Polygon vertices (east, north) of the scaled horizontal covariance ellipse.
pub fn horizontal_covariance_contour(
    position_m: [f64; 3],
    covariance_m2: &[[f64; 3]; 3],
    config: &EntityContourConfig,
) -> Vec<[f32; 2]> {
    let [[a, b, _], [_, d, _], _] = *covariance_m2;
    let half_trace = 0.5 * (a + d);
    let spread = (0.25 * (a - d).powi(2) + b * b).sqrt();
    let (major, minor) = ((half_trace + spread).max(0.0).sqrt(), (half_trace - spread).max(0.0).sqrt());
    let angle = 0.5 * (2.0 * b).atan2(a - d);
    let multiplier = config.contour_sigma_multiplier;
    (0..config.contour_point_count)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / config.contour_point_count as f64;
            let (u, v) = (multiplier * major * t.cos(), multiplier * minor * t.sin());
            [
                (position_m[0] + u * angle.cos() - v * angle.sin()) as f32,
                (position_m[1] + u * angle.sin() + v * angle.cos()) as f32,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIX: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]];

    #[test]
    fn anchor_error_adds_translation_everywhere_and_rotation_across_the_lever() {
        let anchor = AnchorUncertainty { translation_sigma_m: 2.0, rotation_sigma_rad: 0.1, rotation_centre_m: [0.0, 0.0] };
        // 30 m east of the centre: rotation error is all north (0.1 × 30 = 3 m).
        let world = world_frame_covariance([30.0, 0.0, 5.0], &FIX, &anchor);
        assert!((world[0][0] - 5.0).abs() < 1e-9 && (world[1][1] - 14.0).abs() < 1e-9 && world[0][1].abs() < 1e-9);
        assert_eq!(world[2][2], 4.0);
    }

    #[test]
    fn published_covariance_calibrates_by_geometry_then_adds_the_anchor() {
        let calibration = FixCovarianceCalibrationConfig { inside_layout_sigma_scale: 0.5, outside_layout_sigma_scale: 0.25 };
        let anchor = AnchorUncertainty { translation_sigma_m: 1.0, rotation_sigma_rad: 0.0, rotation_centre_m: [0.0, 0.0] };
        let context = RegionContext { layout_centre_m: [0.0, 0.0], layout_radius_m: 10.0, anchor: Some(anchor) };
        let inside = published_covariance([3.0, 4.0, 2.0], &FIX, Some(&context.without_anchor()), &calibration);
        assert_eq!((inside[0][0], inside[2][2]), (0.25, 1.0));
        let outside = published_covariance([30.0, 0.0, 2.0], &FIX, Some(&context), &calibration);
        assert_eq!(outside[1][1], 0.0625 + 1.0);
        let config = EntityContourConfig::default();
        let wider = published_contour([30.0, 0.0, 5.0], &FIX, Some(&context), &config, &calibration);
        let narrower = published_contour([30.0, 0.0, 5.0], &FIX, Some(&context.without_anchor()), &config, &calibration);
        let north_extent = |polygon: &[[f32; 2]]| polygon.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert!(north_extent(&wider) > 3.0 * north_extent(&narrower));
    }
}
