//! The likelihood contour a host publishes for a fix: a polygon on the
//! horizontal covariance ellipse scaled by `EntityContourConfig`'s σ
//! multiplier. Factored out of the host so the statistical coverage gate
//! (minimappr-sim-gates) scores exactly the polygon the map draws.

use super::config::EntityContourConfig;

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
