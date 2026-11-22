//! Layout-level geometry: the search frame, collinearity and GDOP.
//!
//! Collinearity uses node positions (not individual mics): a tetra node's
//! 5 cm aperture can break the left/right mirror through its DOA, but it
//! does not widen the inter-node aperture that ranges the source.

use super::polar_grid::{norm, sub, LayoutFrame};

/// Horizontal scatter of node positions: centroid, principal axis and the
/// two standard deviations along/across it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizontalScatter {
    pub centroid_xy_m: [f64; 2],
    pub principal_axis_azimuth_rad: f64,
    pub major_axis_std_m: f64,
    pub minor_axis_std_m: f64,
}

pub fn horizontal_scatter(node_positions_m: &[[f64; 3]]) -> HorizontalScatter {
    let count = node_positions_m.len().max(1) as f64;
    let mean_x = node_positions_m.iter().map(|p| p[0]).sum::<f64>() / count;
    let mean_y = node_positions_m.iter().map(|p| p[1]).sum::<f64>() / count;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for position in node_positions_m {
        let (dx, dy) = (position[0] - mean_x, position[1] - mean_y);
        xx += dx * dx / count;
        xy += dx * dy / count;
        yy += dy * dy / count;
    }
    let half_trace = 0.5 * (xx + yy);
    let discriminant = (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let major = (half_trace + discriminant).max(0.0);
    let minor = (half_trace - discriminant).max(0.0);
    HorizontalScatter {
        centroid_xy_m: [mean_x, mean_y],
        principal_axis_azimuth_rad: 0.5 * (2.0 * xy).atan2(xx - yy),
        major_axis_std_m: major.sqrt(),
        minor_axis_std_m: minor.sqrt(),
    }
}

impl HorizontalScatter {
    /// sqrt(λ_min / λ_max); 0 for a perfect line or a single node.
    pub fn aspect_ratio(&self) -> f64 {
        if self.major_axis_std_m <= f64::EPSILON {
            0.0
        } else {
            self.minor_axis_std_m / self.major_axis_std_m
        }
    }

    pub fn frame(&self, ground_height_m: f64) -> LayoutFrame {
        LayoutFrame {
            origin_xy_m: self.centroid_xy_m,
            principal_axis_azimuth_rad: self.principal_axis_azimuth_rad,
            ground_height_m,
        }
    }
}

/// Horizontal dilution of precision at a source position, in metres of
/// position per metre of path-difference error (z held by the height prior).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizontalDilution {
    /// sqrt(trace((GᵀG)⁻¹)); infinite when the 2×2 normal matrix is singular.
    pub horizontal_gdop: f64,
    /// Along the centroid → source direction (range dilution).
    pub radial_dilution: f64,
    /// Across it (bearing dilution, in metres at the source).
    pub cross_radial_dilution: f64,
}

/// GDOP from rows gₚ = horizontal part of (u_a − u_b) over the given pairs.
pub fn horizontal_dilution(
    source_position_m: [f64; 3],
    centroid_xy_m: [f64; 2],
    sensor_pairs_m: &[([f64; 3], [f64; 3])],
) -> HorizontalDilution {
    let (mut gxx, mut gxy, mut gyy) = (0.0, 0.0, 0.0);
    for (sensor_a, sensor_b) in sensor_pairs_m {
        let offset_a = sub(source_position_m, *sensor_a);
        let offset_b = sub(source_position_m, *sensor_b);
        let (range_a, range_b) = (norm(offset_a).max(f64::EPSILON), norm(offset_b).max(f64::EPSILON));
        let gx = offset_a[0] / range_a - offset_b[0] / range_b;
        let gy = offset_a[1] / range_a - offset_b[1] / range_b;
        gxx += gx * gx;
        gxy += gx * gy;
        gyy += gy * gy;
    }
    let determinant = gxx * gyy - gxy * gxy;
    let scale_of_normal = (gxx + gyy).max(f64::MIN_POSITIVE);
    if determinant <= scale_of_normal * scale_of_normal * f64::EPSILON {
        return HorizontalDilution {
            horizontal_gdop: f64::INFINITY,
            radial_dilution: f64::INFINITY,
            cross_radial_dilution: f64::INFINITY,
        };
    }
    // (GᵀG)⁻¹ = [[gyy, −gxy], [−gxy, gxx]] / det
    let (cxx, cxy, cyy) = (gyy / determinant, -gxy / determinant, gxx / determinant);
    let radial = [source_position_m[0] - centroid_xy_m[0], source_position_m[1] - centroid_xy_m[1]];
    let radial_length = radial[0].hypot(radial[1]);
    let (ex, ey) = if radial_length > f64::EPSILON {
        (radial[0] / radial_length, radial[1] / radial_length)
    } else {
        (1.0, 0.0)
    };
    let quadratic = |x: f64, y: f64| (x * x * cxx + 2.0 * x * y * cxy + y * y * cyy).max(0.0).sqrt();
    HorizontalDilution {
        horizontal_gdop: (cxx + cyy).max(0.0).sqrt(),
        radial_dilution: quadratic(ex, ey),
        cross_radial_dilution: quadratic(-ey, ex),
    }
}

/// Mirror of a world point across the layout's principal axis line.
pub fn mirror_across_principal_axis(position_m: [f64; 3], frame: &LayoutFrame) -> [f64; 3] {
    let (radius, azimuth, height) = frame.world_to_layout(position_m);
    frame.layout_to_world(radius, std::f64::consts::TAU - azimuth, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_trail_has_tiny_aspect_ratio_and_axis_along_trail() {
        let trail: Vec<[f64; 3]> = (0..5).map(|i| [i as f64 * 6.0, i as f64 * 6.0 + 0.1 * (i % 2) as f64, 1.4]).collect();
        let scatter = horizontal_scatter(&trail);
        assert!(scatter.aspect_ratio() < 0.02);
        assert!((scatter.principal_axis_azimuth_rad - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }

    #[test]
    fn mirror_reflects_across_axis() {
        let frame = LayoutFrame { origin_xy_m: [0.0, 0.0], principal_axis_azimuth_rad: 0.0, ground_height_m: 0.0 };
        let mirrored = mirror_across_principal_axis([4.0, 7.0, 3.0], &frame);
        assert!((mirrored[0] - 4.0).abs() < 1e-9 && (mirrored[1] + 7.0).abs() < 1e-9 && (mirrored[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn far_source_has_larger_radial_than_cross_radial_dilution() {
        let sensors = [[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [10.0, 10.0, 1.0]];
        let mut pairs = Vec::new();
        for a in 0..4 {
            for b in (a + 1)..4 {
                pairs.push((sensors[a], sensors[b]));
            }
        }
        let dilution = horizontal_dilution([5.0, 80.0, 5.0], [5.0, 5.0], &pairs);
        assert!(dilution.radial_dilution > 5.0 * dilution.cross_radial_dilution);
    }
}
