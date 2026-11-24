//! Geodetic <-> local east/north/up conversion, a line-for-line port of
//! Python `minimappr/core/geo.py` `LocalCoordinateFrame` for both of its
//! modes, so a position the sidecar solves in is the position Python stores.
//!
//! Expression order follows the Python source on purpose: both sides run the
//! same libm trig in f64, which keeps them within nanometres instead of the
//! millimetre budget the cross-language parity test enforces
//! (`tests/test_site_geometry_cross_language_parity.py`).

use crate::localization_sensor_entry::{CoordinateProjection, GeodeticOrigin};

pub const WGS84_SEMI_MAJOR_AXIS_M: f64 = 6_378_137.0;
pub const WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;
pub const WGS84_FIRST_ECCENTRICITY_SQUARED: f64 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING);

/// Fixed-point iterations of Python `_ecef_to_lla` (Bowring-style refinement).
const ECEF_TO_GEODETIC_ITERATIONS: usize = 6;
/// Python's divide-by-zero guards in `_ecef_to_lla` / `local_to_geo`.
const PYTHON_DIVISION_GUARD: f64 = 1e-12;
const PYTHON_METRES_PER_DEGREE_GUARD: f64 = 1e-9;

/// Per-origin constants of one site frame, precomputed once like the Python
/// dataclass's `__post_init__`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalEnuProjector {
    origin: GeodeticOrigin,
    projection: CoordinateProjection,
    origin_ecef_m: [f64; 3],
    /// Rows are the east, north and up unit vectors in ECEF.
    ecef_to_enu_rotation: [[f64; 3]; 3],
    metres_per_degree_latitude: f64,
    metres_per_degree_longitude: f64,
}

impl LocalEnuProjector {
    pub fn new(origin: GeodeticOrigin, projection: CoordinateProjection) -> Self {
        let origin_latitude_rad = origin.lat.to_radians();
        // WGS84 series expansion, Python `_meters_per_deg_lat` / `_lon`.
        let metres_per_degree_latitude = 111132.92 - (559.82 * (2.0 * origin_latitude_rad).cos())
            + (1.175 * (4.0 * origin_latitude_rad).cos());
        let metres_per_degree_longitude =
            111412.84 * origin_latitude_rad.cos() - 93.5 * (3.0 * origin_latitude_rad).cos();
        Self {
            origin,
            projection,
            origin_ecef_m: geodetic_to_ecef_m(origin.lat, origin.lon, origin.alt_m),
            ecef_to_enu_rotation: ecef_to_enu_rotation(origin.lat, origin.lon),
            metres_per_degree_latitude,
            metres_per_degree_longitude,
        }
    }

    pub fn origin(&self) -> GeodeticOrigin {
        self.origin
    }

    pub fn projection(&self) -> CoordinateProjection {
        self.projection
    }

    /// Python `LocalCoordinateFrame.geo_to_local`.
    pub fn geodetic_to_local_enu_m(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        match self.projection {
            CoordinateProjection::FlatWgs84Series => [
                (lon_deg - self.origin.lon) * self.metres_per_degree_longitude,
                (lat_deg - self.origin.lat) * self.metres_per_degree_latitude,
                alt_m - self.origin.alt_m,
            ],
            CoordinateProjection::GeodeticEcefEnu => {
                let ecef_m = geodetic_to_ecef_m(lat_deg, lon_deg, alt_m);
                let delta_ecef_m = [
                    ecef_m[0] - self.origin_ecef_m[0],
                    ecef_m[1] - self.origin_ecef_m[1],
                    ecef_m[2] - self.origin_ecef_m[2],
                ];
                multiply_matrix_vector(&self.ecef_to_enu_rotation, delta_ecef_m)
            }
        }
    }

    /// Python `LocalCoordinateFrame.local_to_geo` without its output clamp
    /// (`_safe_geo_point`): callers here build fixtures and tests, where a
    /// silently clamped coordinate would hide a wrong input.
    pub fn local_enu_to_geodetic(&self, position_enu_m: [f64; 3]) -> GeodeticOrigin {
        let [east_m, north_m, up_m] = position_enu_m;
        match self.projection {
            CoordinateProjection::FlatWgs84Series => GeodeticOrigin {
                lat: self.origin.lat
                    + (north_m
                        / self
                            .metres_per_degree_latitude
                            .max(PYTHON_METRES_PER_DEGREE_GUARD)),
                lon: self.origin.lon
                    + (east_m
                        / self
                            .metres_per_degree_longitude
                            .max(PYTHON_METRES_PER_DEGREE_GUARD)),
                alt_m: self.origin.alt_m + up_m,
            },
            CoordinateProjection::GeodeticEcefEnu => {
                let enu_to_ecef = transpose(&self.ecef_to_enu_rotation);
                let rotated_m = multiply_matrix_vector(&enu_to_ecef, position_enu_m);
                ecef_to_geodetic([
                    self.origin_ecef_m[0] + rotated_m[0],
                    self.origin_ecef_m[1] + rotated_m[1],
                    self.origin_ecef_m[2] + rotated_m[2],
                ])
            }
        }
    }
}

/// Site ENU metres of a geodetic point; equals Python
/// `LocalCoordinateFrame(origin, mode).geo_to_local(point)`.
pub fn geodetic_to_local_enu_m(
    origin: &GeodeticOrigin,
    projection: CoordinateProjection,
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
) -> [f64; 3] {
    LocalEnuProjector::new(*origin, projection).geodetic_to_local_enu_m(lat_deg, lon_deg, alt_m)
}

/// Inverse of [`geodetic_to_local_enu_m`] (Python `local_to_geo`, unclamped).
pub fn local_enu_to_geodetic(
    origin: &GeodeticOrigin,
    projection: CoordinateProjection,
    position_enu_m: [f64; 3],
) -> GeodeticOrigin {
    LocalEnuProjector::new(*origin, projection).local_enu_to_geodetic(position_enu_m)
}

/// Python `_lla_to_ecef`.
fn geodetic_to_ecef_m(lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
    let prime_vertical_radius_m = WGS84_SEMI_MAJOR_AXIS_M
        / (1.0 - (WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat)).sqrt();
    [
        (prime_vertical_radius_m + alt_m) * cos_lat * cos_lon,
        (prime_vertical_radius_m + alt_m) * cos_lat * sin_lon,
        (prime_vertical_radius_m * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED) + alt_m) * sin_lat,
    ]
}

/// Python `_ecef_to_lla`.
fn ecef_to_geodetic(ecef_m: [f64; 3]) -> GeodeticOrigin {
    let [x, y, z] = ecef_m;
    let longitude_rad = y.atan2(x);
    let equatorial_distance_m = x.hypot(y);
    let mut latitude_rad =
        z.atan2(equatorial_distance_m * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED));
    let prime_vertical_radius_m = |latitude_rad: f64| {
        let sin_lat = latitude_rad.sin();
        WGS84_SEMI_MAJOR_AXIS_M
            / (1.0 - (WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat)).sqrt()
    };
    for _ in 0..ECEF_TO_GEODETIC_ITERATIONS {
        let radius_m = prime_vertical_radius_m(latitude_rad);
        let altitude_m =
            (equatorial_distance_m / latitude_rad.cos().max(PYTHON_DIVISION_GUARD)) - radius_m;
        latitude_rad = z.atan2(
            equatorial_distance_m
                * (1.0
                    - (WGS84_FIRST_ECCENTRICITY_SQUARED * radius_m
                        / (radius_m + altitude_m).max(PYTHON_DIVISION_GUARD))),
        );
    }
    let radius_m = prime_vertical_radius_m(latitude_rad);
    let altitude_m =
        (equatorial_distance_m / latitude_rad.cos().max(PYTHON_DIVISION_GUARD)) - radius_m;
    GeodeticOrigin {
        lat: latitude_rad.to_degrees(),
        lon: longitude_rad.to_degrees(),
        alt_m: altitude_m,
    }
}

/// Python `_enu_rotation`.
fn ecef_to_enu_rotation(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let (sin_lat, cos_lat) = lat_deg.to_radians().sin_cos();
    let (sin_lon, cos_lon) = lon_deg.to_radians().sin_cos();
    [
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ]
}

fn multiply_matrix_vector(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
}

fn transpose(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGIN: GeodeticOrigin = GeodeticOrigin {
        lat: 45.0,
        lon: -93.0,
        alt_m: 250.0,
    };
    const BOTH_PROJECTIONS: [CoordinateProjection; 2] = [
        CoordinateProjection::FlatWgs84Series,
        CoordinateProjection::GeodeticEcefEnu,
    ];

    #[test]
    fn origin_maps_to_zero_in_both_projections() {
        for projection in BOTH_PROJECTIONS {
            let enu =
                geodetic_to_local_enu_m(&ORIGIN, projection, ORIGIN.lat, ORIGIN.lon, ORIGIN.alt_m);
            assert!(
                enu.iter().all(|axis| axis.abs() < 1e-9),
                "{projection:?}: {enu:?}"
            );
        }
    }

    #[test]
    fn flat_series_uses_python_metres_per_degree() {
        let east_north_up = geodetic_to_local_enu_m(
            &ORIGIN,
            CoordinateProjection::FlatWgs84Series,
            ORIGIN.lat + 0.001,
            ORIGIN.lon + 0.001,
            ORIGIN.alt_m + 3.0,
        );
        // At 45 N: 111132.92 - 1.175 m/deg north, 111412.84 cos45 + 93.5 cos45 m/deg east.
        let expected_north_m = (111_132.92 - 1.175) * 0.001;
        let expected_east_m = (111_412.84 + 93.5) * std::f64::consts::FRAC_1_SQRT_2 * 0.001;
        assert!((east_north_up[1] - expected_north_m).abs() < 1e-6);
        assert!((east_north_up[0] - expected_east_m).abs() < 1e-6);
        assert_eq!(east_north_up[2], 3.0);
    }

    #[test]
    fn projections_agree_near_the_origin_and_round_trip() {
        let local_point_m = [1_200.0, -800.0, 12.0];
        let flat = local_enu_to_geodetic(
            &ORIGIN,
            CoordinateProjection::FlatWgs84Series,
            local_point_m,
        );
        let geodetic_enu = geodetic_to_local_enu_m(
            &ORIGIN,
            CoordinateProjection::GeodeticEcefEnu,
            flat.lat,
            flat.lon,
            flat.alt_m,
        );
        // Earth curvature drops ~0.16 m over 1.4 km; horizontal stays sub-metre.
        assert!((geodetic_enu[0] - local_point_m[0]).abs() < 1.0);
        assert!((geodetic_enu[1] - local_point_m[1]).abs() < 1.0);
        for projection in BOTH_PROJECTIONS {
            let point = local_enu_to_geodetic(&ORIGIN, projection, local_point_m);
            let back =
                geodetic_to_local_enu_m(&ORIGIN, projection, point.lat, point.lon, point.alt_m);
            for axis in 0..3 {
                assert!(
                    (back[axis] - local_point_m[axis]).abs() < 1e-6,
                    "{projection:?}: {back:?}"
                );
            }
        }
    }
}
