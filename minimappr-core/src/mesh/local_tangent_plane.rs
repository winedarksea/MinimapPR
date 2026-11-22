//! Local east-north plane around a session origin (equirectangular).
//! Field Mode islands span at most a few hundred metres, where this
//! approximation is far below GPS error (millimetres at 500 m).

const WGS84_EQUATORIAL_RADIUS_M: f64 = 6_378_137.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTangentPlane {
    pub origin_latitude_deg: f64,
    pub origin_longitude_deg: f64,
}

impl LocalTangentPlane {
    pub fn new(origin_latitude_deg: f64, origin_longitude_deg: f64) -> Self {
        Self {
            origin_latitude_deg,
            origin_longitude_deg,
        }
    }

    fn metres_per_degree_latitude() -> f64 {
        WGS84_EQUATORIAL_RADIUS_M.to_radians()
    }

    fn metres_per_degree_longitude(&self) -> f64 {
        Self::metres_per_degree_latitude() * self.origin_latitude_deg.to_radians().cos()
    }

    pub fn east_north_from_geodetic(&self, latitude_deg: f64, longitude_deg: f64) -> [f64; 2] {
        [
            (longitude_deg - self.origin_longitude_deg) * self.metres_per_degree_longitude(),
            (latitude_deg - self.origin_latitude_deg) * Self::metres_per_degree_latitude(),
        ]
    }

    pub fn geodetic_from_east_north(&self, east_north_m: [f64; 2]) -> (f64, f64) {
        (
            self.origin_latitude_deg + east_north_m[1] / Self::metres_per_degree_latitude(),
            self.origin_longitude_deg + east_north_m[0] / self.metres_per_degree_longitude(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_exact_to_millimetres() {
        let plane = LocalTangentPlane::new(47.6, -122.3);
        let (lat, lon) = plane.geodetic_from_east_north([123.4, -56.7]);
        let back = plane.east_north_from_geodetic(lat, lon);
        assert!((back[0] - 123.4).abs() < 1e-6 && (back[1] + 56.7).abs() < 1e-6);
    }
}
