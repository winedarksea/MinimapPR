//! Flat local-ENU ↔ geodetic conversion, mirroring the `flat` mode of
//! `LocalCoordinateFrame` in `minimappr/core/geo.py` coefficient for
//! coefficient, so `geo_to_local(local_to_geo(p)) == p` on the Python side.

// Series for metres per degree (same coefficients as geo.py).
const METERS_PER_DEGREE_LATITUDE_BASE: f64 = 111_132.92;
const METERS_PER_DEGREE_LATITUDE_COS2: f64 = -559.82;
const METERS_PER_DEGREE_LATITUDE_COS4: f64 = 1.175;
const METERS_PER_DEGREE_LONGITUDE_COS1: f64 = 111_412.84;
const METERS_PER_DEGREE_LONGITUDE_COS3: f64 = -93.5;
const MINIMUM_METERS_PER_DEGREE: f64 = 1e-9;

pub const COORDINATE_MODE_FLAT: &str = "flat";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FlatLocalFrame {
    pub origin: GeoPoint,
    meters_per_degree_latitude: f64,
    meters_per_degree_longitude: f64,
}

impl FlatLocalFrame {
    pub fn new(origin: GeoPoint) -> Self {
        let latitude_rad = origin.lat.to_radians();
        Self {
            origin,
            meters_per_degree_latitude: METERS_PER_DEGREE_LATITUDE_BASE
                + METERS_PER_DEGREE_LATITUDE_COS2 * (2.0 * latitude_rad).cos()
                + METERS_PER_DEGREE_LATITUDE_COS4 * (4.0 * latitude_rad).cos(),
            meters_per_degree_longitude: METERS_PER_DEGREE_LONGITUDE_COS1 * latitude_rad.cos()
                + METERS_PER_DEGREE_LONGITUDE_COS3 * (3.0 * latitude_rad).cos(),
        }
    }

    pub fn local_to_geo(&self, position_m: [f64; 3]) -> GeoPoint {
        GeoPoint {
            lat: self.origin.lat
                + position_m[1]
                    / self
                        .meters_per_degree_latitude
                        .max(MINIMUM_METERS_PER_DEGREE),
            lon: self.origin.lon
                + position_m[0]
                    / self
                        .meters_per_degree_longitude
                        .max(MINIMUM_METERS_PER_DEGREE),
            alt_m: self.origin.alt_m + position_m[2],
        }
    }

    pub fn geo_to_local(&self, point: GeoPoint) -> [f64; 3] {
        [
            (point.lon - self.origin.lon) * self.meters_per_degree_longitude,
            (point.lat - self.origin.lat) * self.meters_per_degree_latitude,
            point.alt_m - self.origin.alt_m,
        ]
    }
}
