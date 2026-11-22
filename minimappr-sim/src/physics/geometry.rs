//! Local ENU geometry (metres; x = east, y = north, z = up) and
//! piecewise-linear trajectories in true session time.

use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: Vec3, factor: f64) -> Vec3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

#[inline]
pub fn norm(a: Vec3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
pub fn distance(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

pub fn lerp(a: Vec3, b: Vec3, fraction: f64) -> Vec3 {
    add(a, scale(sub(b, a), fraction))
}

/// One trajectory vertex: position at a true session time.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Waypoint {
    pub time_s: f64,
    pub position_m: Vec3,
}

/// Piecewise-linear path; held constant before the first and after the last
/// waypoint. A single waypoint is a static point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub waypoints: Vec<Waypoint>,
}

impl Trajectory {
    pub fn stationary(position_m: Vec3) -> Self {
        Self {
            waypoints: vec![Waypoint {
                time_s: 0.0,
                position_m,
            }],
        }
    }

    /// Build from `[t, x, y, z]` rows (sorted by time here).
    pub fn from_rows(rows: &[[f64; 4]]) -> Self {
        let mut waypoints: Vec<Waypoint> = rows
            .iter()
            .map(|row| Waypoint {
                time_s: row[0],
                position_m: [row[1], row[2], row[3]],
            })
            .collect();
        waypoints.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
        Self { waypoints }
    }

    pub fn is_stationary(&self) -> bool {
        self.waypoints
            .windows(2)
            .all(|pair| pair[0].position_m == pair[1].position_m)
    }

    pub fn position_at(&self, time_s: f64) -> Vec3 {
        let points = &self.waypoints;
        match points.len() {
            0 => [0.0; 3],
            1 => points[0].position_m,
            _ => {
                if time_s <= points[0].time_s {
                    return points[0].position_m;
                }
                let last = points[points.len() - 1];
                if time_s >= last.time_s {
                    return last.position_m;
                }
                // partition_point: first waypoint strictly after time_s.
                let upper = points.partition_point(|w| w.time_s <= time_s);
                let (a, b) = (points[upper - 1], points[upper]);
                let span = b.time_s - a.time_s;
                let fraction = if span > 0.0 {
                    (time_s - a.time_s) / span
                } else {
                    1.0
                };
                lerp(a.position_m, b.position_m, fraction)
            }
        }
    }

    /// Mean of the start/end-of-window positions — the static summary used in
    /// ground truth (bundle v1 accepts only static geometry).
    pub fn midpoint_position(&self, start_s: f64, end_s: f64) -> Vec3 {
        self.position_at(0.5 * (start_s + end_s))
    }

    /// Largest speed of any segment (m/s); bounds Doppler/arrival-time drift.
    pub fn max_speed_mps(&self) -> f64 {
        self.waypoints
            .windows(2)
            .map(|pair| {
                let span = pair[1].time_s - pair[0].time_s;
                if span > 0.0 {
                    distance(pair[1].position_m, pair[0].position_m) / span
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max)
    }

    pub fn translated(&self, offset_m: Vec3) -> Self {
        Self {
            waypoints: self
                .waypoints
                .iter()
                .map(|w| Waypoint {
                    time_s: w.time_s,
                    position_m: add(w.position_m, offset_m),
                })
                .collect(),
        }
    }

    pub fn to_rows(&self) -> Vec<[f64; 4]> {
        self.waypoints
            .iter()
            .map(|w| [w.time_s, w.position_m[0], w.position_m[1], w.position_m[2]])
            .collect()
    }
}
