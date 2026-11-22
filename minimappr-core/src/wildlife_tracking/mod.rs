//! Server-track wildlife tracking ported from the Python reference (S3 gate
//! verdict "PASS: port"): the perched random-walk filter, the relocation gate
//! with ghost revert and its concurrency guard, capped species log-odds and
//! the species association gate, composed into [`perched_tracker`].
//!
//! Sans-IO and allocation-light: time is the caller's nanoseconds, no clocks,
//! threads or randomness. Python↔Rust parity: `tests/test_wildlife_tracking_cross_language_parity.py`.

pub mod covariance_math;
pub mod nearest_neighbor_association;
pub mod perched_concurrency;
pub mod perched_filter;
pub mod perched_relocation;
pub mod perched_tracker;
pub mod species_association_gate;
pub mod species_log_odds;
pub mod tracker_config;
pub mod wildlife_track;

pub type TrackId = u64;
pub type Position3 = [f64; 3];
pub type Covariance3 = [[f64; 3]; 3];

pub const NANOSECONDS_PER_SECOND: i64 = 1_000_000_000;

/// `int(seconds * 1e9)` as the reference computes window lengths (truncation).
pub fn seconds_to_nanoseconds(seconds: f64) -> i64 {
    (seconds * NANOSECONDS_PER_SECOND as f64) as i64
}
