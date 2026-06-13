//! Canonical `range_projection_mode` vocabulary, mirrored on the Python side in
//! `minimappr/core/range_projection.py`. Every localization result tags how the
//! radial (range) axis was resolved so downstream consumers can apply a uniform
//! observability/confidence haircut. See `RANGE_PROJECTION_CONTRACT.md` for the
//! shared contract that keeps the two languages from drifting again.

/// Range observed from the data (well-conditioned grid / TDOA fit).
pub const RANGE_REFINED: &str = "range_refined";

/// Range unobservable: the estimate is dominated by the far-field prior rather
/// than the measurements. (Formerly emitted as `prior_projected`.)
pub const RANGE_ASYMPTOTIC: &str = "range_asymptotic";

/// Estimate clamped at the search-grid boundary; range is likewise not
/// observable from the data. (Formerly emitted as `bounded_grid_boundary`.)
pub const RANGE_BOUNDARY: &str = "range_boundary";

/// Returns true when the range axis is not observable for `mode` and the
/// estimate must receive the confidence/observability haircut.
#[allow(dead_code)]
pub fn is_unobservable(mode: Option<&str>) -> bool {
    matches!(mode, Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY))
}
