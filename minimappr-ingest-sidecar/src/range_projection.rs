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

/// Bearing is well-determined; position is projected along the bearing ray at
/// the far-field default range. The radial (range) axis is NOT observed —
/// the covariance is explicitly elongated along the bearing ray. Confidence
/// reflects bearing quality and is NOT penalised by the full unobservable cap.
/// Emitted instead of RANGE_ASYMPTOTIC when the direction-fit residual is low
/// (bearing is tight) even though range curvature is unresolvable.
pub const RANGE_BEARING_PROJECTED: &str = "range_bearing_projected";

/// Confidence/observability caps applied when the range axis is unobservable.
/// These mirror the canonical Python values in `minimappr/core/range_projection.py`
/// and the shared `RANGE_PROJECTION_CONTRACT.md`. RANGE_ASYMPTOTIC / RANGE_BOUNDARY
/// receive the harsh cap (range AND bearing both uncertain); RANGE_BEARING_PROJECTED
/// keeps a high confidence cap (bearing is well-observed) while still driving range
/// observability to the floor.
#[allow(dead_code)]
pub const UNOBSERVABLE_CONFIDENCE_CAP: f32 = 0.20;
#[allow(dead_code)]
pub const UNOBSERVABLE_RANGE_OBSERVABILITY_CAP: f32 = 0.05;
#[allow(dead_code)]
pub const BEARING_PROJECTED_CONFIDENCE_CAP: f32 = 0.85;
#[allow(dead_code)]
pub const BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP: f32 = 0.05;

/// Returns true when the range axis is not observable for `mode`. NOTE: this is
/// true for RANGE_BEARING_PROJECTED as well — its *range* is unobservable — but
/// its confidence cap is the gentle bearing cap, not the harsh unobservable one.
/// Use `confidence_cap_for_mode` to apply the correct, mode-specific cap.
#[allow(dead_code)]
pub fn is_unobservable(mode: Option<&str>) -> bool {
    matches!(
        mode,
        Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY) | Some(RANGE_BEARING_PROJECTED)
    )
}

/// Confidence cap for `mode`, or `None` when the range is observable (no cap).
/// RANGE_BEARING_PROJECTED gets the gentle bearing cap; the other unobservable
/// modes get the harsh cap. Kept in lockstep with the Python haircut so a future
/// Rust-side cap cannot diverge.
#[allow(dead_code)]
pub fn confidence_cap_for_mode(mode: Option<&str>) -> Option<f32> {
    match mode {
        Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY) => Some(UNOBSERVABLE_CONFIDENCE_CAP),
        Some(RANGE_BEARING_PROJECTED) => Some(BEARING_PROJECTED_CONFIDENCE_CAP),
        _ => None,
    }
}

/// Range-observability cap for `mode`, or `None` when the range is observable.
#[allow(dead_code)]
pub fn range_observability_cap_for_mode(mode: Option<&str>) -> Option<f32> {
    match mode {
        Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY) => {
            Some(UNOBSERVABLE_RANGE_OBSERVABILITY_CAP)
        }
        Some(RANGE_BEARING_PROJECTED) => Some(BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP),
        _ => None,
    }
}
