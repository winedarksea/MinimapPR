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
pub const UNOBSERVABLE_CONFIDENCE_CAP: f32 = 0.20;
pub const UNOBSERVABLE_RANGE_OBSERVABILITY_CAP: f32 = 0.05;
pub const BEARING_PROJECTED_CONFIDENCE_CAP: f32 = 0.85;
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
pub fn confidence_cap_for_mode(mode: Option<&str>) -> Option<f32> {
    match mode {
        Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY) => Some(UNOBSERVABLE_CONFIDENCE_CAP),
        Some(RANGE_BEARING_PROJECTED) => Some(BEARING_PROJECTED_CONFIDENCE_CAP),
        _ => None,
    }
}

/// Amplitude/SNR-informed range prior (Phase 1c). Derives a projection distance
/// from the received level via inverse-square spreading::
///
///     r = 10 ^ ((L_ref - L_recv) / 20)
///
/// clamped to `[min_range_m, max_range_m]`. Mirrors the Python
/// `minimappr/core/amplitude_range.py::amplitude_range_prior_m`. Returns
/// `(range_m, was_clamped)`.
pub fn amplitude_range_prior_m(
    received_level_db: f32,
    reference_source_level_db: f32,
    min_range_m: f32,
    max_range_m: f32,
) -> (f32, bool) {
    let lo = min_range_m.max(0.0);
    let hi = max_range_m.max(lo);
    if !received_level_db.is_finite() {
        return (lo, true);
    }
    let raw_range_m = 10f32.powf((reference_source_level_db - received_level_db) / 20.0);
    let clamped_range_m = raw_range_m.clamp(lo, hi);
    let was_clamped = !(lo <= raw_range_m && raw_range_m <= hi);
    (clamped_range_m, was_clamped)
}

/// Range-observability cap for `mode`, or `None` when the range is observable.
pub fn range_observability_cap_for_mode(mode: Option<&str>) -> Option<f32> {
    match mode {
        Some(RANGE_ASYMPTOTIC) | Some(RANGE_BOUNDARY) => {
            Some(UNOBSERVABLE_RANGE_OBSERVABILITY_CAP)
        }
        Some(RANGE_BEARING_PROJECTED) => Some(BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amplitude_prior_inverse_square_table() {
        // 20 dB below reference → 10× farther.
        let (r, clamped) = amplitude_range_prior_m(80.0, 100.0, 5.0, 1000.0);
        assert!((r - 10.0).abs() < 1e-3, "got {r}");
        assert!(!clamped);
        // 40 dB below reference → 100× = clamped up would be 100 (within band).
        let (r, clamped) = amplitude_range_prior_m(60.0, 100.0, 5.0, 1000.0);
        assert!((r - 100.0).abs() < 1e-2, "got {r}");
        assert!(!clamped);
        // At reference level → 1 m, clamped up to the 5 m floor.
        let (r, clamped) = amplitude_range_prior_m(100.0, 100.0, 5.0, 1000.0);
        assert!((r - 5.0).abs() < 1e-6, "got {r}");
        assert!(clamped);
        // 60 dB below → 1000 m at the ceiling boundary (within → not clamped).
        let (r, clamped) = amplitude_range_prior_m(40.0, 100.0, 5.0, 1000.0);
        assert!((r - 1000.0).abs() < 1e-1, "got {r}");
        assert!(!clamped);
        // Way below → clamped to ceiling.
        let (r, clamped) = amplitude_range_prior_m(0.0, 100.0, 5.0, 1000.0);
        assert!((r - 1000.0).abs() < 1e-6, "got {r}");
        assert!(clamped);
    }

    #[test]
    fn amplitude_prior_non_finite_falls_back_to_floor() {
        let (r, clamped) = amplitude_range_prior_m(f32::NAN, 100.0, 5.0, 1000.0);
        assert!((r - 5.0).abs() < 1e-6);
        assert!(clamped);
    }
}
