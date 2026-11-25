//! Compact-array near-field collapse guard; port of
//! `minimappr/core/far_field_model_selection.py`.
//!
//! Why: four mics of a compact (5 cm tetra) array give three independent
//! TDOAs, which a 3-D point always fits exactly, so the solve finds a point a
//! few centimetres out that "explains" timing noise as wavefront curvature.
//! Wherever a compact array's solve lands, the near-field point is replaced by
//! the best plane-wave projection unless it beats that projection by more than
//! the timing-noise floor: that test is the range-observability question, so
//! no distance cutoff applies (a 4-aperture cutoff let collapses 4-6 apertures
//! out through as refined fixes). Multi-node layouts keep the legacy answer.
//! Thresholds must stay equal to the Python constants (parity gate).

use nalgebra::{DVector, Vector3};

use super::levenberg_marquardt_driver::PositionResidualModel;
use super::spatial_constraint_terms::EPSILON;

/// Near-field point must beat the plane wave by this many timing-noise std (RMS).
pub const CURVATURE_EVIDENCE_NOISE_MULTIPLIER: f64 = 1.0;
/// Only arrays at most this wide (metres) are guarded: tetra nodes and stereo
/// phones with margin. Wider layouts observe range; a near/far tie there means
/// inconsistent TDOAs and keeps the legacy low-confidence answer.
pub const COMPACT_ARRAY_MAX_APERTURE_M: f64 = 0.5;

/// A plane-wave projection replacing a collapsed near-field solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarFieldAlternative {
    pub position_m: Vector3<f64>,
    pub residual_rms_seconds: f64,
}

/// Inputs mirroring the Python keyword arguments.
pub struct FarFieldGuardInputs<'model> {
    pub centroid_m: Vector3<f64>,
    pub near_field_position_m: Vector3<f64>,
    pub plane_wave_direction: Option<Vector3<f64>>,
    pub residual_model: &'model dyn PositionResidualModel,
    pub near_field_residual_rms_seconds: f64,
    pub timing_noise_std_seconds: f64,
    pub far_field_radius_m: f64,
    pub array_aperture_m: f64,
}

fn root_mean_square(values: &DVector<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.dot(values) / values.len() as f64).sqrt()
}

fn unit_or_none(vector: Option<Vector3<f64>>) -> Option<Vector3<f64>> {
    let vector = vector?;
    let norm = vector.norm();
    (norm > EPSILON).then(|| vector / norm)
}

/// `Some` = replace the solve with this plane-wave projection; `None` = keep
/// the near-field solution (not compact, or curvature evidence beats the
/// noise).
pub fn far_field_alternative_if_curvature_unsupported(
    inputs: &FarFieldGuardInputs<'_>,
) -> Option<FarFieldAlternative> {
    // `!(a <= b)` so a NaN aperture also leaves the solve untouched.
    if !(inputs.array_aperture_m <= COMPACT_ARRAY_MAX_APERTURE_M) {
        return None;
    }
    let near_offset = inputs.near_field_position_m - inputs.centroid_m;
    let mut best: Option<FarFieldAlternative> = None;
    // Order matters for exact ties: near-offset direction first, as in Python.
    for direction in [
        unit_or_none(Some(near_offset)),
        unit_or_none(inputs.plane_wave_direction),
    ]
    .into_iter()
    .flatten()
    {
        // A fixed far-field radius is a plane wave for a compact array; an
        // unbounded radial search can run off on an asymptotic cost.
        let projected_position_m = inputs.centroid_m + direction * inputs.far_field_radius_m;
        let residual_rms_s =
            root_mean_square(&inputs.residual_model.residuals(&projected_position_m));
        if !residual_rms_s.is_finite() {
            continue;
        }
        if best.map_or(true, |current| {
            residual_rms_s < current.residual_rms_seconds
        }) {
            best = Some(FarFieldAlternative {
                position_m: projected_position_m,
                residual_rms_seconds: residual_rms_s,
            });
        }
    }
    let best = best?;
    let curvature_evidence_s = best.residual_rms_seconds - inputs.near_field_residual_rms_seconds;
    if curvature_evidence_s > CURVATURE_EVIDENCE_NOISE_MULTIPLIER * inputs.timing_noise_std_seconds
    {
        return None;
    }
    Some(best)
}
