//! SRP-PHAT for arbitrary node layouts (tetra arrays + single-mic phones
//! spread over tens of metres), the Field Mode generalization of the
//! tetra-only `srp_phat.rs` (which stays unchanged for the sidecar).
//!
//! Pipeline per analysis window:
//! 1. `clock_resampling`: offset + rate map every clip onto island time.
//! 2. `pair_correlation`: band-limited GCC-PHAT coherence curves + max pyramid.
//! 3. `polar_grid`: coarse polar cells (dense disk, log rings, far-field ring)
//!    scored by max-over-TDOA-interval with `robust_scoring` (Huber per pair).
//! 4. `refinement`: coarse-to-fine branch-and-bound → converged cell → `lm_handoff` seed.
//! 5. `likelihood_region`: height-marginalized contour with temperature,
//!    bearing-fan fallback when range is unobservable.
//! 6. `layout_geometry`: collinearity (dual mirror candidates) and GDOP.

pub mod clock_resampling;
pub mod config;
pub mod layout_geometry;
pub mod layout_input;
pub mod likelihood_region;
pub mod lm_handoff;
pub mod pair_correlation;
pub mod polar_grid;
pub mod refinement;
pub mod region_refinement;
pub mod robust_scoring;
pub mod searcher;

pub use clock_resampling::NodeClockModel;
pub use config::LayoutSrpConfig;
pub use layout_geometry::HorizontalDilution;
pub use layout_input::{IslandAnalysisWindow, LayoutNode, LayoutSrpError, NodeClip};
pub use likelihood_region::{BearingFan, LikelihoodRegion, UncertaintyDisplay};
pub use lm_handoff::{LmSeed, PairTdoaObservation, TdoaSeedRefiner};
pub use searcher::LayoutSrpSearcher;

/// One converged SRP estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceEstimate {
    /// World ENU. Far-field: centroid + direction × grid outer radius.
    pub position_m: [f64; 3],
    pub is_far_field: bool,
    /// From the layout centroid, CCW from +x (east).
    pub world_azimuth_rad: f64,
    pub horizontal_range_m: f64,
    /// −Σ w ρ at the estimate (0 = every pair at its peak).
    pub srp_score: f64,
    /// `srp_score / Σw`, comparable across layouts.
    pub normalized_srp_score: f64,
    /// Branch-and-bound cost and whether it closed within tolerance.
    pub cells_evaluated: usize,
    pub is_certified_within_tolerance: bool,
}

/// Both sides of a collinear layout's principal axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirrorCandidates {
    pub primary: SourceEstimate,
    pub mirror: SourceEstimate,
    /// (primary − mirror score) in likelihood temperatures, all pairs.
    pub score_gap_temperatures: f64,
    /// The gap from rigid-array (intra-node) pairs only, in temperatures of
    /// their own weight (T · Σw_array); the only part trusted to pick a side.
    pub array_score_gap_temperatures: f64,
    /// True only when array pairs favour `primary` by more than the
    /// configured gap. False → report both; never show a confident side.
    pub is_resolved: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutGeometryDiagnostics {
    pub horizontal_aspect_ratio: f64,
    pub is_collinear: bool,
    /// 2 × largest node distance from the centroid.
    pub layout_aperture_m: f64,
    pub dilution_at_estimate: HorizontalDilution,
}

#[derive(Clone, Debug)]
pub struct LayoutSrpResult {
    pub estimate: SourceEstimate,
    pub lm_seed: LmSeed,
    /// Present only for collinear layouts.
    pub mirror: Option<MirrorCandidates>,
    pub geometry: LayoutGeometryDiagnostics,
    pub region: LikelihoodRegion,
    pub display: UncertaintyDisplay,
    /// Search frame; bearings and ranges are from `frame.origin_xy_m`.
    pub frame: polar_grid::LayoutFrame,
    pub coherent_pair_count: usize,
    pub total_pair_weight: f64,
}

/// One-shot convenience: build the searcher and localize one window.
pub fn localize_layout(
    nodes: &[LayoutNode],
    clips: &[NodeClip<'_>],
    window: IslandAnalysisWindow,
    config: &LayoutSrpConfig,
) -> Result<LayoutSrpResult, LayoutSrpError> {
    LayoutSrpSearcher::new(nodes, config)?.localize(clips, window)
}
