//! What every bake-off candidate returns, so scoring treats them alike.

/// The localizers compared. `ProductionGccPhatLsq` and `OnsetClusteringShadow`
/// are the adoption question; the others are diagnostics that come for free
/// (the sidecar's own SRP estimate from the same call) or isolate a cause in
/// production's front-end: the event-span lag cap lifted; and the cap lifted
/// plus a window taper (against the rectangular-cut zero-lag bias) plus a
/// band reaching bird calls (production stops at 3.43 kHz). The decision
/// guard compares onset clustering with the stronger of those two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BakeoffCandidate {
    ProductionGccPhatLsq,
    ProductionSidecarSrp,
    ProductionUncappedWindow,
    ProductionRepairedFrontEnd,
    OnsetClusteringShadow,
}

impl BakeoffCandidate {
    pub const ALL: [BakeoffCandidate; 5] =
        [Self::ProductionGccPhatLsq, Self::ProductionSidecarSrp, Self::ProductionUncappedWindow, Self::ProductionRepairedFrontEnd, Self::OnsetClusteringShadow];

    pub fn label(self) -> &'static str {
        match self {
            Self::ProductionGccPhatLsq => "production_gcc_phat_lsq",
            Self::ProductionSidecarSrp => "production_sidecar_srp",
            Self::ProductionUncappedWindow => "production_uncapped_window",
            Self::ProductionRepairedFrontEnd => "production_repaired_front_end",
            Self::OnsetClusteringShadow => "onset_clustering_shadow",
        }
    }
}

/// One published position. `attribution_cues` say which audio produced it
/// (node, stamped time): the trigger for production, the led onsets for
/// onset clustering. Scoring maps cues to the call sounding there, which is
/// how a classifier label would end up on this position.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFix {
    pub position_m: [f64; 3],
    pub covariance_m2: Option<[[f64; 3]; 3]>,
    pub range_projection_mode: Option<String>,
    pub attribution_cues: Vec<(usize, f64)>,
}

/// One candidate invocation (one solve) and its cost.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CandidateSolveOutcome {
    pub fixes: Vec<CandidateFix>,
    pub wall_time_s: f64,
    pub failure_reason: Option<String>,
    pub gate_rejections: Vec<String>,
}

pub fn matrix3_to_array(matrix: &minimappr_core::tdoa_solver::nalgebra::Matrix3<f64>) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

/// The server's reportability gates every localized handoff passes
/// (`localized_render_localization.py::gate_handoff_localization`): a fix
/// whose range is unobservable is a bearing ray, not a position, and one
/// beyond `localization_max_range_m` of the contributing centroid (or the
/// origin backstop) is demoted to omni. Applied to every candidate alike.
pub fn reportability_gate_rejection(fix: &CandidateFix, contributing_centroid_m: &[f64; 3], config: &ReportabilityGateConfig) -> Option<String> {
    if minimappr_core::range_projection::is_unobservable(fix.range_projection_mode.as_deref()) {
        return Some(format!("gated: range unobservable ({})", fix.range_projection_mode.as_deref().unwrap_or("")));
    }
    let position = fix.position_m;
    if !position.iter().all(|value| value.is_finite()) {
        return Some("gated: non-finite position".to_string());
    }
    let norm = position.iter().map(|value| value * value).sum::<f64>().sqrt();
    let from_centroid = crate::bakeoff_scenes::distance_m(&position, contributing_centroid_m);
    let beyond_origin_backstop = config.origin_backstop_m.is_some_and(|backstop_m| norm > backstop_m);
    (beyond_origin_backstop || from_centroid > config.max_range_from_centroid_m).then(|| "gated: out of range".to_string())
}

/// Server defaults: `localization_max_range_m` (centroid-relative) and the
/// range-proportional covariance ceiling (`localization_std_range_factor`,
/// `localization_position_std_floor_m`, `localization_max_position_std_m`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReportabilityGateConfig {
    pub max_range_from_centroid_m: f64,
    /// The old hard-coded `_SANITY_GATE_ORIGIN_BACKSTOP_M` (removed from the
    /// server in 7191046); `None` = no origin-relative gate, as the app runs.
    pub origin_backstop_m: Option<f64>,
    pub std_ceiling_range_factor: f64,
    pub std_ceiling_floor_m: f64,
    pub std_ceiling_absolute_m: f64,
}

impl Default for ReportabilityGateConfig {
    fn default() -> Self {
        Self {
            max_range_from_centroid_m: 6_000.0,
            origin_backstop_m: None,
            std_ceiling_range_factor: 1.0,
            std_ceiling_floor_m: 30.0,
            std_ceiling_absolute_m: 5_000.0,
        }
    }
}

/// `localization_uncertainty.py::clamp_covariance_eigenvalues_range_proportional`:
/// every eigenvalue capped at the squared range-proportional std ceiling.
fn clamp_covariance_to_range_ceiling(covariance: &[[f64; 3]; 3], range_m: f64, config: &ReportabilityGateConfig) -> [[f64; 3]; 3] {
    use minimappr_core::tdoa_solver::nalgebra::Matrix3;
    let ceiling_std_m = (config.std_ceiling_range_factor * range_m).max(config.std_ceiling_floor_m).min(config.std_ceiling_absolute_m);
    let matrix = Matrix3::from_fn(|row, column| 0.5 * (covariance[row][column] + covariance[column][row]));
    let mut eigen = matrix.symmetric_eigen();
    for value in eigen.eigenvalues.iter_mut() {
        *value = value.min(ceiling_std_m * ceiling_std_m);
    }
    matrix3_to_array(&eigen.recompose())
}

/// Drop gated fixes from an outcome (recording why) and clamp the rest's covariance.
pub fn apply_reportability_gates(outcome: &mut CandidateSolveOutcome, contributing_centroid_m: &[f64; 3], config: &ReportabilityGateConfig) {
    let mut kept = Vec::new();
    for fix in outcome.fixes.drain(..) {
        match reportability_gate_rejection(&fix, contributing_centroid_m, config) {
            Some(reason) => outcome.gate_rejections.push(reason),
            None => {
                let range_m = crate::bakeoff_scenes::distance_m(&fix.position_m, contributing_centroid_m);
                let covariance_m2 = fix.covariance_m2.map(|covariance| clamp_covariance_to_range_ceiling(&covariance, range_m, config));
                kept.push(CandidateFix { covariance_m2, ..fix });
            }
        }
    }
    outcome.fixes = kept;
}
