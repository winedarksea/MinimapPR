//! Every gate threshold and trial-generation constant, grouped by gate.
//! Gate values come from `docs/field_mode_test_strategy.md` §What to assert;
//! where the strategy gives no number, the comment says where it came from.
//! Calibrated *production* parameters (SRP likelihood temperature, mesh
//! contour σ multiplier) live in minimappr-core's own configs, so these
//! gates test what ships.

/// Realistic residual errors a converged field session leaves behind.
#[derive(Debug, Clone, PartialEq)]
pub struct RealisticPerturbationConfig {
    /// Clock-filter offset residual per node (1σ).
    pub clock_residual_std_s: f64,
    /// Layout-solve position residual per node and horizontal axis (1σ).
    pub layout_position_std_m: f64,
    /// Phone height is not measured by the layout (mesh `assumed_phone_height_sigma_m`).
    pub phone_height_std_m: f64,
    /// Phone GPS horizontal error per axis (1σ) under canopy.
    pub gps_error_std_m: f64,
    /// Correlation-peak pick error of a call arrival at good SNR (mesh world default).
    pub arrival_pick_std_s: f64,
}

impl Default for RealisticPerturbationConfig {
    fn default() -> Self {
        Self {
            clock_residual_std_s: 10e-6,
            layout_position_std_m: 0.10,
            phone_height_std_m: 0.10,
            gps_error_std_m: 6.0,
            arrival_pick_std_s: 20e-6,
        }
    }
}

/// Random phone layouts and source placement.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialGeometryConfig {
    pub node_count_range: [usize; 2],
    pub aperture_range_m: [f64; 2],
    /// Reject layouts flatter than this √(λmin/λmax) (collinear trails are
    /// their own case; SRP flags them below 0.15).
    pub minimum_layout_aspect_ratio: f64,
    /// No two phones closer than this fraction of the aperture.
    pub minimum_node_separation_fraction: f64,
    pub phone_height_range_m: [f64; 2],
    /// Inside sources fall within this fraction of the layout radius.
    pub inside_source_radius_fraction: f64,
    pub source_height_range_m: [f64; 2],
    pub layout_rejection_attempts: u64,
}

impl Default for TrialGeometryConfig {
    fn default() -> Self {
        Self {
            node_count_range: [4, 8],
            aperture_range_m: [15.0, 40.0],
            minimum_layout_aspect_ratio: 0.35,
            minimum_node_separation_fraction: 0.15,
            phone_height_range_m: [1.2, 1.6],
            inside_source_radius_fraction: 0.6,
            source_height_range_m: [1.0, 20.0],
            layout_rejection_attempts: 200,
        }
    }
}

/// Monte Carlo accuracy (test strategy row "Accuracy, simulated Monte Carlo").
#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyGateConfig {
    pub inside_median_error_m: f64,
    pub inside_p95_error_m: f64,
    pub outside_gate_range_m: f64,
    pub outside_p95_bearing_error_deg: f64,
    /// Further ranges are reported (not gated): bearing only.
    pub reported_outside_ranges_m: [f64; 2],
}

impl Default for AccuracyGateConfig {
    fn default() -> Self {
        Self {
            inside_median_error_m: 1.5,
            inside_p95_error_m: 3.0,
            outside_gate_range_m: 50.0,
            outside_p95_bearing_error_deg: 5.0,
            reported_outside_ranges_m: [100.0, 200.0],
        }
    }
}

/// Likelihood-contour coverage (row "Uncertainty calibration").
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageGateConfig {
    pub minimum_coverage: f64,
    pub maximum_coverage: f64,
    /// Calibration aims inside the band, nearer its centre than 0.95 would be
    /// on held-out data (the band is asymmetric about 0.95).
    pub calibration_target_coverage: f64,
    /// Below this many trials an aggregate is printed, not gated.
    pub minimum_trials_for_aggregate_gate: usize,
    /// Sharpness: coverage reached by smearing is not calibration. Inside
    /// the layout the drawn region must be a contour (core contour test's
    /// rule) in all but this fraction of trials...
    pub maximum_inside_fan_fraction: f64,
    /// ...and its median area stay below a disk of the P95 accuracy gate.
    pub maximum_inside_median_region_area_m2: f64,
}

impl Default for CoverageGateConfig {
    fn default() -> Self {
        Self {
            minimum_coverage: 0.90,
            maximum_coverage: 0.98,
            calibration_target_coverage: 0.945,
            minimum_trials_for_aggregate_gate: 20,
            maximum_inside_fan_fraction: 0.10,
            maximum_inside_median_region_area_m2: std::f64::consts::PI * 3.0 * 3.0,
        }
    }
}

/// Geometric invariance (row "Geometric invariance": within 1 mm).
#[derive(Debug, Clone, PartialEq)]
pub struct InvarianceGateConfig {
    pub maximum_position_discrepancy_m: f64,
    /// Relative tolerance on covariance eigenvalues after transformation.
    pub maximum_relative_sigma_discrepancy: f64,
    pub maximum_translation_m: f64,
}

impl Default for InvarianceGateConfig {
    fn default() -> Self {
        Self { maximum_position_discrepancy_m: 1e-3, maximum_relative_sigma_discrepancy: 1e-3, maximum_translation_m: 500.0 }
    }
}

/// Chirality (row "Chirality"): never a confident mirror.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiralityGateConfig {
    /// The mesh invariant's flag level (`FLAGGED_CHIRALITY_CONFIDENCE`).
    pub confident_threshold: f64,
    /// Confidence-calibration check: in each bin the wrong rate may exceed
    /// the bin's implied error (1 − lower edge) by at most this many
    /// binomial standard errors.
    pub calibration_slack_standard_errors: f64,
    pub calibration_bin_lower_edges: [f64; 3],
    /// Layout radius range for GPS-only chirality (mesh mirrored_layout uses 5 m).
    pub layout_radius_range_m: [f64; 2],
    pub chirp_range_std_m: f64,
}

impl Default for ChiralityGateConfig {
    fn default() -> Self {
        Self {
            confident_threshold: 0.99,
            calibration_slack_standard_errors: 3.0,
            calibration_bin_lower_edges: [0.6, 0.9, 0.99],
            layout_radius_range_m: [4.0, 15.0],
            chirp_range_std_m: 0.05,
        }
    }
}

/// Graceful degradation (row "Graceful degradation").
#[derive(Debug, Clone, PartialEq)]
pub struct DegradationGateConfig {
    pub full_node_count: usize,
    /// A pocketed phone's arrival is late by a multipath detour in this range.
    pub pocketed_delay_range_s: [f64; 2],
    /// Statistical checks use the LM covariance's Gaussian 95 % ellipse
    /// (declared σ equals true σ in these trials; NEES is gated in the
    /// accuracy gate). Catastrophic: Mahalanobis distance beyond this (σ).
    pub catastrophic_mahalanobis_sigma: f64,
    /// A pocket delay is decisive (robust loss must reject it) beyond this
    /// many declared pair σ; smaller delays are indistinguishable from
    /// layout error and only count toward the aggregate rate.
    pub decisive_outlier_pair_sigma: f64,
    /// σ may shrink by at most this fraction when a node is removed
    /// (linearization point moves; information itself cannot grow)...
    pub maximum_sigma_shrink_fraction: f64,
    /// ...in at most this fraction of removal steps.
    pub maximum_sigma_shrink_step_fraction: f64,
    /// Fraction of muted-node steps allowed outside the previous region.
    pub maximum_outside_previous_region_fraction: f64,
}

impl Default for DegradationGateConfig {
    fn default() -> Self {
        Self {
            full_node_count: 7,
            pocketed_delay_range_s: [0.5e-3, 5e-3],
            catastrophic_mahalanobis_sigma: 4.0,
            decisive_outlier_pair_sigma: 8.0,
            maximum_sigma_shrink_fraction: 0.10,
            maximum_sigma_shrink_step_fraction: 0.05,
            maximum_outside_previous_region_fraction: 0.10,
        }
    }
}

/// Map anchor stability (row "Map anchor stability").
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorGateConfig {
    pub phone_count: usize,
    pub layout_radius_m: f64,
    pub gps_fix_interval_s: f64,
    pub session_duration_s: f64,
    pub jump_time_s: f64,
    pub jump_distance_m: f64,
    /// A multipath jump lasts this long (then the phone's fix recovers);
    /// `anchor_persistent_jump` gates one that never recovers.
    pub jump_duration_s: f64,
    pub maximum_jump_response_m: f64,
    /// Stillness clamp: anchor movement from `settle_time_s` to the jump
    /// (strategy gives 1 m for the jump; same bound here). Earlier movement
    /// is the filter converging (≈ σ_fit·√(1/n₁ − 1/n₂)), not drift.
    pub maximum_still_drift_m: f64,
    pub settle_time_s: f64,
    /// Still-window movement allowed: max(maximum_still_drift_m, this × the
    /// filter's own displacement σ at the window start).
    pub still_drift_reported_sigma_multiple: f64,
    /// Multipath bias correlation (still phones see a slowly varying bias).
    pub gps_bias_fraction_of_error: f64,
}

impl Default for AnchorGateConfig {
    fn default() -> Self {
        Self {
            phone_count: 5,
            layout_radius_m: 10.0,
            gps_fix_interval_s: 5.0,
            session_duration_s: 600.0,
            jump_time_s: 300.0,
            jump_distance_m: 20.0,
            jump_duration_s: 30.0,
            maximum_jump_response_m: 1.0,
            maximum_still_drift_m: 1.0,
            settle_time_s: 150.0,
            still_drift_reported_sigma_multiple: 3.0,
            gps_bias_fraction_of_error: 0.8,
        }
    }
}

/// Loop consistency (row "Loop consistency").
#[derive(Debug, Clone, PartialEq)]
pub struct LoopGateConfig {
    pub sensor_count_range: [usize; 2],
    pub pair_timing_std_s: f64,
    /// Reflection lock adds a detour delay in this range (≥ 10σ, decisive).
    pub reflection_delay_range_s: [f64; 2],
    /// Clean sets lose a pair only through the closure tolerance's tail:
    /// the allowed fraction is the union bound Σ C(n,3)·P(|Z| > k)/trials
    /// plus this many binomial standard errors.
    pub clean_false_rejection_slack_standard_errors: f64,
}

impl Default for LoopGateConfig {
    fn default() -> Self {
        Self {
            sensor_count_range: [4, 7],
            pair_timing_std_s: 20e-6,
            reflection_delay_range_s: [0.3e-3, 3e-3],
            clean_false_rejection_slack_standard_errors: 3.0,
        }
    }
}

/// Resource bounds (row "Resource bounds"). Budgets are about 5× the
/// measured M2 cost under the optimised test profile, so they catch
/// regressions of complexity, not machine noise; the SRP budget is instead
/// a real-time target (see `maximum_srp_solve_s`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceGateConfig {
    pub maximum_arrival_solve_s: f64,
    /// One call every ~3 s on a phone ~3× slower than an M2 ⇒ 1 s on M2.
    /// Measured 0.4–2 s with realistic residuals (the criterion bench's
    /// 54 ms uses true clocks and positions); see sim_gates.md.
    pub maximum_srp_solve_s: f64,
    pub maximum_layout_solve_s: f64,
    pub maximum_session_wall_s: f64,
    pub maximum_srp_peak_heap_bytes: usize,
    pub maximum_arrival_peak_heap_bytes: usize,
    pub maximum_session_peak_heap_bytes: usize,
    /// Doubling the session length may grow peak heap by at most this
    /// factor (the event log grows linearly; everything else must not).
    pub maximum_session_heap_growth_factor: f64,
}

impl Default for ResourceGateConfig {
    fn default() -> Self {
        Self {
            maximum_arrival_solve_s: 0.05,
            maximum_srp_solve_s: 1.0,
            maximum_layout_solve_s: 0.01,
            maximum_session_wall_s: 5.0,
            maximum_srp_peak_heap_bytes: 256 << 20,
            maximum_arrival_peak_heap_bytes: 4 << 20,
            maximum_session_peak_heap_bytes: 256 << 20,
            maximum_session_heap_growth_factor: 2.0,
        }
    }
}

/// Everything above, so a nightly run can override one block.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SimGateConfig {
    pub perturbation: RealisticPerturbationConfig,
    pub geometry: TrialGeometryConfig,
    pub accuracy: AccuracyGateConfig,
    pub coverage: CoverageGateConfig,
    pub invariance: InvarianceGateConfig,
    pub chirality: ChiralityGateConfig,
    pub degradation: DegradationGateConfig,
    pub anchor: AnchorGateConfig,
    pub loop_consistency: LoopGateConfig,
    pub resources: ResourceGateConfig,
}
