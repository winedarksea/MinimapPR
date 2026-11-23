//! Tunables for layout-general SRP-PHAT. Every operational constant lives
//! here so golden-bundle tuning never touches algorithm code.

/// Polar search grid centred on the layout centroid and aligned to its
/// principal horizontal axis (so the grid is invariant to layout rotation).
#[derive(Clone, Debug, PartialEq)]
pub struct PolarGridConfig {
    /// Radial and arc cell size inside `dense_radius_m`.
    pub dense_cell_size_m: f64,
    /// Outer edge of the uniformly dense disk.
    pub dense_radius_m: f64,
    /// Outer edge of the log-spaced rings; beyond it only the far-field ring.
    pub outer_radius_m: f64,
    /// Azimuth sectors per log ring. Log-ring radial ratio is derived from it
    /// (`1 + 2π / count`) so outer cells stay roughly square.
    pub log_ring_azimuth_count: usize,
    /// Fewest sectors in any dense ring; must be even so θ = 0 and θ = π are
    /// sector edges (the collinear mirror split relies on that).
    pub minimum_ring_azimuth_count: usize,
    /// Source height band above local ground (canopy prior, removes the
    /// below-ground mirror of a flat layout).
    pub source_height_above_ground_range_m: [f64; 2],
    /// Coarse height slabs across `source_height_above_ground_range_m`.
    pub height_layer_count: usize,
    /// Used to derive ground height when the caller gives none:
    /// ground = lowest sensor z − this.
    pub assumed_sensor_height_above_ground_m: f64,
    /// Far-field bearing ring: azimuth sectors and elevation band/slabs.
    pub far_field_azimuth_count: usize,
    pub far_field_elevation_range_deg: [f64; 2],
    pub far_field_elevation_layer_count: usize,
}

impl Default for PolarGridConfig {
    fn default() -> Self {
        Self {
            dense_cell_size_m: 2.0,
            dense_radius_m: 30.0,
            outer_radius_m: 200.0,
            log_ring_azimuth_count: 72,
            minimum_ring_azimuth_count: 4,
            source_height_above_ground_range_m: [0.0, 35.0],
            height_layer_count: 8,
            assumed_sensor_height_above_ground_m: 1.5,
            far_field_azimuth_count: 72,
            far_field_elevation_range_deg: [0.0, 40.0],
            far_field_elevation_layer_count: 4,
        }
    }
}

/// Band-limited GCC-PHAT curves and their max-pooled pyramid.
#[derive(Clone, Debug, PartialEq)]
pub struct PairCorrelationConfig {
    /// PHAT weighting band; bins outside are zeroed (a wide PHAT band can
    /// reverse bearings, see `project_localization_band_phat_ceiling`).
    pub band_hz: [f32; 2],
    /// Correlation upsampling factor (zero-padded spectrum) for sub-sample
    /// lags. 2x of a <= 10 kHz band at 48 kHz is ~5x oversampled, enough for
    /// Catmull-Rom interpolation; 4x doubled the inverse-FFT cost for no gain.
    pub upsample_factor: usize,
    /// Extra lag beyond the pair's baseline / c, in seconds (clock residuals).
    pub lag_margin_s: f64,
    /// Spectral bins below this fraction of the strongest in-band cross-power
    /// bin are dropped before whitening (they carry only noise phase).
    pub whitening_floor_fraction: f32,
}

impl Default for PairCorrelationConfig {
    fn default() -> Self {
        Self {
            band_hz: [500.0, 10_000.0],
            upsample_factor: 2,
            lag_margin_s: 2.0e-4,
            whitening_floor_fraction: 1.0e-6,
        }
    }
}

/// Robust per-pair scoring and the likelihood built from it.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustScoringConfig {
    /// Huber knee on the per-pair residual r = 1 − normalized correlation.
    /// Quadratic below (a well-shaped LM-compatible bowl near the optimum),
    /// linear above (one reflection-dominated pair cannot pull the maximum).
    pub huber_knee: f64,
    /// Total weight shared by one multi-mic node's intra-node pairs. Keeps a
    /// tetra node's 6 short-baseline pairs from outvoting inter-node pairs.
    pub intra_node_pair_group_weight: f64,
    /// Likelihood temperature, in units of mean per-pair robust loss:
    /// L ∝ exp(score / (temperature · Σw)). NOT calibrated: the W4-sim-gates
    /// fit (`minimappr-sim-gates` example `calibrate_region_parameters`,
    /// realistic ±10 cm / ±10 µs residuals) found no usable value. Coverage
    /// is 23 % at 0.002, and the 95 % fit (0.0126) only reaches it because the
    /// region collapses from a sub-m² dot into a whole-grid bearing fan
    /// (10⁴–10⁵ m² inside the layout): the Huber score's background floor
    /// swamps the peak once T is large enough to cover residual-shifted
    /// peaks. Kept at 0.002 (compact) until the scoring accounts for layout
    /// and clock σ; see docs/field_mode_progress/sim_gates.md.
    pub likelihood_temperature: f64,
    /// A pair reaches full weight when its peak z-score (peak·√K) is this
    /// many times the noise-only expected maximum √(2 ln L); at 1× it gets 0.
    /// Coherence-based, not level-based: faint but coherent pairs keep weight.
    pub full_weight_significance_ratio: f64,
    /// k in σ-aware scoring: each inter-node pair's curve is max-pooled over
    /// ±k·σ_pair (σ_pair from both nodes' declared layout and clock σ,
    /// `LayoutNode::pair_lag_std_s`) before the Huber loss. Under real
    /// residuals no single point puts every pair on its peak; pooling makes
    /// "within the declared error of its peak" cost nothing, so the score's
    /// peak plateau, not the temperature, absorbs layout/clock error
    /// (closeout D3). No effect when nodes declare σ = 0. 1.25 was chosen on
    /// the calibration training seeds from {1, 1.25, 1.5, 2}: at the
    /// temperature floor it alone covers 83 / 93 / 97 / 99 % of trials, and
    /// 1.25 is the one that leaves the temperature a real job.
    pub declared_sigma_pooling_multiplier: f64,
}

impl Default for RobustScoringConfig {
    fn default() -> Self {
        Self {
            huber_knee: 0.3,
            intra_node_pair_group_weight: 1.0,
            likelihood_temperature: 0.002,
            full_weight_significance_ratio: 3.0,
            declared_sigma_pooling_multiplier: 1.25,
        }
    }
}

/// Coarse-to-fine branch-and-bound refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementConfig {
    /// Best-bound coarse cells whose centres seed the incumbent.
    pub incumbent_seed_cell_count: usize,
    /// Stop once no open cell's bound beats the incumbent by more than this
    /// many likelihood temperatures.
    pub optimality_tolerance_temperatures: f64,
    /// Cell evaluation budget per search (bounds worst-case runtime; the
    /// result reports whether the bound closed before it ran out).
    pub maximum_cell_evaluations: usize,
    /// Cells smaller than this are not split further.
    pub final_cell_radius_m: f64,
    /// Search half-window around the seed-predicted lag when picking each
    /// pair's local peak for the LM handoff, in seconds.
    pub handoff_peak_search_half_window_s: f64,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            incumbent_seed_cell_count: 32,
            optimality_tolerance_temperatures: 0.005,
            maximum_cell_evaluations: 200_000,
            final_cell_radius_m: 0.02,
            handoff_peak_search_half_window_s: 1.0e-4,
        }
    }
}

/// Uncertainty region, bearing-fan fallback and mirror/collinearity checks.
#[derive(Clone, Debug, PartialEq)]
pub struct UncertaintyConfig {
    /// Probability mass the reported likelihood region holds.
    pub contour_coverage_fraction: f64,
    /// Fall back to a bearing fan when the region's radial span divided by
    /// the estimate's range exceeds this.
    pub bearing_fan_relative_range_span: f64,
    /// Also fall back when this fraction of region mass sits on the far ring.
    pub bearing_fan_far_field_mass_fraction: f64,
    /// Layout is collinear when sqrt(λ_min / λ_max) of the node horizontal
    /// scatter falls below this.
    pub collinear_aspect_ratio: f64,
    /// Mirror counts as resolved only when rigid-array (intra-node) pairs
    /// score the winning side above the mirror by more than this many
    /// likelihood temperatures.
    pub mirror_resolution_score_gap_temperatures: f64,
    /// Region cells whose upper-bound likelihood is below this hold no mass
    /// worth refining (e^-9 ≈ 1.2e-4).
    pub region_prune_likelihood: f64,
    /// Region columns are refined until every live cell's pair TDOA
    /// intervals are within this (the envelope then tracks the exact score),
    /// or until their footprint radius is below max(minimum, relative ×
    /// range from the centroid).
    pub region_lag_resolution_s: f64,
    pub region_minimum_cell_radius_m: f64,
    pub region_relative_cell_radius: f64,
    /// Envelope evaluations allowed for region refinement.
    pub maximum_region_cell_evaluations: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            contour_coverage_fraction: 0.95,
            bearing_fan_relative_range_span: 0.6,
            bearing_fan_far_field_mass_fraction: 0.25,
            collinear_aspect_ratio: 0.15,
            mirror_resolution_score_gap_temperatures: 6.0,
            region_prune_likelihood: 1.2e-4,
            region_lag_resolution_s: 2.5e-5,
            region_minimum_cell_radius_m: 0.02,
            region_relative_cell_radius: 0.005,
            maximum_region_cell_evaluations: 300_000,
        }
    }
}

/// Everything `localize_layout` needs besides the audio and layout.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutSrpConfig {
    pub sample_rate_hz: u32,
    pub sound_speed_mps: f64,
    /// Apply each node's clock-rate ratio when mapping its clip onto the
    /// island timeline (in-clip resampling). Off only for A/B tests.
    pub apply_clock_rate_correction: bool,
    /// Ground height (z, metres). None derives it from sensor heights.
    pub ground_height_m: Option<f64>,
    pub grid: PolarGridConfig,
    pub correlation: PairCorrelationConfig,
    pub scoring: RobustScoringConfig,
    pub refinement: RefinementConfig,
    pub uncertainty: UncertaintyConfig,
}

impl Default for LayoutSrpConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            sound_speed_mps: crate::dsp_math::SPEED_OF_SOUND_MPS_F64,
            apply_clock_rate_correction: true,
            ground_height_m: None,
            grid: PolarGridConfig::default(),
            correlation: PairCorrelationConfig::default(),
            scoring: RobustScoringConfig::default(),
            refinement: RefinementConfig::default(),
            uncertainty: UncertaintyConfig::default(),
        }
    }
}
