//! Every tunable of the point-only localizer bake-off (E8), in one place.
//!
//! Production numbers mirror the server and sidecar defaults
//! (`minimappr/config.py`, `minimappr/core/localization_overhaul_config.py`,
//! `actors/dsp_compute_srp_solve.rs`) so the production candidate runs as a
//! point network runs today. Scene numbers describe the simulated sites. The
//! decision rule is fixed here, before any held-out seed is scored.

use minimappr_core::localization_band_policy::LocalizationBandPolicy;

/// Clock classes a point node reports, with the declared 1-sigma offset the
/// server assigns each (`localization_clock_class_uncertainty_seconds`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodeClockClass {
    GpsLocked,
    NtpDisciplined,
    FreeRunning,
}

impl NodeClockClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::GpsLocked => "gps_locked",
            Self::NtpDisciplined => "ntp_disciplined",
            Self::FreeRunning => "free_running",
        }
    }

    /// Server default class uncertainty (`localization_overhaul_config.py`).
    pub fn declared_offset_std_s(self) -> f64 {
        match self {
            Self::GpsLocked => 1.0e-6,
            Self::NtpDisciplined => 2.0e-3,
            Self::FreeRunning => 5.0e-2,
        }
    }
}

/// Layer 1 bird call shapes. Tonal whistles matter: their narrow band makes
/// GCC-PHAT pick a cycle-skipped lag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BirdWaveform {
    FmSweep,
    TonalWhistle,
    Trill,
    Harsh,
}

impl BirdWaveform {
    pub const ALL: [BirdWaveform; 4] = [Self::FmSweep, Self::TonalWhistle, Self::Trill, Self::Harsh];

    pub fn label(self) -> &'static str {
        match self {
            Self::FmSweep => "fm_sweep",
            Self::TonalWhistle => "tonal_whistle",
            Self::Trill => "trill",
            Self::Harsh => "harsh",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneDrawConfig {
    /// Bird call shapes drawn uniformly.
    pub bird_waveforms: Vec<BirdWaveform>,
    pub sample_rate_hz: u32,
    pub node_count_range: [usize; 2],
    /// Site aperture is drawn log-uniformly in this range (metres).
    pub site_spread_range_m: [f64; 2],
    /// Nodes closer than this fraction of the spread are re-drawn.
    pub minimum_node_separation_spread_fraction: f64,
    pub node_height_m: [f64; 2],
    /// Birds sit inside this fraction of the site radius.
    pub source_radius_spread_fraction: f64,
    pub bird_height_m: [f64; 2],
    pub continuous_source_height_m: [f64; 2],
    pub bird_level_db_spl_at_1m: [f64; 2],
    pub continuous_source_level_db_spl_at_1m: [f64; 2],
    pub ambient_noise_db_spl: [f64; 2],
    pub bird_call_duration_s: [f64; 2],
    pub continuous_source_duration_s: f64,
    /// Soft fade-in of a continuous source (no sharp onset, like a drone flying in).
    pub continuous_source_edge_ramp_s: f64,
    pub continuous_source_fundamental_hz: [f64; 2],
    /// Second caller starts this long after the first (simultaneous callers).
    pub second_caller_delay_s: [f64; 2],
    pub second_caller_minimum_separation_spread_fraction: f64,
    /// Event-kind mix: single bird, two birds; the rest are continuous sources.
    pub single_bird_probability: f64,
    pub two_bird_probability: f64,
    /// Clock regime mix: all GPS, all NTP; the rest mix GPS and NTP nodes.
    pub all_gps_probability: f64,
    pub all_ntp_probability: f64,
    /// In a mixed-clock site, chance that one node is free-running.
    pub mixed_site_free_running_node_probability: f64,
    pub ntp_drift_ppm_max: f64,
    pub free_running_drift_ppm_max: f64,
    /// Chance a site has hand-measured nodes at all, and then the chance each
    /// of its nodes was hand-measured (otherwise surveyed).
    pub hand_measured_site_probability: f64,
    pub hand_measured_node_probability: f64,
    pub hand_measured_stated_std_m: [f64; 2],
    pub surveyed_position_std_m: f64,
    /// Noise-only audio before the event (noise floor, onset pre-roll).
    pub lead_in_noise_s: f64,
    pub tail_after_last_arrival_s: f64,
}

impl Default for SceneDrawConfig {
    fn default() -> Self {
        Self {
            bird_waveforms: BirdWaveform::ALL.to_vec(),
            sample_rate_hz: 24_000,
            node_count_range: [3, 8],
            site_spread_range_m: [20.0, 1_000.0],
            minimum_node_separation_spread_fraction: 0.15,
            node_height_m: [1.2, 1.8],
            source_radius_spread_fraction: 0.6,
            bird_height_m: [2.0, 15.0],
            continuous_source_height_m: [10.0, 60.0],
            bird_level_db_spl_at_1m: [85.0, 100.0],
            continuous_source_level_db_spl_at_1m: [80.0, 92.0],
            ambient_noise_db_spl: [30.0, 40.0],
            bird_call_duration_s: [0.2, 0.5],
            continuous_source_duration_s: 3.0,
            continuous_source_edge_ramp_s: 0.4,
            continuous_source_fundamental_hz: [150.0, 300.0],
            second_caller_delay_s: [0.0, 0.25],
            second_caller_minimum_separation_spread_fraction: 0.2,
            single_bird_probability: 0.5,
            two_bird_probability: 0.3,
            all_gps_probability: 0.4,
            all_ntp_probability: 0.4,
            mixed_site_free_running_node_probability: 0.5,
            ntp_drift_ppm_max: 10.0,
            free_running_drift_ppm_max: 50.0,
            hand_measured_site_probability: 0.4,
            hand_measured_node_probability: 0.6,
            hand_measured_stated_std_m: [0.3, 1.5],
            surveyed_position_std_m: 0.1,
            lead_in_noise_s: 1.0,
            tail_after_last_arrival_s: 0.5,
        }
    }
}

/// The live trigger, modelled at cluster level: the first node whose frame
/// RMS stands `relative_db` over its noise floor starts a solve; further
/// crossings inside the cooldown do not.
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerModelConfig {
    pub frame_seconds: f64,
    /// `localization_trigger_noise_floor_relative_db` default.
    pub relative_db: f64,
    /// `trigger_cooldown_seconds` default.
    pub cooldown_seconds: f64,
}

impl Default for TriggerModelConfig {
    fn default() -> Self {
        Self {
            frame_seconds: 0.032,
            relative_db: 8.0,
            cooldown_seconds: 0.8,
        }
    }
}

/// The production point-network path: sidecar GCC-PHAT pair front-end and
/// SRP (`estimate_tetrahedral_steering_with_solve_constraints`), then the
/// server's default `python_cartesian` re-solve of the pair TDOAs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionCandidateConfig {
    /// `localization_window_seconds`: the node-local event window.
    pub node_local_window_seconds: f64,
    /// `localization_coverage_total_cap_seconds`: the event span's lag cap.
    pub coverage_total_cap_seconds: f64,
    /// Shipped per-pair band and taper; every point-network pair is cross-node.
    pub localization_band_policy: LocalizationBandPolicy,
    pub grid_resolution_m: f32,
    pub search_padding_m: f32,
    pub interpolation_factor: u32,
    pub far_field_default_range_m: f64,
    pub far_field_max_range_m: f32,
    /// `localization_solver_floor_relative_db`.
    pub solver_floor_relative_db: f32,
    /// `localization_node_bearing_strength`.
    pub node_bearing_strength: f64,
    /// `localization_obs_factor_bearing_floor`.
    pub bearing_observed_obs_floor: f64,
    /// `rust_tdoa_solve.py`: fewer sensors or pairs than this returns no re-solve.
    pub resolve_minimum_sensor_count: usize,
    pub resolve_minimum_pair_count: usize,
    /// `pair_timing_uncertainty.py::MINIMUM_PAIR_MEASUREMENT_QUALITY`.
    pub minimum_pair_measurement_quality: f64,
    /// Repaired-front-end diagnostic only: Tukey taper fraction of the event
    /// window, and a GCC-PHAT band reaching the bird-call range.
    pub diagnostic_window_taper_fraction: f64,
    pub diagnostic_band_hz: [f32; 2],
}

impl Default for ProductionCandidateConfig {
    fn default() -> Self {
        Self {
            node_local_window_seconds: 0.08,
            coverage_total_cap_seconds: 1.0,
            localization_band_policy: LocalizationBandPolicy::default(),
            grid_resolution_m: 0.5,
            search_padding_m: 2.0,
            interpolation_factor: 4,
            far_field_default_range_m: 50.0,
            far_field_max_range_m: 1_000.0,
            solver_floor_relative_db: 3.0,
            node_bearing_strength: 1.0,
            bearing_observed_obs_floor: 1.0,
            resolve_minimum_sensor_count: 4,
            resolve_minimum_pair_count: 3,
            minimum_pair_measurement_quality: 0.01,
            diagnostic_window_taper_fraction: 0.2,
            diagnostic_band_hz: [50.0, 11_000.0],
        }
    }
}

/// The shadow onset-clustering path (`onset_clustering::localize_onset_clusters`)
/// as the sidecar would run it live, with the R5 clock policy carried into
/// the published covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetCandidateConfig {
    /// Noise-only audio kept before the trigger beyond the layout lag.
    pub window_pre_roll_s: f64,
    /// Audio kept after the far node's latest possible arrival.
    pub window_post_roll_s: f64,
    pub interpolation_factor: usize,
    pub far_field_default_range_m: f64,
    /// Tuned on the tuning seeds: the SRP grid (and range profile) reach at
    /// least this multiple of the layout aperture; 0 keeps the core default.
    pub grid_range_aperture_multiple: f64,
    /// Same noise-onset gate as `OnsetClipLocalizerConfig::minimum_loudest_node_band_snr_db`.
    pub minimum_loudest_node_band_snr_db: f64,
    pub minimum_contributing_nodes: usize,
}

impl Default for OnsetCandidateConfig {
    fn default() -> Self {
        Self {
            window_pre_roll_s: 0.5,
            window_post_roll_s: 1.0,
            interpolation_factor: 4,
            far_field_default_range_m: 50.0,
            grid_range_aperture_multiple: 0.0,
            minimum_loudest_node_band_snr_db: 6.0,
            minimum_contributing_nodes: 3,
        }
    }
}

/// Candidate grid multiples tried on the tuning seeds (0 = core default).
pub const ONSET_GRID_RANGE_MULTIPLES_TRIED: [f64; 2] = [0.0, 1.5];

#[derive(Debug, Clone, PartialEq)]
pub struct ScoringConfig {
    /// A fix "localizes" its call when the horizontal error is under
    /// max(floor, fraction × site aperture).
    pub recall_gate_floor_m: f64,
    pub recall_gate_aperture_fraction: f64,
    /// Stated confidence of the covariance coverage check (horizontal, 2 dof).
    pub coverage_confidence_level: f64,
    /// chi-square(2 dof) quantile at `coverage_confidence_level`.
    pub coverage_chi_square_threshold: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            recall_gate_floor_m: 5.0,
            recall_gate_aperture_fraction: 0.1,
            coverage_confidence_level: 0.95,
            coverage_chi_square_threshold: 5.991_464_547_107_979,
        }
    }
}

/// The pre-registered adoption rule (see `bakeoff_decision.rs` and
/// `docs/point_network_bakeoff.md`). Written before held-out results existed.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRuleConfig {
    /// Onset recall minus production recall must reach this (absolute)…
    pub minimum_recall_gain: f64,
    /// …and the paired-bootstrap lower bound at this one-sided level must be > 0.
    pub bootstrap_confidence_level: f64,
    pub bootstrap_resample_count: usize,
    pub bootstrap_seed: u64,
    /// Matched-call p90 error: onset / production at most this.
    pub maximum_matched_p90_error_ratio: f64,
    /// Onset coverage may trail production's by at most this.
    pub maximum_coverage_shortfall: f64,
    /// Two-caller attribution error rate may exceed production's by at most this.
    pub maximum_attribution_error_increase: f64,
}

impl Default for DecisionRuleConfig {
    fn default() -> Self {
        Self {
            minimum_recall_gain: 0.05,
            bootstrap_confidence_level: 0.95,
            bootstrap_resample_count: 2_000,
            bootstrap_seed: 0x00E8_B00F,
            maximum_matched_p90_error_ratio: 1.10,
            maximum_coverage_shortfall: 0.02,
            maximum_attribution_error_increase: 0.05,
        }
    }
}

/// Seeds: tuning seeds only choose the onset grid multiple; the decision
/// reads held-out seeds alone.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedSplit {
    pub tuning_seeds: Vec<u64>,
    pub held_out_seeds: Vec<u64>,
}

/// Held-out seeds start here so they never collide with tuning seeds.
/// Seeds 1_000_001–1_000_012 were looked at while building the harness, so
/// the held-out range starts above them.
pub const HELD_OUT_SEED_BASE: u64 = 2_000_000;
/// Seeds whose results were seen while building the harness: never held out.
pub const LOOKED_AT_SEEDS: std::ops::RangeInclusive<u64> = 1_000_001..=1_000_012;

impl SeedSplit {
    pub fn counts(tuning_count: u64, held_out_count: u64) -> Self {
        Self::with_held_out_base(tuning_count, held_out_count, HELD_OUT_SEED_BASE)
    }

    /// A nightly run may draw fresh held-out seeds from another base.
    ///
    /// Panics when the held-out range overlaps the tuning seeds or the
    /// looked-at seeds: a decision read from seeds already seen is not held out.
    pub fn with_held_out_base(tuning_count: u64, held_out_count: u64, held_out_base: u64) -> Self {
        let held_out_seeds: Vec<u64> = (held_out_base + 1..=held_out_base + held_out_count).collect();
        if let (Some(&first), Some(&last)) = (held_out_seeds.first(), held_out_seeds.last()) {
            assert!(
                first > tuning_count,
                "held-out seeds {first}..={last} overlap the tuning seeds 1..={tuning_count}"
            );
            assert!(
                last < *LOOKED_AT_SEEDS.start() || first > *LOOKED_AT_SEEDS.end(),
                "held-out seeds {first}..={last} overlap the looked-at seeds {LOOKED_AT_SEEDS:?}"
            );
        }
        Self {
            tuning_seeds: (1..=tuning_count).collect(),
            held_out_seeds,
        }
    }
}

#[cfg(test)]
mod seed_split_tests {
    use super::*;

    #[test]
    fn default_held_out_seeds_are_disjoint_from_tuning_and_looked_at_seeds() {
        let split = SeedSplit::counts(200, 400);
        assert!(split
            .held_out_seeds
            .iter()
            .all(|seed| !split.tuning_seeds.contains(seed) && !LOOKED_AT_SEEDS.contains(seed)));
    }

    #[test]
    #[should_panic(expected = "overlap the tuning seeds")]
    fn held_out_base_inside_the_tuning_seeds_panics() {
        SeedSplit::with_held_out_base(100, 10, 50);
    }

    #[test]
    #[should_panic(expected = "overlap the looked-at seeds")]
    fn held_out_range_over_the_looked_at_seeds_panics() {
        SeedSplit::with_held_out_base(10, 20, 999_995);
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PointBakeoffConfig {
    pub scene: SceneDrawConfig,
    pub trigger: TriggerModelConfig,
    pub production: ProductionCandidateConfig,
    pub onset: OnsetCandidateConfig,
    pub scoring: ScoringConfig,
    pub decision: DecisionRuleConfig,
    pub reportability: crate::bakeoff_candidate_fix::ReportabilityGateConfig,
}
