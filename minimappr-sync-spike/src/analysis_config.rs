//! Every threshold and window of the analyser in one place, with the
//! Phase 1 gate numbers from `docs/field_mode_phone_mesh_plan.md`.

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisConfig {
    pub sample_rate_hz: f64,
    /// Self arrivals are searched this far either side of the emission's
    /// predicted capture sample (AudioTrack/AudioRecord timestamp error plus
    /// output latency uncertainty).
    pub self_search_half_window_s: f64,
    /// A capture-timestamp step larger than this between consecutive
    /// timestamps is a silent drop (Pixel 10 Pro XL timestamps jitter by
    /// ~5 ms; the 2026-11-26 drop was 527 ms).
    pub capture_timestamp_step_threshold_s: f64,
    /// Consecutive self picks whose capture interval differs from their
    /// output-frame interval by more than this many samples (after the
    /// run's median rate ratio) bracket a playback or capture discontinuity.
    /// Rounds overlapping one are left out of ranging and the clock.
    pub self_interval_break_tolerance_samples: f64,
    /// First remote matches use a wide window (start skew between phones);
    /// later ones use the median offset ± this.
    pub remote_refined_half_window_s: f64,
    pub coarse_match_count: usize,
    /// First-match half window when both phones logged wall-clock time
    /// (NTP-synced phones agree to tens of ms).
    pub wall_clock_match_half_window_s: f64,
    /// Longest flight searched for (≈ 34 m at 343 m/s).
    pub maximum_flight_s: f64,
    /// Blind scan (no timestamps): chirps with at least this SNR, spaced at
    /// least this far apart, are taken as arrivals.
    pub blind_scan_minimum_snr_db: f64,
    pub blind_scan_chunk_s: f64,
    /// Used when the manifest gives none (the core default d_ii).
    pub default_self_path_length_m: f64,
    /// Rounds whose ranges differ from the run median by more than this are
    /// counted as outliers (and excluded from σ, reported separately).
    pub range_outlier_threshold_m: f64,
    /// Clock-filter cadences for the one-step prediction residual.
    pub heartbeat_cadence_s: f64,
    /// Leakage is measured in this band (the bird band the classifier sees).
    pub leakage_band_hz: [f64; 2],
    pub gates: GateThresholds,
}

/// Phase 1 gate (plan, "Phase 1 — sync spike").
#[derive(Debug, Clone, PartialEq)]
pub struct GateThresholds {
    /// 10 cm at 20 m.
    pub range_error_at_20_m_m: f64,
    pub range_gate_distance_m: f64,
    /// θ residual over 30 min at the heartbeat cadence.
    pub clock_residual_s: f64,
    pub clock_residual_minimum_duration_s: f64,
    /// Self-onset stability across placements.
    pub self_onset_stability_s: f64,
    /// Leakage into the bird band, dB over the noise floor.
    pub maximum_leakage_over_floor_db: f64,
    /// Locked-screen playback: no self-chirp gap longer than this many periods.
    pub maximum_missing_periods_while_locked: f64,
    /// Items needing three phone models report PARTIAL with fewer.
    pub required_model_count: usize,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            range_error_at_20_m_m: 0.10,
            range_gate_distance_m: 20.0,
            clock_residual_s: 50e-6,
            clock_residual_minimum_duration_s: 1_800.0,
            self_onset_stability_s: 10e-6,
            maximum_leakage_over_floor_db: 3.0,
            maximum_missing_periods_while_locked: 1.5,
            required_model_count: 3,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000.0,
            self_search_half_window_s: 0.030,
            capture_timestamp_step_threshold_s: 0.015,
            self_interval_break_tolerance_samples: 2.0,
            remote_refined_half_window_s: 0.050,
            coarse_match_count: 8,
            wall_clock_match_half_window_s: 0.4,
            maximum_flight_s: 0.1,
            blind_scan_minimum_snr_db: 20.0,
            blind_scan_chunk_s: 1.0,
            default_self_path_length_m: 0.02,
            range_outlier_threshold_m: 0.5,
            heartbeat_cadence_s: 90.0,
            leakage_band_hz: [2_000.0, 11_000.0],
            gates: GateThresholds::default(),
        }
    }
}
