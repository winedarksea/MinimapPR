//! Tunables for onset-frame SRP clustering, mirroring Python
//! `OnsetSrpClusteringConfig` (built from the `onset_*` Settings keys). The
//! defaults are the server defaults (`minimappr/config.py`, W2-onset-srp and
//! S-fixes onset blocks) so a parity case can rely on them.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OnsetClusteringConfig {
    // --- onset detection ---
    pub subband_count: usize,
    pub hop_seconds: f64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
    pub flux_threshold_mad_multiplier: f64,
    pub min_separation_seconds: f64,
    pub max_onsets_per_window: usize,
    pub active_band_snr_db: f64,
    // --- frames and correlation ---
    pub frame_seconds: f64,
    pub frame_pre_roll_seconds: f64,
    pub narrowband_bandwidth_hz: f64,
    pub frame_max_intra_node_weight_share: f64,
    // --- SRP grid ---
    pub srp_azimuth_step_deg: f64,
    pub srp_elevation_step_deg: f64,
    pub srp_grid_min_range_m: f64,
    pub srp_grid_max_range_m: f64,
    pub srp_grid_ring_count: usize,
    pub srp_peaks_per_frame: usize,
    pub srp_peak_min_relative_height: f64,
    pub srp_node_grid_ring_count: usize,
    pub srp_node_grid_min_range_m: f64,
    pub srp_node_grid_max_range_m: f64,
    pub srp_node_grid_azimuth_step_deg: f64,
    pub srp_node_grid_elevation_step_deg: f64,
    // --- clustering ---
    pub cluster_radius_m: f64,
    pub cluster_range_fraction: f64,
    pub cluster_angle_deg: f64,
    pub cluster_min_weight_fraction: f64,
    pub cluster_max_count: usize,
    // --- echo, level and persistence checks ---
    pub echo_min_lag_seconds: f64,
    pub echo_max_lag_seconds: f64,
    pub echo_min_independent_fraction: f64,
    pub level_consistency_margin_db: f64,
    pub persistence_min_independent_onsets: usize,
    pub near_node_level_tolerance_db: f64,
    pub same_source_max_tdoa_difference_seconds: f64,
    // --- ground / canopy prior and height twins ---
    pub ground_prior_enabled: bool,
    pub node_height_above_ground_m: f64,
    pub source_max_height_above_ground_m: f64,
    pub planar_layout_max_flatness_ratio: f64,
    pub height_mirror_min_score_ratio: f64,
    // --- range observability profile ---
    pub range_profile_enabled: bool,
    pub range_profile_min_range_m: f64,
    pub range_profile_max_range_m: f64,
    pub range_profile_sample_count: usize,
    pub range_profile_cone_half_angle_deg: f64,
    pub range_profile_cone_step_deg: f64,
    pub range_profile_support_ratio: f64,
    pub range_observable_max_interval_ratio: f64,
    pub range_observable_interval_floor_aperture_fraction: f64,
}

impl Default for OnsetClusteringConfig {
    fn default() -> Self {
        Self {
            subband_count: 32,
            hop_seconds: 0.01,
            band_min_hz: 500.0,
            band_max_hz: 12_000.0,
            flux_threshold_mad_multiplier: 4.0,
            min_separation_seconds: 0.04,
            max_onsets_per_window: 16,
            active_band_snr_db: 6.0,
            frame_seconds: 0.08,
            frame_pre_roll_seconds: 0.01,
            narrowband_bandwidth_hz: 300.0,
            frame_max_intra_node_weight_share: 0.35,
            srp_azimuth_step_deg: 2.0,
            srp_elevation_step_deg: 10.0,
            srp_grid_min_range_m: 1.0,
            srp_grid_max_range_m: 300.0,
            srp_grid_ring_count: 40,
            srp_peaks_per_frame: 3,
            srp_peak_min_relative_height: 0.6,
            srp_node_grid_ring_count: 8,
            srp_node_grid_min_range_m: 0.5,
            srp_node_grid_max_range_m: 8.0,
            srp_node_grid_azimuth_step_deg: 5.0,
            srp_node_grid_elevation_step_deg: 10.0,
            cluster_radius_m: 3.0,
            cluster_range_fraction: 0.2,
            cluster_angle_deg: 10.0,
            cluster_min_weight_fraction: 0.15,
            cluster_max_count: 4,
            echo_min_lag_seconds: 0.003,
            echo_max_lag_seconds: 0.03,
            echo_min_independent_fraction: 0.5,
            level_consistency_margin_db: 8.0,
            persistence_min_independent_onsets: 2,
            near_node_level_tolerance_db: 10.0,
            same_source_max_tdoa_difference_seconds: 0.0015,
            ground_prior_enabled: true,
            node_height_above_ground_m: 1.5,
            source_max_height_above_ground_m: 50.0,
            planar_layout_max_flatness_ratio: 0.1,
            height_mirror_min_score_ratio: 1.05,
            range_profile_enabled: true,
            range_profile_min_range_m: 0.5,
            range_profile_max_range_m: 1000.0,
            range_profile_sample_count: 96,
            range_profile_cone_half_angle_deg: 3.0,
            range_profile_cone_step_deg: 1.0,
            range_profile_support_ratio: 0.9,
            range_observable_max_interval_ratio: 2.0,
            range_observable_interval_floor_aperture_fraction: 0.5,
        }
    }
}
