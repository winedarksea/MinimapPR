//! Localization-overhaul settings forwarded from Python. Settings key
//! `localization_foo` arrives as env `MINIMAPPR_LOCALIZATION_FOO`
//! (`minimappr/ingest_sidecar_runtime.py`); defaults here must equal the
//! Python `Settings` defaults (the clap literals stay greppable for the
//! Python parity tests, and `cli_defaults_equal_config_defaults` keeps them
//! equal to `Default`).

use std::collections::BTreeMap;

use minimappr_core::localization_sensor_entry::SiteGeometryPush;

/// Per-`time_quality` 1-sigma clock offset defaults (seconds). GPS/PPS is
/// microsecond-class; NTP is millisecond-class; a free-running clock is only
/// bounded by how long ago it was last set, so it gets a deliberately wide
/// default that still admits its pairs (weak clocks widen, never exclude).
const DEFAULT_CLOCK_CLASS_UNCERTAINTY_JSON: &str =
    r#"{"gps_locked":1e-6,"gps_holdover":1e-4,"ntp_disciplined":0.002,"free_running":0.05}"#;

#[derive(Clone, Debug, PartialEq)]
pub struct ClockClassUncertaintySeconds(pub BTreeMap<String, f64>);

#[derive(Clone, Debug, PartialEq)]
pub struct LocalizationSolveConfig {
    pub localization_max_solve_sensors: usize,
    pub localization_min_pair_baseline_m: f32,
    pub localization_cross_node_max_timeline_residual_seconds: f64,
    /// Frames of per-frame timeline residual the node clock tracker keeps;
    /// the reported residual is a percentile over them, so one jittery frame
    /// cannot flip the hard-limit gate.
    pub localization_timeline_residual_window_frames: usize,
    /// Percentile (0-100, nearest rank) of the windowed |residual|; 50 is the median.
    pub localization_timeline_residual_percentile: f64,
    /// Hysteresis: once over the hard limit, the flag clears only when the
    /// windowed residual falls to `limit * ratio` (in 0-1; 1 = no hysteresis).
    pub localization_timeline_residual_over_limit_clear_ratio: f64,
    /// Replaces (does not merge with) the defaults; a class missing from the
    /// map uses `localization_clock_unknown_uncertainty_seconds`.
    pub localization_clock_class_uncertainty_seconds: ClockClassUncertaintySeconds,
    pub localization_clock_unknown_uncertainty_seconds: f64,
    pub localization_timestamp_step_threshold_seconds: f64,
    pub localization_timestamp_step_recovery_holdoff_seconds: f64,
    /// Re-anchor a node's buffer timeline on a confirmed timestamp step
    /// (`minimappr_core::timeline_discipline`); off keeps the pre-2026-11-25
    /// behaviour (residual stays, the hard-limit gate drops the node's pairs).
    pub localization_timeline_step_reanchor_enabled: bool,
    /// Frames on the stepped clock (flagged frame included) before re-anchoring;
    /// a frame back on the old timeline first cancels it as jitter.
    pub localization_timestamp_step_confirm_frames: usize,
    /// Drift slew once the windowed residual passes this fraction of
    /// `localization_cross_node_max_timeline_residual_seconds`; 0 disables.
    pub localization_timeline_drift_slew_fraction: f64,
    pub localization_trigger_noise_floor_relative_enabled: bool,
    pub localization_trigger_noise_floor_relative_db: f32,
    /// Also judge each frame's loudest `localization_trigger_short_block_seconds`
    /// block against the node floor (OR'd; only with the relative trigger on),
    /// so a call shorter than a frame is not diluted by its quiet remainder.
    pub localization_trigger_short_block_excess_enabled: bool,
    pub localization_trigger_short_block_seconds: f64,
    /// Reference-channel RMS floor relative to the node's tracked noise floor.
    /// Kept under the relative trigger's margin so a window the trigger admits
    /// is not then rejected by the solver.
    pub localization_solver_floor_relative_db: f32,
    pub localization_sensor_registry_stale_seconds: f64,
    /// Geometry-derived event span (`minimappr_core::localization_event_timing`):
    /// the coverage wait holds for the cross-node lag plus this delivery skew.
    pub localization_coverage_delivery_margin_seconds: f64,
    /// The span grows by this many sigma of the worst cross-node pair's clock uncertainty.
    pub localization_event_span_clock_margin_sigma: f64,
    /// Memory/latency safety bound on the derived cross-node lag.
    pub localization_event_span_max_cross_node_lag_seconds: f64,
    /// Event solves parked at once per cluster buffer; the oldest is
    /// dispatched (superseded) when one more arrives.
    pub localization_coverage_max_pending_per_buffer: usize,
    /// `MINIMAPPR_SITE_GEOMETRY_JSON`: same body as `POST /api/v1/dsp/geometry`.
    pub site_geometry_at_launch: Option<SiteGeometryPush>,
}

impl Default for LocalizationSolveConfig {
    fn default() -> Self {
        Self {
            localization_max_solve_sensors: 16,
            localization_min_pair_baseline_m: 0.02,
            localization_cross_node_max_timeline_residual_seconds: 0.010,
            localization_timeline_residual_window_frames: 16,
            localization_timeline_residual_percentile: 50.0,
            localization_timeline_residual_over_limit_clear_ratio: 0.8,
            localization_clock_class_uncertainty_seconds: parse_clock_class_uncertainty_json(
                DEFAULT_CLOCK_CLASS_UNCERTAINTY_JSON,
            )
            .expect("default clock-class uncertainty JSON is valid"),
            localization_clock_unknown_uncertainty_seconds: 0.005,
            localization_timestamp_step_threshold_seconds: 0.002,
            localization_timestamp_step_recovery_holdoff_seconds: 30.0,
            localization_timeline_step_reanchor_enabled: true,
            localization_timestamp_step_confirm_frames: 3,
            localization_timeline_drift_slew_fraction: 0.5,
            localization_trigger_noise_floor_relative_enabled: true,
            localization_trigger_noise_floor_relative_db: 8.0,
            localization_trigger_short_block_excess_enabled: false,
            localization_trigger_short_block_seconds: 0.005,
            localization_solver_floor_relative_db: 3.0,
            localization_sensor_registry_stale_seconds: 120.0,
            localization_coverage_delivery_margin_seconds: 1.0,
            localization_event_span_clock_margin_sigma: 3.0,
            localization_event_span_max_cross_node_lag_seconds: 10.0,
            localization_coverage_max_pending_per_buffer: 8,
            site_geometry_at_launch: None,
        }
    }
}

#[derive(clap::Args, Clone, Debug)]
pub struct LocalizationSolveArgs {
    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_MAX_SOLVE_SENSORS",
        default_value_t = 16
    )]
    localization_max_solve_sensors: usize,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_MIN_PAIR_BASELINE_M",
        default_value_t = 0.02
    )]
    localization_min_pair_baseline_m: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_CROSS_NODE_MAX_TIMELINE_RESIDUAL_SECONDS",
        default_value_t = 0.010
    )]
    localization_cross_node_max_timeline_residual_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMELINE_RESIDUAL_WINDOW_FRAMES",
        default_value_t = 16
    )]
    localization_timeline_residual_window_frames: usize,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMELINE_RESIDUAL_PERCENTILE",
        default_value_t = 50.0
    )]
    localization_timeline_residual_percentile: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMELINE_RESIDUAL_OVER_LIMIT_CLEAR_RATIO",
        default_value_t = 0.8
    )]
    localization_timeline_residual_over_limit_clear_ratio: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_CLOCK_CLASS_UNCERTAINTY_SECONDS",
        default_value = DEFAULT_CLOCK_CLASS_UNCERTAINTY_JSON,
        value_parser = parse_clock_class_uncertainty_json
    )]
    localization_clock_class_uncertainty_seconds: ClockClassUncertaintySeconds,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS",
        default_value_t = 0.005
    )]
    localization_clock_unknown_uncertainty_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMESTAMP_STEP_THRESHOLD_SECONDS",
        default_value_t = 0.002
    )]
    localization_timestamp_step_threshold_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMESTAMP_STEP_RECOVERY_HOLDOFF_SECONDS",
        default_value_t = 30.0
    )]
    localization_timestamp_step_recovery_holdoff_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMELINE_STEP_REANCHOR_ENABLED",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    localization_timeline_step_reanchor_enabled: bool,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMESTAMP_STEP_CONFIRM_FRAMES",
        default_value_t = 3
    )]
    localization_timestamp_step_confirm_frames: usize,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TIMELINE_DRIFT_SLEW_FRACTION",
        default_value_t = 0.5
    )]
    localization_timeline_drift_slew_fraction: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    localization_trigger_noise_floor_relative_enabled: bool,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_DB",
        default_value_t = 8.0
    )]
    localization_trigger_noise_floor_relative_db: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TRIGGER_SHORT_BLOCK_EXCESS_ENABLED",
        default_value_t = false,
        action = clap::ArgAction::Set
    )]
    localization_trigger_short_block_excess_enabled: bool,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_TRIGGER_SHORT_BLOCK_SECONDS",
        default_value_t = 0.005
    )]
    localization_trigger_short_block_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB",
        default_value_t = 3.0,
        allow_negative_numbers = true
    )]
    localization_solver_floor_relative_db: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_SENSOR_REGISTRY_STALE_SECONDS",
        default_value_t = 120.0
    )]
    localization_sensor_registry_stale_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_COVERAGE_DELIVERY_MARGIN_SECONDS",
        default_value_t = 1.0
    )]
    localization_coverage_delivery_margin_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_EVENT_SPAN_CLOCK_MARGIN_SIGMA",
        default_value_t = 3.0
    )]
    localization_event_span_clock_margin_sigma: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_EVENT_SPAN_MAX_CROSS_NODE_LAG_SECONDS",
        default_value_t = 10.0
    )]
    localization_event_span_max_cross_node_lag_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_COVERAGE_MAX_PENDING_PER_BUFFER",
        default_value_t = 8
    )]
    localization_coverage_max_pending_per_buffer: usize,

    #[arg(long, env = "MINIMAPPR_SITE_GEOMETRY_JSON", value_parser = parse_site_geometry_json)]
    site_geometry_json: Option<SiteGeometryPush>,
}

impl LocalizationSolveArgs {
    pub fn into_config(self) -> LocalizationSolveConfig {
        LocalizationSolveConfig {
            localization_max_solve_sensors: self.localization_max_solve_sensors,
            localization_min_pair_baseline_m: self.localization_min_pair_baseline_m,
            localization_cross_node_max_timeline_residual_seconds: self
                .localization_cross_node_max_timeline_residual_seconds,
            localization_timeline_residual_window_frames: self
                .localization_timeline_residual_window_frames,
            localization_timeline_residual_percentile: self
                .localization_timeline_residual_percentile,
            localization_timeline_residual_over_limit_clear_ratio: self
                .localization_timeline_residual_over_limit_clear_ratio,
            localization_clock_class_uncertainty_seconds: self
                .localization_clock_class_uncertainty_seconds,
            localization_clock_unknown_uncertainty_seconds: self
                .localization_clock_unknown_uncertainty_seconds,
            localization_timestamp_step_threshold_seconds: self
                .localization_timestamp_step_threshold_seconds,
            localization_timestamp_step_recovery_holdoff_seconds: self
                .localization_timestamp_step_recovery_holdoff_seconds,
            localization_timeline_step_reanchor_enabled: self
                .localization_timeline_step_reanchor_enabled,
            localization_timestamp_step_confirm_frames: self
                .localization_timestamp_step_confirm_frames,
            localization_timeline_drift_slew_fraction: self
                .localization_timeline_drift_slew_fraction,
            localization_trigger_noise_floor_relative_enabled: self
                .localization_trigger_noise_floor_relative_enabled,
            localization_trigger_noise_floor_relative_db: self
                .localization_trigger_noise_floor_relative_db,
            localization_trigger_short_block_excess_enabled: self
                .localization_trigger_short_block_excess_enabled,
            localization_trigger_short_block_seconds: self.localization_trigger_short_block_seconds,
            localization_solver_floor_relative_db: self.localization_solver_floor_relative_db,
            localization_sensor_registry_stale_seconds: self
                .localization_sensor_registry_stale_seconds,
            localization_coverage_delivery_margin_seconds: self
                .localization_coverage_delivery_margin_seconds,
            localization_event_span_clock_margin_sigma: self
                .localization_event_span_clock_margin_sigma,
            localization_event_span_max_cross_node_lag_seconds: self
                .localization_event_span_max_cross_node_lag_seconds,
            localization_coverage_max_pending_per_buffer: self
                .localization_coverage_max_pending_per_buffer
                .max(1),
            site_geometry_at_launch: self.site_geometry_json,
        }
    }
}

/// `{"<time_quality>": seconds, ...}`; every value finite and non-negative.
fn parse_clock_class_uncertainty_json(raw: &str) -> Result<ClockClassUncertaintySeconds, String> {
    let by_class: BTreeMap<String, f64> =
        serde_json::from_str(raw).map_err(|error| format!("invalid JSON object: {error}"))?;
    if let Some((class, value)) = by_class
        .iter()
        .find(|(_, value)| !value.is_finite() || **value < 0.0)
    {
        return Err(format!(
            "{class}: uncertainty must be finite and >= 0, got {value}"
        ));
    }
    Ok(ClockClassUncertaintySeconds(by_class))
}

/// A malformed launch geometry fails startup: silently solving in a frame
/// Python did not ask for would publish positions it then rejects.
fn parse_site_geometry_json(raw: &str) -> Result<SiteGeometryPush, String> {
    serde_json::from_str(raw).map_err(|error| format!("invalid site geometry: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct LocalizationSolveCli {
        #[command(flatten)]
        localization_solve: LocalizationSolveArgs,
    }

    #[test]
    fn cli_defaults_equal_config_defaults() {
        let cli = LocalizationSolveCli::try_parse_from(["sidecar"]).unwrap();
        assert_eq!(
            cli.localization_solve.into_config(),
            LocalizationSolveConfig::default()
        );
    }

    #[test]
    fn clock_class_map_and_site_geometry_parse_from_flags() {
        let cli = LocalizationSolveCli::try_parse_from([
            "sidecar",
            "--localization-clock-class-uncertainty-seconds",
            r#"{"gps_locked": 2e-6}"#,
            "--localization-solver-floor-relative-db",
            "-3",
            "--site-geometry-json",
            r#"{"site_frame_id":"s","geometry_push_revision":1,
                "site_origin":{"lat":45.0,"lon":-93.0,"alt_m":0.0},
                "projection":"geodetic_ecef_enu","node_placements":{}}"#,
        ])
        .unwrap();
        let config = cli.localization_solve.into_config();
        assert_eq!(
            config.localization_clock_class_uncertainty_seconds.0,
            BTreeMap::from([("gps_locked".to_string(), 2e-6)])
        );
        assert_eq!(config.localization_solver_floor_relative_db, -3.0);
        assert_eq!(
            config
                .site_geometry_at_launch
                .map(|push| push.site_frame_id),
            Some("s".to_string())
        );
    }

    #[test]
    fn negative_clock_uncertainty_is_rejected() {
        assert!(parse_clock_class_uncertainty_json(r#"{"gps_locked": -1.0}"#).is_err());
        assert!(parse_site_geometry_json("{}").is_err());
    }
}
