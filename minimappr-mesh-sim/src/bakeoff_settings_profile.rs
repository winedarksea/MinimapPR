//! Named solver/gate settings the bake-off can run, and the block every
//! report prints so a result always says which settings produced it.
//!
//! `CurrentProduction` is the application's defaults (asserted against the
//! Python and sidecar sources by `tests/bakeoff_settings_parity.rs`).
//! `HistoricalPreKilometreRange` keeps the values E8 still carried after the
//! km-range change (7191046) moved the app's envelope; it reproduces the
//! recorded 0.491 recall run (`docs/point_network_bakeoff.md`).

use std::fmt::Write as _;

use crate::bakeoff_candidate_fix::ReportabilityGateConfig;
use crate::bakeoff_config::{PointBakeoffConfig, ProductionCandidateConfig};

/// Pre-km-range envelope: SRP far-field search and covariance ceiling.
const HISTORICAL_FAR_FIELD_MAX_RANGE_M: f32 = 1_000.0;
const HISTORICAL_MAX_RANGE_FROM_CENTROID_M: f64 = 1_200.0;
const HISTORICAL_ORIGIN_BACKSTOP_M: f64 = 5_000.0;
const HISTORICAL_STD_CEILING_ABSOLUTE_M: f64 = 1_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BakeoffSettingsProfile {
    #[default]
    CurrentProduction,
    HistoricalPreKilometreRange,
}

impl BakeoffSettingsProfile {
    pub const ALL: [BakeoffSettingsProfile; 2] = [Self::CurrentProduction, Self::HistoricalPreKilometreRange];

    pub fn label(self) -> &'static str {
        match self {
            Self::CurrentProduction => "current_production",
            Self::HistoricalPreKilometreRange => "historical_pre_km_range",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| profile.label() == label)
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::CurrentProduction => "the application's current solver and gate defaults",
            Self::HistoricalPreKilometreRange => {
                "E8's settings before they were brought in line with the km-range envelope (recorded 0.491 recall run, 73f1d0d)"
            }
        }
    }
}

impl PointBakeoffConfig {
    /// The full bake-off config with the named settings; everything not part
    /// of the solver/gate envelope (scenes, scoring, decision) is shared.
    pub fn for_settings_profile(profile: BakeoffSettingsProfile) -> Self {
        let mut config = Self::default();
        config.settings_profile = profile;
        if profile == BakeoffSettingsProfile::HistoricalPreKilometreRange {
            config.production = ProductionCandidateConfig {
                far_field_max_range_m: HISTORICAL_FAR_FIELD_MAX_RANGE_M,
                resolve_far_field_max_range_m: None,
                ..config.production
            };
            config.reportability = ReportabilityGateConfig {
                max_range_from_centroid_m: HISTORICAL_MAX_RANGE_FROM_CENTROID_M,
                origin_backstop_m: Some(HISTORICAL_ORIGIN_BACKSTOP_M),
                std_ceiling_absolute_m: HISTORICAL_STD_CEILING_ABSOLUTE_M,
                ..config.reportability
            };
        }
        config
    }
}

fn optional_metres(value: Option<f64>) -> String {
    value.map_or_else(|| "unbounded".to_string(), |metres| format!("{metres} m"))
}

/// The effective solver and gate settings, as a markdown table.
pub fn effective_settings_markdown(config: &PointBakeoffConfig) -> String {
    let production = &config.production;
    let gate = &config.reportability;
    let band = &production.localization_band_policy;
    let mut text = format!(
        "### Effective settings\n\nProfile: `{}` ({}).\n\n| setting | value |\n|---|---|\n",
        config.settings_profile.label(),
        config.settings_profile.description()
    );
    let rows: Vec<(&str, String)> = vec![
        ("node-local window (s)", format!("{}", production.node_local_window_seconds)),
        ("event span max cross-node lag (s)", format!("{}", production.event_span_max_cross_node_lag_seconds)),
        (
            "cross-node band (Hz) / taper / PHAT-β",
            format!("{}–{} / {} / {}", band.band_min_hz, band.cross_node_band_max_hz, band.cross_node_window_taper_fraction, band.cross_node_phat_beta),
        ),
        ("SRP grid resolution / padding (m)", format!("{} / {}", production.grid_resolution_m, production.search_padding_m)),
        ("GCC interpolation factor", format!("{}", production.interpolation_factor)),
        ("far-field default range (m)", format!("{}", production.far_field_default_range_m)),
        ("SRP far-field max range (m)", format!("{}", production.far_field_max_range_m)),
        ("re-solve radial search bound", optional_metres(production.resolve_far_field_max_range_m)),
        ("solver floor relative (dB)", format!("{}", production.solver_floor_relative_db)),
        ("re-solve minimum sensors / pairs", format!("{} / {}", production.resolve_minimum_sensor_count, production.resolve_minimum_pair_count)),
        ("trigger relative dB / cooldown (s)", format!("{} / {}", config.trigger.relative_db, config.trigger.cooldown_seconds)),
        ("gate: max range from centroid (m)", format!("{}", gate.max_range_from_centroid_m)),
        ("gate: origin backstop", gate.origin_backstop_m.map_or_else(|| "none".to_string(), |metres| format!("{metres} m"))),
        (
            "covariance ceiling: range factor / floor / absolute (m)",
            format!("{} / {} / {}", gate.std_ceiling_range_factor, gate.std_ceiling_floor_m, gate.std_ceiling_absolute_m),
        ),
        ("gate: unobservable range", "dropped (scoring protocol; the app keeps it with a confidence haircut)".to_string()),
    ];
    for (name, value) in rows {
        let _ = writeln!(text, "| {name} | {value} |");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_profile_differs_only_in_the_range_envelope() {
        let current = PointBakeoffConfig::for_settings_profile(BakeoffSettingsProfile::CurrentProduction);
        let historical = PointBakeoffConfig::for_settings_profile(BakeoffSettingsProfile::HistoricalPreKilometreRange);
        assert_eq!(current, PointBakeoffConfig::default());
        assert_eq!(historical.scene, current.scene);
        assert_eq!(historical.scoring, current.scoring);
        assert_eq!(historical.production.far_field_max_range_m, 1_000.0);
        assert_eq!(historical.production.resolve_far_field_max_range_m, None);
        assert_eq!(historical.reportability.origin_backstop_m, Some(5_000.0));
    }

    #[test]
    fn profile_labels_round_trip_and_reach_the_settings_block() {
        for profile in BakeoffSettingsProfile::ALL {
            assert_eq!(BakeoffSettingsProfile::from_label(profile.label()), Some(profile));
            let block = effective_settings_markdown(&PointBakeoffConfig::for_settings_profile(profile));
            assert!(block.contains(profile.label()));
        }
    }
}
