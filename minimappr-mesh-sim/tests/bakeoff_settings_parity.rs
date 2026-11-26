//! The bake-off's `current_production` settings must be the application's
//! defaults. Reads the named constants out of the Python config and the
//! sidecar sources (by name, not by line), so a changed app default fails
//! here instead of silently leaving E8 measuring a stale envelope.

use std::path::PathBuf;

use minimappr_mesh_sim::bakeoff_config::PointBakeoffConfig;
use minimappr_mesh_sim::bakeoff_settings_profile::BakeoffSettingsProfile;

fn repository_file(relative_path: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(relative_path);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn parse_number(text: &str) -> f64 {
    let cleaned: String = text.trim().trim_end_matches(',').chars().filter(|character| *character != '_').collect();
    cleaned.parse().unwrap_or_else(|error| panic!("not a number {text:?}: {error}"))
}

/// `NAME = 5_000.0` at module level of a Python file.
fn python_module_constant(source: &str, name: &str) -> f64 {
    let prefix = format!("{name} = ");
    let line = source.lines().find(|line| line.starts_with(&prefix)).unwrap_or_else(|| panic!("{name} not found"));
    parse_number(line[prefix.len()..].split('#').next().unwrap_or_default())
}

/// `    field: float = 1.0` in a Python settings class; every occurrence must agree.
fn python_settings_literal_default(source: &str, field: &str) -> f64 {
    let prefix = format!("{field}: float = ");
    let values: Vec<f64> = source
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix(&prefix))
        .map(|value| parse_number(value.split('#').next().unwrap_or_default()))
        .collect();
    assert!(!values.is_empty(), "{field} default not found");
    assert!(values.windows(2).all(|pair| pair[0] == pair[1]), "{field} defaults disagree: {values:?}");
    values[0]
}

/// The `default_value_t` of the clap argument read from env var `env_name`.
fn sidecar_cli_default(source: &str, env_name: &str) -> f64 {
    let marker = format!("env = \"{env_name}\"");
    let after = &source[source.find(&marker).unwrap_or_else(|| panic!("{env_name} not found")) + marker.len()..];
    let value_start = after.find("default_value_t = ").expect("default_value_t follows the env") + "default_value_t = ".len();
    let literal: String = after[value_start..].chars().take_while(|character| character.is_ascii_digit() || matches!(character, '.' | '_' | '-')).collect();
    parse_number(&literal)
}

/// `field: 5000.0,` in a Rust struct literal (the sidecar's `Default`).
fn rust_struct_literal_field(source: &str, field: &str) -> f64 {
    let prefix = format!("{field}: ");
    let line = source
        .lines()
        .map(str::trim_start)
        .find(|line| line.starts_with(&prefix) && !line.contains("f32,") && !line.contains("f64,"))
        .unwrap_or_else(|| panic!("{field} literal not found"));
    parse_number(&line[prefix.len()..])
}

#[test]
fn current_production_is_the_default_profile() {
    assert_eq!(PointBakeoffConfig::default(), PointBakeoffConfig::for_settings_profile(BakeoffSettingsProfile::CurrentProduction));
}

#[test]
fn range_envelope_matches_the_python_overhaul_defaults() {
    let overhaul = repository_file("minimappr/core/localization_overhaul_config.py");
    let config = PointBakeoffConfig::default();
    let far_field_max_range_m = python_module_constant(&overhaul, "DEFAULT_LOCALIZATION_FAR_FIELD_MAX_RANGE_M");
    assert_eq!(f64::from(config.production.far_field_max_range_m), far_field_max_range_m);
    assert_eq!(config.production.resolve_far_field_max_range_m, Some(far_field_max_range_m));
    assert_eq!(config.reportability.max_range_from_centroid_m, python_module_constant(&overhaul, "DEFAULT_LOCALIZATION_MAX_RANGE_M"));
    assert_eq!(config.reportability.std_ceiling_absolute_m, python_module_constant(&overhaul, "DEFAULT_LOCALIZATION_MAX_POSITION_STD_M"));
    assert_eq!(f64::from(config.production.solver_floor_relative_db), python_module_constant(&overhaul, "DEFAULT_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB"));
    assert_eq!(config.trigger.relative_db, python_module_constant(&overhaul, "DEFAULT_NOISE_FLOOR_RELATIVE_TRIGGER_SNR_DB"));
    assert_eq!(
        config.production.event_span_max_cross_node_lag_seconds,
        python_module_constant(&overhaul, "DEFAULT_LOCALIZATION_EVENT_SPAN_MAX_CROSS_NODE_LAG_SECONDS")
    );
}

#[test]
fn gate_and_seed_defaults_match_the_python_settings() {
    let settings = repository_file("minimappr/config.py");
    let config = PointBakeoffConfig::default();
    assert_eq!(config.production.far_field_default_range_m, python_settings_literal_default(&settings, "localization_far_field_default_range_m"));
    assert_eq!(config.reportability.std_ceiling_range_factor, python_settings_literal_default(&settings, "localization_std_range_factor"));
    assert_eq!(config.reportability.std_ceiling_floor_m, python_settings_literal_default(&settings, "localization_position_std_floor_m"));
    assert_eq!(config.trigger.cooldown_seconds, python_settings_literal_default(&settings, "trigger_cooldown_seconds"));
}

#[test]
fn the_server_bounds_the_resolve_and_has_no_origin_backstop() {
    // The re-solve's radial search is bounded by the same envelope setting.
    let handoff = repository_file("minimappr/core/localized_render_localization.py");
    assert!(handoff.contains("far_field_max_range_m=host.settings.localization_far_field_max_range_m"));
    // The reportability gate is centroid-relative only (7191046 removed the backstop).
    let fusion_node = repository_file("minimappr/core/fusion_node.py");
    assert!(!fusion_node.contains("ORIGIN_BACKSTOP"));
    assert_eq!(PointBakeoffConfig::default().reportability.origin_backstop_m, None);
}

#[test]
fn srp_defaults_match_the_sidecar_cli_and_worker_defaults() {
    let sidecar_cli = repository_file("minimappr-ingest-sidecar/src/main.rs");
    let sidecar_worker = repository_file("minimappr-ingest-sidecar/src/dsp_preprocess.rs");
    let production = PointBakeoffConfig::default().production;
    let expected = [
        ("MINIMAPPR_LOCALIZATION_FAR_FIELD_MAX_RANGE_M", "localization_far_field_max_range_m", f64::from(production.far_field_max_range_m)),
        ("MINIMAPPR_LOCALIZATION_FAR_FIELD_DEFAULT_RANGE_M", "localization_far_field_default_range_m", production.far_field_default_range_m),
        ("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M", "localization_srp_grid_resolution_m", f64::from(production.grid_resolution_m)),
        ("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M", "localization_search_padding_m", f64::from(production.search_padding_m)),
    ];
    for (env_name, worker_field, bakeoff_value) in expected {
        assert_eq!(sidecar_cli_default(&sidecar_cli, env_name), bakeoff_value, "sidecar CLI {env_name}");
        assert_eq!(rust_struct_literal_field(&sidecar_worker, worker_field), bakeoff_value, "sidecar worker {worker_field}");
    }
    assert_eq!(sidecar_cli_default(&sidecar_cli, "MINIMAPPR_GCC_PHAT_INTERP_FACTOR"), f64::from(production.interpolation_factor));
    assert_eq!(rust_struct_literal_field(&sidecar_worker, "gcc_phat_interp_factor"), f64::from(production.interpolation_factor));
}
