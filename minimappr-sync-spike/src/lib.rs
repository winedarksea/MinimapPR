//! Phase 1 sync spike analyser: turns two phones' chirp-lab recordings into
//! the Phase 1 gate evidence (plan `docs/field_mode_phone_mesh_plan.md`,
//! "Phase 1 — sync spike"; run matrix A0–A9 in
//! `docs/field_mode_progress/phase1_sync_spike.md`).
//!
//! Flow: [`bundle_loader`] ([`capture_timeline`]) → [`round_pairing`] (picks via [`arrival_scan`])
//! → [`capture_continuity`] → [`pair_ranging_series`] / [`clock_residuals`] / [`onset_stability`] /
//! [`spectral_leakage`] / [`channel_map`] → [`run_analysis::RunReport`] →
//! [`gate_evaluation`] → [`report_writer`].

pub mod analysis_config;
pub mod arrival_scan;
pub mod bundle_loader;
pub mod bundle_schema;
pub mod bundle_validation;
pub mod capture_continuity;
pub mod capture_timeline;
pub mod channel_map;
pub mod clock_residuals;
pub mod gate_evaluation;
pub mod onset_stability;
pub mod pair_ranging_series;
pub mod report_writer;
pub mod round_pairing;
pub mod run_analysis;
pub mod spectral_leakage;
