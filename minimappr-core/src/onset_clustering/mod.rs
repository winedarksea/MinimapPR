//! Onset-frame SRP clustering with echo checks (Field Mode server-track S2),
//! ported from the Python reference `minimappr/core/onset_*.py`.
//!
//! Entry point: [`pipeline::localize_onset_clusters`] takes one analysis
//! window of multi-sensor audio on a shared sample timeline and returns the
//! candidate birds (accepted / tentative / echo, with reasons), each with a
//! refined position, covariance, range-projection mode, SRP range interval and
//! height-twin verdict. Sans-IO and synchronous: no threads, clocks or files,
//! so the same code runs in the sidecar, on phones and in WASM.
//!
//! Parity: `tests/test_onset_clustering_cross_language_parity.py` compares
//! this module against the Python pipeline through the sidecar's
//! `onset-clustering-oracle` subcommand (see
//! `docs/field_mode_progress/port_server_track_onset.md`).

pub mod attribution_cascade;
pub mod attribution_decision;
pub mod attribution_evidence;
pub mod bounded_trf;
mod bounded_trf_step;
mod cluster_finalization;
pub mod config;
pub mod echo_checks;
mod frame_assembly;
pub mod frame_correlation;
pub mod geometry_prior;
pub mod geometry_resolution;
pub mod height_and_range;
pub mod numeric;
pub mod onset_detection;
pub mod pipeline;
pub mod python_set_order;
pub mod refinement;
#[cfg(test)]
mod scenario_tests;
pub mod srp_grid;
pub mod tdoa_votes;
pub mod trace;
pub mod types;
pub mod uncertainty;

pub use config::OnsetClusteringConfig;
pub use pipeline::{localize_onset_clusters, localize_onset_clusters_traced, OnsetClusteringInput};
pub use trace::OnsetClusteringTrace;
pub use types::{ClusterStatus, LocalizedOnsetCluster, OnsetClusteringResult};
