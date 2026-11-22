//! Server-track bird entities ported from the Python reference (S6 gate
//! verdict "PASS: port the entity model", with the S-fixes absorption and
//! concurrency fixes): one entity per bird moving Heard nearby → Direction
//! known → Pinpointed, covariance-gated (Mahalanobis) absorption, concurrent
//! inconsistent fixes as two birds, far-field fixes as bearing evidence, and
//! the bound track's fused position. [`entity_pipeline`] composes it with the
//! perched tracker in FusionNode order for the mesh host.
//!
//! Not ported: the notification grace window and maintenance timer (host
//! shell timers), API/WS projection (geo frames live in the shell).
//! Python↔Rust parity: `tests/test_wildlife_tracking_cross_language_parity.py`.

pub mod absorption_geometry;
pub mod config;
pub mod entity_pipeline;
pub mod matching;
pub mod model;
pub mod registry;
pub mod registry_spatial;
pub mod reporting_decision;
