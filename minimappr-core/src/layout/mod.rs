//! Field Mode rigid layout (plan: "Layout solve", "Map anchoring",
//! "Mirror images").
//!
//! Chirp ranges → `classical_mds` initial guess → `robust_layout_refine`
//! (heights from `barometric_height`) → `gps_anchor_fit` places the rigid
//! layout on the map (mirror check, chirality confidence) →
//! `anchor_filter` smooths east/north/rotation, clamped while the group is
//! still; `filtered_map_anchor` adds per-phone jump holding and frame
//! changes for the mesh. Inputs are distances, not waveforms, so clap or
//! audible ranges work the same.

pub mod anchor_filter;
pub mod barometric_height;
pub mod classical_mds;
pub mod filtered_map_anchor;
pub mod gps_anchor_fit;
pub mod robust_layout_refine;
