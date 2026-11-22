//! Cartesian TDOA and joint TDOA+DOA position solvers.
//!
//! Two layers:
//! - [`cartesian_tdoa_solve`]: a numerical port of the server's Python
//!   `minimappr/core/cartesian_tdoa.py` (multi-start nonlinear least squares,
//!   radial range refinement, SVD covariance, range-projection modes and
//!   confidence). Parity is gated by `tests/test_tdoa_solver_cross_language_parity.py`
//!   through the sidecar's `tdoa-solver-oracle` subcommand. It adds an optional
//!   canopy-height prior (off in parity mode).
//! - [`joint_objective`]: the Field Mode mixed-node solver over
//!   [`node_descriptor::NodeDescriptor`]s (tetra arrays, mono and stereo phones):
//!   whitened TDOA, DOA and stereo-cone residuals with a covariance that refuses
//!   to invert an unobservable geometry.
//!
//! Supporting pieces: [`loop_consistency`] (argmax-TDOA reflection filter),
//! [`bearing_fusion`] (port of `multi_node_bearing_fusion.fuse_bearings`).
//! All nonlinear refinement goes through the maintained MINPACK-derived
//! `levenberg-marquardt` crate via [`levenberg_marquardt_driver`].
//!
//! Sign convention everywhere: `tdoa_seconds = (|x − a| − |x − b|) / c`,
//! i.e. arrival time at sensor A minus arrival time at sensor B.

pub mod bearing_fusion;
pub mod bounded_scalar_minimizer;
pub mod canopy_height_prior;
pub mod cartesian_initial_positions;
pub mod cartesian_solve_quality;
pub mod cartesian_tdoa_solve;
pub mod far_field_model_selection;
pub mod joint_objective;
pub mod joint_residual_terms;
pub mod levenberg_marquardt_driver;
pub mod loop_consistency;
pub mod measurement_types;
pub mod node_descriptor;
pub mod radial_range_refinement;
pub mod robust_linear_algebra;
pub mod scipy_trf_compatible;
pub mod spatial_constraint_terms;

#[cfg(test)]
mod cartesian_solver_tests;
#[cfg(test)]
mod degenerate_input_tests;
#[cfg(test)]
mod joint_objective_tests;
#[cfg(test)]
mod loop_consistency_tests;
#[cfg(test)]
mod test_support;

use std::fmt;

/// Re-exported so callers use the exact nalgebra version in this API.
pub use nalgebra;

/// Every recoverable failure in the solver stack. Solvers return these instead
/// of panicking so a degenerate geometry on a phone is a reported outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum TdoaSolverError {
    NonFiniteInput(&'static str),
    EmptySystem(&'static str),
    DecompositionDidNotConverge(&'static str),
    Singular {
        smallest_eigenvalue: f64,
    },
    IllConditioned {
        condition_number: f64,
    },
    /// Fewer measurements than the method needs (e.g. bearing solve < 3 pairs).
    InsufficientMeasurements {
        needed: usize,
        available: usize,
    },
    /// A direction solve collapsed to a zero vector.
    DegenerateDirection,
    UnknownSensor(String),
    UnknownNode(String),
    MicIndexOutOfRange {
        node_id: String,
        mic_index: usize,
    },
    /// Two nodes in one TDOA residual do not share a clock domain.
    ClockDomainMismatch {
        node_a: String,
        node_b: String,
    },
    InvalidNodeDescriptor(String),
    /// No start point converged.
    RefinementDidNotConverge,
    RadialRefinementNoFiniteCandidate,
}

impl fmt::Display for TdoaSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TdoaSolverError {}
