# W2-core-solver: notes

Branch: `fm/core-solver` (off `field-mode`).

## What landed

New module `minimappr-core/src/tdoa_solver/` (one `pub mod tdoa_solver;` line in `lib.rs`):

| File | Role |
|---|---|
| `cartesian_tdoa_solve.rs`, `cartesian_solve_quality.rs` | Port of `minimappr/core/cartesian_tdoa.py::solve_cartesian_tdoa`: algebraic / plane-wave / bearing-ray / prior starts, multi-start refine, radial range refinement, SVD covariance, range-projection mode, observability, confidence, GDOP. Optional canopy prior (off in parity mode). |
| `scipy_trf_compatible.rs` | Port of SciPy unbounded TRF (`trf_no_bounds` + `solve_lsq_trust_region`, soft-L1 loss, 2-point finite-difference Jacobian). Parity engine; see below for why. |
| `levenberg_marquardt_driver.rs` | `PositionResidualModel` trait + adapter onto the `levenberg-marquardt` crate (MINPACK `lmder` port). Used by the joint solver and as the Cartesian `LevenbergMarquardt` engine. |
| `radial_range_refinement.rs`, `bounded_scalar_minimizer.rs` | Port of `radial_range_refinement.py`; SciPy's bounded Brent (`minimize_scalar(method="bounded")`) ported line by line. |
| `spatial_constraint_terms.rs`, `measurement_types.rs` | TDOA / bearing / amplitude residual rows + Jacobians, plane-wave direction, ray intersection; Python dataclass mirrors. |
| `canopy_height_prior.rs` | Hinge prior on `z ∈ [ground, ground + 35 m]` + z-mirror / lifted band probe starts. |
| `loop_consistency.rs` | Greedy reflection filter on argmax pair TDOAs (`τ_AB + τ_BC + τ_CA ≈ 0`), with diagnostics for the SRP path. |
| `bearing_fusion.rs` | Port of `multi_node_bearing_fusion.fuse_bearings` (closed form; the async TTL store stays server-side). |
| `node_descriptor.rs` | `NodeDescriptor` (kind tetra / phone-mono / phone-stereo / generic sub-array, position + std, body-frame mic offsets, body→world quaternion + attitude std, clock domain + offset std, speaker). |
| `joint_residual_terms.rs`, `joint_objective.rs` | Joint whitened TDOA + DOA objective: inter-node mic-pair TDOAs (same clock domain only), intra-node stereo cones (exact hyperboloid between the phone's two mics, rotated by its IMU attitude), array DOAs (body az/el or world vector), world bearing rays (fusion cones as a DOA term), optional canopy prior. Multi-start LM, `(JᵀJ)⁻¹` covariance that errors on an unobservable geometry, near-equal-cost mirror reported as `ambiguous_alternate_position_m`. |
| `robust_linear_algebra.rs` | Non-panicking SVD / min-norm lstsq / symmetric eigen / SPD inverse with condition check. |

Sidecar: `minimappr-ingest-sidecar/src/tdoa_solver_oracle.rs` + two lines in `main.rs`
(`mod` and the `tdoa-solver-oracle` dispatch). Batch JSON in, JSON out; `inf` → `null`.

Python: `tests/tdoa_solver_parity_cases.py` (cases, Python runner, golden writer:
`python -m tests.tdoa_solver_parity_cases`), `tests/fixtures/tdoa_solver_parity_goldens.npz`,
`tests/test_tdoa_solver_cross_language_parity.py` (Rust vs goldens, and Python vs goldens).

Dependencies (labelled block in `minimappr-core/Cargo.toml`): `nalgebra 0.33` (no default
features, `std`), `levenberg-marquardt 0.14` (no default features). Both build for
`wasm32-unknown-unknown` and `aarch64-linux-android`.

## Parity

14 golden cases: 9 Cartesian (distributed 4/6-node, canopy source, near-field tetra, far tetra,
far tetra + bearing prior, two tetras + bearing and amplitude constraints, collinear line,
radial refinement off + far-field prior), 2 fusion successes, 3 fusion rejections
(near_parallel, stale, out_of_range). Gate: 1e-5 relative, with an absolute floor of 1e-5 × the
array's largest magnitude.

**Review addendum (promotion review):** the onset-srp stream's compact-array collapse guard
(`minimappr/core/far_field_model_selection.py`, hooked into `solve_cartesian_tdoa`) is ported as
`tdoa_solver/far_field_model_selection.rs` with the same constants. None of the original 9 cases
triggered it, so 3 tetra cases were added (17 cases total): the guard fires for a 400 m source and a
1 m source, and declines for a refined solve just outside the 4-aperture collapse zone. With the
Rust guard disabled, case 9 fails parity. Start-point construction moved to
`cartesian_initial_positions.rs` to keep `cartesian_tdoa_solve.rs` under 500 lines.

Measured worst relative error, Rust vs Python goldens: position 4.9e-8, covariance 1.4e-7,
scalars (confidence, GDOP, residual RMS, radial observability, condition number) 3.2e-7.
Range-projection mode and fusion rejection reasons match exactly.

The collinear case is compared on rotation invariants (coordinate along the array axis, distance
from it, covariance eigenvalues). Its solution is a circle about the axis; SciPy drifts along that
null space on finite-difference rounding, so the azimuth around the axis is noise in both
languages.

## Deviation: why there is a SciPy-TRF port and not only LM

The task asked for LM with 1e-5 parity. Both are not possible on compact-array geometries,
because the Python solver does not converge there:

- `solve_cartesian_tdoa` gives SciPy residuals in seconds (~1e-6). SciPy's `gtol` test is
  absolute (`‖Jᵀf‖∞ < 1e-9`), so on a 5 cm tetra the gradient is under `gtol` at once. TRF
  stops after 0–3 evaluations, often *at the start point* (status 1, nfev 1).
- The range-projection logic after the solve is calibrated on that early stop. Radial refinement
  is adopted only when it beats the multi-start fit, so an unconverged start leaves room for it
  to win and produce `range_asymptotic` → `range_bearing_projected`.
- A converged LM finds the true minimum, which on a compact array is a noise-driven near-field
  point. Radial refinement cannot beat it, so the result stays `range_refined`. Measured on the
  golden cases with the LM engine: a tetra source at (25, 18, 6) m came back as `range_refined`
  at 0.45 m range (Python: `range_bearing_projected`, projected to ~50 m along the bearing);
  a near-field source at (0.6, 0.5, 1.8) m came back `range_refined` at (1.20, 1.01, 2.11) m,
  where Python gives `range_bearing_projected` at ~50 m.

So `CartesianTdoaSolveRequest.refinement_engine` offers both:
`ScipyTrfParity` (the default, server-equivalent, used by the oracle) and
`LevenbergMarquardt`. On well-conditioned distributed layouts the two agree within 1e-5
(unit test `engines_agree_on_noisy_well_conditioned_layout`). The joint objective always uses
LM. Its residuals are whitened, so the covariance shows a compact array's weak range directly.
It does not rely on the TRF artifact.

Other numerically negligible deviations inside the Cartesian port: Python's `np.linalg.solve` /
`inv` in fusion are replaced by an eigen-decomposition inverse (condition ≤ 1e4 enforced first).
The LM engine uses analytic Jacobians and no soft-L1, because with second-scale residuals
soft-L1's weight differs from 1 by ~1e-10.

## Robustness choices

- Every linear solve goes through `robust_linear_algebra`: non-finite input, SVD/eigen
  non-convergence, a singular matrix and condition > limit each give a `TdoaSolverError`, never a
  panic. Python's m < 3 Jacobian shape crash in `_covariance_from_svd` is avoided by zero-row
  padding, which leaves the singular values unchanged.
- LM uses `scale_diag = false`. All parameters are metres, and MINPACK's column-norm scaling
  turned a ~1e-17 Jacobian column on a collinear symmetry axis into a 1e18 m step.
- The joint and canopy solves add vertical starts (±z, plus lifted band probes). On a coplanar
  layout ∂r/∂z = 0 in the layout plane, so in-plane starts can never reach a canopy source.

## Tests

- Rust (`cargo test -p minimappr-core tdoa_solver`): 27 tests. They cover noise-free recovery
  (both engines), LM/TRF agreement, collinear flagged not refined, degenerate / insufficient /
  unknown-sensor errors, and the canopy prior picking the above-ground mirror on a flat layout.
  Loop consistency gets clean, one reflection-locked pair (exactly that pair rejected, 3 of 10
  loops broken) and NaN cases. The joint solver gets tetra DOA + 2 mono, rotated stereo cones,
  single stereo phone → explicit unobservable error, coplanar mirror reported unless the canopy
  prior breaks it, clock-domain mismatch, and fusion cones as DOA terms = closed form. There are
  Monte Carlo covariance checks for both solvers, plus unit tests for the LM driver, Brent, SVD,
  eigen and SPD-inverse helpers.
- Python: `tests/test_tdoa_solver_cross_language_parity.py` (2 tests).

## Open issues

- **Server-side finding:** the Python Cartesian solver's compact-array solves are not converged
  (absolute `gtol` on second-scale residuals). Scaling residuals by `1/sample_time_std_s`
  (or passing `x_scale`/`gtol` suited to the scale) would fix it. Its range-mode thresholds would
  then need recalibrating, which is why it was not changed here. Once fixed, the Rust parity
  engine can switch to LM and `scipy_trf_compatible.rs` can go.
- The LM engine on a compact array alone reports `range_refined` at a noise-driven near-field
  minimum (above). Do not use it for single-array Cartesian solves before that classification
  is reworked. Field Mode should use `joint_objective`.
- Optional free speed-of-sound parameter (plan §Joint objective, "optionally") is not
  implemented. The objective is position-only.
- Stereo-phone spatial aliasing above ~1.1 kHz (multiple valid delays) is not modelled. The
  caller must pass the resolved delay.
- Loop consistency with only 3 sensors (one loop) cannot tell which pair is bad; it drops the
  weakest-correlation pair. This is documented in the function.
- iOS target not built (not installed, same as core-extract).
