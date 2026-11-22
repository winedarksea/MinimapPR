# W4-port-server-track, S2 sub-stream: onset-SRP clustering + echo checks in `minimappr-core`

Branch `fm/port-onset` (from `field-mode` @ 2eb0a18, "Merge review/sim-gates fixes into field-mode").
Measured 2026-11-22. This is the S2 half of W4-port-server-track. S3 (perched tracker, log-odds,
association gate), S5 decisions, S6 entities, mesh-host wiring and the Layer 2 scenarios are on
`fm/port-server-track`.

## What was ported

The whole Python `localize_onset_clusters` pipeline as it stands after `fm/s-fixes-onset`, plus the
S5 per-cluster level evidence that the clustering attaches when attribution is on:

| Python reference | Rust (`minimappr-core/src/onset_clustering/`) |
| --- | --- |
| `onset_detection.py` (sub-band flux history, onsets, onset bands) | `onset_detection.rs` |
| `onset_frame_correlation.py` (noise spectra, band-limited ML-weighted GCC-PHAT, envelope for narrowband) | `frame_correlation.rs` |
| `onset_srp_grid.py` (direction / spatial grids, lag cache, SRP, greedy NMS peaks) | `srp_grid.rs` |
| `onset_srp_geometry_prior.py` (node plane, ground/canopy slab, node shells) | `geometry_prior.rs` |
| `onset_srp_clustering.py` orchestration | `pipeline.rs` (window), `frame_assembly.rs` (frames, frame peaks), `cluster_finalization.rs` (per cluster) |
| `cluster_frame_peaks`, `onset_cluster_echo_checks.py`, `onset_near_node_level_check.py` | `echo_checks.rs` |
| `onset_cluster_refinement.py` (SRP scores, TDOA votes, seed-only result, single-array Cartesian solve, multi-node soft-L1 TRF) | `tdoa_votes.rs`, `refinement.rs` |
| `onset_height_resolution.py`, `onset_range_observability.py` | `height_and_range.rs` |
| `onset_cluster_geometry_resolution.py` (intra-node weight cap, window grid, geometry verdicts, window holds) | `geometry_resolution.rs` |
| `localization_uncertainty.covariance_from_jacobian`, `range_observability_from_covariance` | `uncertainty.rs` |
| `onset_cluster_attribution_evidence.py` (S5 evidence, not the decision) | `attribution_evidence.rs` |
| `onset_srp_types.py` | `types.rs`, `config.rs` (defaults = server Settings defaults) |
| SciPy `least_squares(trf, soft_l1, bounds=...)` (`trf_bounds`, `select_step`) | `bounded_trf.rs`, `bounded_trf_step.rs` |
| CPython small-int `set` iteration order (reference-sensor tie-break) | `python_set_order.rs` |
| stage trace for parity | `trace.rs` |

Reused from core rather than duplicated: `tdoa_solver::solve_cartesian_tdoa` (single compact array,
SciPy-TRF parity engine), `tdoa_solver::scipy_trf_compatible` (unbounded TRF iterates; its
`TrfOutcome` now also returns SciPy's robust-scaled `result.jac`, and its soft-L1 /
trust-region helpers are `pub(crate)` for the bounded port), `robust_linear_algebra` (sorted SVD,
min-norm lstsq, ascending eigh), `bearing_fusion::stabilize_covariance`,
`far_field_model_selection::COMPACT_ARRAY_MAX_APERTURE_M`, `range_projection::is_unobservable`.

Not reused, on purpose: `srp_layout::pair_correlation` / `polar_grid` and `gcc_phat` are f32,
whole-clip, node-anchored and laid out differently. The reference's candidate indices, NMS order
and curve values have to match exactly for decision-level parity, so the f64 template-vs-search
form was ported instead. `stage1::noise_floor` is a streaming gated tracker, while the reference
uses a per-window 20th percentile.

## Entry point (for the mesh host)

```rust
use minimappr_core::onset_clustering::{localize_onset_clusters, OnsetClusteringConfig, OnsetClusteringInput};

pub struct OnsetClusteringInput {
    pub sensor_signals: BTreeMap<String, Vec<f64>>,      // sensor id -> samples, shared timeline, equal length
    pub sensor_positions_m: BTreeMap<String, [f64; 3]>,  // ENU metres
    pub sensor_node_ids: BTreeMap<String, String>,       // sensor -> node (missing = own node)
    pub sample_rate_hz: u32,
    pub sound_speed_mps: f64,
    pub interpolation_factor: usize,                     // server: gcc_phat_interp_factor = 4
    pub far_field_default_range_m: f64,                  // server: 50 m
    pub consumed_onset_samples_by_node: BTreeMap<String, Vec<usize>>, // dedupe across overlapping windows
    pub attribution_audible_snr_db: Option<f64>,         // Some(3.0) attaches S5 level evidence
}

pub fn localize_onset_clusters(input: &OnsetClusteringInput, config: &OnsetClusteringConfig) -> OnsetClusteringResult;
// localize_onset_clusters_traced(input, config, Some(&mut OnsetClusteringTrace)) also records the stages.
```

`OnsetClusteringResult { clusters: Vec<LocalizedOnsetCluster>, onset_count, frame_count,
consumed_onset_samples_by_node }`. Each cluster has:

- `status`: `Accepted` / `Tentative` / `Echo`, plus `status_reasons`. Only accepted clusters
  should surface, and the window's strongest cluster is never suppressed;
- `position_m`, `covariance_m2`;
- `range_projection_mode` (`range_refined` / `range_bearing_projected`), `range_observability`,
  `range_interval_m` (the SRP-supported range interval from the layout centroid);
- `height_resolution`;
- `onset_samples` / `template_node_ids` (the led onsets), `band_min_hz` / `band_max_hz`;
- `loudest_node_id`, the node to classify from;
- `contributing_node_ids`, `independent_onset_count`, `attribution_evidence`.

To feed it from the mesh: each node's clip, placed on the island clock, becomes that node's
sensors. Phones are one-sensor nodes. A Sirith tetra contributes its four channels with one node id.

## Parity (`tests/test_onset_clustering_cross_language_parity.py`)

Oracle: sidecar `onset-clustering-oracle` (`minimappr-ingest-sidecar/src/onset_clustering_oracle.rs`).
Signals and positions travel as base64 little-endian f64, so the inputs are bit-exact.
Goldens: `python -m tests.onset_clustering_parity_cases` writes
`tests/fixtures/onset_clustering_parity_goldens.npz` (1.0 MB).

**Inputs.** Synthetic scenes from `minimappr.sim.synthetic_entity_scenarios`, rendered at 12 kHz,
over the 0.9-1.8 s window. The audio is quantised to PCM16 and stored, so the input is exact on any
machine.

**Stage trace.** Captured on the Python side by wrapping the reference module's stage functions
(`detect_onsets`, `_build_frame`, `cluster_frame_peaks`, the echo checks, the grid builders), so
Python's behaviour is unchanged. A mismatch names the first stage that differs.

| # | Window | Path exercised |
| --- | --- | --- |
| 0 | one robin, 3 tetras (planar) | spatial grid + shells, unbounded soft-L1 TRF, mirror by prior, range profile, echo cluster |
| 1 | same, node-a only | direction grid, `solve_cartesian_tdoa`, cone rescale, echo clusters |
| 2 | robin + tanager overlapping, attribution on | 15 frames, two accepted birds, seed-only result (LM rejected by score), bearing-projected range, S5 evidence |
| 3 | robin + trunk echo image | echo rejection, bearing-projected secondary |
| 4 | robin, 4 nodes with one raised 7 m (non-planar), attribution on | **bounded** TRF (z slab), no mirror |
| 5 | case 0 with case 0's consumed onsets | dedupe: zero frames |

Results: all 6 windows agree at every stage. The decisions match exactly: onsets, frame
reference sensors, the frame each peak came from, cluster membership, statuses and reasons, modes,
height verdicts, classification nodes, arrivals.

Measured worst numeric deviations:

| Quantity | Tolerance (why) | Worst measured |
| --- | --- | --- |
| onset strength, curve weight sum / peak / sum, SRP peak weight | 1e-7 rel (f64 FFT pipelines; only summation order and FFT rounding differ) | 6.7e-13 |
| RMS bandwidth | 1e-7 rel, 1e-6 Hz abs (one-bin band: exactly 0 vs 4.5e-13 Hz) | 4.5e-13 Hz |
| position, TDOAs, residual, range observability, range interval, evidence geometry | 1e-5 rel (TRF iterates and pinv; the solver gate's tolerance) | position 3.8e-7, residual 1.4e-6 |
| covariance, GDOP | 1e-4 rel to the largest element (pinv(JᵀJ) of a √eps forward-difference Jacobian; ~1e-8 rounding noise times condition numbers up to ~1e5 on range-elongated / height-ambiguous fixes) | 1.9e-5 |
| confidence | 1e-5 rel on −ln(c) (c = exp(−rms / 4 periods) turns a 1e-6 residual difference into ~1e-5 for weak fits) | 2e-5 rel on c |

A second test re-runs Python against the goldens (`test_python_onset_clustering_still_matches_goldens`).
Runtime: about 30 s for both tests with the debug sidecar. Most of that is the 15-frame overlap
window: the spatial grid has 106 k candidates per frame, about 8 s unoptimised.

## Details that parity forced (and why)

- **SciPy `f_scale`.** soft-L1 with `f_scale = c` is reproduced by dividing the residuals by `c` and
  dividing `gtol` by `c²`. The iterates are otherwise scale-invariant. SciPy's gradient test is
  absolute, and on second-scale residuals it stops early; the reference relies on that. The bounded
  port also converts `theta = max(0.995, 1 − |g|)` back to original units.
- **`result.jac` is the robust-scaled Jacobian.** The covariance uses it, as the reference does, not
  the raw Jacobian.
- **CPython set order.** `Counter(... for index in cluster.frame_indices).most_common(1)` breaks
  reference-sensor ties by set iteration order. For example, `{3, 9}` iterates as `[9, 3]`.
  `python_set_order.rs` emulates the 3.12 table; unit vectors were checked against CPython.
- **Python rounding.** `int(round(x))` is round-half-even (`round_ties_even`), and floor division
  on negative arrivals uses `div_euclid`. `np.hanning` uses its cosine form, `np.percentile` its
  lerp rule, `np.interp` its exact-knot rule, and `geomspace` pins its endpoints.
- **The f32 core constant `BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP`** is 0.05000000074 as f64,
  so the onset path uses its own f64 0.05.

## Rust tests (`cargo test -p minimappr-core onset_clustering`, 9)

- numeric helpers: percentile, interp, irfft, arange/geomspace;
- CPython set order;
- analytic-delay scenarios: a 3-tetra fix lands within 1.5 m with `range_refined` and a range
  interval, a single tetra gives the bearing within 10° with `range_bearing_projected`, consumed
  onsets give zero frames;
- bounded TRF equals unbounded for an interior solution and respects a tight slab.

## Commands

```
CARGO_TARGET_DIR=<repo>/target cargo test -p minimappr-core                 # 289 tests (lib + integration), 0 failed
CARGO_TARGET_DIR=<repo>/target cargo build -p minimappr-core --lib --target wasm32-unknown-unknown   # ok
CARGO_TARGET_DIR=<repo>/target cargo build -p minimappr-core --lib --target aarch64-linux-android    # ok
CARGO_TARGET_DIR=<repo>/target cargo build -p minimappr-core --lib --target aarch64-apple-ios        # ok
python -m tests.onset_clustering_parity_cases                                # regenerate goldens
CARGO_TARGET_DIR=<repo>/target python -m pytest tests/ -k parity -q -p no:cacheprovider   # 40 passed, 1 skipped
```

## Deviations

- **Node order.** Python orders nodes by their first sensor id (dict insertion order) in a few
  places: nearest/farthest tie-breaks, node-centroid lists. Rust orders them by node id. These
  differ only if node ids sort differently from their sensors' ids. They never do with the
  `node:chN` convention, and they could only matter on exact distance ties.
- **The fixture is 1.0 MB** (PCM16 audio for four renders). That is bigger than the other parity
  fixtures (≤ 0.3 MB), but it keeps the input exact across machines without re-rendering.

## Remaining / not ported

- **LOCO masking** (`onset_cluster_masking.py`) and the attribution stage. The classifier rerun is
  a host callback, and the decision logic is S5 on `fm/port-server-track`.
  `ClusterAttributionEvidence` (with `node_onset_arrivals` and `predicted_snr_db`) is the input the
  decision and masking need.
- **The FusionNode seam** (`onset_srp_localization_stage.py`): span waiting, consumed-onset memory
  with TTL, and candidate assembly. A mesh host replaces it. Pass
  `consumed_onset_samples_by_node` from the previous window, as the stage's `_remember` does.
- **Mesh-host wiring.** The host's `ClipSetLocalizer` seam is the natural place. That is the
  parent stream's job; see the entry point above.

## Open issues

- The pipeline is CPU-heavy on phones: about 106 k spatial candidates per frame × pairs. This port
  keeps the reference's grid for parity. A phone profile (coarser azimuth step, fewer rings) is a
  config change (`OnsetClusteringConfig`), but it needs its own accuracy run.
- Clippy style lints remain in the new module (needless range loops, one `too_many_arguments` on
  `build_frame`). Behaviour is unaffected, and no `cargo clippy -D warnings` gate exists for core.
