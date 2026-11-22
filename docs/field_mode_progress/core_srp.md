# W2-core-srp: SRP-PHAT for arbitrary node layouts

Branch `fm/core-srp` (off `field-mode`). Row "W2-core-srp" in `build_out_plan.md`.

## What exists

New module tree `minimappr-core/src/srp_layout/` (one `pub mod srp_layout;` line in `lib.rs`).
`srp_phat.rs` is **untouched**, so the sidecar's tetra path and its parity tests are unaffected.

| Module | Responsibility |
| --- | --- |
| `config.rs` | `LayoutSrpConfig` and sub-configs (grid, correlation, scoring, refinement, uncertainty). Every tunable lives here |
| `clock_resampling.rs` | `NodeClockModel` (anchor sample, anchor island time, rate ratio). One Kaiser-windowed sinc pass (16 taps each side, 1024 phases) maps each clip onto the island timeline, applying offset and rate together. Integer-aligned disciplined clips are copied bit-exactly |
| `layout_input.rs` | `LayoutNode` (position, world-frame mic offsets, clock, pair weight), `NodeClip`, `IslandAnalysisWindow`, errors. Pair plan: all intra-node pairs of each array (sharing `intra_node_pair_group_weight`) plus one inter-node pair per node pair, between the nodes' first mics |
| `pair_correlation.rs` | Band-limited GCC-PHAT coherence curves (absolute coherence, perfect = 1, noise ≈ 1/√K), upsampled 2× through a zero-padded spectrum, Catmull-Rom interpolation, and a dyadic max-pool pyramid for O(1) **max over a TDOA interval** |
| `polar_grid.rs` | `LayoutFrame` (centroid, principal-axis azimuth, ground z). `SearchCell::{Near, FarField}`: dense 2 m rings to 30 m, log rings (ratio 1 + 2π/72) to 200 m, 8 height slabs over 0–35 m above ground, and a far-field bearing ring of 72 × 4 plane-wave patches over 0–40° elevation. Conservative per-pair TDOA interval bounds (second-order Taylor bound, capped by the Lipschitz bound) |
| `robust_scoring.rs` | score = −Σ wₚ ρ_δ(1 − cₚ/peakₚ) with Huber ρ (knee 0.3). wₚ = node weights × a coherence-significance factor (0 at the noise-only expected peak √(2 ln L)/√K, 1 at 3× that). This is coherence-based, not level-based, so faint but coherent pairs keep their weight |
| `refinement.rs` | Coarse-to-fine **best-first branch-and-bound**: pop the highest envelope bound, halve every axis, keep children whose bound beats the best exact score (the incumbent), stop when nothing open can beat it by more than the tolerance |
| `region_refinement.rs` | Adaptive refinement of horizontal columns for the contour. Whole columns split (footprint into 4, live slabs halved while thicker than the footprint) until the envelope is tight (TDOA intervals ≤ 25 µs) or the footprint is below max(2 cm, 0.5 % of range). Far-field columns refine in azimuth and elevation |
| `likelihood_region.rs` | L = exp((S − S_best)/(T·Σw)), height-marginalized Σ_z L·Δz per column, HPD set holding 95 % of the mass, `contains_xy`. Bearing-fan fallback when the estimate is far-field, the region reaches the grid edge, far-field mass is above 25 %, or the radial span is above 0.6 × max(range, aperture) |
| `layout_geometry.rs` | Node horizontal scatter (aspect ratio √(λmin/λmax), principal axis), horizontal GDOP (total, radial, cross-radial), mirror across the axis |
| `lm_handoff.rs` | `LmSeed` (seed position, or far-field direction, plus per-pair `PairTdoaObservation`s taken at the **local** curve peak nearest the seed's predicted lag, the pair weight, relative peak coherence, height bounds, Huber knee) and `trait TdoaSeedRefiner` for W2-core-solver to implement. No LM is implemented here |
| `searcher.rs` | `LayoutSrpSearcher::new` (grid and per-cell×pair interval table, built once per layout and reused across clips and onset frames) and `localize`. It also has `exact_scores_at` for diagnostics |
| `mod.rs` | Result types (`SourceEstimate`, `MirrorCandidates`, `LayoutGeometryDiagnostics`, `LayoutSrpResult`) and `localize_layout` |

### Collinear layouts

When aspect < 0.15, each side of the principal axis is refined separately, and both are returned as `MirrorCandidates`. The mirror counts as resolved only on evidence from **rigid-array (intra-node) pairs**, measured in their own temperature (T · Σw_array) with a threshold of 6.

Reasoning: in the sim, a mono line with ±15 cm cross-trail jitter "resolves" the mirror by 67 T, because the sim's geometry and clocks are perfect. In the field those tens of centimetres sit inside the layout solve's own position error, so that preference is not trustworthy. With this rule:

- a mono line is never resolved;
- a tetra on the line resolves by 43–70 T, to the correct side.

## Results (Apple M2, 2026-11-22)

Scenes: sim, 5 phones about 24 m across (irregular pentagon), 92 dB SPL calls (trill, FM sweep, harsh), ambient 35 dB, ground reflection, ISO absorption. Contour tests also enable random trunks. Clock models are set to the true device clocks, standing in for a converged clock filter.

| Check | Result |
| --- | --- |
| Inside 5-phone group, 8 sources at z 2–20 m | median **0.01 m**, P95 **0.03 m** horizontal (gate 1.5 / 3 m) |
| 3 phones + 2 tetra, same sources | median 0.01 m, P95 0.02 m |
| 50 / 100 / 150 m, 3 azimuths each | bearing error ≤ **0.01°** (gate 5°); range within 0.7 m |
| 400 m (far field), 3 azimuths | bearing error 0.22–1.11°; display = bearing fan, half-width ≈ 1–1.5° |
| Contour coverage, 5 inside + 4 outside, with trunks | 9/9 covered. Inside, the region is 0.0–0.6 m² and drawn as a contour. At 40–60 m it is a wedge (radial span 8–13 m). At 120–180 m it reaches the grid edge and switches to a fan |
| Collinear trail, 5 mono phones, bird 15 m off-trail | collinear flagged; both candidates reported; true candidate within 0.08 m; **never resolved** |
| Trail with a tetra in the middle | resolved to the true side, primary error 0.10–0.20 m |
| ±60 ppm drift, anchor 10 s before the call | rate-corrected median **0.007 m**; uncorrected median 1.06 m (worst 1.26 m) |

**Runtime per solve** (release, `cargo bench -p minimappr-core --bench layout_srp`; ~0.6 s window, 19 296 coarse cells): see the table at the end of this file. Breakdown for 5 phones inside the layout, from an ad-hoc timing run:

| Stage | Time |
| --- | --- |
| Alignment and curves | ≈ 9 ms |
| Coarse scan | ≈ 4 ms |
| Branch-and-bound (~54 k cell evaluations) | ≈ 27 ms |
| Region refinement (~65 k evaluations) | ≈ 20 ms |
| `LayoutSrpSearcher::new` (once per layout) | 3–4.5 ms |

Two optimisations brought this down from ~220 ms:

- per-cell geometry is now computed once instead of per pair;
- correlation upsampling dropped from 4× to 2×.

## Commands

- `cargo test --workspace`: green. Counts: core 78 unit tests + 9 scenario tests; sidecar 116; sim 20.
- `cargo build -p minimappr-core --target wasm32-unknown-unknown` and `--target aarch64-linux-android`: pass.
- `python -m pytest tests/ -k "cross_language_parity or rust_tdoa_solve or ingest_parity" -q -p no:cacheprovider`: 42 passed, 1 skipped.
- `cargo clippy -p minimappr-core --all-targets`: no warnings in `srp_layout` (all 14 remaining warnings are in pre-existing files).

## Deviations

- **Branch-and-bound instead of a fixed-width beam.** The first version was a beam. Big coarse cells all hit the maximum envelope bound together, so the beam dropped the true cell on ties: truth scored ≈ 0 while the estimate scored −2.5. Best-first branch-and-bound with an incumbent fixes this. The result reports `cells_evaluated` and `is_certified_within_tolerance`; the latter goes false if the 200 k evaluation budget runs out.
- **The likelihood region is refined adaptively, not read off the coarse grid.** Coarse 2 m × 4.4 m envelopes span metres of path difference, so a region read from them covered the whole disk. Pruned cells carry no mass: their upper bound times a large area would otherwise swamp a centimetre-scale true region.
- **Default temperature is 0.002, not fitted.** At 0.02 the background floor (cells where one pair is off) dominated. The plan's golden-bundle fit is still pending (see open issues).
- **Inter-node pairs use each node's first mic only.** This keeps the pair count O(nodes²).
- **`minimappr-sim` is a dev-dependency.** It sits in a labelled block in `minimappr-core/Cargo.toml`. The sim does not depend on core, so there is no cycle. `rayon` and `zip` reach dev builds only; wasm and android lib builds are unaffected.
- **New `layout_srp` criterion bench.** It shares `tests/srp_layout_support/mod.rs` through `#[path]`.

## Open issues

- **Runtime is ~54 ms per solve (99 ms with 22 pairs), not the plan's "few ms".** Branch-and-bound and region refinement dominate, because the envelope bounds stay loose for big cells. Ideas:
  - tighter anisotropic bounds (per-axis gradients instead of a bounding ball);
  - reuse the BnB-explored cells for the region;
  - a coarser `region_lag_resolution_s`;
  - cap the region at an onset-frame budget.
- **Temperature and coverage are uncalibrated.** On clean sim, regions are tiny and coverage is trivially 100 %. T, `region_lag_resolution_s` and the fan thresholds need the golden-bundle / nightly Monte Carlo fit (90–98 % gate) once Layer 3 bundles exist.
- **Not built** (planned elsewhere or later):
  - narrowband / tonal-whistle envelope path (cycle skipping);
  - per-bin SNR soft PHAT weights;
  - onset-weighted short frames and spatial clustering (W3);
  - stereo-phone cone pairs (they work as a 2-mic `LayoutNode`, but there is no orientation handling).
- **Tests use the true clocks as the clock model.** Sensitivity to clock-filter residuals and layout-position error is not yet characterised. Tests with ±10 µs / ±10 cm perturbations would be the next step.
- **The far-field ring's mass prior is arbitrary.** Each far-field bearing gets the area of the outermost ring sector, a heuristic.

## Release bench (criterion, median)

| Case | `LayoutSrpSearcher::new` | `localize` |
| --- | --- | --- |
| 5 phones, source inside (10 pairs) | 2.8 ms | **54 ms** |
| 5 phones, source at ~100 m | 3.1 ms | **54 ms** |
| 3 phones + 2 tetra, inside (22 pairs) | 4.4 ms | **99 ms** |

## Review fixes (branch `review/core-srp`, 2026-11-22)

- **Envelope was not admissible.** `max_coherence_over_interval` bounded the stored samples, but the exact score reads the Catmull-Rom interpolant. Near a sharp peak the interpolant overshoots both neighbouring samples, and for sub-step intervals only three points were probed. A cell's envelope could therefore fall below its own exact score, and branch-and-bound could prune the true cell. The envelope now pools exact per-segment cubic maxima, padded by a few f32 ulps. It is also tighter than the old "+1 step" slack, so region refinement uses about 30 % fewer evaluations. One visible effect: the 120 m contour case is now drawn as a wedge contour instead of a fan (relative span 0.47 < 0.6).
- **Mirror side.** The primary candidate used to be the all-pair winner even when the array pairs decisively favoured the other side. That reported a wrong-side primary and marked it unresolved. Now, when |array gap| exceeds the threshold, the array picks the side. `score_gap_temperatures` can therefore be negative. Covered by `tests/srp_layout_perturbed.rs`, where the phone-jitter model is mirrored: all-pair gaps are −47 and −55 T, while the array resolves to the true side.
- **Input validation.** `LayoutSrpSearcher::new` now rejects non-finite geometry, clocks and weights, a zero sample rate, and a non-positive sound speed (`LayoutSrpError::InvalidInput`). A significance ratio ≤ 1 no longer produces NaN weights. Edge cases are covered in `src/srp_layout/edge_case_tests.rs`: 1 sensor, coincident nodes, a lone tetra, empty audio, NaN audio, and a zero window.
- **Perturbed inputs (±10 cm layout, ±10 µs clock).** Median error is 0.08 m, but the worst case is 3.0 m (a z = 15 m source), and **contour coverage is 2 of 4**. This confirms that T = 0.002 is overconfident once inputs are realistic. Calibrating T on residual-perturbed sims or golden bundles is a prerequisite before showing contours to users.
- **Open: the far-field ring is a plane-wave model.** For sources just past `outer_radius_m`, inter-node wavefront curvature (|a⊥|² − |b⊥|²)/(2R) reaches about 0.3 m at 200 m and 0.18 m at 400 m for this 24 m layout. That is many PHAT peak widths. The far-ring exact score and bounds ignore it, which likely explains the 0.2–1.1° far-field bearing errors against ≤ 0.01° at 150 m. Options: extend the log rings, or give far cells a finite range band with a curvature term (which then needs an angular, not lag-based, convergence test).
