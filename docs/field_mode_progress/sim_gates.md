# W4-sim-gates: statistical gate suite

Branch `fm/sim-gates` (off `field-mode` at "Merge fm/attribution into field-mode"). Row "W4-sim-gates" in `build_out_plan.md`; test strategy §What to assert.

## What exists

New workspace crate **`minimappr-sim-gates`** (test and tooling only; nothing depends on it, so core's wasm32/android builds are untouched). Each gate is a library function `fn(&GateRun) -> GateOutcome` over a seed list. The fixed-seed `cargo test` subset and the nightly runner share that one implementation.

| Module | Role |
|---|---|
| `gate_config` | Every threshold and trial constant, grouped by gate (`SimGateConfig`). Realistic residuals: clock 10 µs, layout 10 cm, phone height 10 cm, GPS 6 m, call pick 20 µs (all 1σ). |
| `gate_seeds` | Three disjoint seed ranges: calibration training starts at 5 000 000, PR seeds start at 1, nightly start = Unix time mod 5 000 000. `MINIMAPPR_SIM_GATES_SEED` narrows every gate to one trial. `GateOutcome` keeps per-seed failures (always enforced), aggregate failures (enforced from 20 trials up) and contributing seeds. Reports print repro commands. Includes the parallel trial runner. |
| `trial_geometry` | Seeded irregular layouts (4–8 phones, 15–40 m, aspect ≥ 0.35), inside/at-range sources, planar rigid motions (rotation, translation, mirror), Mahalanobis helpers. Every draw is a pure hash of (seed, purpose, index), so node-dropped and moved twins reuse the same noise. |
| `arrival_trials` | Arrival-time trials through the shipping `mesh::ArrivalTimeLocalizer`, with residual-perturbed nodes declared at their true σ. |
| `mesh_region_trials` | End-to-end Layer 2 sessions (random rings of 4–7 phones, GPS σ 6 m, calls alternating inside and outside at 20–60 m). Each fix is scored as the **gauge of the published contour polygon** (verified against a point-in-polygon test on `fix_contour` output), in the relative frame. World-frame gauges are kept as a diagnostic. |
| `srp_scene`, `srp_trials` | Layer 1 rendered scenes (5 phones, 18–30 m, 92 dB calls, trunks, ambient, ground) with residual-perturbed layout and clocks. The drawn region (contour or fan) is checked against the truth. Also: the LM-polished estimate, region area, and a per-trial minimum covering temperature (binary search on a log ladder). |
| `layout_chain` | Noisy ranges → `solve_layout` → `fit_layout_to_gps`, and a reflection test. |
| `heap_accounting` | Thread-local counting global allocator (opt-in per binary) for peak-heap measurement. |
| `gates/*` | The gates below, plus the `GATES` catalogue (name, test target, PR seed count, nightly fraction). |
| `tests/statistical_{accuracy,coverage,geometry,mesh,resources}.rs` | One `#[test]` per gate on its fixed PR seeds. A catalogue check makes sure every gate has a test. |
| `examples/nightly_sim_gates.rs`, `scripts/nightly_sim_gates.sh [UNITS] [START] [GATE]` | Nightly runner. Each gate runs `units × nightly_fraction` seeds. Per-seed failures print their `cargo test` repro; aggregate failures print the nightly command plus contributing seeds. Exit code 1 on any failure. |
| `examples/calibrate_region_parameters.rs` | Fits both region parameters on training seeds and reports training and held-out coverage, plus SRP sharpness. |
| `examples/inspect_srp_trial.rs` | One SRP trial with and without residuals and trunks, to attribute a failing seed. |

### Core changes (minimal, each explained)

1. **`mesh/fix_contour.rs` (new) + `EntityContourConfig` in `MeshConfig`.** The host's published contour used two hard-coded consts (12 points, 2σ). It is now one function that the gate scores directly, and the σ multiplier is a config key holding the **fitted value 0.52** (it was 2.0). `host_state_machine` calls the function; there is no other behaviour change.
2. **`arrival_time_localizer`: polar grid azimuth starts at the layout's principal axis.** Before, the grid was east-aligned, and the invariance gate found rotated twins of one scene landing in different LM basins **7–11 m apart** (3 of 40 seeds). After the fix: 0 of 400 nightly.
3. **`srp_seed_refiner`: height-band drift term bounded by the geometry.** A pinned-height re-solve can run off along an unobservable range direction. One reached 3·10⁸ m, which gave a σ of 10⁸ m that also depended on the frame (covariance eigenvalues differed by 54–87 % between rigid twins). The drift is now added only when it is ≤ max(sensor aperture, the fix's range from the sensor centroid). Otherwise the linearized σ, which already shows the weak range, stands.
4. **`srp_layout` `likelihood_temperature`: value unchanged (0.002), doc comment rewritten** with the calibration finding (below).

## Commands, counts, timings (Apple M2, 2026-11-22)

- `cargo test -p minimappr-sim-gates`: 5 lib + 11 gate tests (2 ignored as known failures) + 1 catalogue test. Test time **≈ 3 s** (build excluded) under the workspace test profile.
- `scripts/nightly_sim_gates.sh 200 100000`: all 11 gates, **≈ 25 s**; exit 1 with 3 gates failing (degradation, SRP coverage, anchor seed 100105), see table. Scale per gate: accuracy 400, degradation 400, mesh coverage 200 sessions / 772 fixes, SRP coverage 200 rendered trials, invariance 400, SRP+LM invariance 20, chirality 2 000, anchor 200, islands 100 sessions, loops 2 000, resources 4.
- `cargo run --release -p minimappr-sim-gates --example calibrate_region_parameters -- --mesh-sessions 300 --srp-trials 120 --held-out-mesh 100 --held-out-srp 48`: mesh fit 2 s, SRP fit ≈ 5 min.
- Wall-time checks in `resource_bounds` run only in builds without debug assertions (`--release`, nightly). Heap checks run whenever the counting allocator is installed.

## Calibration

### Mesh published contour: `EntityContourConfig::contour_sigma_multiplier` = **0.52** (fitted)

The minimum covering multiplier is exact per fix (the polygon gauge), so the fit is the 94.5 % order statistic over training fixes.

| Set | Fixes | Coverage at 0.52 | Coverage at the old 2.0 |
|---|---|---|---|
| Training (seeds 5 000 000+, 300 sessions) | 1 151 | **0.944** (fit 0.521) | 1.000 |
| Held-out PR (seeds 1–100) | 382 | **0.955** | 1.000 |
| Nightly (seeds 100 000–100 199) | 772 | **0.946** (inside 0.902, outside 0.990) | n/a |

The distribution of minimum covering multipliers: median 0.14, P90 0.40, P95 0.54, P98 0.73. **The LM covariance overstates the error by about 4–7× in σ.** The old 2σ polygon covered 100 %, which is outside the band on the conservative side. The multiplier absorbs this. Likely cause: conservative declared σ in the mesh (phone height σ 0.25 m, clock filter σ, layout σ), plus the height-band drift term. A tracker or NEES consumer of `CallFix::covariance_m2` sees the same overstatement (see open issues).

Flagged mirrors are exempt, as in the mesh invariants (28 of 800 nightly; 0 unflagged). In the world frame, with the anchor's GPS error charged, coverage is **0.4 %**: a 6 m GPS anchor moves the drawn region metres. The map draws phones with the same anchor, so the gate scores the relative frame, as `ScenarioOutcome::scored_fixes` does.

### SRP likelihood temperature: **NOT calibratable; left at 0.002**

Fit on 120 rendered training trials (half inside the layout, half at 40–150 m) with ±10 cm / ±10 µs residuals and trunks. Each trial's minimum covering T comes from a binary search on a 31-rung log ladder from 0.001 to 1.

| | Coverage | Inside sharpness |
|---|---|---|
| Shipped T = 0.002, training | **0.233** | dots of 0.0–0.2 m² |
| Shipped T = 0.002, PR seeds 1–24 / nightly 200 | **0.125 / 0.180** | 0 fans, median area 0.01–0.02 m² |
| Fitted T = 0.0126, training / 48 held-out | 0.950 / **0.958** | **23 of 24 inside trials drew a bearing fan; median area 61 000 m²** |

Coverage jumps from a dot straight to a whole-grid fan. On a single trial (seed 5 000 008, error 0.11 m) the region is under 0.05 m² up to T = 0.0063 and then a fan of tens of thousands of m² at 0.0079. In between, it never covers. With clean inputs (true positions and clocks), T = 0.0126 also smears the core contour test's inside source over 115 m² and exhausts the region budget. That breaks `srp_layout_contour`'s compactness gate, which is how this was caught before shipping it.

Cause: the score is Σ Huber(1 − c/peak). Under realistic residuals each pair's PHAT peak moves by up to 0.4 ms (10 cm ≈ 0.29 ms per node), so no single point gets every pair to its peak. The temperature that covers that inconsistency also lifts the background floor (cells where one or two pairs are off pay only a bounded Huber cost), and with thousands of such cells their mass swamps the peak. A scalar temperature cannot calibrate this surface.

Decision: **the fitted value is not shipped** (that would ship whole-map fans and pass coverage by smearing). The gate checks coverage *and* sharpness (inside fans ≤ 10 %, inside median area ≤ π·3² m²), so both values fail it honestly. The PR test is `#[ignore]` with that reason, and nightly runs it. Needed: score-side tolerance for declared layout/clock σ (for example, max-pool each pair's curve over ±k·σ_pair before scoring), or a region derived from the LM covariance, then refit. The mesh path is not affected: it publishes the LM-ellipse polygon calibrated above, and the clip-SRP `ClipSetLocalizer` is not wired yet.

SRP accuracy itself is within the strategy gates under residuals: inside median 0.27–0.34 m, P95 1.7–2.2 m; outside bearing P95 1.1–3.4°. But residuals cost about 100× against true inputs (inside error 0.00–0.01 m becomes 0.1–2.5 m), driven by height/horizontal coupling on near-flat layouts. Runtime: 0.03–0.8 s single-threaded (median 0.5 s; budget 1 s) against the criterion bench's 54 ms. Branch-and-bound prunes far less well when the pair peaks disagree.

## Gate table

PR = fixed seeds in `cargo test`; nightly = `scripts/nightly_sim_gates.sh 200 100000` (each gate scales the 200 units by its nightly fraction).

| Gate | Test target | PR seeds → result | Nightly → result | Key numbers |
|---|---|---|---|---|
| `monte_carlo_accuracy` | statistical_accuracy | 200 → PASS | 400 → PASS | Inside median 0.18 m, P95 0.73 m (gates 1.5 / 3 m). Bearing P95 1.45° at 50 m (gate 5°), 1.3° at 100 m, 1.3° at 200 m (394/400 solved). LM NEES mean 1.2 (2 dof; gate band 0.2–3) |
| `graceful_degradation` | statistical_accuracy | 100 → **FAIL (ignored, known)** | 400 → **FAIL** | Muted phone outside old region 5 % (gate 10 %): PASS. Pocketed phone outside clean region 27–35 % (gate 10 %). Decisive pockets (≥ 8 pair σ) rejected only 58–60 %, dragging the estimate 5–24σ. σ shrank > 10 % on 16–19 % of node removals (gate 5 %). 3 phones: median 1.7 m, 7–11 % refused, rare 4.7σ misses |
| `mesh_contour_coverage` | statistical_coverage | 100 sessions / 382 fixes → PASS 0.955 | 200 / 772 → PASS 0.946 | Band 0.90–0.98. Inside calls 0.90, outside 0.99. World frame 0.4 % (diagnostic) |
| `srp_contour_coverage` | statistical_coverage | 24 → **FAIL (ignored, known)** 0.125 | 200 → **FAIL** 0.180 | See SRP calibration. Accuracy sub-gates pass |
| `geometric_invariance` | statistical_geometry | 100 → PASS | 400 → PASS, 0 discrepancies | Proper and mirrored motion, ±500 m translation. Arrival fixes within 1 mm and σ eigenvalues within 1e-3; anchored layout within 1 mm. Failed 3/40 before core fix 2 and 1/40 before core fix 3 |
| `srp_lm_invariance` | statistical_geometry | 4 → PASS | 20 → PASS | Rendered twin scenes, SRP + LM: worst 3.3e-6 m (gate 1 mm) |
| `chirality` | statistical_geometry | 400 → PASS | 2 000 → PASS | GPS-fit wrong side: 0/259 at ≥ 0.99; 4.4 % in [0.9, 0.99); 29 % in [0.6, 0.9) (all within calibration slack). Bearing pairs locked 1 740/2 000, 0 wrong |
| `anchor_stability` | statistical_geometry | 30 → PASS | 200 → **1 seed FAIL** (100 105: 1.18 m) | 20 m / 30 s jump against its no-jump twin: median 0.32 m, max 1.18 m (gate 1 m); PR seed 38 is 1.13 m. Still window 150–300 s: median 0.54 m, within max(1 m, 3 × the filter's σ). Unfiltered per-round fit (what the mesh publishes): median 7 m, max 24 m. Persistent jump: median 1.9 m, max 7.2 m (reported) |
| `island_isolation` | statistical_mesh | 20 → PASS | 100 → PASS | 2–3 clumps 150–300 m apart, calls in and between clumps, half with a walk-in merge. 0 cross-clump clip sets, 0 stale or mixed domains, 0 pre-merge domain reuse; 50/50 merges ran a call on the merged domain |
| `loop_consistency` | statistical_mesh | 300 → PASS | 2 000 → PASS | Reflection-locked pair caught 2 000/2 000 and never costs an extra pair. Clean false rejection 5.1 % against a union-bound limit of 6.1 % |
| `resource_bounds` | statistical_resources | 1 → PASS | 4 → PASS | Arrival LM 0.7–2 ms, 3 KiB peak. Layout solve < 0.1 ms. SRP 0.03–0.5 s, 11–14 MiB peak (budget 1 s). Dawn chorus 120 s → 240 s: 0.04–0.17 s, peak heap 1.37 → 2.16 MiB (×1.58 for ×2 length, gate ×2). 33/73 calls all localized, 0 dropped, no silent loss |

Failures from before each fix, and what the gates caught on the way:
- invariance basins (fixed in core, change 2);
- runaway drift σ (fixed in core, change 3);
- the SRP temperature trap (not shipped);
- flagged-mirror tail in the mesh contour fit (exempted, as in the mesh invariants).

## Open issues

- **SRP likelihood region is not calibratable by temperature** (above). This blocks drawing SRP contours to users. Needed: σ-aware pair scoring or an LM-derived region, then refit with `calibrate_region_parameters` and un-ignore `srp_contour_coverage`.
- **Pocketed or late phone not rejected** by `ArrivalTimeLocalizer` / `JointLmSeedRefiner` in about 40 % of decisive cases (3.5–4.8 ms, ≥ 8 pair σ), worst when the late phone is the pair reference (node 0, equal σ): pairs are all formed against it, so its delay is shared and no single residual stands out. The χ²/dof > 9 leave-one-out retry does not fire often enough. Ideas: reference-free pairing (all pairs with a correct covariance), or leave-one-out driven by a per-node residual test. Repro: `MINIMAPPR_SIM_GATES_SEED=26 cargo test --release -p minimappr-sim-gates --test statistical_accuracy graceful_degradation -- --ignored --nocapture`.
- **σ is not information-monotone**: removing a phone shrank σ by > 10 % on 16–19 % of removals (up to 72 %). The height-band drift term is a nonlinear quantity, not information. Shared with the next item.
- **Mesh LM covariance overstates the error by 4–7× in σ** (median contour gauge 0.14 at multiplier 1). The 0.52 multiplier fixes the drawn contour, but `CallFix::covariance_m2` should get the same scale before a tracker or NEES consumes it. Better: find why the declared σ are conservative (assumed phone height σ 0.25 m, clock σ, the drift term) and refit.
- **Anchor transient 20 m jump moves the filtered anchor > 1 m in about 1 of 200 seeds** (PR-range seed 38: 1.13 m; nightly 100 105: 1.18 m). A 20 m jump at σ 6 m is 3.3σ: the 3σ hard drop sometimes misses it after the fit absorbs part, and Huber still pulls about 2 m per fit. A persistent jump reaches 1.9 m median, 7 m max. Ideas: per-phone residual history while still (a jump relative to its own past), or a lower hard-drop threshold for still groups. PR seeds run 1–30 (green). Per the strategy's seed policy, 38 and 100 105 join the list after a fix.
- **The mesh publishes the unfiltered per-round anchor** (`island_sync_engine` refits each round; `MapAnchorFilter` is not used). On the same transient jump that moves the map 7 m median, 24 m max. Wire `MapAnchorFilter` in, aligning frames with `layout_frames::previous_to_new_frame`.
- **World-frame contour coverage is 0.4 %.** The GPS anchor's uncertainty is not in any drawn region. The map stays self-consistent (phones move with birds), but absolute positions (export, cross-island, federation) need the anchor σ added.
- **SRP runtime** under realistic residuals is 0.03–0.8 s per solve on M2 (median 0.5 s), against 54 ms in the clean bench. The resource gate's 1 s budget is a real-time target. Mobile will need the branch-and-bound / region ideas in core_srp.md.
- Wall-time resource checks are skipped under debug assertions, so `cargo test --workspace` (test profile) enforces only heap bounds; `--release` and nightly enforce both.
- Not covered here: tonal cycle-skip fixtures (needs the unbuilt envelope path); the clock-sync residual gate (core-sync owns it); golden-bundle replay (Layer 3).
