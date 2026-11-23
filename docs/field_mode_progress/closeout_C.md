# Closeout Phase C: mesh fixes gated in simulation (C1–C5)

Branch `field-mode-closeout`, one commit per step:

| Step | Commit | Gate result |
| --- | --- | --- |
| C1 filtered map anchor | `4062266` | `anchor_stability` passes; `anchor_persistent_jump` un-ignored and passes |
| C2 anchor error in polygons | `11fc825` | new `mesh_contour_world_coverage` gate (left ignored in C2, un-ignored in C3) |
| C3 mesh LM covariance | `5f22dbe` | NEES gate now bands the median; `mesh_contour_coverage_by_class` and `mesh_contour_world_coverage` un-ignored and pass |
| C4 late-phone rejection | `bf9b03b` | **`graceful_degradation` still ignored** (better, not passing) |
| C5 successor restores entities | `be52a7d` | new mesh-sim scenario `successor_restores_entities` passes |

Suites at `be52a7d`:
- Rust (`cargo test --workspace`): 573 passed, 2 ignored. The baseline was 559 passed, 4 ignored.
  The two still ignored are `graceful_degradation` and `srp_contour_coverage` (Phase D).
- Python: see the end of this note.

How the numbers were taken:
- "PR" means the fixed `cargo test` seeds.
- "Nightly N@S" means N seed units from start seed S (`scripts/nightly_sim_gates.sh N S [gate]`).
- "Before" numbers come from a scratch build of `3cd6f78`, the branch head before Phase C.

## Seed policy deviation

`PULL_REQUEST_SEEDS` (`minimappr-mesh-sim/src/seeded_random.rs`) belongs to the Layer 2 scenario suite. Gate seeds 38 and 100105 are
`minimappr-sim-gates` seeds and mean nothing to the scenarios. So the gate catalogue gained a per-gate `regression_seeds` list,
which `pull_request_seeds(count, regression_seeds)` appends to the PR range:
- `anchor_stability` and `anchor_persistent_jump` now run seeds 1–30 plus 38 and 100105.
- Seed 26 (`graceful_degradation`) is already inside that gate's PR range 1–100.

## C1: `MapAnchorFilter` in the mesh

**What was built**
- **New `layout/filtered_map_anchor.rs` (`FilteredMapAnchor`).** Each round's `fit_layout_to_gps` goes through `MapAnchorFilter`.
  - `MapAnchorFilter::reframe` (new) re-frames the filter into each new MDS solve with T_new = T_old ∘ F⁻¹, using the new
    `RigidTransform2d::compose` and `inverse`.
  - The island engine passes `layout_frames::previous_to_new_frame` as F. When there is no common frame, it resets.
- **Still-phone jump handling.**
  - While the group is still, each phone keeps a running mean of its own GPS fixes. This is an EWMA with weight 0.05, about
    100 s at a 5 s fix rate, and a plain average until that window fills.
  - A fix more than **2 fix-σ** from that mean is a jump. This is the plan's "lower hard-drop threshold while still": the fit's
    own drop is 3σ. The multipath bias cancels against the phone's own past, so only jitter is left in the deviation.
  - The jump (the fix minus the pre-jump mean, averaged while it lasts) is subtracted from the phone's fixes. A held phone is
    released only below **1σ** (hysteresis).
  - If a majority of phones jump together, nothing is held. That is the group moving, not multipath.
  - Any motion clears the histories.
- **Mesh wiring.**
  - `IslandState` owns the tracker. `refresh_anchor(island, fixes, now_ms, group_is_still, config)` gets stillness from member
    motion states in `host_sync_pipeline`.
  - `island.anchor` and `layout.anchor` now hold the *filtered* fit.
  - A chirality lock from bearings resets the filter only if it contradicts it. If it agrees, it only marks the anchor as locked,
    so the same fixes are not fused twice.
  - The anchor code moved to `mesh/island_anchor.rs`, which keeps `island_sync_engine.rs` at 385 lines.
- **Gate.** `anchor_gate` now drives `FilteredMapAnchor` on unaligned fresh layouts, re-framing each round as the mesh does. It
  no longer pre-aligns layouts and calls the raw filter.

**Deviations**
- Subtracting the jump, rather than dropping the phone or freezing its fix, was chosen after measuring:
  - Dropping the phone moves a 10 m, 5-phone fit by about bias/4, and up to 0.25 rad of rotation.
  - Freezing the fix at its mean fails differently: the held phone then has no jitter while its twin does, and the twin
    comparison drifts.

**Gates**

| Gate | Before | After |
| --- | --- | --- |
| `anchor_stability` PR (1–30) | pass (seed 38 fails: 1.13 m) | pass, plus 38 and 100105 |
| `anchor_stability` nightly 3 × 1000 (starts 100000, 200000, 1300000) | 104 / 3000 seeds fail | 1 / 3000 (seed 200465, a still-window drift of 4.05 m with no jump, which fails identically before) |
| Transient jump response, nightly | median 0.32 m, max 1.58 m | median 0.08 m, max 0.69 m |
| `anchor_persistent_jump` PR (1–30, 38, 100105) | 26 / 30 seeds fail; 38 → 6.02 m, 100105 → 6.42 m | pass: median 0.19 m, max 0.62 m |
| `anchor_persistent_jump` nightly 3 × 1000 | 2378 / 3000 fail (median 1.9 m, max 9.9 m) | **5 / 3000 fail** (median 0.18 m, max 1.76 m) |
| Unfiltered per-round fit (diagnostic) | median 7 m | median 2.1–2.4 m |

`anchor_persistent_jump` still fails about 1.7 seeds per 1000. The jump is held in those runs, but the per-round fit's rotation on
a 10 m layout under 6 m GPS is close to chaotic: twin fits differ by up to 2.5 rad. The no-jump twin's filtered rotation wanders
about 0.1 rad over 300 s, and the twins drift apart through it. The 200-seed nightly at 100000 passes. A run of the default 200
seeds then fails about 30 % of the time. **Needs a decision:** accept that nightly flake rate, or replace the filter's
independent-fit model with a fit to time-averaged fixes while still (not done here).

## C2: anchor error in map polygons

**What was built**
- **New in `mesh/fix_contour.rs`.**
  - `AnchorUncertainty` and `world_frame_covariance`: Σ_world = Σ_fix + σ_t²·I + σ_θ²·r⊥r⊥ᵀ. Here r is the fix's offset from the
    anchored layout centroid, which is the anchor fit's rotation centre.
  - In C3, `RegionContext` (layout centre, radius and anchor) and `published_covariance` replaced the bare anchor.
- **Callers.** The island engine exposes the filtered anchor's error. The call pipeline stamps it on each request and record.
  Both publish paths draw the result:
  - `host_state_machine`, for per-call updates;
  - `host_entity_tracking`, through `EntityUpdateContext::for_islands`.
- **Published anchor σ.** The filter assumes independent rounds, so its σ shrinks as 1/√n. Multipath bias does not average
  down. The published σ² is ρ²·σ_fit² + (1 − ρ²)·σ_filter², with `correlated_gps_error_fraction` ρ (see C3 for the value).
- **Gate.** The new `mesh_contour_world_coverage` scores the published polygon against the world truth.
  - **Bug found in the old world diagnostic:** it compared host-frame fixes with site-frame truth. The host's ENU plane is
    centred on the first GPS fix it sees, not on the site origin.
  - The truth now goes through geodetic coordinates into the host plane. The old "world coverage 0.4 %" was mostly that frame
    offset.

**Gates**
- The relative-frame `mesh_contour_coverage` is unchanged, at 0.946 nightly.
- World coverage with the 0.52 multiplier was 0.19. The gauge, though, was already Gaussian-shaped: median 1.06 and P95 2.47,
  against 1.18 and 2.45 for a calibrated 2-D Gaussian.
- The 0.52 multiplier absorbed an overstated Σ_fix, so it under-drew the honest anchor term. The gate stayed ignored until C3.

## C3: mesh LM covariance

**What was built (core)**
- `assumed_phone_height_sigma_m` changed from 0.25 to **0.115**: handheld ±0.2 m uniform, σ = 0.4/√12, as the plan asked.
- **Drift inflation.** `srp_seed_refiner` now clamps the height-band drift at its geometric bound, keeping its direction,
  instead of dropping it above the bound.
- **Height marginalization reworked.** This goes beyond the plan's literal list; the reason is below.
  - Old: Σ_h = (JᵀJ)⁻¹_hh + ΔΔᵀ/12.
  - New: Σ_h = (I_hh)⁻¹ + g gᵀ·Var(z), where g = Δ/band is the horizontal shift per metre of height, and Var(z) is the height
    variance the data leave once combined with the band taken as a uniform prior (Schur complement minus the weak prior's
    share).
  - Why: the old term charged the whole band even when the TDOAs pin the height, and it added the drift to a block that already
    spread the weak 15 m prior into x and y.
  - Measured under correctly declared noise: without the drift term the NEES median is 1.01; with the old term it is 0.42.
  - The data get credit for height only when they can vouch for it: the solved height lies inside the band, and there are at
    least 3 pairs (this second condition was added in C4).
  - Otherwise the old conservative term stays. The first condition comes from mesh `offline_host_loss` seed 42, where a
    misplaced node produced a 22 m height. The second comes from 3-phone fixes, which landed 5–10σ out.
- **Published contour.** One scalar multiplier could not calibrate both inside and outside calls, so the split is a per-geometry
  **σ scale on the fix covariance** (`FixCovarianceCalibrationConfig`). The polygon is drawn at the **Gaussian** 12-gon
  multiplier.
  - `contour_sigma_multiplier` is **2.465**: computed, not fitted, as the 12-gon whose 2-D Gaussian content is the 0.945
    calibration target.
  - `inside_layout_sigma_scale` is **0.695** and `outside_layout_sigma_scale` is **0.315**. Both were fitted on 300 training
    sessions (seeds 5 000 000+, 1 152 fixes) by `examples/calibrate_region_parameters.rs`, which now prints per-class fits.
  - Inside means within the island layout's circumradius of its centroid. Host and truth classes agreed on 576 of 576.
  - Why scales and not two multipliers: the multiplier also scales the anchor term, which is honest. Fitting the scale to Σ_fix
    alone keeps the world polygon calibrated.
- **What `CallFix::covariance_m2` holds.** It stays the raw LM covariance; the calibration applies only to what is drawn.
  - Applying the scales to `CallFix` itself broke mesh scenario invariants. That is the "4σ + 0.5 m" rule, on
    `offline_host_loss` seed 42.
  - So trackers and NEES consumers still see the raw mesh σ: about 1.4× overstated inside the layout and about 3× outside.
- **`correlated_gps_error_fraction`** is **1.0**, the conservative bound. With 0.8, world coverage was 0.88. Like canopy
  multipath, the simulator's GPS model is about 99.6 % bias.
- **NEES gate.** It now bands the **median** at **[0.95, 1.85]** and caps the mean at 3.
  - The band is 2·ln 2 = 1.386 ± 3 standard errors of a χ²₂ sample median at the 200-seed PR count (SE = 2/√n = 0.14).
  - Equivalently, σ mis-scaled outside about 0.83×–1.16× fails.
  - The old mean band (0.2–3) passed a Σ overstated by 1.8× in σ. Both constants now live in `AccuracyGateConfig`.

**Gates**

| Gate | Before | After |
| --- | --- | --- |
| NEES median, nightly 400@100000 / 300000 / 700000 / 1300000 / 2100000 | 0.44 / 0.43 / 0.51 / … (fails new band) | 1.03 / 1.13 / 1.17 / 1.20 / 1.29 (pass) |
| NEES median, PR | 0.60 | 1.30 |
| `mesh_contour_coverage_by_class` PR | inside 0.92, outside 0.99 (fail) | 0.974 / 0.958 |
| `mesh_contour_coverage_by_class` nightly 200@100000, 300000, 700000, 1300000, 2100000 | fail (outside 0.99–0.997) | **5 / 5 pass**; inside 0.94–0.97, outside 0.95–0.98 |
| `mesh_contour_world_coverage` PR | — | 0.947 |
| `mesh_contour_world_coverage` nightly, the same 5 starts | — | 0.930 / 0.916 / 0.937 / 0.934 / **0.898** (**4 / 5 pass**) |
| Held-out (PR 100 sessions) | — | all 0.966, inside 0.974, outside 0.958, world 0.947 |

**Needs a decision: world coverage sits low in the band (mean about 0.92) and failed 1 of 5 sweeps.** The world gauge has a heavier
tail than a Gaussian (P95 2.6–2.9 against 2.45). It is un-ignored because it passes on PR seeds and on most nightly runs.

## C4: late and pocketed phone rejection

**What was built**
- `ArrivalTimeLocalizer::localize` solves with every arrival and with each node left out. It keeps the solve with the lowest
  χ² + 10.83 × (nodes it does without, whether left out or rejected by the refiner).
  - So a node is dropped only when χ² falls by more than the 99.9 % point of χ²₁ for it. That is a one-outlier likelihood-ratio
    test.
  - Charging refiner rejections stops a subset that kept the late node, by rejecting two good ones, from winning.
  - The old χ²/dof > 9 trigger is gone (`node_rejection_chi_square_drop` replaces `maximum_reduced_chi_square`).
- New unit test: a 3 ms pocket on the pair reference (an edge node) is rejected. It fails on the old localizer.

**Deviations**
- A deletion-residual test (predict node i from a fit without it) was built first and dropped. Fits without a good node slip into
  another height basin and make it fire on clean data: the inside P95 error went to 3.4 m.
- **Finding:** over a flat layout, a late node near the centre is absorbed almost exactly by a higher source. A 4 ms pocket raised
  the source 8 m at χ² 1.2. No TDOA test can see that case.

**Gates (`graceful_degradation`, still ignored)**

| | PR (100) before | PR after | Nightly 400@100000 before | Nightly after |
| --- | --- | --- | --- | --- |
| Decisive pockets rejected | 21 / 35 | 33 / 35 | 101 / 173 | **159 / 173** |
| Decisive pocket dragged > 4σ (seed failures) | 11 | 4 | 40 | 13 |
| Pocketed estimate outside clean 95 % region (limit 10 %) | 35 % | 17 % | 27 % | 15.8 % |
| σ shrink > 10 % on removal (limit 5 %) | 0.160 | 0.028 | 0.187 | 0.057 |
| Few-node fixes > 4σ from truth | 2 | 5 | 9 | 17 |
| Muted phone: 6-phone fix > 4σ from the 7-phone one | 0 | 0 | 0 | 2 |

- Repro seed 26 is now rejected; before, it was dragged 22.5σ. Even so, the 6-phone fix lies 4.6σ from the 7-phone one (the rule
  is 4σ). Rejecting a critical node correctly can still leave the nested fix more than 4σ of the full fix's σ away.
- The few-node tail grew because C3's σ is honest rather than inflated.
- **Left ignored**, with the precise comment in `tests/statistical_accuracy.rs`.

## C5: successor host restores entities

**What was built**
- **`host_restoration`.** Collects the latest non-expired `BirdEntityUpdate` per (island, wire id), each island's last
  `ClockModel`, and the session's ENU origin (from the latest `LayoutModel`).
- **Seed API.** New `BirdEntityRegistry::seed_entity` and `shift_entity_times` (`bird_entity/registry_seed.rs`), plus
  `WildlifeEntityPipeline::seed_entity`.
- **`take_over`.** New `mesh/host_entity_restoration.rs`, called from `take_over`:
  - restores the old wire ids;
  - rebuilds the covariance from the contour: the vertex second moment is k²/2·Σ, divided by the inside scale², with the anchor
    share left in, so it errs wide;
  - sets `clock_anchors` from `last_heard_reference_sample_index` at takeover time;
  - at an island's first call, if the successor references a different node, moves the seeded times onto the new reference
    clock through that node's clock in the dead host's model.
- **Same ENU origin.** The successor now adopts the dead host's ENU origin. Without that, a restored entity and the new fixes were
  about 14 m apart, because each host centred its plane on a different first GPS fix.
- No wire change.

**Gate**
- New `host_scenarios::successor_restores_entities`: one bird calls at 14 and 20 s; the host dies in the following round; the bird
  calls again at 50 and 60 s. It asserts:
  - two host tenures;
  - entity outcomes under both hosts (so the check is not vacuous);
  - `entity_invariants`;
  - `duplicate_entities == 0`.
- Before: 13 / 13 PR seeds fail with one duplicate. After: 13 / 13 PR seeds pass, plus 100 / 100 extra seeds (1000–1099).

**Known limits**
- Entities are keyed by island id. A successor whose island tracker numbers the group differently starts that island empty.
- A bearing-only entity comes back as heard-nearby, because the bearing's origin position is not on the wire.
- Calls pending at the dead host are still lost.

## Regression to flag: mesh-sim nightly invariants (needs a decision)

`nightly_mesh_seeds --seeds 200 --start-seed 500000`, across all scenarios:
- Before (`3cd6f78`): 6 failures in 4 600 runs.
- After: **29 failures in 4 800 runs**:
  - `island_split_merge` 13;
  - `offline_host_loss` 10 (was 3);
  - `thermal_shedding` 2;
  - `route_change` 1;
  - `network_faults` 2 (unchanged);
  - `flock_echo` 1 (unchanged).
- Almost all new failures are "error outside reported region (4σ + 0.5 m)" on degraded-geometry calls: after a merge, a rejoin
  or a handover.
- Example: `island_split_merge` seed 500023, call 2.
  - Before: error 2.78 m with σ 1.17 m.
  - After: the same error, 2.77 m, with σ 0.25 m.
- **The error was there before. The old, roughly 4× overstated LM σ covered it by accident.**
- Most of it (13 → 1 in `island_split_merge`) comes from the phone-height σ change, 0.25 → 0.115.
- PR seeds pass.
- Pre-existing cause found on the way: in `offline_host_loss` seed 42, after the wanderers return, node 2's layout position is
  wrong by about 19 m relative to the others. This happens at `3cd6f78` too.

Options:
1. keep the honest σ and fix the post-merge, rejoin and handover layout errors (follow-up work);
2. restore 0.25 m (refit the scales), which re-hides them;
3. grow node σ after membership changes.

I kept the plan's 0.115 m.

## Python suite

`python -m pytest tests/ --ignore=tests/test_soak_harness.py` at `be52a7d`: 2083 passed, 9 skipped (baseline 2083 / 9). Phase C touches no Python and no parity golden.
