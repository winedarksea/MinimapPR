# Final review: field-mode closeout (Phases B, C, D)

Reviewer: an independent agent with fresh context. Branch `field-mode-closeout`, 20 implementer commits on main `df95986`,
plus the review commits below. Inputs: the plan, `git diff main..field-mode-closeout`, and closeout_B1/B2/C/D.md.
I treated the notes as claims and checked them against the code. Phase A files were read but not committed.

## How it was checked
- Read every non-mechanical hunk of the diff. For B5 I compared the sorted line multiset of `dsp_worker.rs` before the
  split (16c8fc4) with the union of the nine split files (802079a). The only differences are `pub(super)` visibility,
  `#[path]`/`mod`/`use` glue, module doc comments, rustfmt rewraps and three dropped banner comments.
- **Sabotage checks.** I broke the key line, ran the step's test, then restored the line:

  | Step | Line broken | Test | Result |
  | --- | --- | --- | --- |
  | B1 | removed `or math.isnan(probability)` | `test_nan_score_skipped_without_poisoning_label` | FAIL |
  | B2 | removed the own-top-label exemption | `test_own_top_label_survives_a_same_species_sibling` | FAIL |
  | B4 | kept `run_srp`/render on the parked frame | `render_on_a_parked_trigger_frame_…` | FAIL |
  | C5 | skipped `restore_entities` in `take_over` | `successor_restores_entities` | FAIL on every PR seed, 1 duplicate entity |
  | D3 | `declared_sigma_pooling_multiplier = 0` | `srp_contour_coverage` | FAIL, coverage 2/24 = 0.083 |
  | B2 seam (new test, below) | the last sibling skips the parked ones | `test_fusion_node_assembles_every_parked_sibling` | FAIL |

- **Gate thresholds, before and after.**
  - The coverage band (0.90–0.98) and the anchor and accuracy limits are unchanged.
  - The NEES gate is *tighter*. It was a mean band of [0.2, 3]. It is now a median band of [0.95, 1.85] plus a mean ≤ 3.
  - No seed special-casing.
  - No simulator ground truth changed. Two scoring frames changed, and both are justified:
    - C2 carries the world truth into the host's ENU plane through geodetic coordinates. The old diagnostic compared
      across two different planes, which was a real bug.
    - D3 tells the SRP searcher the σ that its trial is perturbed with (see finding M3).
- **Config registration.** Both new Python keys (B2 flag and timeout; B3 ratio) are in `config.py` (default, env,
  validation), `settings_store.py`, `main.py`, `config_groups.py` and `pipeline_graph.py`. `test_config_structured.py`
  passes. All new Rust constants sit in config structs:
  - `OnsetClipLocalizerConfig`
  - `FilteredMapAnchorConfig`
  - `FixCovarianceCalibrationConfig`
  - `RobustScoringConfig::declared_sigma_pooling_multiplier`
  - `ArrivalTimeLocalizer::node_rejection_chi_square_drop`
  - `MeshAttributionConfig::window_label_reconciliation_enabled`
  - `AccuracyGateConfig` NEES band

  The only bare consts are named physical or format constants (PCM16 full scale, ellipse moment factor) and
  memory bounds.
- **Flags.** No new flag defaults on. The server and mesh B2 flags both default to `false`. The Layer 2 entity scenarios
  switch the mesh flag on for themselves (`entity_scenarios.rs` `run_scene`). That is test configuration, not a default.

## Verdict per step

| Step | Verdict | Reasoning |
| --- | --- | --- |
| B1 NaN log-odds | OK | `isnan` guard as specified. The test fails without it. |
| B2 window label reconcile | OK-with-deviation | The rule matches the plan, with one sound refinement: a cluster keeps its own top label. Without it, two same-species birds in one window would strip each other's labels. Python's top label is `classification.label`; Rust uses the argmax of attributed scores. These agree for every server classifier. The parity case is real (the implementer sabotage-tested the Rust side; I sabotage-tested the Python rule). Seeds 5007/10015/10187 were added. The server gate shows no recall loss and 0 swaps. The FusionNode seam had no pytest coverage; review commit `7d72fe7` adds it. Low findings L1–L2. |
| B3 collinear mirror guard | OK | Faithful port. Applied only when every *contributing* node is single-mic: a non-contributing rigid node does not exempt the fix, which is correct. Reuses `range_asymptotic` (bearing-only, no track) rather than adding a new state, which is sound. The replay tests check the guard's own tag, which is robust. There is no master flag (by plan), so it is live at ratio 0.2 (note N1). |
| B4 render on the deferred payload | OK | The render rides `PendingLocalization` as `{end_ns, window_seconds}` and is re-extracted at release or supersede. `localization_only_template` is unchanged, which is a harmless deviation. Memory cost is two numbers per pending buffer. The test is sabotage-verified. The live per-trigger 1.0 check still needs node traffic on Bluefin (Phase A blocked). |
| B5 split | OK | Behaviour-neutral (multiset diff above). Over 500 lines: `dsp_worker.rs` 573 (plan target < 600) and `dsp_worker_coverage_race_tests.rs` 554. |
| C1 filtered map anchor | OK-with-deviation | `MapAnchorFilter` is fed every round. `reframe` is T_old∘F⁻¹; I checked `compose`/`inverse` algebra, mirror case included. The deviation, subtracting a held phone's jump rather than a lower hard-drop threshold, is measured and justified. `anchor_persistent_jump` is un-ignored. Seeds 38 and 100105 are added through a new per-gate `regression_seeds`. That is a sound reading of the seed policy, since `PULL_REQUEST_SEEDS` belongs to Layer 2. |
| C2 anchor error in polygons | OK | Σ_world = Σ_fix + σ_t²I + σ_θ²r⊥r⊥ᵀ as specified. Both callers use `published_contour`. The gate scores with the same `published_covariance` function the host draws with. The relative-frame gate still passes. |
| C3 mesh LM covariance | OK-with-deviation | Phone-height σ is 0.115, the drift is clamped, and NEES is median-banded (tighter); `mesh_contour_coverage_by_class` is un-ignored. Deviations: height marginalization reworked (Schur-complement Var(z), sound); per-geometry σ scales on the *published* contour instead of one multiplier; `CallFix` keeps the raw covariance. Each is defensible, but see H1 (mesh nightly invariant regression) and M2 (the calibration is a sim-fitted shrink). |
| C4 late-phone rejection | PROBLEM (plan gate not met, honestly reported) | Leave-one-out with a χ²₁(99.9 %) drop penalty, which counts refiner rejections. This is a proper one-outlier likelihood-ratio test, and a defensible replacement for the plan's per-node residual test. Decisive pockets rejected rose from 58 % to 92 % nightly. `graceful_degradation` is **still `#[ignore]`**; the plan said to un-ignore it. The ignore comment is precise and nothing was loosened. Cost: n+1 LM solves on every call (was 1 unless inconsistent), fine for ≤ 8 phones. |
| C5 successor restores entities | OK | Keeps old wire ids. Clock anchors come from `last_heard_reference_sample_index`, rebased onto a new reference node at the first call. Covariance is rebuilt from the contour, with no wire change. The successor also adopts the dead host's ENU origin, a needed extra. The scenario is sabotage-verified. See M1 on the rebuilt covariance. |
| D1 OnsetClipSetLocalizer | OK-with-deviation | Decode, then a clock-mapped windowed-sinc resample. I checked the mapping: rate ratio 1/(1−ε) anchored at the first sample, and the unit test compares it with `NodeClockMapping` directly. Then onset clustering, LM with per-pair σ (25 µs floor plus both clocks), χ²/dof inflation, attribution evidence moved to 48 kHz reference samples, and a classifier callback. Two deviations, both sound and driven by D2 measurements: a window fails only when no cluster fixes, and seed-only clusters are kept. Also an audibility gate at 6 dB, below the 10 dB trigger, so faint long-range calls are not blocked. Review commit `50f8da5` hardens the decode cache key. |
| D2 rendered-audio Layer 2 mode | OK | Opt-in (`AcousticMode::RenderedAudio`), 7 ignored nightly tests, about 2.5 s per run. Only clip audio is rendered; sync, Stage-1 and the classifier stay on arrival times, which is scoped and documented. Scoring uses truth only in the harness. The results (multi-bird failures) are a finding, not an implementation defect. See the claim check below. |
| D3 σ-aware SRP scoring | OK | Exact scores max-pool over ±k·σ_pair; envelope intervals widen by the same amount; intra-node pairs get 0; σ_pair uses a correct conservative bound. k = 1.25 was chosen on training seeds and T refit to 0.0013. `srp_contour_coverage` is un-ignored and fails without pooling (sabotage-verified). See M3: no production caller declares σ. |

## Claim check: "the server would behave the same on phone layouts" (closeout_D.md, D2)

**Partly true. As worded, it invites the wrong inference that tetra-array servers share the multi-bird failure.**

**What is true.** The algorithm is shared. `localize_onset_clusters` is a parity port, frame assembly is the same
(`onset_srp_clustering.py:100-144` vs `onset_clustering/frame_assembly.rs`), and both allow envelope correlation on
baselines longer than a compact array. A server site made only of single-mic/POINT nodes would show the same behaviour.
Such a site also cannot classify at the default `beamformed_classification_min_sensor_count=2`.

**Why tetra servers differ:**
- **Intra-node pairs.** Each onset frame is built from the template node's reference sensor against *every* other
  sensor. On a tetra that includes the node's own 3 sibling mics: 5 cm baselines, with a maximum lag of about 0.15 ms.
  That is below a carrier fringe period up to about 3 kHz, so those pairs give an unambiguous bearing.
- **Their weight share.** `cap_intra_node_weight_share` (`onset_srp_clustering.py:143`, 0.35) keeps up to 35 % of the
  frame weight on those pairs. That bearing excludes the off-axis fringe basins of the widely spaced inter-node pairs.
- **What phones lack.** A phone frame has *only* inter-node pairs, so nothing anchors which fringe is right. That
  matches the D2 dumps: 12–60 m wrong basins confirmed by LM with a small σ.

**Empirical support.** The server gate's multi-bird tetra scenes show no such failure (closeout_B2.md table): 0 swaps,
0 duplicates, 0 merges, sub-metre medians on `onset_overlap_two_species`, `onset_same_band_pair`,
`onset_same_band_two_species` and `gate_countersinging_pair`.

**Conclusion.** The decision D2 asks for (clip SRP, wideband/envelope inter-node frames, or arrival-time picks) concerns
**single-mic meshes**. It is not evidence against the server's tetra pipeline.

## Findings, by severity

### High
- **H1. Mesh-sim nightly invariant failures rose from 6/4600 to 29/4800 after C3** (closeout_C.md; `config.rs:231`
  `assumed_phone_height_sigma_m: 0.115`).
  - Almost all are "error outside reported region (4σ + 0.5 m)" after a merge, rejoin or handover.
  - The error was there before; the overstated σ hid it.
  - PR seeds pass, so CI does not see it. This is honest σ exposing real layout error in degraded geometry, but on real
    phones it means overconfident fixes right after membership changes.
  - **Needs a user decision**, one of the three options in closeout_C.md. Preferred: fix the layout errors, or grow
    node σ after membership changes. Restoring 0.25 m would only re-hide it.

### Medium
- **M1. C5 rebuilds outside-layout entity covariance too tight** (`host_entity_restoration.rs:59`).
  - `covariance_from_contour` always divides by the *inside* scale² (0.695²). An entity published outside the layout
    used the outside scale (0.315²), so its fix share comes back at about 0.2× the raw variance (σ about 2.2× too small).
    The old comment claimed it "errs wide"; review commit `ea6893b` corrects the comment.
  - Risk: a restored far bird's next fix may fail to associate, which brings back the duplicate C5 exists to prevent.
    The scenario's bird sits inside the ring, so the test does not see this.
  - Options: divide by `min(inside, outside)²` (errs wide always, at some merge risk inside), or classify by the
    restored position against the first post-takeover layout. Not changed: it is a behaviour choice.
- **M2. The C3 published contour is a sim-fitted shrink.**
  - Inside 0.695 and outside 0.315 multiply the raw LM σ *down* (`config.rs:323`), fitted on Layer 2 noise models.
  - If real phones have more error than the sim (canopy multipath on clocks, speaker/mic bias, the deferred
    speaker→mic clock bias), polygons will under-cover.
  - Trackers still use the raw σ, about 1.4–3× overstated, so the published polygon and the tracker disagree about the
    same fix.
  - Recommend recalibrating against the first real phone sessions before relying on polygon coverage. Longer term,
    remove the scales by fixing the σ model they stand in for.
- **M3. D3 is calibrated for declared σ, which no production code declares.**
  - `LayoutSrpSearcher` has no production caller today; only gates, tests and benches use it. Every
    `LayoutNode { … }` outside the gate's trials passes `position_std_m = 0`, `clock_offset_std_s = 0`.
  - At σ = 0, pooling is off. With the new T = 0.0013 (`srp_layout/config.rs:120`), coverage collapses to 2/24
    (measured), worse than the old 0.002.
  - The gate also declares exactly the σ it perturbs with (`srp_trials.rs:76`), an optimistic oracle.
  - When a host wires `srp_layout`, it must pass the island's real layout and clock σ, or refit T.
  - Recommend a constructor that requires σ, or a debug assertion that inter-node pairs declare σ > 0, so this cannot
    be forgotten.

### Low
- **L1. Parked B2 siblings never record a chunk-dispatch outcome** (`fusion_node.py:1692`).
  - The worker `continue`s before `_record_chunk_dispatch_outcome`. When the sibling is later assembled (by the last
    sibling or the timeout task), no outcome is recorded for it.
  - Effect: a BirdNET chunk retry is never re-enabled from a held, non-actionable sibling. That is conservative (fewer
    retries) and only applies with chunking plus the B2 flag on.
- **L2. Held B2 siblings are assembled inside the last sibling's classification timeout** (`fusion_node.py:2198`).
  - A timeout or exception partway through loses the remaining held siblings, because their window is already finished.
  - The window is bounded and rare, but there is no per-sibling isolation, unlike `_emit_released_reconciliation_window`,
    which catches exceptions per sibling.
- **L3. The D2 thread-local overrides are not restored on panic** (`scenarios/mod.rs:74-95`).
  - A panicking audio-mode run leaves `ACOUSTIC_MODE_OVERRIDE` set for later runs on that test thread.
  - Harness-only. A drop guard would fix it.
- **L4. D3 nightly over-coverage fails in about 1 of 5 sweeps** (0.985 > 0.98 at n = 200). This is sampling spread
  around a fit that lands exactly on 0.945; documented in closeout_D.md.
- **L5. C1 `anchor_persistent_jump` flakes at about 1.7 seeds per 1000** (about 30 % of 200-seed nightly runs fail).
  Documented decision in closeout_C.md.
- **L6. C3 `mesh_contour_world_coverage` sits low in its band** (mean about 0.92; failed 1 of 5 nightly sweeps at
  0.898). Documented.

### Notes (no change recommended)
- **N1.** B3 has no master flag, by plan: the ratio key set to 0 is the off switch. It changes behaviour only for fixes
  whose contributing nodes are *all* single-mic and near one line. Those become bearing rays instead of points that may
  be mirror images. That is the safe direction, and a single-mic-only site cannot classify by default anyway. I see no
  problem.
- **N2.** The C1–C5 mesh changes (filtered anchor, calibration, LOO localizer, entity restoration) are unflagged
  behaviour changes in `minimappr-core`'s mesh. The plan asked for no flags, and the mesh has no deployed shell.
- **N3.** B4 frame-0 caveat (pre-existing): a never-streamed cluster member empties the omni render.
- **N4.** `fusion_node.py` grows to 3864 lines (+109). It was already far over the AGENTS.md 500-line target; B2 split
  its own code into helpers but kept them in the class.

## Review commits
| Commit | What |
| --- | --- |
| `7d72fe7` | `tests/test_onset_window_label_reconciliation.py::test_fusion_node_assembles_every_parked_sibling`: replays `onset_overlap_two_species` with the B2 flag on and asserts every parked sibling is later assembled. It is sabotage-verified, and it is the first pytest coverage of the FusionNode hold/release seam. |
| `50f8da5` | The `OnsetClipSetLocalizer` aligned-window cache is keyed on (call id, island, window centre/half-width, clip nodes) instead of the call id alone, so a localizer shared across host tenures cannot reuse a stale decode. |
| `ea6893b` | Corrects the C5 `covariance_from_contour` comment (M1); comment only. |

## Recommended, not changed
1. H1: choose a C3 follow-up for degraded-geometry σ after membership changes.
2. M1: decide the restored-covariance scale for outside-layout entities.
3. M2: plan a real-session recalibration of the contour scales.
4. M3: make declared σ mandatory when `srp_layout` gets a production caller.
5. C4: `graceful_degradation` stays ignored. The remaining failures (1–3 σ delays absorbed by height over flat layouts;
   nested-fix 4σ rule on critical nodes) need a gate-semantics or geometry decision, not more tuning.
6. L1/L2: if the B2 flag goes on in the live trial, record chunk outcomes for released siblings and isolate per-sibling
   failures in the last-sibling path.
7. Wording for Phase E: say D2's failure applies to single-mic meshes, not to the tetra server (claim check above).

## Suites (branch head, with the review commits)
- `cargo test --workspace`: **581 passed, 0 failed, 8 ignored** (as expected: `graceful_degradation` + 7 opt-in audio entity scenarios).
- `pytest tests/ --ignore=tests/test_soak_harness.py`: **2084 passed, 9 skipped** (2083 expected + the 1 review test).
